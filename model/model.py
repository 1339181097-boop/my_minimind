from transformers import PretrainedConfig, PreTrainedModel,GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class CaveManMindConfig(PretrainedConfig):
    model_type = "cavemanmind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        intermediate_size: int = None, # pyright: ignore[reportargsumentType] # type: ignore
        hidden_size: int = 512,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.head_dims = hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )





import torch
from torch import nn
from typing import Optional, Tuple, List, Union
from transformers.activations import ACT2FN
import torch.nn.functional as F
import math
class RMSnorm(nn.Module):
    def __init__(self,dim: int, eps: float = 1e-5) -> None:
        super().__init__()    
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))  # weight是可学习的参数
                                                        # 要用nn.Parametes定义
    def _norm(self,x):
        return torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*x
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)  # 半精度训练容易溢出
                                                            # float32全精度更稳定，但结束后记得返回float16


# 提前用非常长的end计算好cos and sin,然后切片使用
def pre_compute_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    # 先计算角频率0-dim//2个
    freqs=1.0/(rope_base**torch.arange(0,dim,2).float()/dim)
    seq_idx=torch.arange(0,end,device=freqs.device)
    # 计算corr_dim
    if rope_scaling is not None:
        original_max,beta_fast,beta_slow,factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1),
            rope_scaling.get("factor", 4)
        )

        corr_dim=next((i for i in range(0,dim//2) if 2*torch.pi/freqs[i]>original_max),dim//2)
        
        # 计算scale
        power=torch.arange(0,dim//2,device=freqs.device).float()/max(dim//2,1)
        beta=beta_slow+(beta_fast-beta_slow)*power
        scale=torch.where(
            torch.arange(0,dim//2,device=freqs.device)<corr_dim,
            (beta*factor-beta+1)/(beta*factor),
            1.0/factor
        )
        # 角度乘上scale计算后续
        freqs=freqs*scale
    idx_freqs=torch.outer(seq_idx,freqs)
    idx_freqs2=torch.cat([idx_freqs,idx_freqs],dim=1)
    freqs_cos=idx_freqs2.cos().unsqueeze(0).unsqueeze(2)
    freqs_sin=idx_freqs2.sin().unsqueeze(0).unsqueeze(2)
    return freqs_cos,freqs_sin

def rota_half(x):
    dim=x.shape[-1]
    x1=x[...,:dim//2]
    x2=x[...,dim//2:]
    return torch.cat([-x2,x1],dim=-1)
# 进来的是分完头（batch_size,seq_len,num_heads,head_dim)
def apply_rotary_pos_emb(q,k,freqs_cos,freqs_sin):
    q_embed=q*freqs_cos+rota_half(q)*freqs_sin
    k_embed=k*freqs_cos+rota_half(k)*freqs_sin
    return q_embed,k_embed



# 将1/n_rep的kv重复n_rep次
def repeat_kv(x,n_rep):
    if n_rep==1:
        return x
    else:   # 可以用expend和reshape优化计算速度
        x=torch.repeat_interleave(x,n_rep,dim=2)
        return x
    
# GQA

class Attention(nn.Module):
    def __init__(self,args:CaveManMindConfig) -> None:
        super().__init__()
        self.q_heads=args.num_attention_heads
        self.k_v_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads%args.num_key_value_heads==0
        self.head_dims=args.hidden_size//args.num_attention_heads
        self.n_rep=args.num_attention_heads//args.num_key_value_heads

        self.w_q=nn.Linear(args.hidden_size,args.hidden_size,bias=False)
        self.w_k=nn.Linear(args.hidden_size,args.head_dims*args.num_key_value_heads,bias=False)
        self.w_v=nn.Linear(args.hidden_size,args.head_dims*args.num_key_value_heads,bias=False)
        self.w_o=nn.Linear(args.hidden_size,args.hidden_size,bias=False)

        self.dropout=args.dropout
        self.attn_dropout=nn.Dropout(args.dropout)
        self.restnet_dropout=nn.Dropout(args.dropout)
        self.flash=hasattr(torch.nn.functional,'scaled_dot_product_attention')and args.flash_attention
     
     # 进来的是经过embedding层的x（batch_size,seq_len,hidden_size)
    def forward(self,X,
                position_embeddings,
                past_kv=None,
                use_cache=False,
                attention_mask=None):
        # 投影qkv
        Q,K,V=self.w_q(X),self.w_k(X),self.w_v(X)
        batch_size,seq_len,_=X.shape
        # 分头
        q=Q.reshape(batch_size,seq_len,self.q_heads,self.head_dims)
        k=K.reshape(batch_size,seq_len,self.k_v_heads,self.head_dims)
        v=V.reshape(batch_size,seq_len,self.k_v_heads,self.head_dims)
        # q，k位置编码（先取出全部长度的cos，sin）RoPE 必须在 KV Cache 更新之前做
        cos,sin=position_embeddings
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        # kv缓存
        if past_kv is not None:
            past_k,past_v=past_kv
            k=torch.cat([past_k,k],dim=1)   # 拼接k，v的序列长度
            v=torch.cat([past_v,v],dim=1)
        # 关键修正：无论是否拼接，只要开启 use_cache，就要更新 past_kv
        past_kv=(k,v) if use_cache else None
        # 转置并重复k，v n_rep次
        q=q.transpose(1,2)
        k=repeat_kv(k,self.n_rep).transpose(1,2)
        v=repeat_kv(v,self.n_rep).transpose(1,2)
        # 计算注意力（因果掩码必须写q,k,v shape均为(batch_size,num_heads,seq_len,head_dim)
        if self.flash and (seq_len > 1) and (past_kv is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dims)
        # 因果掩码（需要自己生成）
            if seq_len>1:          # 二维即可，从右向左广播
                tri_mask=torch.triu(torch.ones(seq_len,seq_len,device=X.device),diagonal=1)
                # masked_fill 是 Tensor类自带的从成员方法（True表示要掩码的位置）
                scores=scores.masked_fill(tri_mask.bool(),float('-inf'))
            # padding掩码(attention_mask是二维的，但在四维的scores中全都一样，直接复制)
            if attention_mask is not None:
                extended_mask=attention_mask.unsqueeze(1).unsqueeze(2) # 给pad的地方全部变成-inf
                scores+=(1.0-extended_mask)*(float('-inf'))
            attention_weights=F.softmax(scores,dim=-1)
            attention_weights=self.attn_dropout(attention_weights)
            output=torch.matmul(attention_weights,v)
    # FlashAttn 和 手动计算都得到了 (b, h, s, d)，统一处理
        # 合并头部
        output=output.transpose(1,2).contiguous().reshape(batch_size,seq_len,-1)
        output=self.w_o(output)
        return self.restnet_dropout(output),past_kv



class FeedForward(nn.Module):
    def __init__(self, args:CaveManMindConfig) -> None:
        super().__init__()
        self.hidden_size=args.hidden_size
        if args.intermediate_size is None:
            intermediate_size=int(8*args.hidden_size/3)
            args.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.dropout=nn.Dropout(args.dropout)
        self.act_fn=ACT2FN[args.hidden_act]  
          # ACT2FN是transformers里定义的激活函数映射表
    def forward(self,X):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(X))*self.up_proj(X)))





# 拼接transformerblock
class CaveManMindBlock(nn.Module):
    def __init__(self,layer_id:int,config:CaveManMindConfig) -> None:
        super().__init__()
        self.config=config
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim=config.hidden_size//config.num_attention_heads
        self.self_attn=Attention(config)
        self.layer_id=layer_id
        self.input_layernorm=RMSnorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm=RMSnorm(config.hidden_size,config.rms_norm_eps)
        self.mlp=FeedForward(config)
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # [关键点1] 残差连接 进入transformer的输入
        residual = hidden_states
        # [关键点2] Pre-Norm (前置归一化)
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
        past_key_value, use_cache, attention_mask

        )
        hidden_states = residual + hidden_states   # x+Attention(x)


        # 第二步：MLP (FeedForward) 子层

        norm_x = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(norm_x)
        hidden_states = hidden_states + mlp_out  # x + MLP(x)

        return hidden_states, present_key_value
    

class CaveManMindModel(nn.Module):
    def __init__(self,config:CaveManMindConfig) -> None:
        super().__init__()
        self.config=config
        self.vocab_size,self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers
        )
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)
        self.layers=nn.ModuleList(
            [CaveManMindBlock(i,config) for i in range(self.num_hidden_layers)]
        )
        self.norm=RMSnorm(config.hidden_size,config.rms_norm_eps)
    # rope预计算（计算整个长度的cos和sin）
        freqs_cos,freqs_sin=pre_compute_cis(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        # 注册参数的缓冲区
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargss   # 兼容性
                ):
        batch_size,seq_len=input_ids.shape # type: ignore
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers) # type: ignore
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0 # type: ignore
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # 计算位置编码切片（从已经计算好的非常长的cos和sin中切片）
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_len], # type: ignore
            self.freqs_sin[start_pos:start_pos + seq_len] # type: ignore
        )
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)): # type: ignore
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if hasattr(l.mlp, 'aux_loss')], # type: ignore
            torch.tensor(0.0, device=hidden_states.device)
        ) # pyright: ignore[reportCallIssue]
        return hidden_states, presents, aux_loss


# 封装与huggingface的接口
class CaveManMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = CaveManMindConfig

    def __init__(self, config: CaveManMindConfig = None): # type: ignore
        self.config = config or CaveManMindConfig()
        super().__init__(self.config)
        self.model = CaveManMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享（省显存，语义对齐）
        self.model.embed_tokens.weight = self.lm_head.weight
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **kargs):
        hidden_states, past_key_values,aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kargs
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss=None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous() # 扔掉最后一个
            shift_labels = labels[:, 1:].contiguous()    # 扔掉第一个
            logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            labels_flat=shift_labels.view(-1)
         # CrossEntropyLoss：输入 (Input): (N, C) —— N 个样本，C 个类别，
         # 目标 (Target): (N) —— N 个正确答案
            loss=nn.functional.cross_entropy(logits_flat,labels_flat,ignore_index=-100)
        # 如果你有 MoE，这个 loss 就不为 0，需要加到总 loss 里去优化
            if aux_loss is not None:
                loss += aux_loss


        return CausalLMOutputWithPast(  
                                        loss=loss, # pyright: ignore[reportArgumentType]
                                        logits=logits,
                                        past_key_values=past_key_values, # type: ignore
                                        hidden_states=hidden_states) 
 











































































































