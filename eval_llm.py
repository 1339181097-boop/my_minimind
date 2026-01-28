import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model import CaveManMindConfig, CaveManMindForCausalLM
from trainer.trainer_utils import setup_seed
from transformers import PreTrainedTokenizerFast
warnings.filterwarnings('ignore')

def init_model(args):
    # 1. 加载分词器
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<s>"
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    # ------------------ 诊断信息打印 ------------------
    real_vocab_size = len(tokenizer)+8
    print(f"\n[诊断] 分词器真实词表大小: {real_vocab_size}")
    print(f"[诊断] 默认配置词表大小: 6400")
    
    # 2. 动态初始化模型（关键：强制使用真实大小）
    config = CaveManMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        vocab_size=real_vocab_size  # <--- 强制扩容
    )
    model = CaveManMindForCausalLM(config)
    print(f"[诊断] 模型已初始化，Embedding层大小: {model.model.embed_tokens.weight.shape}")

    # 3. 智能加载权重（自动填充缺失的 embedding）
    ckp = f'./out/{args.load_mark}'
    print(f"[诊断] 正在加载权重: {ckp}")
    state_dict = torch.load(ckp, map_location=args.device)
    
    # === 自动外科手术：如果权重里的词表小于分词器，自动补 0 ===
    ckpt_vocab_size = state_dict['model.embed_tokens.weight'].shape[0]
    if ckpt_vocab_size < real_vocab_size:
        print(f"[警告] 权重词表({ckpt_vocab_size}) 小于 分词器({real_vocab_size})，正在自动补全...")
        pad_size = real_vocab_size - ckpt_vocab_size
        
        # 1. 补全 embed_tokens
        old_embed = state_dict['model.embed_tokens.weight']
        # 生成很小的随机数填充，避免 NaN
        pad_embed = torch.randn(pad_size, old_embed.shape[1], device=args.device) * 0.01 
        new_embed = torch.cat([old_embed, pad_embed], dim=0)
        state_dict['model.embed_tokens.weight'] = new_embed
        
        # 2. 补全 lm_head (如果存在且未绑定)
        if 'lm_head.weight' in state_dict:
            old_head = state_dict['lm_head.weight']
            # 输出层通常补 0 比较安全，避免预测出奇怪的词
            pad_head = torch.zeros(pad_size, old_head.shape[1], device=args.device)
            new_head = torch.cat([old_head, pad_head], dim=0)
            state_dict['lm_head.weight'] = new_head
            
    # ========================================================

    model.load_state_dict(state_dict, strict=False)
    return model.to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind模型推理与对话")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    
    # 👇👇👇 【必须补上这一行】 👇👇👇
    parser.add_argument('--load_mark', default='pretrain_512.pth', type=str, help="加载权重的具体文件名")
    # 👆👆👆
    
    parser.add_argument('--weight', default='pretrain', type=str, help="权重名称前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE外推")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值")
    parser.add_argument('--historys', default=0, type=int, help="历史对话轮数")
    parser.add_argument('--show_speed', default=1, type=int, help="显示速度")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--tokenizer_path', type=str, default='./model/tokenizer.json', help='分词器路径')
    
    args = parser.parse_args()
    
    prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    
    # 诊断通过后，开始交互
    print(f"\n[系统] 模型加载完成，当前运行设备: {args.device}")
    
    input_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(2026)
        if input_mode == 0: print(f'💬: {prompt}')
        
        # 处理历史对话
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})

        templates = {"conversation": conversation, "tokenize": False, "add_generation_prompt": True}
        if args.weight == 'reason': templates["enable_thinking"] = True 
        
        # 预训练模型（pretrain）没有对话模板，直接拼接
        # 如果 bos_token 是 None，就用空字符串代替，防止报错
        bos = tokenizer.bos_token if tokenizer.bos_token else ""
        inputs = tokenizer.apply_chat_template(**templates) if args.weight != 'pretrain' else (bos + prompt)
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🤖: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1.0
        )
        
        # 解码并打印速度
        gen_len = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_len / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')


if __name__ == "__main__":
    main()