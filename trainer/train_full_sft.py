import os
import sys
import time
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_base import CaveManMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler


__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time() # 记录本epoch开始时间
    # 1. 遍历 DataLoader
    # loader: 之前做好的那个dataset及其dataloader，像传送带一样源源不断送入 batch 数据
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 2. 数据搬运：CPU -> GPU
        # 张量必须在同一个设备上才能运算
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 3. 动态调整学习率 (Learning Rate Schedule)
        # get_lr 是一个余弦退火（Cosine Annealing）函数
        # 逻辑：训练刚开始步长（增益）大，快速收敛；后期步长小，精细微调。
        # 信号处理类比：这就好比 LMS 算法中的变步长因子 \mu。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        
        # 将计算好的 lr 应用到优化器组中
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 4. 混合精度上下文 (Mixed Precision)
        # autocast_ctx: 自动将部分运算转为 FP16（半精度），以节省显存并加速，
        # 同时保持关键部分（如 Softmax）在 FP32 以维持精度。
        with autocast_ctx:
            # 5. 前向传播 (Forward Pass)
            # 这里的 model 就是 MiniMind，输入 input_ids，同时传入 labels 计算 Loss。
            # res (Result) 包含了 logits, loss 等信息
            res = model(input_ids, labels=labels)
            
            # 6. 损失聚合
            # aux_loss (Auxiliary Loss): 这是 Mixture of Experts (MoE) 架构特有的负载均衡损失。
            # 如果你跑的是普通 Dense 模型，这个值为 0。
            loss = res.loss + res.aux_loss
            
            # 7. 梯度累积 (Gradient Accumulation)
            # 这是一个显存优化技巧。如果显存不够报错（OOM），不要急着减小模型，先减小 batch_size，同时增大 accumulation_steps。保持 batch_size * accumulation_steps （即等效 Batch Size）在 64~128 左右，这样训练才稳定
            # 假设显卡只能塞下 Batch Size=4，但你想跑 Batch Size=64 的效果，
            # 你就累积 16 次 (16*4=64) 的梯度，然后再更新一次参数。
            loss = loss / args.accumulation_steps

        # 8. 反向传播 (Backward Pass)
        # scaler.scale: 混合精度训练中防止梯度下溢（Underflow），先将 Loss 放大。
        scaler.scale(loss).backward()

        # 9. 只有当累积够了步数，才真正更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # unscale_: 将放大的梯度还原
            scaler.unscale_(optimizer)
            
            # 10. 梯度裁剪 (Gradient Clipping)
            # 关键！防止“梯度爆炸”。SFT 数据质量参差不齐。如果遇到某条“脏数据”导致 Loss 突然飙升，梯度裁剪能救你一命，防止之前训练的成果被一次巨大的梯度更新给破坏掉。
            # 信号处理类比：限幅器（Limiter），防止误差信号过大把系统搞震荡了。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 11. 迈出一步 (Update Weights)
            scaler.step(optimizer)
            scaler.update() # 更新 scaler 的缩放因子

            # 12. 梯度清零
            # set_to_none=True 比 .zero_grad() 稍微快一点，直接释放内存
            optimizer.zero_grad(set_to_none=True)


        # 13. 日志打印
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原真实的 Loss数值（因为之前为了累积除过了）
            current_loss = loss.item() * args.accumulation_steps
            # 剥离出 logits_loss (预测下一个词的准确度) 和 aux_loss (专家负载均衡度)
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            # ... (计算剩余时间，打印 logs)
            Logger(f'Epoch:[...] ...')
            if wandb: wandb.log({...})

        # 14. 模型保存 (Checkpointing)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval() # 切换到评估模式（关闭 Dropout 等）
            
            # 兼容 MoE 命名
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 15. DDP 解包
            # 如果用了多卡分布式训练 (DDP)，模型被包裹了一层，需要用 .module 取出来
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model) # 兼容 torch.compile
            
            state_dict = raw_model.state_dict()
            
            # 16. 保存权重
            # .half().cpu(): 转成半精度并移动到 CPU 内存，减小保存文件的大小（pth文件更小）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            # 还会调用 lm_checkpoint 保存 optimizer 状态，以便断点续训
            lm_checkpoint(..., scaler=scaler)
            
            model.train() # 保存完切回训练模式
            del state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CaveManMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 检查是否是多卡并行（DDP）。如果是单卡运行，local_rank 就是 0。
    local_rank = init_distributed_mode()
    
    # 绑定设备，你的 4090 就是 cuda:0
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    
    # 固定随机种子 (Random Seed)
    # 通信类比：这就像设定伪随机序列发生器的初始状态，确保你的实验结果是可复现的。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))


    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True) # 创建输出目录
    
    # 实例化模型配置图纸（此时还没有权重，只是空壳配置）
    lm_config = CaveManMindConfig(hidden_size=args.hidden_size,  num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    
    # 自动断点续训逻辑 (Resume Logic)
    # 如果 args.from_resume=1，它会去检查目录下有没有上次没跑完的 checkpoints。
    ckp_data = lm_checkpoint(...) if args.from_resume==1 else None


    # ========== 3. 设置混合精度 ==========
    # 这一步是为了后面 with autocast_ctx: 做准备
    # 如果检测到是 bfloat16，就创建一个 bfloat16 的上下文环境。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)




    # ========== 5. 定义模型、数据、优化器 ==========
    # A. 初始化模型 (Init Model)
    # 这一行调用 init_model，加载了 'pretrain' 的权重文件。
    # 此时，模型的大脑里已经有了预训练的知识。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # B. 编译加速 (Torch Compile)
    # PyTorch 2.0 的黑科技，类似 JIT 编译，把动态图变成静态图优化，推理/训练速度提升 20%~50%。
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        
    # C. 加载数据 (Load Dataset)
    # 实例化了我们刚才讲的 SFTDataset。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # D. 分布式采样器 (Sampler)
    # 如果是多卡，需要把数据切分给不同的卡。单卡则为 None。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # E. 梯度缩放器 (Scaler)
    # 配合混合精度使用，防止梯度太小消失。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # F. 优化器 (Optimizer)
    # 使用 AdamW，这是 LLM 训练的标准配置。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)




    # ========== 6. 从ckp恢复状态 ==========
    # 如果之前意外中断了（比如 AutoDL 实例关机了），这里会把模型权重、优化器步数全加载回来。
    # 就像玩游戏读档一样。
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)


    # ========== 7. DDP包模型 ==========
    # 如果是多卡训练，需要用 DistributedDataParallel 把模型包起来。
    # 它负责在反向传播时，把所有显卡的梯度求平均（All-Reduce），实现同步更新。
    if dist.is_initialized():
        # 忽略 RoPE 的频率缓存，不进行广播
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"} # type: ignore
        model = DistributedDataParallel(model, device_ids=[local_rank])


    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 每一轮 Epoch 开始前，设置 sampler 的 epoch，保证随机数洗牌不同
        train_sampler and train_sampler.set_epoch(epoch)
        
        # 这里的逻辑主要是为了处理“断点续训”时的数据对齐问题
        # 如果是从第 100 步中断的，下次启动要跳过前 100 条数据。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        # 构造 DataLoader
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        # === 启动引擎 ===
        # 调用我们刚才细讲的 train_epoch 函数
        if skip > 0: 
            # 如果是续训，告诉它这轮只有剩下的一点数据了
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            # 正常训练
            train_epoch(epoch, loader, len(loader), 0, wandb)