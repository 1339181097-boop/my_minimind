🧠 MiniMind: 极简大模型全生命周期实战 (Pretrain -> SFT -> DPO)

本项目实现了一个拥有 25M (0.025B) 参数的极简微型大语言模型（MiniMind）的完整训练闭环。项目在单张 RTX 4090 (24GB) 上完成了从“词表初始化 -> 预训练 -> 指令微调（短文本+长文本） -> 偏好对齐 (DPO)”的全阶段流程，并对超参数与显存管理进行了深度调优。

🚀 项目亮点

全栈闭环训练：完整复现了现代 LLM 的训练基线（Pipeline），涵盖 Pretrain、SFT (512 & 2048 长度) 以及 DPO 阶段。

极限硬件压榨：在 24GB 物理显存的限制下，通过动态调整 batch_size 与 accumulation_steps，成功攻克 2048 长文本微调阶段的显存分页（Memory Paging）降速陷阱，找到了 4090 的最佳“甜点位”。

全流程可视化：集成了 SwanLab 进行实验跟踪，包含 Loss、Learning Rate 及 DPO 阶段特有的 Reward Margin 等多维度指标可视化。

🛠️ 硬件与环境配置

核心组件

配置详情

GPU

1x NVIDIA GeForce RTX 4090 (24GB VRAM)

环境

AutoDL, PyTorch 2.x, CUDA 12.x

监控工具

SwanLab, nvtop

模型参数量

~25.83M (千万级极小体量)

📈 训练阶段解析与可视化

阶段一：预训练 (Pre-training)

目标：让模型掌握基础的语言规律和词汇关系，从“白痴”变成“懂语法的复读机”。

参数：seq_len=512, batch_size=128, lr=5e-4

表现：模型快速收敛，Loss 从最初的 ~7.0 断崖式下降，最终稳定在 2.2 左右。

(注：请将预训练的图片重命名为 pretrain_loss.png 并放在 images/ 目录下)

阶段二：指令微调 - 短文本 (SFT-512)

目标：教会模型听懂人类的指令格式，学会问答和对话交互。

参数：seq_len=512, batch_size=128, accumulation_steps=2, epochs=2

表现：基于 680 万条数据，采用等效 Batch Size 256 进行微调，Loss 平滑下降，初步具备指令跟随能力。

(注：请将 SFT 512 的图片重命名为 sft_512_loss.png 并放在 images/ 目录下)

阶段三：指令微调 - 长文本挑战 (SFT-2048)

目标：突破上下文窗口限制，提升模型对复杂长文的理解与生成能力。

性能调优记录：在 4090 上直接拉大 Batch 会导致显存借用系统 RAM（引发断崖式降速）。经测试，将组合设为 batch_size=16 + accumulation_steps=16，成功将显存占用控制在 11.9 GB，保持 100% GPU 利用率并实现了最快 Epoch 耗时。

表现：长文本数据方差极大，Loss 呈现早期高频震荡（2.5 ~ 3.3 之间），随后逐步收敛。

阶段四：偏好对齐 (DPO - Direct Preference Optimization)

目标：通过人类偏好数据（Chosen vs Rejected），赋予模型“高情商”，抑制有害或敷衍的回答。

参数：seq_len=512, batch_size=8, accumulation_steps=8, epochs=1

表现：DPO 阶段算力消耗翻倍，退回 512 长度单轮训练以防过拟合（Reward Hacking）。Loss 曲线呈现特有形态（约在 0.69 震荡），模型成功实现价值观对齐。

(注：请将 DPO 的图片重命名为 dpo_loss.png 并放在 images/ 目录下)

💻 快速复现 (Quick Start)

项目中所有阶段均采用模块化脚本启动，以下为各阶段满血运行的核心指令示例：

1. 预训练 (Pretrain)

python -m trainer.trainer_pretrain \
    --data_path dataset/pretrain.jsonl \
    --save_dir out \
    --max_seq_len 512 \
    --batch_size 128 \
    --use_wandb


2. 核心微调 (SFT 2048甜点位)

python -m trainer.train_full_sft \
    --from_weight sft_512 \
    --data_path dataset/sft_2048.jsonl \
    --save_dir out \
    --save_weight sft_2048 \
    --max_seq_len 2048 \
    --batch_size 16 \
    --accumulation_steps 16 \
    --epochs 1 \
    --use_wandb


3. 后台挂机指南 (针对云服务器)
为防止网络中断导致训练失败，推荐使用 Linux 原生进程挂起大法：

启动训练后按 Ctrl + Z 暂停进程。

输入 bg 将进程放入后台。

输入 disown 剥离终端绑定，即可安心关机。

📜 结语

本项目证明了即使是极小参数量（25M）的模型，在严谨的数据工程和合理的超参数配置下，也能走通当前最前沿的 LLM 训练范式，展现出大模型“麻雀虽小，五脏俱全”的基础能力。