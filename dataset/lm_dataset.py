from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples) # type: ignore

    def __getitem__(self, index):
        sample = self.samples[index] # type: ignore
        # 手动添加特殊符号bos，eos
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # pad为还没到达最大长度的部分
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # label相较于input_ids shift一位在model里进行
        labels = input_ids.clone()
        # pad填上-100，计算loss时自动忽略
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer  # 传入的分词器，用于将文本转为数字
        self.max_length = max_length # 序列最大长度，超过截断，不足补0
        # 加载数据：从jsonl文件中读取训练数据，split='train'表示读取训练集
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        
        # === 核心逻辑：定义“信号提取”的特征码 ===
        # bos_id: 并不是简单的bos_token，而是构造了 '<s>assistant\n' 这样的序列
        # 这意味着：模型看到这个标志，就知道“轮到我（助手）说话了”，这是计算Loss的起点特征。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        
        # eos_id: 构造 '</s>\n' 这样的序列
        # 这意味着：模型看到这个标志，就知道“这句话说完了”，这是计算Loss的终点特征。
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples) # type: ignore
    def create_chat_prompt(self, cs):
        # cs 是 conversation 列表，例如 [{'role': 'user', 'content': '...'}, {'role': 'assistant', ...}]
        messages = cs.copy()
        # 检查是否有 function calling（工具调用）的定义，通常医药大模型需要挂载知识库查询工具
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        
        # === 核心逻辑：应用聊天模板 ===
        # 这一行非常关键。它利用 tokenizer 内部定义的 Jinja2 模板，将 list 格式的对话
        # 自动拼接成一个长字符串。例如：
        # "<|user|>\n你好<|end|>\n<|assistant|>\n你好，请问有什么帮您？<|end|>"
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,            # 这里先不转数字，只拼字符串
            add_generation_prompt=False, # 训练时不需要添加生成的引导符
            tools=tools
        )
    
    # 将label仅设为assistant回答部分
    def generate_labels(self, input_ids):
        # 1. 初始化标签：全为 -100
        # 在 PyTorch 的 CrossEntropyLoss 中，ignore_index 默认为 -100。
        # 意味着标签为 -100 的位置，不计算 Loss，不进行反向传播。
        labels = [-100] * len(input_ids)
        
        i = 0
        while i < len(input_ids): 
            # 2. 寻找回答的起点（帧头检测）
            # 遍历 input_ids，寻找是否匹配 self.bos_id（即 '<s>assistant\n'）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id) # 真正需要学习的内容，是从 'assistant\n' 之后开始的
                end = start
                
                # 3. 寻找回答的终点（帧尾检测）
                while end < len(input_ids):
                    # 寻找 self.eos_id（即 '</s>\n'）
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 4. 填充有效标签（Payload 解调）
                # 将 input_ids 中 [start, end] 这一段复制给 labels
                # 这一段就是模型应该输出的“正确答案”
                # 注意：min(..., self.max_length) 是防止越界
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                
                # 更新指针，继续寻找下一轮对话（如果是多轮对话，会有多个 user/assistant 交互）
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index] # type: ignore # 取出一条原始数据
        prompt = self.create_chat_prompt(sample['conversations']) # 1. 格式化成字符串
        
        # 2. 分词（Tokenization）
        # 将字符串转为数字列表，并强制截断到 max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # 3. Padding（补齐）
        # 如果长度不够 max_length，后面补 pad_token_id (通常是0)
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 4. 生成对应的标签（调用上面的核心函数）
        labels = self.generate_labels(input_ids)
        
        # 5. 返回 Tensor
        # input_ids: 模型的输入 X
        # labels:    模型的监督信号 Y (其中 -100 的地方被 Mask 掉了)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
            #  Assistant 既在输入里（作为背景上下文），也在 Label 里（作为考试标准）

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 获取 assistant 专属的起始和结束 Token ID，用于后续精确匹配并定位回复内容的边界
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.data = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.data) # type: ignore

    def __getitem__(self, index):
        item = self.data[index] # type: ignore
        chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = item['rejected']  # 同上
        # 利用 tokenizer 内置的聊天模板，将字典格式的对话拼接成标准字符串格式。tokenize=False 表示暂不转 ID。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        # 符合对话模板的字符串
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        # 将拼接好的字符串进行 Tokenize，统一截断/填充到 max_length
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        
        chosen_input_ids = chosen_encoding['input_ids']
        # 核心：生成掩码，区分“指令/问题”与“模型回复”。
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # 输入序列舍去最后一个，标签序列去掉第一个，mask与标签同样的处理
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        # 全部用0填充，只有assistant回复的部分才标记为1，表示这些位置需要计算Loss
        loss_mask = [0] * len(input_ids)     
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask







