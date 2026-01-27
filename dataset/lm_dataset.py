from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainedDataset(Dataset): # 产出：Tensor([512]) —— 这是一个一维向量
    def __init__(self,data_path,tokenizer,max_length=512) :
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # split='train': 加载训练集部分
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index] # pyright: ignore[reportIndexIssue]
        tokens = self.tokenizer(
            str(sample['text']), 
            add_special_tokens=False, 
            max_length=self.max_length - 2, # 留下bos和eos的位置
            truncation=True
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100 # 计算loss时忽略pad部分
        return input_ids,labels
    
    def __len__(self):
        return len(self.samples) # pyright: ignore[reportArgumentType]
    
