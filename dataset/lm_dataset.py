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
        # shift在model里进行
        labels = input_ids.clone()
        # pad填上-100，计算loss时自动忽略
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
    
