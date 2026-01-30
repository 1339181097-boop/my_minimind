from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainedDataset(Dataset): # 产出：Tensor([512]) —— 这是一个一维向量
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # 提取每一行内容放到sample
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 用tokenizer进行编码
        # 超过max_length的截断，不到的填充
        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids.squeeze()
        # 忽略padding产生的Y
        loss_mask = input_ids != self.tokenizer.pad_token_id
        # X,Y无需错位，model会错位
        X = torch.tensor(input_ids, dtype=torch.long)
        Y = torch.tensor(input_ids, dtype=torch.long)
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        return X, Y, loss_mask
    
