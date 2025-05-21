import torch
from torch.utils.data import Dataset

class DualClassificationDataset(Dataset):
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset
    
    def __len__(self):
        return len(self.tokenized_dataset)
    
    def __getitem__(self, idx):
        item = self.tokenized_dataset[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "hazard_label": torch.tensor(item["hazard_label"]),
            "product_label": torch.tensor(item["product_label"])
        }