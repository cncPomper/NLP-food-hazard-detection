from torch.utils.data import Dataset
import torch

# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, texts, hazard_labels, product_labels, tokenizer, max_length=128):
        self.texts = texts
        self.hazard_labels = hazard_labels
        self.product_labels = product_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mappings
        unique_hazards = sorted(list(set(hazard_labels)))
        unique_products = sorted(list(set(product_labels)))
        
        self.hazard_to_id = {h: i for i, h in enumerate(unique_hazards)}
        self.product_to_id = {p: i for i, p in enumerate(unique_products)}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hazard_label': torch.tensor(self.hazard_to_id[self.hazard_labels[idx]], dtype=torch.long),
            'product_label': torch.tensor(self.product_to_id[self.product_labels[idx]], dtype=torch.long)
        }