from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch.nn as nn

# Simple BERT model
class SimpleBERT(nn.Module):
    def __init__(self, model_name, num_hazards, num_products):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.hazard_head = nn.Linear(hidden_size, num_hazards)
        self.product_head = nn.Linear(hidden_size, num_products)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.hazard_head(pooled), self.product_head(pooled)