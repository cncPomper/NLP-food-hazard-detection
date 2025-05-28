from dataset.dual_classification_dataset import DualClassificationDataset
import torch
import torch.nn as nn
from transformers import AutoModel
from datetime import datetime

from datasets import load_dataset
from utils import *
from torch.utils.data import DataLoader

import wandb

import os
import gc

from tqdm import tqdm


class Qwen(nn.Module):
    def __init__(self, model_name, cfg):
        super().__init__()
        
        self.path = "https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/blob/main/data/"
        
        self.model_name = model_name
        self.cfg = cfg
        now = datetime.now()
        f = now.strftime("%Y-%m-%d_%H:%M:%S")
        self.experiment_name = 'model_{}_{}'.format(cfg["exp_type"], f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.best_f1 = 0
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.data_files = {
            "train": os.path.join(self.path, "incidents_train.csv?raw=true"),
            "valid": os.path.join(self.path, "incidents_valid.csv?raw=true"),
            "test": os.path.join(self.path, "incidents_test.csv?raw=true")
        }
        
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'saved_models')):
            os.mkdir(os.path.join(self.cfg['main_path'], 'saved_models'))
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'saved_models',
                                           '{}_models'.format(self.cfg['exp_type']))):
            os.mkdir(os.path.join(self.cfg['main_path'], 'saved_models',
                                  '{}_models'.format(self.cfg['exp_type'])))
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'experiment_logs')):
            os.mkdir(os.path.join(self.cfg['main_path'], 'experiment_logs'))
        if not os.path.exists(os.path.join(self.cfg['main_path'], 'experiment_logs',
                                           '{}_logs'.format(self.cfg['exp_type']))):
            os.mkdir(os.path.join(self.cfg['main_path'], 'experiment_logs',
                                  '{}_logs'.format(self.cfg['exp_type'])))

        self.create_dataloaders()
        self.create_model()
        self.model.gradient_checkpointing_enable()  # Good, you have this!
        self.create_optimizer()
        self.create_loggers()
    
    def create_model(self):
        
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            device_map='auto',
            trust_remote_code=True
        )
        
        # Get hidden size properly
        hidden_size = self.model.config.hidden_size
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Two classification heads
        self.hazard_classifier = nn.Linear(hidden_size, self.num_hazards).to(self.device)
        self.product_classifier = nn.Linear(hidden_size, self.num_products).to(self.device)
    
    def create_optimizer(self):
    # Separate parameters for base model and classifiers
        from torch.optim.lr_scheduler import CosineAnnealingLR
        optimizer_grouped_parameters = [
            {'params': self.model.parameters(), 'lr': self.cfg["learning_rate"]/10},
            {'params': self.hazard_classifier.parameters(), 'lr': self.cfg["learning_rate"]},
            {'params': self.product_classifier.parameters(), 'lr': self.cfg["learning_rate"]},
        ]
        
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg["epochs"],
            eta_min=1e-6
        )
    
    def create_loggers(self):
        """
        Creating logs for training and validation losses.
        """
        self.train_loss_log = []
        self.val_loss_log = []
    
    def forward(self, input_ids, attention_mask):
        
        with torch.autocast(device_type=self.device):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True  # Important for getting hidden states
            )
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        
        # Classification heads
        hazard_logits = self.hazard_classifier(pooled_output.float())
        product_logits = self.product_classifier(pooled_output.float())
        
        return {
            "hazard_logits": hazard_logits,
            "product_logits": product_logits
        }
    
    def train(self):
        torch.cuda.empty_cache()
        gc.collect()
        
        scaler = torch.amp.GradScaler(enabled=self.device=='cuda')
        wandb.watch(self.model, log_freq=100)

        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):
                self.optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                hazard_labels = batch["hazard_label"].to(self.device)
                product_labels = batch["product_label"].to(self.device)
                
                with torch.autocast(device_type=self.device):
                    outputs = self.forward(input_ids, attention_mask)

                    hazard_loss = self.loss_fn(outputs["hazard_logits"], hazard_labels)
                    product_loss = self.loss_fn(outputs["product_logits"], product_labels)
                    loss = hazard_loss + product_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)

                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=0.5,
                    norm_type=2.0,
                    error_if_nonfinite=False
                )
                
                scaler.step(self.optimizer)
                scaler.update()
                
                # Check for NaN in outputs
                if torch.isnan(loss).any():
                    print("NaN detected in loss!")
                    break

                # Check gradient norms
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                print(f"Gradient norm: {total_norm:.4f}")
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            val_score = self.handle_validation_batches()
            
            self.epoch_curr = epoch
            
            wandb.log({
                "Epoch": epoch,
                "Avg_loss": avg_loss,
                "loss": total_loss,
                "valid_f1_for_epoch": val_score,
                # "lr": self.scheduler.get_last_lr()[0]
            })
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Score={val_score:.4f}")

        self.checkpoint()        
        
        final_score = self.handle_validation_batches()
        wandb.log({"Valid F1": final_score})
    
    def predict(self):
        final_score = self.handle_validation_batches(is_valid=False)
        return final_score
        
    def handle_validation_batches(self, is_valid=True):
        self.model.eval()
        all_hazard_preds = []
        all_product_preds = []
        all_hazard_true = []
        all_product_true = []
        
        with torch.no_grad():
            loader = self.valid_loader if is_valid else self.test_loader
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                with torch.autocast(device_type=self.device):
                    outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get predictions
                    hazard_preds = torch.argmax(outputs["hazard_logits"], dim=1).cpu().numpy()
                    product_preds = torch.argmax(outputs["product_logits"], dim=1).cpu().numpy()
                    
                    all_hazard_preds.extend(hazard_preds)
                    all_product_preds.extend(product_preds)
                    all_hazard_true.extend(batch["hazard_label"].numpy())
                    all_product_true.extend(batch["product_label"].numpy())
                
        
        # Convert to numpy arrays for easier manipulation
        all_hazard_preds = np.array(all_hazard_preds)
        all_product_preds = np.array(all_product_preds)
        all_hazard_true = np.array(all_hazard_true)
        all_product_true = np.array(all_product_true)
        
        # Calculate macro-F1 scores
        f1_hazard = f1_score(all_hazard_true, all_hazard_preds, average='macro')
        
        # Calculate product F1 score only for instances where hazard prediction is correct
        correct_hazard_mask = all_hazard_preds == all_hazard_true
        
        if sum(correct_hazard_mask) > 0:
            f1_product = f1_score(
                all_product_true[correct_hazard_mask], 
                all_product_preds[correct_hazard_mask], 
                average='macro'
            )
        else:
            f1_product = 0.0
        
        # Final score as per the requirement
        final_score = (f1_hazard + f1_product) / 2
        
        self.val_loss_log.append(final_score)
        
        return final_score
        
    
    def map_labels(self, example):
        hazard_label = self.hazard_label_mapping.get(example["hazard-category"], -1)
        product_label = self.product_label_mapping.get(example["product-category"], -1)
        return {"hazard_label": hazard_label, "product_label": product_label}
    
    def create_dataloaders(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token   
            
        # Make sure to properly clean and preprocess text
        def preprocess_text(text):
            if isinstance(text, list):
                return [' '.join(str(t).split()) for t in text]
            return ' '.join(str(text).split())
        
        # Proper tokenization function
        def tokenize_function(examples):
            texts = [f"{title} {text}" for title, text in zip(examples["title"], examples["text"])]
            texts = preprocess_text(texts)
            inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=256,  # Increased from 128
                return_tensors="pt"
            )
            return inputs
        
        # Load and process dataset
        self.dataset = load_dataset("csv", data_files=self.data_files)
        
        # Apply preprocessing
        self.dataset = self.dataset.map(
            lambda x: {"text": preprocess_text(x["text"])},
            batched=True
        )
        
        self.hazard_categories = self.dataset["train"].unique("hazard-category")
        self.product_categories = self.dataset["train"].unique("product-category")
        
        self.num_hazards = len(self.hazard_categories)
        self.num_products = len(self.product_categories)
        
        print(f"Number of hazard categories: {self.num_hazards}")
        print(f"Number of product categories: {self.num_products}")

        # Create label mappings
        self.hazard_label_mapping = {cat: idx for idx, cat in enumerate(self.hazard_categories)}
        self.product_label_mapping = {cat: idx for idx, cat in enumerate(self.product_categories)}
        
        self.dataset = self.dataset.map(self.map_labels)
        
        tokenized_datasets = self.dataset.map(tokenize_function, batched=True)
        
        self.train_dataset = DualClassificationDataset(tokenized_datasets["train"])
        self.valid_dataset = DualClassificationDataset(tokenized_datasets["valid"])
        self.test_dataset = DualClassificationDataset(tokenized_datasets["test"])

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.cfg["batch_size"], 
            shuffle=True
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=self.cfg["batch_size"]
        )

        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.cfg["batch_size"]
        )
        
    def checkpoint(self):
        """
        Function to determine whether to save the current model.
        If the current mean average precision is 0.001 higher than the best value, the model is saved.
        """
        torch.save(self.model.state_dict(),
                    os.path.join(self.cfg['main_path'], 'saved_models', '{}_models'.format(self.cfg['exp_type']),
                                'model_{}_ep_cur_{}.pt'.format(self.experiment_name, self.epoch_curr)))