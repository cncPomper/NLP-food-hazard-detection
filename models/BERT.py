from dataset.dual_classification_dataset import DualClassificationDataset
import torch
import torch.nn as nn
from transformers import AutoModel
from time import time
from datasets import load_dataset
from utils import *
from torch.utils.data import DataLoader

import json

import os
import gc

class BERTWithDualHeads(nn.Module):
    def __init__(self, bert_model_name, cfg):
        super().__init__()
        
        self.path = "https://github.com/food-hazard-detection-semeval-2025/food-hazard-detection-semeval-2025.github.io/blob/main/data/"
        
        self.bert_model_name = bert_model_name
        self.cfg = cfg
        
        self.experiment_name = "bert_git"
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.best_f1 = 0
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.data_files = {
            "train": os.path.join(self.path, "incidents_train.csv?raw=true"),
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
        self.create_optimizer()
        self.create_loggers()                
    
    def create_model(self):        
        self.model = AutoModel.from_pretrained(self.bert_model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Two classification heads
        self.hazard_classifier = nn.Linear(self.model.config.hidden_size, self.num_hazards)
        self.product_classifier = nn.Linear(self.model.config.hidden_size, self.num_products) 
    
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["learning_rate"])

    def create_loggers(self):
        """
        Creating logs for training and validation losses.
        """
        self.train_loss_log = []
        self.val_loss_log = []
    
    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get the [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        # Pass through the classification heads
        hazard_logits = self.hazard_classifier(pooled_output)
        product_logits = self.product_classifier(pooled_output)
        
        return {
            "hazard_logits": hazard_logits,
            "product_logits": product_logits
        }
        
    def train(self, save_logs=True):
        for epoch in range(self.cfg["epochs"]):
            # self.model.train()
            print(f"Epoch {epoch+1}")
            total_loss = 0
                        
            for batch in self.train_loader:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                hazard_labels = batch["hazard_label"].to(self.device)
                product_labels = batch["product_label"].to(self.device)
                
                # Forward pass
                outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
                
                # print(outputs)
                
                # Calculate loss
                hazard_loss = self.loss_fn(outputs["hazard_logits"], hazard_labels)
                product_loss = self.loss_fn(outputs["product_logits"], product_labels)
                loss = hazard_loss + product_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.train_loader)
            self.train_loss_log.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.cfg['epochs']}, Loss: {avg_loss:.4f}")
            
            # Evaluate after each epoch
            val_epoch = (epoch + 1) % 2 == 0 or epoch == self.cfg['epochs'] - 1
            if val_epoch:
                self.handle_training_batches()

                self.checkpoint()
        if save_logs:
            with open('./experiment_logs/{}_logs/{}.json'.format(self.cfg['exp_type'], self.experiment_name), 'w') as f:
                json.dump({'train_loss_log': self.train_loss_log, 'val_loss_log': self.val_loss_log}, f)
    
    def handle_training_batches(self):
        self.model.eval()
        all_hazard_preds = []
        all_product_preds = []
        all_hazard_true = []
        all_product_true = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
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
        
    
    def map_labels(self, example):
        hazard_label = self.hazard_label_mapping.get(example["hazard-category"], -1)
        product_label = self.product_label_mapping.get(example["product-category"], -1)
        return {"hazard_label": hazard_label, "product_label": product_label}            
    
    def create_dataloaders(self):
        """
        Creating data loader for training and validation loops.
        :param is_train: whether the data loader is for a training loop
        """
        # flushing down the previous loaders and related variables
        gc.collect()
        
        self.dataset = load_dataset("csv", data_files=self.data_files)
        
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
        self.test_dataset = DualClassificationDataset(tokenized_datasets["test"])
        # print(self.cfg)
        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.cfg["batch_size"], 
            shuffle=True
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
        print(self.val_loss_log)
        if self.val_loss_log[-1] >= self.best_f1 + 1e-3:

            torch.save(self.model.state_dict(),
                       os.path.join(self.cfg['main_path'], 'saved_models', '{}_models'.format(self.cfg['exp_type']),
                                    'model_{}.pt'.format(self.experiment_name)))

            self.best_f1 = self.val_loss_log[-1]