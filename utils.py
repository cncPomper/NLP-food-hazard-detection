from transformers import AutoTokenizer
import numpy as np
import torch
from sklearn.metrics import f1_score

# def tokenize_function(examples):
#     # Using only the title field for tokenization as specified
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
#     return tokenizer(
#         examples["title"],
#         padding="max_length",
#         truncation=True,
#         max_length=128  # As specified in requirements
#     )
    
def import_dataset_from_pt():
    pass

def evaluate_model(model, test_loader, device):
    model.eval()
    all_hazard_preds = []
    all_product_preds = []
    all_hazard_true = []
    all_product_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
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
    
    print(f"Hazard Macro-F1: {f1_hazard:.4f}")
    print(f"Product Macro-F1 (for correct hazards): {f1_product:.4f}")
    print(f"Final Score: {final_score:.4f}")
    
    return final_score