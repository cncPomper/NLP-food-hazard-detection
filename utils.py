from transformers import AutoTokenizer
import numpy as np
import torch
from sklearn.metrics import f1_score

import pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

def preprocess(df):
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['combined_text'] = df['combined_text'].str.lower()
    df['combined_text'] = df['combined_text'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True)
    return df

def prepare_text_features(df, config):
    """Przygotowanie features tekstowych"""
    if config['use_both_text_title']:
        # Łączenie title + text
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    else:
        df['combined_text'] = df['title'].fillna('')
    
    # Podstawowe czyszczenie
    df['combined_text'] = df['combined_text'].str.lower()
    df['combined_text'] = df['combined_text'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True)
    
    return df['combined_text']

def simple_but_effective_preprocessing(df):
    """Back to basics - what actually worked"""
    # Simple combination
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

    # Basic cleaning only
    df['combined_text'] = df['combined_text'].str.lower()
    df['combined_text'] = df['combined_text'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True)

    return df

def compute_food_hazard_score(hazards_true, products_true, hazards_pred, products_pred):
    """Oficjalna funkcja oceny z konkursu"""
    # F1 dla hazards
    f1_hazards = f1_score(hazards_true, hazards_pred, average='macro')
    
    # F1 dla products tylko gdy hazard prediction jest poprawny
    correct_hazard_mask = hazards_pred == hazards_true
    
    if sum(correct_hazard_mask) > 0:
        f1_products = f1_score(
            products_true[correct_hazard_mask],
            products_pred[correct_hazard_mask],
            average='macro'
        )
    else:
        f1_products = 0.0
    
    final_score = (f1_hazards + f1_products) / 2
    
    return {
        'f1_hazards': f1_hazards,
        'f1_products': f1_products,
        'final_score': final_score
    }
    
# Weighted ensemble predictions
def weighted_ensemble(pred1, pred2, w1, w2):
    ensemble_pred = []
    for i in range(len(pred1)):
        # Simple weighted voting with fallback to XGBoost if tie
        votes = {pred1[i]: w1, pred2[i]: w2}
        if pred1[i] == pred2[i]:
            ensemble_pred.append(pred1[i])  # Both agree
        else:
            # Choose by weight
            best_pred = max(votes.keys(), key=lambda x: votes[x])
            ensemble_pred.append(best_pred)
    return np.array(ensemble_pred)

def create_safe_features(df):
    """Create safe numeric features that work across all datasets"""
    features = pd.DataFrame(index=df.index)
    
    # Text statistics
    features['text_length'] = df['combined_text'].str.len()
    features['word_count'] = df['combined_text'].str.split().str.len()
    features['title_length'] = df['title'].fillna('').str.len()
    features['title_word_count'] = df['title'].fillna('').str.split().str.len()
    
    # Safe ratios (avoid division by zero)
    features['title_text_ratio'] = features['title_length'] / (features['text_length'] + 1)
    features['title_words_ratio'] = features['title_word_count'] / (features['word_count'] + 1)
    
    # Temporal features (safe defaults)
    features['year'] = df['year'].fillna(2020)
    features['month'] = df['month'].fillna(6)
    features['day'] = df['day'].fillna(15)
    
    # Advanced temporal features
    features['is_summer'] = ((features['month'] >= 6) & (features['month'] <= 8)).astype(int)
    features['is_winter'] = ((features['month'] <= 2) | (features['month'] == 12)).astype(int)
    features['is_recent'] = (features['year'] >= 2020).astype(int)
    features['is_weekend_day'] = ((features['day'] % 7) < 2).astype(int)  # Rough weekend approximation
    
    # Safe country features (only major ones that appear in all datasets)
    major_countries = ['us', 'ca', 'au', 'uk']  # Most common in food safety data
    for country in major_countries:
        features[f'country_is_{country}'] = (df['country'].fillna('').str.lower() == country).astype(int)
    
    # Text content features (semantic indicators)
    features['has_recall'] = df['combined_text'].str.contains('recall', case=False, na=False).astype(int)
    features['has_contamination'] = df['combined_text'].str.contains('contaminat', case=False, na=False).astype(int)
    features['has_allergen'] = df['combined_text'].str.contains('allergen|allergy', case=False, na=False).astype(int)
    features['has_bacteria'] = df['combined_text'].str.contains('salmonella|listeria|e\.coli', case=False, na=False).astype(int)
    features['has_undeclared'] = df['combined_text'].str.contains('undeclared', case=False, na=False).astype(int)
    features['has_foreign'] = df['combined_text'].str.contains('foreign|object|metal|plastic', case=False, na=False).astype(int)
    features['has_mislabel'] = df['combined_text'].str.contains('mislabel|mislab|wrong|incorrect', case=False, na=False).astype(int)
    
    # Advanced text features
    features['avg_word_length'] = df['combined_text'].apply(lambda x: np.mean([len(word) for word in str(x).split()] or [0]))
    features['exclamation_count'] = df['combined_text'].str.count('!').fillna(0)
    features['question_count'] = df['combined_text'].str.count('\?').fillna(0)
    features['number_count'] = df['combined_text'].str.count(r'\d').fillna(0)
    
    # Fill any remaining NaN values
    features = features.fillna(0)
    
    return features

def enhanced_text_preparation(df):
    """Enhanced text preparation"""
    # Basic text combination
    df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Enhanced cleaning
    df['combined_text'] = df['combined_text'].str.lower()
    # Keep important punctuation patterns for food safety
    df['combined_text'] = df['combined_text'].str.replace(r'[^\w\s\-\(\)]', ' ', regex=True)
    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True)
    df['combined_text'] = df['combined_text'].str.strip()
    
    return df

def prepare_text(df):
    """Simple text preparation"""
    texts = []
    for _, row in df.iterrows():
        # Combine title and text
        text = str(row['title']) + " " + str(row.get('text', ''))
        # Basic cleaning
        text = text.lower().replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Remove extra spaces
        texts.append(text)
    return texts