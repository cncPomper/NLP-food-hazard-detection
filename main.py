import argparse
import json

from models.BERT import BERTWithDualHeads
from utils import *

import os
import pandas as pd
import pathlib

from datetime import datetime

import wandb

# print(f"Login is {wandb.login()}")

def main(run_type, cfg):
    # model_pretrained = "bert-large-uncased"
    model_pretrained = "bert-base-uncased"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if run_type == 'train':
        # Initialize model
        wandb.init(project="NLP", name=f"{model_pretrained}-{cfg['learning_rate']}-{cfg['epochs']}-{cfg['batch_size']}")
        
        print("Initializing model...")
        
        trainer = BERTWithDualHeads(model_pretrained, cfg)
        print(trainer.train_loader)
        
        print(f"Using device: {device}")
                
        trainer.to(device)
                
        # Train model
        print("Training model...")
        trainer.train()
        
        
        print("Saving model...")
        now = datetime.now()
        f = now.strftime("%Y-%m-%d_%H:%M:%S")
        torch.save(trainer.state_dict(), os.path.join(cfg['main_path'], 'saved_models', f"bert_large_model_{f}.pt"))
    elif run_type == 'test':
        model = BERTWithDualHeads(model_pretrained, cfg)
        model_name = os.path.join(cfg['main_path'], 'saved_models', f"{cfg['test_file']}")
        print(model_name)
        print("Loading from model...")
        model.load_state_dict(torch.load(model_name, map_location='cpu'))
        model.to(device)

        print("Testing model...")
        f1 = model.predict()
        print(f"F1-score: {f1}")
        # evaluate(experiment_name, cfg['exp_type'], cfg['main_path'], cfg['emb_size'], cfg['loss'])
        pass
    else:
        print('Run type not understood.')

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Training and evaluation code for Re-MOVE experiments.')
    parser.add_argument('-rt',
                        '--run_type',
                        type=str,
                        default='train',
                        choices=['train', 'test'],
                        help='Whether to run the training or the evaluation script.')
    parser.add_argument('-mp',
                        '--main_path',
                        type=str,
                        default='{}'.format(path),
                        help='Path to the working directory.')
    parser.add_argument('--exp_type',
                        type=str,
                        default='bert',
                        choices=['bert'],
                        help='Type of experiment to run.')
    parser.add_argument('-noe',
                        '--epochs',
                        type=int,
                        default=None,
                        help='Number of epochs for training.')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=None,
                        help='Batch size for training iterations.')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=None,
                        help='Base learning rate for the optimizer')
    parser.add_argument('-tf',
                        '--test_file',
                        type=float,
                        default=None,
                        help='Test file from a model')

    args = parser.parse_args()
    
    with open(os.path.join(path, 'config.json')) as f:
        cfg = json.load(f)
        
    print(cfg)
        
    for key in args.__dict__.keys():
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    print(args)    
    main(args.run_type, args.__dict__)
    # print(torch.cuda.is_available())