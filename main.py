import argparse
import json

from models.BERT import BERTWithDualHeads
from utils import *

import os
import pandas as pd
import pathlib


def main(run_type, cfg):
    
    if run_type == 'train':
        # Initialize model
        print("Initializing model...")
        trainer = BERTWithDualHeads("bert-base-uncased", cfg)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        trainer.to(device)
                
        # Train model
        print("Training model...")
        trainer.train()
        
        print("Saving model...")
        # torch.save(trainer.state_dict(), os.path.join(path, "bert_food_hazard_model.pt"))
    elif run_type == 'test':
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
                        choices=['train'],
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

    args = parser.parse_args()
    
    with open(os.path.join(path, 'config.json')) as f:
        cfg = json.load(f)
        
    for key in args.__dict__.keys():
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
        
    main(args.run_type, args.__dict__)
    # print(torch.cuda.is_available())