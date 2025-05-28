import argparse
import json

from models.BERT import BERTWithDualHeads
from models.qwen import Qwen

from utils import *

import os
import pandas as pd
import pathlib

from datetime import datetime
import kagglehub

from transformers import AutoModelForSequenceClassification

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(run_type, cfg):
    # model_pretrained = "bert-large-uncased"
    if (cfg["exp_type"] == 'qwen'):
        model_pretrained = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")
        model = Qwen(model_pretrained, cfg)
    elif (cfg["exp_type"] == 'bert'):
        model_pretrained = ""
        model = BERTWithDualHeads(model_pretrained, cfg)
    elif (cfg["bert_large"] == 'bert_large'):
        model_pretrained = "bert-large-uncased"
        model = BERTWithDualHeads(model_pretrained, cfg)
    else:
        pass
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_model_name = f"{model_pretrained}-learing_rate_{cfg['learning_rate']}-epochs_{cfg['epochs']}-batch_size_{cfg['batch_size']}"
        
    if run_type == 'train':
        print(f"Login is {wandb.login()}")
        # Initialize model
        wandb.init(project="NLP", name=file_model_name)
        
        print("Initializing model...")
        
        # model = BERTWithDualHeads(model_pretrained, cfg)
        model = Base(model_pretrained, cfg)
        # print(trainer.train_loader)
        
        print(f"Using device: {device}")
                
        # model.to(device)
                
        # Train model
        print("Training model...")
        model.train()
        
        print("Predict model...")
        f1 = model.predict()
        wandb.log({"F1-score":f1})
        
        print("Saving model...")
        torch.save(model.state_dict(), os.path.join(cfg['main_path'], 'saved_models', f"{cfg['exp_type']}_models", f"{file_model_name}.pt"))
        
    elif run_type == 'test':
        if (cfg["exp_type"] == 'qwen'):
            model_pretrained = kagglehub.model_download("qwen-lm/qwen-3/transformers/0.6b")
            model = Qwen(model_pretrained, cfg)
        elif (cfg["exp_type"] == 'bert'):
            model_pretrained = ""
            model = BERTWithDualHeads(model_pretrained, cfg)
        elif (cfg["bert_large"] == 'bert_large'):
            model_pretrained = "bert-large-uncased"
            model = BERTWithDualHeads(model_pretrained, cfg)
        else:
            pass

        model_name = os.path.join(cfg['main_path'], 'saved_models', f"{cfg['exp_type']}_models", f"{cfg['test_file']}")
        print(model_name)
        print("Loading from model...")
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)

        print("Testing model...")
        f1 = model.predict()
        print(f"F1-score: {f1}")

        api = wandb.Api()

        run = api.run("-piotr-kitlowski-/NLP/utjo2cii")
        run.config["Test F1-score"] = f1
        run.update()

    else:
        print('Run type not understood.')

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.absolute()

    parser = argparse.ArgumentParser(description='Training and evaluation code for NLP experiments.')
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
                        choices=['bert', 'bert_large', 'qwen'],
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
                        type=str,
                        default=None,
                        help='Test file from a model')

    args = parser.parse_args()
    
    with open(os.path.join(path, 'config.json')) as f:
        cfg = json.load(f)
        
    # print(cfg)
        
    for key in args.__dict__.keys():
        if getattr(args, key) is None:
            setattr(args, key, cfg[key])
    # print(args)    
    main(args.run_type, args.__dict__)