# NLP-food-hazard-detection

Authors: Michał Kaczmaryk, Hubert Podlewski i Piotr Kitłowski

## Usage
```bash
usage: main.py [-h] [-rt {train,test}] [-mp MAIN_PATH] [--exp_type {bert,bert_large,qwen}] [-noe EPOCHS] [-bs BATCH_SIZE] [-lr LEARNING_RATE] [-tf TEST_FILE]

Training and evaluation code for NLP experiments.

optional arguments:
  -h, --help            show this help message and exit
  -rt {train,test}, --run_type {train,test}
                        Whether to run the training or the evaluation script.
  -mp MAIN_PATH, --main_path MAIN_PATH
                        Path to the working directory.
  --exp_type {bert,bert_large,qwen}
                        Type of experiment to run.
  -noe EPOCHS, --epochs EPOCHS
                        Number of epochs for training.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training iterations.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Base learning rate for the optimizer
  -tf TEST_FILE, --test_file TEST_FILE
                        Test file from a model
```

## Installation
```bash
git clone git@github.com:cncPomper/NLP-food-hazard-detection.git
```


```bash
conda create -n nlp python=3.9 anaconda
conda activate nlp
```

```bash
pip install -r requirements.txt
```

## Training
```bash
python main.py -rt train --exp_type qwen -noe 5
```

## Notebooks
Everything is explained in the folder `/notebooks`