# bert-kaggle-stackoverflow

This project tries to predict if a question from StackOverflow will be closed. There are 5 classes:
- open
- not a real question
- off topic
- not constructive
- too localized

One of the main challenges is that the dataset is hugely imbalanced with 98% of the data belonging to `open`.

Undersampling was performed and two approaches were used: 
- **Binary classification**: to predict whether the question will be closed or not (`85% accuracy`)
- **Multiclass classification**: to classify the question into one of the five categories (`65% accuracy`)

## Installation

1. Install Python dependencies
```console
$ pipenv install
```
2. Download dataset **Predict Closed Questions on Stack Overflow** from [Kaggle](https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/data)

3. Log to [wandb](https://wandb.ai/home) to track metrics

```console
$ wandb login
```

## Training

```console
$ pipenv run python train.py predict-closed-questions-on-stack-overflow/train.csv
```
