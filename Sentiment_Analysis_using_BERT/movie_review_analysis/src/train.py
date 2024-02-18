import config
import dataset
import torch
import pandas as pd

from model import BERTBasedUncased
from sklearn import model_selection

def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df.sentiment = df.sentiment.apply(
        lambda x:1 if x=="positive" else 0
        )
    
    df_train, df_valid = model_selection.train_test_split(
        df,
        test_size=0.1,
        random_state=123,
        stratify=df.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values,
        target=df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers = 4
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values,
        target=df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers = 1
    )

    device = torch.device("cuda")
    model = BERTBasedUncased()

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'param': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'param': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]


