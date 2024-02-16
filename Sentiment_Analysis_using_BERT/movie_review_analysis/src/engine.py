from tqdm import tqdm
import torch
import torch.nn as nn

def loss_fn(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)

def train_fn(data_loader, model, optimizer, device, accumulation_steps):
    model.train()
    
    for bi, d in tqdm(enumerate(data_loader), total= len(data_loader)):
        ids = d['ids']
        mask = d["masks"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zer_grad()
        outputs = model(
            ids = ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

def eval_fn():
    pass