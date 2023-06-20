import os
import re
from torch.utils.data import random_split, DataLoader
import torch

def ck_dir_exist(pth):
    if pth is None:
        return 
    if not os.path.exists(os.path.dirname(pth)):
        os.makedirs(os.path.dirname(pth))
        
def get_epoch_from_path(pth):
    matches = re.findall(r"\d+", pth)
    if matches:
        last_num = int(matches[-1])
        return last_num
    return -1

def split_dataset(dataset, n,batch_size,n_worker):
    split_size = len(dataset) // n
    dataloader = []
    for split in random_split(dataset,[split_size] * n):
        loader = DataLoader(split,batch_size=batch_size,shuffle=True,num_workers=n_worker)
        dataloader.append(loader)
    return dataloader

def epoch_loss(model,tqdm,device):
    model.eval()
    with torch.no_grad():
        e_loss = 0.
        n_iter = 0
        for x, _ in tqdm:
            x = x.to(device)
            loss = model(x)
            n_iter +=1 
            e_loss += loss.item()
            tqdm.set_description(f"eval epoch loss")
            tqdm.set_postfix({
                "epoch_loss": f"{e_loss / n_iter:.4f}",
                })
        return e_loss / n_iter