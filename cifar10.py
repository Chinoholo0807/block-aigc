from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from model.unet import NaiveUnet
from model.ddpm import DDPM
import util
import os

def train_cifar10(
    device: str = "cuda:2", 
    # ckpt_load_pth: str = 'checkpoint/cifar/ckpt_18_.pt', 
    ckpt_load_pth: str = "",
    ckpt_dir_pth: str = "checkpoint/cifar/",
    matrix_dir_pth : str = "matrix/cifar/",
    sample_dir_pth : str = "sample/cifar/",
    log_dir_pth : str = "log/cifar/",
    n_epoch: int = 100, 
    n_T: int = 1000,
    beta_1: float = 1e-4,
    beta_2: float = 0.02,
    batch_size: int = 256,
    num_workers: int = 16,
    lr: float = 1e-5,
    n_sample = 8
) -> None:
   
    util.ck_dir_exist(matrix_dir_pth)
    util.ck_dir_exist(ckpt_dir_pth)
    util.ck_dir_exist(log_dir_pth)
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128),
                betas=(beta_1, beta_2), n_T=n_T)
    start_epoch = 0
    if os.path.isfile(ckpt_load_pth):
        print(f"load model from {ckpt_load_pth}")
        ddpm.load_state_dict(torch.load(ckpt_load_pth,map_location=device))
        start_epoch = util.get_epoch_from_path(ckpt_load_pth)+1
        print(f"start_epoch is {start_epoch}")
    ddpm.to(device)

    tf = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10(
        "./data/cifar/",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    matrix_file = os.path.join(matrix_dir_pth,'cifar10.txt')
    for e in range(n_epoch):
        e = e + start_epoch
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        running_loss = 0.
        n_iter = 0
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            running_loss = running_loss + loss.item()
            n_iter = n_iter+1
            pbar.set_description(f"epoch: {e}")
            pbar.set_postfix({
                "loss": f"{loss_ema:.4f}",
                "iter_loss": f"{loss.item()}"
                })
            optim.step()
        epoch_loss = util.epoch_loss(ddpm,pbar,device)
        avg_iter_loss = running_loss / n_iter
        print(f"epoch={e},loss_ema={loss_ema:.4f},avg_iter_loss={avg_iter_loss},epoch_loss={epoch_loss}")
        if e % 4 == 0:
            ckpt_file = os.path.join(ckpt_dir_pth,"ckpt_" + str(e) + "_.pt")
            torch.save(ddpm.state_dict(), ckpt_file)
        with open(matrix_file,'a+') as f:
            f.write(f'{e},{loss},{avg_iter_loss},{epoch_loss}\n')
    

def eval_cifar10(
    device: str = "cuda:2", 
    # ckpt_load_pth: str = 'checkpoint/cifar/ckpt_18_.pt', 
    ckpt_load_pth: str = "checkpoint/cifar/ckpt_39_.pt",
    ckpt_dir_pth: str = "checkpoint/cifar/",
    matrix_dir_pth : str = "matrix/cifar/",
    sample_dir_pth : str = "sample/cifar/",
    log_dir_pth : str = "log/cifar/",
    n_epoch: int = 20, 
    n_T: int = 1000,
    beta_1: float = 1e-4,
    beta_2: float = 0.02,
    batch_size: int = 256,
    num_workers: int = 16,
    lr: float = 1e-5,
    n_sample = 8
):
    util.ck_dir_exist(sample_dir_pth)
    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128),
                betas=(beta_1, beta_2), n_T=n_T)
    if os.path.isfile(ckpt_load_pth):
        print(f"load model from {ckpt_load_pth}\n")
        ddpm.load_state_dict(torch.load(ckpt_load_pth,map_location=device))
    else:
        print("no model load")
        return 
    ddpm.to(device)
    ddpm.eval()
    epoch = util.get_epoch_from_path(ckpt_load_pth)
    with torch.no_grad():
        xset = ddpm.sample(n_sample, (3, 32, 32), device)
        grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
        sample_file = os.path.join(sample_dir_pth,f"ddpm_sample_cifar_{epoch}.png")
        save_image(grid, sample_file)
    
if __name__ == "__main__":
    train_cifar10()
    # eval_cifar10()