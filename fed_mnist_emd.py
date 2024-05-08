from typing import Dict, Optional, Tuple
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10,MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from model.unet import NaiveUnet
from model.ddpm import DDPM
from model.unet_complex import UNet
import util
import os
import copy
import datatime
class Client:
    def __init__(
        self, 
        device,
        model,
        lr,
        local_epoch = 1,
    ):
        self.device = device
        self.model = model
        self.model.to(device)
        self.local_epoch = local_epoch
        self.opti = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataloader: DataLoader):
        self.model.train()
        loss_ema_list = []
        avg_iter_loss_list = []
        # test_key = 'eps_model.up2.model.1.conv.3.weight'
        # print('before train:',self.model.state_dict()[test_key].view(-1)[:8])
        pbar = tqdm(dataloader)
        for iter in range(self.local_epoch):
            loss_ema = None
            iter_loss_sum = 0
            n_iter = 0
            for x, _ in pbar:
                self.opti.zero_grad()
                x = x.to(self.device)
                loss = self.model(x)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                iter_loss_sum = iter_loss_sum + loss.item()
                n_iter = n_iter + 1
                pbar.set_postfix({
                    "loss": f"{loss_ema:.4f}",
                    "iter_loss": f"{loss.item()}"
                })
                self.opti.step()
            loss_ema_list.append(loss_ema)
            avg_iter_loss_list.append(iter_loss_sum / n_iter)
        # print('after train:',self.model.state_dict()[test_key].view(-1)[:8])
        return sum(loss_ema_list)/len(loss_ema_list),sum(avg_iter_loss_list)/len(avg_iter_loss_list)
    
    def eval_loss(self,datalaoder:DataLoader):
        with torch.no_grad():
            pbar = tqdm(datalaoder)
            iter_loss_sum = 0
            n_iter = 0
            loss_ema = None
            for x, _ in pbar:
                self.opti.zero_grad()
                x = x.to(self.device)
                loss = self.model(x)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                iter_loss_sum = iter_loss_sum + loss.iterm()
                n_iter = n_iter +1
                pbar.set_postfix({
                    "loss": f"{loss_ema:.4f}",
                    "iter_loss": f"{loss.item()}"
                })
                self.opti.step()
            return loss_ema, iter_loss_sum / n_iter
    # def sample(self, n_samples: int, input_shape: Tuple[int, int, int]):
    #     with torch.no_grad():
    #         xh = self.model.sample(n_samples, input_shape, self.device)
    #     return xh

SEED = 24
np.random.seed(SEED)
torch.manual_seed(SEED)
# path parameters
device = "cuda:3"
# ckpt_load_pth = "checkpoint/fed_mnist/ckpt_cplx_150_.pt"
ckpt_load_pth = ""
dataset_pth = "data/cifar/"
folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
folder_path = os.path.join("extra","fed_emd",folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
ckpt_save_pth = os.path.join(folder_path,"ckpt.pt")
matrix_save_pth = os.path.join(folder_path,"matrix.txt")
sample_save_pth = os.path.join(folder_path,"sample.png")
log_save_pth = os.path.join(folder_path,"log.txt")
emd_src_folder_pth = os.path.join(folder_path,"emd","src")
if not os.path.exists(emd_src_folder_pth):
    os.makedirs(emd_src_folder_pth)
emd_dst_folder_pth = os.path.join(folder_path,"emd","dst")
if not os.path.exists(emd_dst_folder_pth):
    os.makedirs(emd_dst_folder_pth)
# training parameters
n_epoch = 200
ckpt_save_freq = 50
local_epoch = 1
n_T = 1000
beta_1 = 1e-4
beta_2 = 0.02
batch_size = 256
num_workers = 16
lr = 1e-5

# fed parameters
n_sample = 16
n_client = 2
total_samples = 1000
emd_delta = 0.1

def train_fed_mnist():
    ddpm = DDPM(eps_model=UNet(n_T),
            betas=(beta_1, beta_2), n_T=n_T,regular=False)
    start_epoch = 0
    if os.path.isfile(ckpt_load_pth):
        print(f"load model from {ckpt_load_pth}")
        ddpm.load_state_dict(torch.load(ckpt_load_pth,map_location=device))
        start_epoch = util.get_epoch_from_path(ckpt_load_pth)+1
    print(f"start_epoch is {start_epoch}")
    print("emd_delta=",emd_delta,"total_samples=",total_samples)
    ddpm.to(device)
    glb_model = ddpm
    clients = [Client(device,copy.deepcopy(ddpm),lr,local_epoch) for _ in range(n_client)]
    
    tf = transforms.Compose(
        [
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset = CIFAR10(
        dataset_pth,
        train=True,
        download=True,
        transform=tf,
    )
    dataloaders = util.split_dataset_with_emd(dataset,total_samples,emd_delta,batch_size,num_workers)
    glb_dataloader = dataloaders[-1]
 
    for e in range(1,n_epoch+1):
        e = e + start_epoch
        ddpm.train()
        loss_ema_arr = []
        loss_iter_arr = []
        for cli,dataloader in zip(clients,dataloaders):
            # load global model
            cli.model.load_state_dict(copy.deepcopy(glb_model.state_dict()))
            loss_ema,loss_iter = cli.train(dataloader)
            loss_ema_arr.append(loss_ema)
            loss_iter_arr.append(loss_iter)
        # aggregate the global model
        glb_model = copy.deepcopy(clients[0].model)
        for cli in clients[1:]:
            for glb_param,loc_param in zip(glb_model.parameters(),cli.model.parameters()):
                glb_param.data +=loc_param.data
        for glb_param in glb_model.parameters():
            glb_param.data /= n_client
            
        
        avg_loss_ema = sum(loss_ema_arr)/len(loss_ema_arr)
        avg_loss_iter = sum(loss_iter_arr)/len(loss_iter_arr)
        if e % ckpt_save_freq == 0 or e == n_epoch:
            ckpt_file = os.path.join(folder_path,"ckpt_e" + str(e) + ".pt")
            torch.save(glb_model.state_dict(), ckpt_file)
        epoch_loss = util.epoch_loss(glb_model,tqdm(glb_dataloader),device)
        print(f'epoch={e},avg_loss_ema={avg_loss_ema},avg_loss_iter={avg_loss_iter},epoch_loss={epoch_loss}')
        with open(matrix_save_pth,'a+') as f:
            f.write(f'{e},{avg_loss_ema},{avg_loss_iter},{epoch_loss}\n')
    
    print("---training finished---")
    
def eval_fed_mnist():
    util.ck_dir_exist(sample_dir_pth) 
    ddpm = DDPM(eps_model=UNet(n_T),
        betas=(beta_1, beta_2), n_T=n_T,regular=False)
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
        sample_file = os.path.join(sample_dir_pth,f"ddpm_sample_fed_cifar_cplx{epoch}.png")
        save_image(grid, sample_file)

    
if __name__ == "__main__":
    train_fed_mnist()
    