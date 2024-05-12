import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10,MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from model.unet import NaiveUnet
from model.ddpm import DDPM
from model.unet_complex import UNet

from typing import Dict, Optional, Tuple
import util
import os
import copy
import random
import datetime
from tqdm import tqdm
class Client:
    def __init__(
        self, 
        device,
        model,
        gaussian_diffusion,
        lr,
        local_epoch = 1,
    ):
        self.device = device
        self.model = model
        self.model.to(device)
        self.local_epoch = local_epoch
        self.opti = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gassian_diffusion = gaussian_diffusion

    def train(self, dataloader: DataLoader):
        self.model.train()
        loss_ema_list = []
        avg_iter_loss_list = []
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
random.seed(SEED)
torch.manual_seed(SEED)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--ckpt_load_pth', type=str, default='/home/gao/haiwu/block-aigc/checkpoint/cifar/ckpt_cplx150_.pt')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--total_samples', type=int, default=2000)
parser.add_argument('--emd_delta', type=float, default=0.1)
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_known_args()[0]

device = args.device
ckpt_load_pth = args.ckpt_load_pth

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
n_epoch = 300
ckpt_save_freq = 300
local_epoch = 2
n_T = 1000
beta_1 = 1e-4
beta_2 = 0.02
batch_size = 256
num_workers = 16
lr = 1e-5

# fed parameters
n_sample = args.n_sample
n_client = 2
total_samples = args.total_samples
emd_delta = args.emd_delta


parameters = {
    'n_epoch': n_epoch,
    'ckpt_load_pth': ckpt_load_pth,
    'ckpt_save_freq': ckpt_save_freq,
    'local_epoch': local_epoch,
    'n_T': n_T,
    'beta_1': beta_1,
    'beta_2': beta_2,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'lr': lr,
    'n_sample': n_sample,
    'n_client': n_client,
    'total_samples': total_samples,
    'emd_delta': emd_delta,
    'is_test': args.test,
    'device': device,
}
import json
json_str = json.dumps(parameters, indent=4)
with open(os.path.join(folder_path,'parameters.json'), 'w') as json_file:
    json_file.write(json_str)
    
def train_fed_mnist():
    ddpm = DDPM(eps_model=UNet(n_T),
            betas=(beta_1, beta_2), n_T=n_T,regular=False)
    start_epoch = 0
    if os.path.isfile(ckpt_load_pth):
        print(f"load model from {ckpt_load_pth}")
        ddpm.load_state_dict(torch.load(ckpt_load_pth,map_location=device))
        # start_epoch = util.get_epoch_from_path(ckpt_load_pth)+1
    print(f"start_epoch is {start_epoch}")
    print("emd_delta=",emd_delta,"total_samples=",total_samples)
    ddpm.to(device)
    glb_model = ddpm
    clients = [Client(device,copy.deepcopy(ddpm),lr,local_epoch) for _ in range(n_client)]
    
    if args.dataset == 'mnist':
        dataset_pth = "data/mnist/"
        tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        dataset = MNIST(
            dataset_pth,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
    elif args.dataset == 'cifar10':
        dataset_pth = "data/cifar/"
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
    
    if args.test is False:
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
    # sample data from dataset
    dataset = glb_dataloader.dataset
    for i in range(n_sample):
        idx = random.randint(0,len(dataset)-1)
        image,label = dataset[idx]
        grid = make_grid(image, normalize=True, value_range=(-1, 1), nrow=1)
        sample_save_pth = os.path.join(emd_src_folder_pth,f"{i}.png")
        save_image(grid, sample_save_pth)
    print("---sample from dataset finished---")
    
    # sample data from glb_model
    with torch.no_grad():
        for i in range(n_sample):
            xset = glb_model.sample(1, (3, 32, 32), device)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=1)
            sample_save_pth = os.path.join(emd_dst_folder_pth,f"{i}.png")
            save_image(grid, sample_save_pth)
            print("sample %d image from glb_model"%(i+1))
    print(f"python -m pytorch_fid {emd_dst_folder_pth} {emd_src_folder_pth} --device {device}")
    # python -m pytorch_fid extra/fed_emd/2024-05-08-20:58:06/emd/dst extra/fed_emd/2024-05-08-20:58:06/emd/src --device cuda:2
    

    
if __name__ == "__main__":
    train_fed_mnist()
    