import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10,MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from model.unet import NaiveUnet
from model.ddpm import DDPM
from model.ddpm_mnist import UNetModel,GaussianDiffusion

from typing import Dict, Optional, Tuple
import util
import os
import json
import copy
import random
import datetime
from tqdm import tqdm
class Client:
    def __init__(
        self, 
        device,
        model,
        lr,
        local_epoch = 1,
        timesteps=500,
    ):
        self.device = device
        self.model = model
        self.model.to(device)
        self.epochs = local_epoch
        self.opti = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.timesteps = timesteps
        self.gassusian_diffusion = GaussianDiffusion(timesteps)

    def train(self, dataloader: DataLoader):
        self.model.train()
        loss_li = []
        for epoch in range(self.epochs):
            for step, (images, labels) in enumerate(dataloader):
                self.opti.zero_grad()

                batch_size = images.shape[0]
                images = images.to(self.device)

                # sample t uniformally for every example in the batch
                #随机生成batch_size个（0~timesteps）的t（对于每次训练数据，我们是随机对第其中一个t时刻的加噪过程进行训练和预测）
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                #输入unet模型，样本图像，和t计算损失
                loss = self.gassusian_diffusion.train_losses(self.model, images, t)
                    #先随机生成一个正太分布(作为我们的加噪的正太分布）
                    #将输入的图像images作为x_start
                    #通过前向加噪，对输入的图像加入t时刻的噪声（此时生成的噪声作为我们的基准噪声）
                    #通过unet，输入上一步的基准噪声，和时间步t，我们进行对基准噪声的预测
                    #损失函数计算的就是我们的预测噪声和基准噪声之间的差距，采用的是每个像素点的均方差的计算

                
                #每次训练模型都是让我们的unet模型的参数进行优化，让我们的unet模型最终可以根据给定一个加噪了t次后的图像，和t，去生成一个对于这个基准噪声的预测。（也就是，我们的unet模型能生成和加入的噪声十分相似的噪声）
                loss.backward()
                loss_li.append(loss.item())
                self.opti.step()
        return sum(loss_li) / len(loss_li)
    
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

SEED = 1024
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:3')
parser.add_argument('--ckpt_load_pth', type=str, default='')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--n_sample', type=int, default=100)
parser.add_argument('--datasize', type=int, default=2000)
parser.add_argument('--emd_delta', type=float, default=0.1)
parser.add_argument('--test', type=bool, default=False)
parser.add_argument('--glb_epoch', type=int, default=20)
parser.add_argument('--local_epoch', type=int, default=30)
parser.add_argument('--sample_seed', type=int, default=SEED)
parser.add_argument('--seed', type=int, default=SEED)
args = parser.parse_known_args()[0]
seed = args.seed
sample_seed = args.sample_seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

device = args.device
ckpt_load_pth = args.ckpt_load_pth
folder_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
folder_path = os.path.join("extra","fed_mnist_emd",folder_name)
ckpt_save_pth = os.path.join(folder_path,"ckpt.pt")
matrix_save_pth = os.path.join(folder_path,"matrix.txt")
sample_save_pth = os.path.join(folder_path,"sample.png")
log_save_pth = os.path.join(folder_path,"log.txt")
emd_src_folder_pth = os.path.join(folder_path,"emd","src")
emd_dst_folder_pth = os.path.join(folder_path,"emd","dst")
# training parameters
glb_epoch = args.glb_epoch
ckpt_save_freq = glb_epoch
local_epoch = args.local_epoch
n_T = 500
beta_1 = 1e-4
beta_2 = 0.02
batch_size = 256
num_workers = 16
lr = 5e-4

# fed parameters
n_sample = args.n_sample
n_client = 2
datasize = args.datasize
emd_delta = args.emd_delta


parameters = {
    'seed': seed,
    'sample_seed': sample_seed,
    'glb_epoch': glb_epoch,
    'ckpt_load_pth': ckpt_load_pth,
    'ckpt_save_freq': ckpt_save_freq,
    'local_epoch': local_epoch,
    'glb_epoch': glb_epoch,
    'n_T': n_T,
    'beta_1': beta_1,
    'beta_2': beta_2,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'lr': lr,
    'n_sample': n_sample,
    'n_client': n_client,
    'datasize': datasize,
    'emd_delta': emd_delta,
    'is_test': args.test,
    'device': device,
}



def init_path():
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(emd_src_folder_pth):
        os.makedirs(emd_src_folder_pth)
    if not os.path.exists(emd_dst_folder_pth):
        os.makedirs(emd_dst_folder_pth)
    json_str = json.dumps(parameters, indent=4)
    with open(os.path.join(folder_path,'parameters.json'), 'w') as json_file:
        json_file.write(json_str)
        
def train_fed_mnist():
    init_path()
    model = UNetModel(
        in_channels=1,
        model_channels=96,
        out_channels=1,
        channel_mult=(1, 2, 2),
        attention_resolutions=[]
    )
    model.to(device)
    
    start_epoch = 0
    if os.path.isfile(ckpt_load_pth):
        print(f"load model from {ckpt_load_pth}")
        model.load_state_dict(torch.load(ckpt_load_pth,map_location=device))
    print(f"start_epoch is {start_epoch}")
    print(f"emd_delta={emd_delta}, datasize={datasize}")
    
    glb_model = model
    clients = [Client(device,copy.deepcopy(model),lr,local_epoch,timesteps=n_T) for _ in range(n_client)]
    
    
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
    # dataloaders = util.split_dataset_with_emd(dataset,datasize,emd_delta,batch_size,num_workers)
    dataloaders = util.split_dataset_with_delta_ugly(dataset,datasize,emd_delta,batch_size,num_workers)
    glb_dataloader = dataloaders[-1]
    dataloaders = dataloaders[:-1]
    if args.test is False:
        for e in range(1,glb_epoch+1):
            e = e + start_epoch
            loss_li = []
            for cli,dataloader in zip(clients,dataloaders):
                # load global model
                cli.model.load_state_dict(copy.deepcopy(glb_model.state_dict()))
                loss = cli.train(dataloader)
                loss_li.append(loss)
                print(f"client {cli} train finish, loss is {loss}")
            # aggregate the global model
            glb_model = copy.deepcopy(clients[0].model)
            for cli in clients[1:]:
                for glb_param,loc_param in zip(glb_model.parameters(),cli.model.parameters()):
                    glb_param.data +=loc_param.data
            for glb_param in glb_model.parameters():
                glb_param.data /= n_client
            # print(f"global model updated")
            
            loss = sum(loss_li) / len(loss_li)
            if e % ckpt_save_freq == 0 or e == glb_epoch:
                ckpt_file = os.path.join(folder_path,"ckpt_e" + str(e) + ".pt")
                torch.save(glb_model.state_dict(), ckpt_file)
            print(f'epoch={e},loss={loss}')
            with open(matrix_save_pth,'a+') as f:
                f.write(f'{e},{loss}\n')
        
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
        gassusian_diffusion = clients[0].gassusian_diffusion
        torch.manual_seed(sample_seed)
        generated_images =gassusian_diffusion.sample(glb_model,28,batch_size=n_sample,channels=1,timesteps=n_T)
        imgs = generated_images[-1].reshape(n_sample, 28, 28)
        print(imgs.shape)
        for i in range(n_sample):
            xset = torch.tensor(imgs[i].reshape(1,28,28))
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=1)
            sample_save_pth = os.path.join(emd_dst_folder_pth,f"{i}.png")
            save_image(grid, sample_save_pth)
            print("sample %d image from glb_model"%(i+1))
    print(f"python fed_mnist_emd.py --datasize={datasize} --emd_delta={emd_delta} --test=True --ckpt_load_pth={folder_path}/ckpt_e{glb_epoch}.pt --device={device}")
    print("----------------")
    print(f"echo {datasize} && echo {emd_delta} && python -m pytorch_fid {emd_dst_folder_pth} {emd_src_folder_pth} --device {device}")
    # print(f"python -m pytorch_fid {emd_dst_folder_pth} {emd_src_folder_pth} --device {device}")
    # python -m pytorch_fid extra/fed_emd/2024-05-08-20:58:06/emd/dst extra/fed_emd/2024-05-08-20:58:06/emd/src --device cuda:2
    

def generate_testimg():
    
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
    label_dict = util.create_label_dict(dataset)
    # 从dataset中抽样
    for label in label_dict.keys():
        for i in range(10):
            idx = random.randint(0,len(label_dict[label])-1)
            image = label_dict[label][idx]
            grid = make_grid(image, normalize=True, value_range=(-1, 1), nrow=1)
            sample_save_pth = os.path.join(emd_src_folder_pth,f"{label}_{i}.png")
            save_image(grid, sample_save_pth)
if __name__ == "__main__":
    train_fed_mnist()
    # generate_testimg()
    