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


def sample_data(label_dict, label, num_samples):
    import random
    # 从指定标签中随机抽取指定数量的样本
    data = random.sample(label_dict[label], num_samples)
    # 返回 (num_sample, dim_1, .. , dim_n)的张量
    data = torch.stack(data)
    data = data.squeeze(1)
    return data
def 
def calculate_emd(labels1, labels2, labels_cnt=10):
    # 计算labels1的频率向量
    cnt1 = torch.zeros(labels_cnt)
    for label in labels1:
        cnt1[label] += 1
    cnt1 = cnt1 / len(labels1)
    # 计算labels2的频率向量
    cnt2 = torch.zeros(labels_cnt)
    for label in labels2:
        cnt2[label] += 1
    cnt2 = cnt2 / len(labels2)
    # print(f"cnt1 = {cnt1}", f"cnt2 = {cnt2}")
    # 计算EMD
    emd = (cnt1-cnt2).abs().sum()
    return emd
def split_dataset_with_emd(dataset, total_sampels, delta, batch_size, n_worker, n_client=2,label_cnt=10):
    num_samples = total_samples // label_cnt
    more = delta * num_samples
    
    