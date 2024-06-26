import os
import re
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch
import numpy as np
from collections import defaultdict
import random
random.seed(24)
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
    return 0

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

    # 从指定标签中随机抽取指定数量的样本
    data = random.sample(label_dict[label], num_samples)
    # 返回 (num_sample, dim_1, .. , dim_n)的张量
    data = torch.stack(data)
    # data = data.squeeze(1)
    print(f"Sampled {num_samples} images from label {label} with shape {data.shape}")
    return data
def create_label_dict(dataset):
    label_dict = defaultdict(list)
    for image,label in dataset:
        label_dict[label].append(image)
    return label_dict
def calculate_emd(labels1, labels2, labels_cnt=10):
    # 计算labels1的频率向量
    cnt1 = torch.zeros(labels_cnt)
    for label in labels1:
        cnt1[label] += 1
    print("label1 cnt:",cnt1)
    cnt1 = cnt1 / len(labels1)
    print("lalel1 freq:",cnt1)
    # 计算labels2的频率向量
    cnt2 = torch.zeros(labels_cnt)
    for label in labels2:
        cnt2[label] += 1
    print("label2 cnt:",cnt2)
    cnt2 = cnt2 / len(labels2)
    print("lalel2 freq:",cnt2)
    # print(f"cnt1 = {cnt1}", f"cnt2 = {cnt2}")
    # 计算EMD
    emd = (cnt1-cnt2).abs().sum()
    return emd

def split_dataset_with_emd(dataset, datasize, delta, batch_size, n_worker, n_client=2,label_cnt=10):
    label_dict = create_label_dict(dataset)
    for label in label_dict:
        print(f"Label {label} has {len(label_dict[label])} images")
    label_datasize = datasize // label_cnt
    more = delta * label_datasize
    images_a = []
    labels_a = []
    images_b = []
    labels_b = []
    images_all = []
    labels_all = []
    for label in range(label_cnt): 
        sampled_data = sample_data(label_dict, label, label_datasize)
        # 打印第一个元素的值
        print("label=",label,",sample=",sampled_data[0].sum())
        # print(f"first={label_datasize//2+int(more)}, second={label_datasize//2-int(more)}")
        # print("label=",label,"label_cnt//4=",label_cnt//4)
        if label < label_cnt // 2:
            images_a.append(sampled_data[:label_datasize//2+int(more)])
            images_b.append(sampled_data[label_datasize//2+int(more):])
            labels_a.append(torch.full((label_datasize//2+int(more),), label))
            labels_b.append(torch.full((label_datasize//2-int(more),), label))
        else :
            images_a.append(sampled_data[:label_datasize//2-int(more)])
            images_b.append(sampled_data[label_datasize//2-int(more):])
            labels_a.append(torch.full((label_datasize//2-int(more),), label))
            labels_b.append(torch.full((label_datasize//2+int(more),), label))
        images_all.append(sampled_data)
        labels_all.append(torch.full((label_datasize,), label))
        # if(label < delta):
        #     images_a.append(sampled_data)
        #     labels_a.append(torch.full((label_datasize,), label))
        # else:
        #     images_a.append(sampled_data[:label_datasize//2])
        #     labels_a.append(torch.full((label_datasize//2,), label))
        #     images_b.append(sampled_data[label_datasize//2:])
        #     labels_b.append(torch.full((label_datasize//2,), label))
    images_a = torch.cat(images_a, dim=0)
    labels_a = torch.cat(labels_a, dim=0)
    images_b = torch.cat(images_b, dim=0)
    labels_b = torch.cat(labels_b, dim=0)
    images_all = torch.cat(images_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    print("a:",images_a.shape, labels_a.shape)
    print("b:",images_b.shape, labels_b.shape)
    print("all:",images_all.shape, labels_all.shape)
    emd_a = calculate_emd(labels_a, labels_all)
    emd_b = calculate_emd(labels_b, labels_all)
    print("EMD between two a and all:", emd_a)
    print("EMD between two b and all:", emd_b)
    print("avg EMD:", (emd_a+emd_b)/2,", delta:", delta, "more:", more)
    dataloaders = []
    loader_a = DataLoader(TensorDataset(images_a, labels_a), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    loader_b = DataLoader(TensorDataset(images_b, labels_b), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    loader_all = DataLoader(TensorDataset(images_all, labels_all), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    dataloaders.append(loader_a)
    dataloaders.append(loader_b)
    dataloaders.append(loader_all)
    return dataloaders

def split_dataset_with_delta(dataset, total_samples, delta, batch_size, n_worker, n_client=2,label_cnt=10):
    delta = int(delta)
    label_dict = create_label_dict(dataset)
    for label in label_dict:
        print(f"Label {label} has {len(label_dict[label])} images")
    label_datasize = total_samples // label_cnt
    images_a = []
    labels_a = []
    images_b = []
    labels_b = []
    images_all = []
    labels_all = []
    for label in range(label_cnt): 
        sampled_data = sample_data(label_dict, label, label_datasize)
        # 打印第一个元素的值
        if label < delta:
            images_a.append(sampled_data)
            labels_a.append(torch.full((label_datasize,), label))
        elif label >= label_cnt - delta:
            images_b.append(sampled_data)
            labels_b.append(torch.full((label_datasize,), label))
        else:
            images_a.append(sampled_data[:label_datasize//2])
            labels_a.append(torch.full((label_datasize//2,), label))
            images_b.append(sampled_data[label_datasize//2:])
            labels_b.append(torch.full((label_datasize//2,), label))
        images_all.append(sampled_data)
        labels_all.append(torch.full((label_datasize,), label))
    images_a = torch.cat(images_a, dim=0)
    labels_a = torch.cat(labels_a, dim=0)
    images_b = torch.cat(images_b, dim=0)
    labels_b = torch.cat(labels_b, dim=0)
    images_all = torch.cat(images_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    print(images_a.shape, labels_a.shape)
    print(images_b.shape, labels_b.shape)
    print(images_all.shape, labels_all.shape)
    emd_a = calculate_emd(labels_a, labels_all)
    emd_b = calculate_emd(labels_b, labels_all)
    print("EMD between two a and all:", emd_a)
    print("EMD between two b and all:", emd_b)
    print("avg EMD:", (emd_a+emd_b)/2,", delta:", delta)
    dataloaders = []
    loader_a = DataLoader(TensorDataset(images_a, labels_a), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    loader_b = DataLoader(TensorDataset(images_b, labels_b), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    loader_all = DataLoader(TensorDataset(images_all, labels_all), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    dataloaders.append(loader_a)
    dataloaders.append(loader_b)
    dataloaders.append(loader_all)
    return dataloaders

def split_dataset_with_delta_ugly(dataset, total_samples, delta, batch_size, n_worker, n_client=2,label_cnt=10):
    delta = int(delta)
    label_dict = create_label_dict(dataset)
    for label in label_dict:
        print(f"Label {label} has {len(label_dict[label])} images")
    label_datasize = total_samples // label_cnt
    images_a = []
    labels_a = []
    images_b = []
    labels_b = []
    images_all = []
    labels_all = []
    for label in range(label_cnt//2): 
        
        # 打印第一个元素的值
        sampled_data = sample_data(label_dict, label, label_datasize)
        if label < delta:
            images_a.append(sampled_data[:label_datasize//4])
            labels_a.append(torch.full((label_datasize//4,), label))
            images_b.append(sampled_data[label_datasize//4:label_datasize//2])
            labels_b.append(torch.full((label_datasize//4,), label))
            
            images_all.append(sampled_data)
            labels_all.append(torch.full((label_datasize,), label))
            
            back_label = label + label_cnt//2
            back_sampled_data = sample_data(label_dict, back_label, label_datasize//2)
            images_a.append(back_sampled_data[:label_datasize//4])
            labels_a.append(torch.full((label_datasize//4,), back_label))
            images_b.append(back_sampled_data[label_datasize//4:])
            labels_b.append(torch.full((label_datasize//4,), back_label))
        else:
            images_a.append(sampled_data[:label_datasize//2])
            labels_a.append(torch.full((label_datasize//2,), label))
            images_b.append(sampled_data[label_datasize//2:])
            labels_b.append(torch.full((label_datasize//2,), label))
            images_all.append(sampled_data)
            labels_all.append(torch.full((label_datasize,), label))
    images_a = torch.cat(images_a, dim=0)
    labels_a = torch.cat(labels_a, dim=0)
    images_b = torch.cat(images_b, dim=0)
    labels_b = torch.cat(labels_b, dim=0)
    images_all = torch.cat(images_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    print(images_a.shape, labels_a.shape)
    print(images_b.shape, labels_b.shape)
    print(images_all.shape, labels_all.shape)
    emd_a = calculate_emd(labels_a, labels_all)
    emd_b = calculate_emd(labels_b, labels_all)
    print("EMD between two a and all:", emd_a)
    print("EMD between two b and all:", emd_b)
    print("avg EMD:", (emd_a+emd_b)/2,", delta:", delta)
    dataloaders = []
    loader_a = DataLoader(TensorDataset(images_a, labels_a), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    loader_b = DataLoader(TensorDataset(images_b, labels_b), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    loader_all = DataLoader(TensorDataset(images_all, labels_all), batch_size=batch_size, shuffle=True, num_workers=n_worker)
    dataloaders.append(loader_a)
    dataloaders.append(loader_b)
    dataloaders.append(loader_all)
    return dataloaders