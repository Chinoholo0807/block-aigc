import numpy as np
import torch

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# 训练节点相关
NUM_NODES = 20  # number of nodes
TOTAL_C_RANGE = np.arange(2,10)  # 节点总计算资源范围 这里为节点的CPU数量
SINGLE_D_RANGE = np.arange(50, 500, step=100)  # 单类数据的持有范围
NUM_D_TYPES = 10 # 总共有多少类数据

K = [ 
     0, 
     1.05929976, # k1
     0.00754265,  # k2
     0.12761403,  # k3
     -0.1480588,  # k4
     1.13964024 , # k5
     0.42965643, # k6
     0.986706, # k7
     0.01, # k8
     0, # k9
     1.5 # k10
     ]
# 训练数据收益函数
def REWARD_D(total_ds, avg_emd):
    f_avg_emd = K[4]* avg_emd + K[5]
    return  K[6] * (f_avg_emd  - K[1]*np.exp(-K[2]*np.power(K[3]*total_ds,K[7]*f_avg_emd)) )
# 训练时间收益函数
def REWARD_T(t_occ):
    return  1 - np.exp(-K[8]*(t_occ-K[9]))
# 训练总收益函数
def REWARD_FOR_TASK(total_ds, avg_emd, t_occ):
    reward_d =  REWARD_D(total_ds, avg_emd) 
    reward_t = REWARD_T(t_occ)
    # print(f"total_ds:{total_ds}, avg_emd:{avg_emd}, t_occ:{t_occ}, reward_d:{reward_d}, reward_t:{reward_t}, reward_total:{reward_d*reward_t}")
    return K[10]*reward_d * reward_t

# 训练任务发布者相关
NUM_USERS = 1000  # number of users to serve
LOCATION_RANGE = [(0, 0), (100, 100)]  # [(x_min, y_min), (x_max, y_max)]

# 训练任务相关
LAMBDA = 0.001  # λ for Poisson distribution
TOTAL_TIME = 1000000  # time duration of an episode
TASK_C_RANGE = np.arange(1, 2)  # 任务占用的计算资源范围 这里直接取1
TASK_R_LOCAL_RANGE = np.arange(10, 20, step=5)
TASK_R_GLOBAL_RANGE = np.arange(150, 250, step=20)
TASK_R_SLOT = 10  # 单轮本地训练花费时间 这里先取固定值
TASK_N_MAX_RANGE = np.arange(2,4) # 参与任务最多节点数

# For task
NUM_TASK_TYPES = 5  # number of task types available
IMG_CHW = (3, 218, 178)  # (n_channel, height, width)
IMG_BUFFER = 8 * 2 ** 10  # 8KBytes per image, JPEG format, for storage and transmission
GPU_MEM_OCCUPY = 4000 * 2 ** 20  # 7468MB GPU memory occupation per image and per run
GPU_UTILITY = 1.  # GPU-Util of 100%, full load
CPU_MEM_OCCUPY = 2000 * 2 ** 20  # 4980MB CPU memory occupation per image and per run
CPU_UTILITY = 0.1  # CPU-Util of 10%
CRASH_PENALTY_COEF = 2.  # The penalty unit value for crash
CRASH_PENALTY_BASE = 2. 
# Runtime for each image. The value is proportional to t_T in the diffusion algorithm.
RUNTIME = lambda t: (0.001 * t ** 2 + 2.5 * t - 14) * 60
