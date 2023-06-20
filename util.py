import os
import re

def ck_dir_exist(pth):
    if pth is None:
        return 
    if not os.path.exists(os.path.dirname(pth)):
        os.makedirs(os.path.dirname(pth))
        
def get_epoch_from_pth(pth):
    matches = re.findall(r"\d+", pth)
    if matches:
        last_num = int(matches[-1])
        return last_num
    return -1