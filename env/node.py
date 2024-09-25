from .task import TaskType
from .config import *


class Node:

    def __init__(self, sid, task_type_id):
        self._sid = sid
        self._task_type = TaskType(task_type_id)
        self._serving_tasks = []
        self._terminated_tasks = {'crashed': [], 'finished': []}
        self._num_crashed = 0
        self._num_finished = 0
        # 节点资源
        self._total_c = np.random.choice(TOTAL_C_RANGE) # 持有计算资源
        self._total_d = np.random.choice(SINGLE_D_RANGE, size=NUM_D_TYPES) # 持有数据资源

    @property
    def id(self):
        return self._sid

    @property
    def total_c(self):
        return self._total_c

    @property
    def used_c(self):
        total  = 0
        for task in self._serving_tasks:
            if not task.crashed and not task.finished:
                total += task.c
        assert total <= self._total_c,\
            f"Node {id}'s used c {total} >= total c {self._total_c}"
        return total

    @property
    def total_d(self):
        return self._total_d
    
    @property
    def total_ds(self):
        return sum(self._total_d)
    
    @property
    def norm_total_ds(self):
        return self.total_ds / (max(SINGLE_D_RANGE) * NUM_D_TYPES)
    
    @property
    def norm_total_d(self):
        return self.total_d / self.total_ds
    
    @property
    def available_c(self):
        return self._total_c - self.used_c

    def is_enough(self, task):
        return self.available_c >= task.c

    @property
    def norm_total_c(self):
        return self._total_c / max(TOTAL_C_RANGE)

    @property
    def norm_available_c(self):
        return self.available_c / self._total_c

    def task_summary(self):
        num_serving = len(self._serving_tasks)
        num_crashed = len(self._terminated_tasks['crashed'])
        num_finished = len(self._terminated_tasks['finished'])
        crashed_total_c = sum(
            task.c for task in self._terminated_tasks['crashed']
        )
        finished_total_c = sum(
            task.c for task in self._terminated_tasks['finished']
        )
        crashed_total_reward = sum(
            self.calculate_reward(task) for task in self._terminated_tasks['crashed']
        )
        finished_total_reward = sum(
            self.calculate_reward(task) for task in self._terminated_tasks['finished']
        )
        return {
            "total": num_serving + num_crashed + num_finished,
            "serving": num_serving,
            "crashed": num_crashed,
            "finished": num_finished,
            "crashed_total_c": crashed_total_c,
            "crashed_total_reward": crashed_total_reward,
            "finished_total_c": finished_total_c,
            "finished_total_reward": finished_total_reward
        }

    def check_finished(self, curr_time):
        num_finished = 0
        for running_task_ in self._serving_tasks[:]:
            if running_task_.can_finished(curr_time):
                running_task_.set_finished()
                self._terminated_tasks['finished'].append(running_task_)
                self._serving_tasks.remove(running_task_)
                num_finished += 1
            elif running_task_.crashed:
                self._terminated_tasks['crashed'].append(running_task_)
                self._serving_tasks.remove(running_task_)
                self._num_crashed += 1
            elif running_task_.finished:
                self._terminated_tasks['finished'].append(running_task_)
                self._serving_tasks.remove(running_task_)
                self._num_finished +=1
        return num_finished
    
    def calculate_reward(self, task):  
        return 0.0 # reward由task直接计算

    def assign_task_precheck(self, task):
        if task.c > self.available_c:
            return False
        return True
    
    # 将task分配给node 
    def assign_task(self, task, curr_time):
        reward = self.calculate_reward(task)
        # No enough resources, node crashes
        if task.c > self.available_c:
            penalty = CRASH_PENALTY_COEF
            # 有可能会影响到其他node上的task 
            for running_task_ in self._serving_tasks:
                running_task_.crash(curr_time)
                self._terminated_tasks['crashed'].append(running_task_)
                penalty += (1 - running_task_.progress()) * CRASH_PENALTY_COEF
            self._serving_tasks.clear()
            self._num_crashed += 1
            return -penalty
        
        # task记录在node中 node记录在task中
        self._serving_tasks.append(task)
        task.add_node(self)
        return reward

    def reset(self):
        self._serving_tasks.clear()
        self._terminated_tasks['crashed'].clear()
        self._terminated_tasks['finished'].clear()
        self._num_crashed = 0

    @property
    def vector(self):
        # (total_c, available_c)
        return np.hstack([self.norm_total_d, self.norm_total_c, self.norm_total_ds, self.norm_available_c])

    @property
    def info(self):
        return {
            'id': self.id,
            'task_type': self._task_type,
            'task_serving': len(self._serving_tasks),
            'task_finished': len(self._terminated_tasks['finished']),
            'task_crashed': len(self._terminated_tasks['crashed']),
            'total_c': self._total_c,
            'used_c': self.used_c,
            'available_c': self.available_c,
            'num_crashed': self._num_crashed
        }
