from .task import TaskType,TaskState
from .config import *


class Node:

    def __init__(self, sid, task_type_id):
        self._sid = sid
        self._task_type = TaskType(task_type_id)
        self._serving_tasks = []
        self._terminated_tasks = {TaskState.CRASHED: [], TaskState.FINISHED: []}
        self._num_node_crashed = 0 # 当前node crash的次数
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
        total_used_c  = 0
        for task in self._serving_tasks:
            if task.running:
                total_used_c += task.c
        assert total_used_c <= self._total_c,\
            f"Node {id}'s used c {total_used_c} >= total c {self._total_c}"
        return total_used_c

    @property
    def total_d(self):
        return self._total_d
    
    @property
    def norm_total_d(self):
        # 归一化后即为数据资源的分布
        return self.total_d / self.total_ds
    
    @property
    def total_ds(self):
        return sum(self._total_d)
        
    @property
    def norm_total_ds(self):
        return self.total_ds / (max(SINGLE_D_RANGE) * NUM_D_TYPES)

    @property
    def available_c(self):
        return self.total_c - self.used_c
    
    @property
    def norm_available_c(self):
        return self.available_c / self._total_c

    @property
    def norm_total_c(self):
        return self._total_c / max(TOTAL_C_RANGE)
    
    @property
    def vector(self):
        return np.hstack([self.norm_total_d, self.norm_total_c, self.norm_total_ds, self.norm_available_c])

    # 根据task的状态更新node的task列表
    def update_task_list(self, task):
        if task.crashed:
            if task in self._serving_tasks:
                self._serving_tasks.remove(task)
            if task not in self._terminated_tasks[TaskState.CRASHED]:
                self._terminated_tasks[TaskState.CRASHED].append(task)
        elif task.finished:
            if task in self._serving_tasks:
                self._serving_tasks.remove(task)
            if task not in self._terminated_tasks[TaskState.FINISHED]:
                self._terminated_tasks[TaskState.FINISHED].append(task)
        
    def is_enough(self, task):
        return task.c <= self.available_c

    def task_summary(self):
        num_serving = len(self._serving_tasks)
        num_crashed = len(self._terminated_tasks[TaskState.CRASHED])
        num_finished = len(self._terminated_tasks[TaskState.FINISHED])
        crashed_total_c = sum(
            task.c for task in self._terminated_tasks[TaskState.CRASHED]
        )
        finished_total_c = sum(
            task.c for task in self._terminated_tasks[TaskState.FINISHED]
        )
        # 计算reward时 选取第一个node作为task的leader node计算reward
        crashed_total_reward = sum(
            task.reward for task in self._terminated_tasks[TaskState.CRASHED]
        )
        finished_total_reward = sum(
            task.reward for task in self._terminated_tasks[TaskState.FINISHED]
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
        
    def task_summary_unique(self):
        num_serving = sum(
            1 if task.assigned_nodes[0] == self else 0 for task in self._serving_tasks 
        )
        num_crashed = sum(
            1 if task.assigned_nodes[0] == self else 0 for task in self._terminated_tasks[TaskState.CRASHED]
        )
        num_finished = sum(
            1 if task.assigned_nodes[0] == self else 0 for task in self._terminated_tasks[TaskState.FINISHED]
        )
        crashed_total_c = sum(
            task.c for task in self._terminated_tasks[TaskState.CRASHED]
        )
        finished_total_c = sum(
            task.c for task in self._terminated_tasks[TaskState.FINISHED]
        )
        crashed_total_reward = sum(
            task.reward if task.assigned_nodes[0] == self else 0 for task in self._terminated_tasks[TaskState.CRASHED]
        )
        finished_total_reward = sum(
            task.reward if task.assigned_nodes[0] == self else 0 for task in self._terminated_tasks[TaskState.FINISHED]
        )
        return {
            "total": num_serving + num_crashed + num_finished,
            "serving": num_serving,
            "crashed": num_crashed,
            "finished": num_finished,
            "num_node_crashed": self._num_node_crashed,
            "crashed_total_c": crashed_total_c,
            "crashed_total_reward": crashed_total_reward,
            "finished_total_c": finished_total_c,
            "finished_total_reward": finished_total_reward
        }

    def check_finished(self, curr_time):
        for running_task_ in self._serving_tasks[:]:
            if running_task_.can_finished(curr_time):
                # running -> finished
                running_task_.set_finished()
               
            for node in running_task_.assigned_nodes:
                node.update_task_list(running_task_)
    
    # 将task分配给node 
    def assign_task(self, task, curr_time):
        reward = 0.
        # 资源不足, node直接crash
        if not self.is_enough(task):
            penalty = CRASH_PENALTY_COEF
            # 该node上其他task也crash
            for running_task_ in self._serving_tasks:
                if running_task_.running:
                    running_task_.set_crashed(curr_time)
                    
                for node in running_task_.assigned_nodes:
                    node.update_task_list(running_task_)
                if running_task_.crashed:
                    penalty += (1 - running_task_.progress()) * CRASH_PENALTY_COEF
            self._serving_tasks.clear()
            self._num_node_crashed += 1
            return -penalty
        
        # task记录在node中 node记录在task中
        self._serving_tasks.append(task)
        task.add_node(self)
        return reward

    def reset(self):
        self._serving_tasks.clear()
        self._terminated_tasks[TaskState.CRASHED].clear()
        self._terminated_tasks[TaskState.FINISHED].clear()
        self._num_node_crashed = 0


    @property
    def info(self):
        return {
            'id': self.id,
            'task_type': self._task_type,
            'task_serving': len(self._serving_tasks),
            'task_finished': len(self._terminated_tasks[TaskState.FINISHED]),
            'task_crashed': len(self._terminated_tasks[TaskState.CRASHED]),
            'total_c': self._total_c,
            'used_c': self.used_c,
            'available_c': self.available_c,
            'num_node_crashed': self._num_node_crashed
        }
