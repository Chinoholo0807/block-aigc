from .config import *
from enum import Enum, auto

class TaskType:

    def __init__(self, type_id):
        self._num_types = NUM_TASK_TYPES
        assert type_id < self._num_types
        self._type_id = type_id

    @property
    def one_hot(self):
        return np.eye(self._num_types)[self._type_id]
    
class TaskState(Enum):
    RUNNING = auto()
    CRASHED = auto()
    FINISHED = auto()

class Task:

    def __init__(self, task_id, arrival_time,d, c,r_local, r_global, n_max):
        self._task_id = task_id
        self._arrival_time = arrival_time
        self._d = d # 任务数据分布
        self._c = c # 计算资源占用
        self._task_type = None
        self._n_max = n_max
        self._total_r = r_global * r_local # 全局训练轮数 * 本地训练轮数
        self._runtime = self._total_r * TASK_R_SLOT # 任务运行时间,即资源占用时间
        self._crash_time = -1
        self._state = TaskState.RUNNING
        self._assigned_nodes = [] # 任务分配的节点

        # The following info are not currently considered
        self._num_cpu = 1
        self._num_gpu = 1
        self._cpu_mem = CPU_MEM_OCCUPY
        self._gpu_mem = GPU_MEM_OCCUPY
        self._data_type = 'image'
        self._data_bytes = IMG_BUFFER
        self._data_shape = IMG_CHW
        
    @property
    def assigned_nodes(self):
        return self._assigned_nodes
    
    @property
    def arrival_time(self):
        return self._arrival_time

    @property
    def task_type(self):
        return self._task_type.one_hot

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d
    
    @property
    def ds(self):
        return sum(self._d)
    
    @property
    def norm_c(self):
        return self._c / max(TASK_C_RANGE)

    @property
    def norm_runtime(self):
        return self._runtime / (max(TASK_R_LOCAL_RANGE) * max(TASK_R_GLOBAL_RANGE) * TASK_R_SLOT)
    
    @property
    def norm_d(self):
        return self._d / self.ds
    
    @property
    def norm_n_max(self):
        return self._n_max / max(TASK_N_MAX_RANGE)
    
    @property
    def state(self):
        return self._state
    
    @property
    def finished(self):
        return self._state == TaskState.FINISHED
    
    @property
    def crashed(self):
        return self._state == TaskState.CRASHED
    
    @property
    def running(self):
        return self._state == TaskState.RUNNING

    @property
    def vector(self):
        assert self._task_type is not None, \
            f"Please set task type for task {self._task_id} first"
        vec = np.hstack([self.norm_d, self.norm_c, self.norm_runtime, self.norm_n_max])
        return vec
    
    @property
    def reward(self):
        assert len(self._assigned_nodes) > 0,\
            "No assigned nodes for the task"
        if self.crashed:
            penalty = CRASH_PENALTY_BASE
            penalty += (1 - self.progress()) * CRASH_PENALTY_COEF
            return penalty
        emds = []
        total_ds = 0
        for node in self._assigned_nodes:
            freq_node = node.total_d / node.total_ds
            freq_task = self.d / self.ds
            emd = np.abs(freq_node - freq_task).sum()
            emds.append(emd)
            total_ds  += node.total_ds
        avg_emd = sum(emds) / len(emds)
        return REWARD_FOR_TASK(total_ds, avg_emd, self._total_r)
    
    def set_task_type(self, type_id):
        self._task_type = TaskType(type_id)

    # 指派task给node
    def add_node(self, node):
        # 确保node不在assigned_nodes中
        assert node not in self._assigned_nodes, \
            f"Node {node.id} has been assigned to task {self._task_id}"
        self._assigned_nodes.append(node)

    def can_finished(self, curr_time):
        return self.running and curr_time >= self._arrival_time + self._runtime

    def set_finished(self):
        assert self.running, \
            f"This task {self._task_id} has been crashed"
        self._state = TaskState.FINISHED

    def set_crashed(self, curr_time):
        assert self.running, \
            f"This task {self._task_id} has been finished"
        self._crash_time = curr_time
        self._state = TaskState.CRASHED

    def progress(self, curr_time=None):
        if self.finished:
            return 1.

        if self.crashed:
            return (self._crash_time - self._arrival_time) / self._runtime

        assert curr_time, "Current time unknown"
        assert curr_time >= self._arrival_time,\
            f"Current time {curr_time} < task arrival time {self._arrival_time} with runtime {self._runtime}"
        return (curr_time - self._arrival_time) / self._runtime


class TaskGenerator:

    def __init__(self):
        self._task_id_counter = 0
        self._lambda = LAMBDA
        self._total_time = TOTAL_TIME
        self._total_task = 0
        self._task_arrival_time = None
        self.reset()

    def reset(self):
        self._task_id_counter = 0
        self._total_task = np.random.poisson(self._lambda * self._total_time)
        self._task_arrival_time = np.hstack(
            [[0], np.sort(np.random.random(self._total_task) * self._total_time)])
        self._task_arrival_time = self._task_arrival_time.astype(np.int64)
        self._total_task = len(self._task_arrival_time)

    def __next__(self):
        # generate a task
        task_id = self._task_id_counter
        assert task_id < self._total_task, "number of tasks out of range"

        arrival_time = self._task_arrival_time[task_id]
        required_d = np.random.choice(SINGLE_D_RANGE, size=NUM_D_TYPES) # 任务数据分布
        required_c = np.random.choice(TASK_C_RANGE) # 计算资源占用
        required_r_local = np.random.choice(TASK_R_LOCAL_RANGE)
        required_r_global = np.random.choice(TASK_R_GLOBAL_RANGE)
        required_n_max = np.random.choice(TASK_N_MAX_RANGE)
        task = Task(task_id, arrival_time,required_d, required_c, required_r_local, required_r_global, required_n_max)

        self._task_id_counter += 1
        terminate = True if self._task_id_counter == self._total_task else False
        return task, terminate