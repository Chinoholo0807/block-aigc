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
        self._d = d
        self._c = c
        self._c_range = TASK_C_RANGE
        self._r_global_range = TASK_R_GLOBAL_RANGE
        self._r_local_range = TASK_R_LOCAL_RANGE
        self._n_max_range = TASK_N_MAX_RANGE
        self._task_type = None
        self._r_slot = TASK_R_SLOT
        self._n_max = n_max
        self._total_r = r_global * r_local
        self._runtime = self._total_r * TASK_R_SLOT # 任务运行时间,即资源占用时间
        self._crashed = False
        self._crash_time = -1
        self._finished = False
        self._state = TaskState.RUNNING
        self._assigned_nodes = []

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
        return self._c / self._c_range[-1]

    @property
    def norm_runtime(self):
        return self._runtime / (max(self._r_global_range)*  max(self._r_local_range)*TASK_R_SLOT)
    
    @property
    def norm_d(self):
        return self._d / self.ds
    
    @property
    def norm_n_max(self):
        return self._n_max / self._n_max_range
    
    @property
    def finished(self):
        return self._finished
    
    @property
    def crashed(self):
        return self._crashed

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
        return REWARD_TOTAL(total_ds, avg_emd, self._total_r)
    
    def set_task_type(self, type_id):
        self._task_type = TaskType(type_id)

    # add_node说明当前节点加入到task中
    def add_node(self, node):
        self._assigned_nodes.append(node)

    def can_finished(self, curr_time):
        return not self._crashed and curr_time >= self._arrival_time + self._runtime

    def set_finished(self):
        assert not self._crashed, \
            f"This task {self._task_id} has been crashed"
        self._finished = True

    def crash(self, curr_time):
        self._crash_time = curr_time
        self._crashed = True
        
    def crash_boardcast(self, curr_time):
        for node in self._assigned_nodes:
            node.crash_task(self,curr_time)

    def progress(self, curr_time=None):
        if self._finished:
            return 1.

        if self._crashed:
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
        self._c_range = TASK_C_RANGE
        self._r_local_range = TASK_R_LOCAL_RANGE
        self._r_global_range = TASK_R_GLOBAL_RANGE
        self._n_max_range = TASK_N_MAX_RANGE
        self.reset()

    def reset(self):
        self._task_id_counter = 0
        self._total_task = np.random.poisson(self._lambda * self._total_time)
        self._task_arrival_time = np.hstack(
            [[0], np.sort(np.random.random(self._total_task) * self._total_time)])
        self._task_arrival_time = self._task_arrival_time.astype(np.int64)
        self._total_task = len(self._task_arrival_time)

    def __next__(self):
        task_id = self._task_id_counter
        assert task_id < self._total_task, "number of tasks out of range"

        arrival_time = self._task_arrival_time[task_id]
        required_d = np.random.choice(SINGLE_D_RANGE, size=NUM_D_TYPES)
        required_c = np.random.choice(self._c_range)
        required_r_local = np.random.choice(self._r_local_range)
        required_r_global = np.random.choice(self._r_global_range)
        required_n_max = np.random.choice(self._n_max_range)
        task = Task(task_id, arrival_time,required_d, required_c, required_r_local, required_r_global, required_n_max)

        self._task_id_counter += 1
        terminate = True if self._task_id_counter == self._total_task else False
        return task, terminate