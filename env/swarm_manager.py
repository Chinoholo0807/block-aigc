import os
import platform
from .user import User
from .node import Node
from .task import TaskGenerator
from .config import *


class SwarmManager:

    def __init__(self):
        self._n_users = NUM_USERS
        self._n_nodes = NUM_NODES
        self._users = [
            User(uid, 0) for uid in range(self._n_users)]
        self._nodes = [
            Node(sid, 0) for sid in range(self._n_nodes)]
        self._task_generator = TaskGenerator()
        self._querying_user = None
        self.next_user_task()

    def check_finished(self, curr_time):
        num_finished = 0
        for node in self._nodes:
            num_finished += node.check_finished(curr_time)
        return num_finished

    def next_user_task(self):
        self._querying_user = np.random.choice(self._users)
        # 随机生成一个task给user
        task, terminate = next(self._task_generator)
        self._querying_user.add_task(task)

        curr_time = task.arrival_time
        self.check_finished(curr_time)
        return curr_time, terminate

    # 将task分配给node
    @DeprecationWarning
    def assign(self, sid, curr_time):
        assert self._querying_user, \
            "No querying user found, call next_user_task first"
        
        if type(sid) is np.ndarray:
            assert len(sid) == 1
            sid = int(sid.item())

        node = self._nodes[sid]
        assert node.id == sid, \
            f"Node (id={node.id}) not match assigned sid={sid}"
        assert self._querying_user.task.arrival_time == curr_time, \
            f"Arrival time mismatch: {self._querying_user.task.arrival_time} != {curr_time}"

        reward = node.assign_task(self._querying_user.task, curr_time)
        return reward
    
    # 将task分配给nodes
    def assign_multi(self, sids, curr_time):
        assert self._querying_user, \
            "No querying user found, call next_user_task first"

        # assert len(sids) ==1,\
        #     "Only one node can be assigned currently"
        task  = self._querying_user.task
        crashed_nodes = []
        for sid in sids:
            node = self._nodes[sid]
            if node.assign_task_precheck(task) == False :
                crashed_nodes.append(node)
        # 分配此次task后存在node会crash
        if(len(crashed_nodes) != 0):
            total_penalty = 0.0
            for node in crashed_nodes:
                penalty = node.assign_task(task, curr_time)
                assert penalty <= 0.0,\
                    f"Penalty should less than 0.0, current is {penalty}"
                total_penalty += penalty
            return total_penalty
        # 否则分配正常 计算对应的reward
        for sid in sids:
            node = self._nodes[sid]
            node.assign_task(task,curr_time) 
        return task.reward

    @property
    def vector(self):
        assert self._querying_user, \
            "No querying user found, call next_user_task first"

        # 所有node的状态
        vec = [node.vector for node in self._nodes]
        # 当前task的状态
        vec += [self._querying_user.vector]
        # （1,n) + (1,n) -> (1,2n)
        vec = np.hstack(vec)
        return vec

    def reset(self):
        [user.reset() for user in self._users]
        [node.reset() for node in self._nodes]
        self._task_generator.reset()
        self.next_user_task()
        return self.vector

    @property
    def total_c_available(self):
        return sum([node.total_c for node in self._nodes])

    @property
    def total_c_serving(self):
        return sum([node.used_c for node in self._nodes])

    @property
    def most_available_node(self):
        arr = [node.available_c  for node in self._nodes]
        # ret = np.argsort(arr)[::-1]
        
        arr = []
        for node in self._nodes:
            if node.available_c == 0:
                arr.append(1)
            else:
                arr.append(node.norm_total_ds)
        ret = np.argsort(arr)
        # print("ret " , ret)
        return ret

    @property
    def available_node(self):
        arr = [node.available_c for node in self._nodes]
        arr = []
        num_crashvoid = 0
        num_crashvoid_threshhold = int(NUM_NODES*0.4)
        for node in self._nodes:
            if node.available_c == 0 and num_crashvoid < num_crashvoid_threshhold:
                arr.append(1)
                num_crashvoid = num_crashvoid +1
            else:
                arr.append(np.random.rand())
        ret = np.argsort(arr)
        # print("ret " , ret)
        return ret
    
    
    @property
    @DeprecationWarning
    def best_reward_service_provider(self):
        best_reward_ = 0.
        best_sid_ = 0
        for service_provider in self._nodes:
            if service_provider.is_enough(self._querying_user.task):
                reward_ = service_provider.calculate_reward(self._querying_user.task)
                if reward_ > best_reward_:
                    best_reward_ = reward_
                    best_sid_ = service_provider.id
        return best_sid_

    def monitor(self):
        cmd = 'cls' if platform.system().lower() == "windows" else 'clear'
        os.system(cmd)
        WIDTH = 98
        print()

        # service provider info
        HEAD = " SID | Task Serving | " \
               "\033[0;32mTask Finished\033[0m | " \
               "\033[0;31mTask Crashed\033[0m | " \
               "Total C | Used C | Available C | " \
               "\033[0;31mNum Crashed\033[0m "

        print("-" * WIDTH)
        print(f"\033[7mNode\033[0m "
              f"(Total C Avaliable {self.total_c_available})".center(WIDTH))
        print("-" * WIDTH)
        print(HEAD)
        print("-" * WIDTH)

        for node in self._nodes:
            info = node.info
            print(f"{str(info['id']).center(6)}"
                  f"{str(info['task_serving']).center(15)}"
                  f"\033[0;32m{str(info['task_finished']).center(16)}\033[0m"
                  f"\033[0;31m{str(info['task_crashed']).center(15)}\033[0m"
                  f"{str(info['total_c']).center(10)}"
                  f"{str(info['used_c']).center(9)}"
                  f"{str(info['available_c']).center(14)}"
                  f"\033[0;31m{str(info['num_crashed']).center(14)}\033[0m")

        print("-" * WIDTH)

        # task info
        total_tasks = 0
        total_serving = 0
        total_crashed = 0
        total_finished = 0
        crashed_total_c = 0
        crashed_total_reward = 0
        finished_total_c = 0
        finished_total_reward = 0
        for node in self._nodes:
            info = node.task_summary()
            total_serving += info['serving']
            total_crashed += info['crashed']
            total_finished += info['finished']
            total_tasks += info['total']
            crashed_total_c += info['crashed_total_c']
            finished_total_c += info['finished_total_c']
            crashed_total_reward += info['crashed_total_reward']
            finished_total_reward += info['finished_total_reward']
        assert total_tasks == total_serving + total_crashed + total_finished

        print(f"\033[7mTask\033[0m".center(6), end='')
        print(f"Total: {total_tasks}".center(14), end='')
        print(f"Serving: {total_serving}".center(13), end='')
        print(f"\033[0;31mCrashed: {total_crashed} "
              f"(TotalC: {crashed_total_c}, Reward: {int(crashed_total_reward)})\033[0m"
              .center(43), end='')
        print(f"\033[0;32mFinished: {total_finished} "
              f"(TotalC: {finished_total_c}, Reward: {int(finished_total_reward)})\033[0m"
              .center(45))
        print("-" * WIDTH)

        # user info
        print(f"\033[7mUser\033[0m".center(6), end='')
        print(f"Total Users: {str(self._n_users)}".center(22), end='')
        print(f"Total Serving C: {str(self.total_c_serving)}".center(36))
        print("-" * WIDTH)