import torch
import numpy as np
from torch.nn.functional import one_hot
from tianshou.policy import BasePolicy
from tianshou.data import Batch
from typing import Any, Dict, List, Type, Optional, Union, Callable

from env.config import NUM_NODES


class GreedyPolicy(BasePolicy):
    """Implementation of greedy policy. This policy assign user tasks to service
    providers using greedy strategy defined in greedy_act_func.

    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    """

    def __init__(
            self,
            greedy_act_func: Optional[Callable[[], np.ndarray]],
            dist_fn: Type[torch.distributions.Distribution],
            action_scaling: bool = True,
            action_bound_method: str = "clip",
            **kwargs: Any
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs)
        self.greedy_act_func = greedy_act_func
        self.dist_fn = dist_fn

    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            **kwargs: Any
    ) -> Batch:
        """Compute action with maximum available resources"""
        # print("obs.shape", batch.obs.shape, batch.obs.shape[0] )
        greedy = self.greedy_act_func()
        # print(type(greedy))
        if  isinstance(greedy,np.ndarray):
            greedy = torch.from_numpy(greedy.copy())
        # print("greedy:", greedy)
        base = torch.linspace(1,0,NUM_NODES)
        mean = torch.zeros_like(base)
        mean[greedy] = base
        mean = mean.reshape(1,NUM_NODES)
        std =  torch.ones((1,NUM_NODES))
        logits, hidden = (mean,std), None
        # print("mean:", mean)
        
        # convert to probability distribution
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)

        # use deterministic policy
        if self.action_type == "discrete":
            act = logits.argmax(-1)
        elif self.action_type == "continuous":
            act = logits[0]
        # print("act:",act)
        # assert act.equal(act_), f"Action mismatch: {act_} != {act}"
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn(
            self,
            batch: Batch,
            batch_size: int,
            repeat: int,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        return {"loss": [0.]}