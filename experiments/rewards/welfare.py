import igraph as ig
import pandas as pd
from typing import List
from ._utils import get_tt_hops_com_dfs
from experiments.constants.travel_metric import TravelMetric
from .base_reward import BaseReward


class WelfareReward(BaseReward):

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        pass

    def _reward_scaling(self, reward: float) -> float:
        return reward
