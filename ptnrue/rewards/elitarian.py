import igraph as ig
from .utilitarian import UtilitarianReward
from .egalitarian import EgalitarianTheilReward
from .base_reward import BaseReward


class ElitarianReward(BaseReward):

    def _evaluate(self, g: ig.Graph, *args, **kwargs) -> float:
        ur = UtilitarianReward(census_data=self.census_data, groups=self.groups,
                               com_threshold=self.com_threshold)
        # eg = EgalitarianTheilReward(census_data=self.census_data, groups=self.groups,
        #                             com_threshold=self.com_threshold)
        return ur.evaluate(g)  # + eg(g)

    def _reward_scaling(self, reward: float) -> float:
        return reward
