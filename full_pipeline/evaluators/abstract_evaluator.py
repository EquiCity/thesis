from abc import ABC, abstractmethod
from .candidate_solution import CandidateSolution
from .reward import Reward


class AbstractEvaluator(ABC):

    @abstractmethod
    def evaluate(self, solution: CandidateSolution) -> Reward:
        pass
