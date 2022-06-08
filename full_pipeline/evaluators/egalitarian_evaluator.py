from .abstract_evaluator import AbstractEvaluator
from .candidate_solution import CandidateSolution
from .reward import Reward


class EgalitarianEvaluator(AbstractEvaluator):

    def __init__(self, ):
        pass

    def evaluate(self, solution: CandidateSolution) -> Reward:
        pass
