from abc import ABC, abstractmethod

from full_pipeline.evaluators.candidate_solution import CandidateSolution
from ..evaluators.abstract_evaluator import AbstractEvaluator
# from ..evaluators.egalitarian_evaluator import EgalitarianEvaluator


class AbstractModel(ABC):

    @abstractmethod
    @property
    def evaluator(self) -> AbstractEvaluator:
        pass

    @abstractmethod
    @property
    def graph(self) -> AbstractEvaluator:
        pass

    def evaluate(self, solution: CandidateSolution):
        self.evaluator.evaluate(solution)

    @abstractmethod
    def infer(self) -> CandidateSolution:
        pass

    @abstractmethod
    def compute_solution(self) -> CandidateSolution:
        pass
