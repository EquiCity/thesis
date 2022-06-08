import igraph as ig
from ..evaluators.abstract_evaluator import AbstractEvaluator
from ..models.abstract_model import AbstractModel

import logging

logger = logging.getLogger(__name__)


class ExperimentRunner:

    def __init__(self, graph: ig.Graph, model: AbstractModel, evaluator: AbstractEvaluator):
        self.graph = graph
        self.model = model
        self.evaluator = evaluator

    def run(self):
        pass
