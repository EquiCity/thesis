from dataclasses import dataclass
from typing import Union, List

import igraph as ig


@dataclass
class CandidateSolution:

    graph: ig.Graph
    _solution: ig.EdgeSeq

    @property
    def solution(self) -> ig.EdgeSeq:
        return self._solution

    @solution.setter
    def solution(self, edge_set: Union[ig.EdgeSeq, List[str]]) -> None:
        if isinstance(edge_set, list):
            if isinstance(edge_set[0], str):
                es = self.graph.es.select(key_in=edge_set)
                if not len(es) == len(edge_set):
                    raise LookupError(f"Tried to find {edge_set=} but got instead "
                                      f"(diff: {abs(len(es) - len(edge_set))}) {es=}")
            else:
                raise TypeError(f"a solution should be an igraph.EdgeSeq or a list of str with the edge ids, "
                                f"not {type(edge_set[0])}")
        else:
            es = edge_set
        if not len(self.graph.subgraph_edges(es).es) == len(es):
            raise ValueError(f"Could not identify edges")

        self.solution = es
