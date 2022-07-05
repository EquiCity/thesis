import igraph as ig
import pandas as pd
from typing import Tuple
import numpy as np


# import ray
# from typing import List
#
# @ray.remote
# class GraphActor:
#     def __init__(self, graph: ig.Graph):
#         self.graph = graph
#
#     def set_graph(self, v):
#         self.graph = v
#
#     def get_graph(self):
#         return self.graph
#
#
# @ray.remote
# class ShortestPathComputer:
#     def __init__(self, graph_actor):
#         self.graph_actor = graph_actor
#
#     def get_shortes_path(self, v_rc: ig.Vertex, v_poi_list: List[ig.Vertex]):
#         return ray.get(self.graph_actor.get_graph.remote())


def evaluate_graph(g: ig.Graph) -> pd.DataFrame:
    nb_nodes = g.vs.select(type_eq='res_node')
    poi_nodes = g.vs.select(type_eq='poi_node')




    shortest_paths_tt = g.shortest_paths(nb_nodes, poi_nodes, weights='tt')

    # Travel Time
    tt_mx = np.array(shortest_paths_tt)
    # Assign max over both dimensions to inf values
    tt_mx[tt_mx == np.inf] = tt_mx.max(1).max()

    # Number of hops
    hops_mx = []
    for v_rc in nb_nodes:
        shortest_path_edges = g.get_shortest_paths(v_rc, poi_nodes, weights='tt', output='epath')
        hops_mx.append([len(es) for es in shortest_path_edges])

    hops_mx = np.array(hops_mx)
    hops_mx[hops_mx == 0] = 1

    df_tt = pd.DataFrame(tt_mx, columns=poi_nodes['name'])
    df_tt['metric'] = 'travel_time'
    df_tt['rc'] = nb_nodes['name']

    df_hops = pd.DataFrame(hops_mx, columns=poi_nodes['name'])
    df_hops['metric'] = 'hops'
    df_hops['rc'] = nb_nodes['name']

    return pd.concat([df_tt, df_hops], axis=0)


def generate_samples(metric_df: pd.DataFrame, inh_per_group: pd.DataFrame) -> pd.DataFrame:
    merged_df = pd.merge(metric_df, inh_per_group, how='left', left_on='rc', right_on='neighborhood')

    city = pd.DataFrame(columns=['metric_value', 'group'])

    for group in merged_df.loc[:, merged_df.columns.str.contains('n_')].columns:
        samples_group = np.repeat(merged_df['average'].values, merged_df[group])
        samples_group = samples_group.reshape((-1, 1))
        labels = np.repeat(group.replace('n_', ''), samples_group.shape[0]).reshape((-1, 1))
        group_individuals = np.hstack([samples_group, labels])
        df = pd.DataFrame(group_individuals, columns=['metric_value', 'group'])
        city = pd.concat([city, df], axis=0)

    return city


def get_tt_hops_com_dfs(g: ig.Graph, census_data: pd.DataFrame,
                        com_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_df = evaluate_graph(g)
    groups_census = census_data.drop(columns=['n_inh', 'geometry'])

    metric_df['average'] = metric_df.loc[:, metric_df.columns.str.contains('POI')].mean(axis=1)
    tt_samples = generate_samples(metric_df[metric_df['metric'] == 'travel_time'], groups_census)
    hops_samples = generate_samples(metric_df[metric_df['metric'] == 'hops'], groups_census)

    value_cols = metric_df.loc[:, metric_df.columns.str.contains('POI')]
    metric_df['average'] = (value_cols < com_threshold).sum(axis=1)
    com_samples = generate_samples(metric_df[metric_df['metric'] == 'travel_time'], groups_census)

    return tt_samples, hops_samples, com_samples


def series_min_max_norm(sr: pd.Series):
    return (sr - sr.min()) / (sr.max() - sr.min())
