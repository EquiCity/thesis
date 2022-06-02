import igraph as ig
import pandas as pd
from typing import Tuple
import numpy as np


def evaluate_graph(g: ig.Graph) -> pd.DataFrame:
    nb_nodes = g.vs.select(type_eq='res_node')
    poi_nodes = g.vs.select(type_eq='poi_node')

    tt_mx = np.zeros((len(nb_nodes), len(poi_nodes)))
    hops_mx = np.zeros((len(nb_nodes), len(poi_nodes)))

    failed = {}

    for i, o in enumerate(nb_nodes):
        for j, d in enumerate(poi_nodes):
            # Travel Time
            tt = g.shortest_paths(o, d, weights='tt')[0][0]
            if tt == np.inf:
                if not failed.get(f"{o['node_id']}_tt", None) == d['node_id']:
                    failed[f"{o['node_id']}_tt"] = d["node_id"]
            else:
                tt_mx[i, j] = tt
            # Number of hops
            edges = g.get_shortest_paths(o, d, weights='tt', output='epath')[0]
            if edges == np.inf or not edges:
                if not failed.get(f"{o['node_id']}_edges", None) == d["node_id"]:
                    failed[f"{o['node_id']}_edges"] = d["node_id"]
            else:
                # TODO: consider whether this is correct. Consider that we cannot make any assumption over the
                # TODO: fitness of the number of edges here as we don't take into consideration whether it's a
                # TODO: really long one or not.
                hops_mx[i, j] = len(edges)

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
