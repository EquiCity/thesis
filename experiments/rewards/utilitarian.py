from typing import List
import igraph as ig
import numpy as np
import pandas as pd


def utilitarian(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None) -> float:
    """

    Args:
        g:
        census_data:
        groups:

    Returns:

    """
    # TODO: Look into multi-obj. learning but for now just consider one at a time or with weights
    tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g, census_data)

    groups = tt_samples.group.unique() if not groups else groups

    tt_samples_hat = tt_samples
    hops_samples_hat = hops_samples
    com_samples_hat = com_samples

    tt_samples_hat['metric_value'] = series_min_max_norm(tt_samples.metric_value.astype(float))
    hops_samples_hat['metric_value'] = series_min_max_norm(hops_samples.metric_value.astype(float))
    com_samples_hat['metric_value'] = series_min_max_norm(com_samples.metric_value.astype(float))

    reward = 0

    for group in groups:
        reward += -float(np.mean(tt_samples_hat[tt_samples_hat.group == group]['metric_value'])) \
                  - float(np.mean(hops_samples_hat[hops_samples_hat.group == group]['metric_value'])) \
                  + float(np.mean(com_samples_hat[com_samples_hat.group == group]['metric_value']))

    return reward
