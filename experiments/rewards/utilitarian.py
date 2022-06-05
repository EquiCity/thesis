from typing import List
import igraph as ig
import numpy as np
import pandas as pd
from ._utils import get_tt_hops_com_dfs, series_min_max_norm
from experiments.constants.travel_metric import TravelMetric
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def utilitarian(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None,
                metrics: List[TravelMetric] = None, com_threshold: float = 12) -> float:
    """

    Args:
        com_threshold:
        g:
        census_data:
        groups:

    Returns:

    """
    # TODO: Look into multi-obj. learning but for now just consider one at a time or with weights
    tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g, census_data, com_threshold)
    metrics_values = {
        TravelMetric.TT.value: tt_samples,
        TravelMetric.HOPS.value: hops_samples,
        TravelMetric.COM.value: com_samples
    }

    metrics = metrics if metrics is not None else [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]

    metrics_names = [t.value for t in metrics]
    # TODO: integrate this
    metrics_values = {metrics_name: metrics_values[metrics_name] for metrics_name in metrics_names}

    groups = list(tt_samples.group.unique()) if not groups else groups

    tt_samples_hat = tt_samples
    hops_samples_hat = hops_samples
    com_samples_hat = com_samples

    # tt_samples_hat['metric_value'] = series_min_max_norm(tt_samples.metric_value.astype(float))
    # hops_samples_hat['metric_value'] = series_min_max_norm(hops_samples.metric_value.astype(float))
    # com_samples_hat['metric_value'] = series_min_max_norm(com_samples.metric_value.astype(float))

    reward = 0

    for group in groups:
        tt_reward = float(tt_samples_hat[tt_samples_hat.group == group]['metric_value'].values.astype(float).mean())
        tt_reward *= TravelMetric.TT.value in metrics_names
        hops_reward = float(hops_samples_hat[hops_samples_hat.group == group]['metric_value'].values.astype(float).mean())
        hops_reward *= TravelMetric.HOPS.value in metrics_names
        com_reward = float(com_samples_hat[com_samples_hat.group == group]['metric_value'].values.astype(float).mean())
        com_reward *= TravelMetric.COM.value in metrics_names

        group_reward = - tt_reward - hops_reward + com_reward
        # logger.info(f"computed reward {group_reward} for group {group}")
        reward += group_reward

    return reward
