from typing import List
import igraph as ig
import pandas as pd
import statsmodels.api as sm
from ._utils import get_tt_hops_com_dfs
from matplotlib import pyplot as plt
import numpy as np
from inequality.theil import TheilD
from experiments.constants.travel_metric import TravelMetric


def egalitarian_jsd(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None,
                      metrics: List[TravelMetric] = None, com_threshold: float = 12) -> float:
    """

    Args:
        com_threshold:
        g:
        census_data:
        groups:

    Returns:

    """
    tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g, census_data, com_threshold)

    groups = list(tt_samples.group.unique()) if not groups else groups
    assert isinstance(groups, list)

    # fit KDE (sklearn) on each component
    kdes = {group: {metric: None for metric in ['tt', 'hops', 'com']} for group in groups}
    kde_mixtures = {metric: None for metric in ['tt', 'hops', 'com']}

    for metric, metric_df in zip(['tt', 'hops', 'com'], [tt_samples, hops_samples, com_samples]):
        fig, ax = plt.subplots()
        fig.suptitle(f"Plot for {metric=}")
        for group in groups:
            X = metric_df[metric_df.group == group].drop(columns='group').astype(float).to_numpy()
            kde = sm.nonparametric.KDEUnivariate(X)
            kde.fit(bw=0.2)
            kdes[group][metric] = kde
            # score_samples returns the log of the probability density
            ax.plot(kde.support, kde.density, lw=3, label=f"KDE from samples {group=}", zorder=10, color=group)
            ax.scatter(
                X,
                np.abs(np.random.randn(X.size)) / 40,
                marker="x",
                color=group,
                zorder=20,
                label=f"Samples {group=}",
                alpha=0.5,
            )
            ax.legend(loc="best")
            ax.grid(True, zorder=-5)
        plt.show()

    for metric, metric_df in zip(['tt', 'hops', 'com'], [tt_samples, hops_samples, com_samples]):
        X = metric_df.drop(columns='group').astype(float).to_numpy()
        kde = sm.nonparametric.KDEUnivariate(X)
        kde.fit(bw=0.2)
        kde_mixtures[metric] = kde

    reward = 0
    for metric in ['tt', 'hops', 'com']:
        n_dist = len(kdes.keys())
        reward += kde_mixtures[metric].entropy - 1 / n_dist * sum([kdes[group][metric].entropy for group in kdes])

    return -reward


def egalitarian_theil(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None,
                      metrics: List[TravelMetric] = None, com_threshold: float = 12) -> float:
    tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g, census_data, com_threshold)
    metrics_values = {
        TravelMetric.TT.value: tt_samples,
        TravelMetric.HOPS.value: hops_samples,
        TravelMetric.COM.value: com_samples
    }

    metrics = metrics if metrics is not None else [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]

    metrics_names = [t.value for t in metrics]
    metrics_values = {metrics_name: metrics_values[metrics_name] for metrics_name in metrics_names}

    groups = list(tt_samples.group.unique()) if not groups else groups

    assert isinstance(groups, list)

    # fit KDE (sklearn) on each component
    theil_inequality = {metric: None for metric in metrics_names}

    for metric, metrics_values_key in zip(metrics_names, metrics_values):
        metric_df = metrics_values[metrics_values_key]
        X = metric_df.drop(columns='group').astype(float).to_numpy()
        Y = metric_df.group
        theil_inequality[metric] = TheilD(X, Y).T

    return -sum([theil_inequality[k] for k in theil_inequality])
