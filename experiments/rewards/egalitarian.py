from typing import List
import igraph as ig
import pandas as pd
import statsmodels.api as sm


def egalitarian(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None) -> float:
    """

    Args:
        g:
        census_data:
        groups:

    Returns:

    """
    tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g, census_data)

    groups = list(tt_samples.group.unique()) if not groups else groups
    assert isinstance(groups, list)

    # fit KDE (sklearn) on each component
    kdes = {group: {metric: None for metric in ['tt', 'hops', 'com']} for group in groups}
    kde_mixtures = {metric: None for metric in ['tt', 'hops', 'com']}

    for group in groups:
        for metric, metric_df in zip(['tt', 'hops', 'com'], [tt_samples, hops_samples, com_samples]):
            X = metric_df[metric_df.group == group].drop(columns='group').astype(float).to_numpy()
            kde = sm.nonparametric.KDEUnivariate(X)
            kde.fit(bw=0.2)
            kdes[group][metric] = kde

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
