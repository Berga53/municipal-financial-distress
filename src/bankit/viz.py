"""Summary statistics and plotting helpers used by results.ipynb."""

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse


def summarize_metric(metric_series):
    return pd.Series({
        'min': metric_series.min(),
        'p5': np.percentile(metric_series, 5),
        'p25': np.percentile(metric_series, 25),
        'p50': np.percentile(metric_series, 50),
        'p75': np.percentile(metric_series, 75),
        'p95': np.percentile(metric_series, 95),
        'max': metric_series.max(),
        'mean': metric_series.mean(),
        'std': metric_series.std()
    })


def plot_confidence_ellipse(x, y, ax, n_std=1.0, **kwargs):
    if len(x) < 2:
        return  # Can't draw ellipse
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    if cov.shape == (2, 2):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(vals)
        ellipse = Ellipse((mean_x, mean_y), width, height, angle=theta, **kwargs)
        ax.add_patch(ellipse)
