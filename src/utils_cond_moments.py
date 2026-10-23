import numpy as np
import pandas as pd

import AD_modules


def cond_moments_long_table(
        host_id, abx_labels, cond_moments, observed_dates, obs, cutoff_adj_sf,
        abx_exposure, output_vars, coordinate=0, starting_date=0, epsilon=1e-4,
        replace_values=None, scores=None):
    """One row per observed (infant, day) with the quantities behind the normal score.

    The conditional standard deviation is corrected exactly as in normal scoring
    (AD_modules.get_corrected_var); multiplying it by the scaling factor ``sf``
    gives the standard deviation that the left-tail score uses. The filled
    time-zero value is never an observation and is excluded.

    :param host_id: Infant identifiers [infant].
    :param abx_labels: Exposure labels [infant].
    :param cond_moments: Conditional moments [time, infant, coordinate, moment].
    :param observed_dates: Boolean observation mask [time, infant].
    :param obs: Observed values [time, infant, coordinate].
    :param cutoff_adj_sf: Standard-deviation scaling factors [time, infant].
    :param abx_exposure: Cumulative exposure counts [infant, time].
    :param output_vars: Names identifying the conditional moments.
    :param coordinate: Coordinate to tabulate.
    :param starting_date: Day of the first time step.
    :param epsilon: Minimum variance used in normal scoring.
    :param replace_values: Optional variance replacements from normal scoring.
    :param scores: Optional anomaly scores [infant, time], added as ``score``.
    :return: pandas.DataFrame sorted by infant (input order) and day.
    """
    mean, variance = AD_modules.get_cond_exp_var(cond_moments, output_vars)
    replacement = (np.expand_dims(replace_values, 1)
                   if replace_values is not None else None)
    variance = AD_modules.get_corrected_var(
        np.expand_dims(variance.copy(), 3), epsilon, replacement).squeeze(3)
    observed = np.asarray(observed_dates, dtype=bool).copy()
    observed[0] = False
    rows = []
    for i in range(len(host_id)):
        for t in np.flatnonzero(observed[:, i]):
            row = {
                "host_id": host_id[i],
                "abx": abx_labels[i],
                "day": int(t + starting_date),
                "obs": obs[t, i, coordinate],
                "cond_mean": mean[t, i, coordinate],
                "cond_std": np.sqrt(variance[t, i, coordinate]),
                "sf": cutoff_adj_sf[t, i],
                "abx_exposure": abx_exposure[i, t],
            }
            if scores is not None:
                row["score"] = scores[i, t]
            rows.append(row)
    columns = ["host_id", "abx", "day", "obs", "cond_mean", "cond_std", "sf",
               "abx_exposure"] + (["score"] if scores is not None else [])
    return pd.DataFrame(rows, columns=columns)
