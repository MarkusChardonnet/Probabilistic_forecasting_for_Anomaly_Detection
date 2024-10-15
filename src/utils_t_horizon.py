# Helper functions for time horizon reliability evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils_eval_score import (
    _add_month_bins,
    _transform_scores,
)


def transform_cutoff_scores(c_scores, days_per_month):
    # transform scores from wide to long format
    c_scores_t = _transform_scores(c_scores, add_col_not_melt=["use_obs_until_day"])
    c_scores_t = _add_month_bins(c_scores_t)

    c_scores_t["cutoff_month"] = (
        (c_scores_t["use_obs_until_day"] / days_per_month).round().astype(int)
    )

    # drop all rows with no scores available
    c_scores_t = c_scores_t.dropna(subset=["score"]).copy()

    return c_scores_t


def enrich_scores(c_scores_t):
    """Retrieve information about previous observations before the cutoff date"""
    # get last observation and number of observations before cutoff per host
    c_scores_t["days_since_last_obs_before_cutoff"] = (
        c_scores_t["day"] - c_scores_t["use_obs_until_day"]
    )

    grouped_scores_per_host = c_scores_t.groupby("host_id")[
        "days_since_last_obs_before_cutoff"
    ]

    days_since_last_obs_before_cutoff = grouped_scores_per_host.apply(
        lambda x: x[x < 0].max()
    )
    nb_obs_before_cutoff = grouped_scores_per_host.apply(
        lambda x: (x < 0).sum()
    ).rename("nb_obs_before_cutoff")

    cutoff_info = days_since_last_obs_before_cutoff.to_frame()
    cutoff_info = pd.merge(
        cutoff_info, nb_obs_before_cutoff.to_frame(), left_index=True, right_index=True
    )

    # merge per host info to scores
    c_scores_t.drop(columns=["days_since_last_obs_before_cutoff"], inplace=True)
    c_scores_t_m = pd.merge(c_scores_t, cutoff_info, how="left", on="host_id")

    # add days since cutoff
    c_scores_t_m["days_since_cutoff"] = (
        c_scores_t_m["day"] - c_scores_t_m["use_obs_until_day"]
    ).astype(float)

    # add months since cutoff
    c_scores_t_m["months_since_cutoff"] = (
        c_scores_t_m["month5_bin"] - c_scores_t_m["cutoff_month"]
    ).astype(float)

    # monthly rounding
    # round to full months for simplicity. note: added 0.01 since lots of 0.5
    # would otw be rounded down leading to uneven sample distribution
    c_scores_t_m["months_since_cutoff"] = c_scores_t_m["months_since_cutoff"] + 0.01
    c_scores_t_m["months_since_cutoff"] = np.round(
        c_scores_t_m["months_since_cutoff"], 0
    )
    # fix -0.0: these are samples that were obtained prior to abx exposure!
    # TODO: can be ignored if we go with 0.5 months resolution
    bool_sample_prior = np.logical_and(
        c_scores_t_m["month5_bin"] < c_scores_t_m["cutoff_month"],
        c_scores_t_m["months_since_cutoff"] == -0.0,
    )
    c_scores_t_m.loc[bool_sample_prior, "months_since_cutoff"] = -1.0

    return c_scores_t_m


def plot_cutoff_date_distribution(c_scores_all, flag=""):
    unique_cutoffs = c_scores_all[["host_id", "use_obs_until_day"]].drop_duplicates()
    unique_cutoffs["use_obs_until_day"] = unique_cutoffs["use_obs_until_day"].astype(
        int
    )
    cutoff_counts = unique_cutoffs["use_obs_until_day"].value_counts().sort_index()
    plt.figure(figsize=(20, 5))
    sns.barplot(x=cutoff_counts.index, y=cutoff_counts.values)
    plt.title(f"Distribution of cutoff values {flag}")
    plt.xlabel("cutoff (days)")
    plt.ylabel("Count")
    plt.show()


def sample_from_each_group(group, sample_sizes, seed):
    """Sample sample_sizes.values rows from each group based on the column."""
    # Get the sample size for this group from the sample_sizes dictionary
    n = sample_sizes.get(group.name, 0)
    # Ensure there are enough rows to sample
    if len(group) < n:
        raise ValueError(
            f"Not enough rows to sample for cutoff_month {group.name}. Needed {n}, but only {len(group)} available."
        )
    # Sample n rows from the group
    return group.sample(n=n, random_state=seed)


def display_two_distributions(orig_values, sampled_values):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # original
    axes[0].hist(orig_values, bins=30, edgecolor="black", color="blue")
    axes[0].set_title("Original distribution")
    axes[0].set_ylabel("Frequency")

    # sampled
    axes[1].hist(sampled_values, bins=30, edgecolor="black", color="orange")
    axes[1].set_title("Sampled distribution")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
