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
    # round to full months for simplicity. note: added 0.01 since lots of 0.5
    # would otw be rounded down leading to uneven sample distribution
    c_scores_t_m["months_since_cutoff"] = c_scores_t_m["months_since_cutoff"] + 0.01
    c_scores_t_m["months_since_cutoff"] = np.round(
        c_scores_t_m["months_since_cutoff"], 0
    )
    # fix -0.0 artifact
    c_scores_t_m["months_since_cutoff"] = c_scores_t_m["months_since_cutoff"].replace(
        {-0.0: 0.0}
    )
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

def plot_cutoff_host_id_distribution(c_scores_all, flag=""):
    # Count unique host_ids per use_obs_until_day
    unique_cutoffs = c_scores_all.groupby("use_obs_until_day")["host_id"].nunique().reset_index()
    unique_cutoffs.columns = ["use_obs_until_day", "unique_host_count"]
    
    # Convert use_obs_until_day to integer
    unique_cutoffs["use_obs_until_day"] = unique_cutoffs["use_obs_until_day"].astype(int)
    
    # Sort by use_obs_until_day
    unique_cutoffs = unique_cutoffs.sort_values("use_obs_until_day")
    
    # Create the plot
    plt.figure(figsize=(20, 5))
    sns.barplot(x="use_obs_until_day", y="unique_host_count", data=unique_cutoffs)
    plt.title(f"Distribution of unique host_ids per cutoff value {flag}")
    plt.xlabel("Cutoff (days)")
    plt.ylabel("Count of unique host_ids")
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
    sampled = group.sample(n=n, random_state=seed)
    while sampled.host_id.nunique() != n:
        print(
            "Resampling since multiple host_ids were selected for cutoff value {group.name}"
        )
        sampled = group.sample(n=n, random_state=seed)
    return sampled


def sample_unique_host_ids(cutoff_host_mapping, sample_sizes, seed=42):
    selected_host_ids = set()
    sampled_frames = []

    # Optionally, you can sort the groups to maximize the chance of successful sampling
    # For example, process groups with the smallest sample_sizes first
    groups = cutoff_host_mapping.groupby('cutoff_month')
    # sorted_groups = sorted(groups, key=lambda x: sample_sizes.get(x[0], 0))

    for group_name, group in groups:
        n = sample_sizes.get(group_name, 0)

        # Remove rows with host_ids already selected
        available_group = group[~group['host_id'].isin(selected_host_ids)]

        if len(available_group) < n:
            raise ValueError(
                f"Not enough unique host_ids to sample for cutoff_month {group_name}. "
                f"Needed {n}, but only {len(available_group)} unique host_ids available."
            )

        sampled_group = available_group.sample(n=n, random_state=seed)
        sampled_frames.append(sampled_group)
        selected_host_ids.update(sampled_group['host_id'])

    # Concatenate the sampled groups
    cutoff_host_mapping_subset = pd.concat(sampled_frames).reset_index(drop=True)
    return cutoff_host_mapping_subset


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
