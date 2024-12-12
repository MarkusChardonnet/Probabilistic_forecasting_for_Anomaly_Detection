import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, wilcoxon

plt.rcParams.update({"font.family": "DejaVu Sans"})
plt.style.use("tableau-colorblind10")


def _transform_scores(
    scores_wide: pd.DataFrame, add_col_not_melt: list = None
) -> pd.DataFrame:
    """Transform scores_wide from wide to long"""
    ls_cols_not_to_melt = ["host_id", "abx"]
    if add_col_not_melt is not None:
        ls_cols_not_to_melt += add_col_not_melt

    scores = scores_wide.melt(
        id_vars=ls_cols_not_to_melt, var_name="day", value_name="score"
    )
    scores["day"] = scores["day"].str.extract(r"ad_score_day-(\d+)")[0].astype(int)
    scores.sort_values(["abx", "host_id", "day"], inplace=True)

    return scores


def _add_month_bins(scores: pd.DataFrame, days_per_month=30.437) -> pd.DataFrame:
    """Add month bins to scores"""
    scores["month_bin"] = (scores["day"] / days_per_month).round().astype(int)
    scores["month5_bin"] = (scores["day"] / days_per_month * 2).round() / 2
    return scores


def _get_all_scores(path_to_scores, split="train", limit_months=None):
    """
    Get all split scores for all one-step & multi-step predictions from
    path_to_scores. If limit_months is set, only returns scores up to this
    months limit.
    """
    scores = []
    score_types = list(range(1, 4)) + ["False"]
    for i in score_types:
        scores.append(pd.read_csv(f"{path_to_scores}{split}_ad_scores_{i}_coord-0.csv"))

    scores_list = [_transform_scores(x) for x in scores]

    scores_all = scores_list[0].copy()
    scores_all = scores_all.join(scores_list[1][["score"]], rsuffix="_2", how="left")
    scores_all = scores_all.join(scores_list[2][["score"]], rsuffix="_3", how="left")
    scores_all.rename(columns={"score": "score_1"}, inplace=True)
    # add one-step scores
    scores_all = scores_all.join(scores_list[3][["score"]], how="left")
    scores_all.rename(columns={"score": "score_0"}, inplace=True)

    scores_all = _add_month_bins(scores_all)

    if limit_months is not None:
        scores_all = scores_all[scores_all["month5_bin"] <= limit_months].copy()

    return scores_all


def _add_md_from_ft(abx_scores_flat, ft_name, path_to_data="../data/original_data/"):
    """
    Add metadata matching samples over time from ft to abx_scores_flat
    """
    ft_df = pd.read_csv(f"{path_to_data}{ft_name}.tsv", sep="\t", index_col=0)
    ft_df["age_days"] = ft_df["age_days"].astype(int)
    ft_df.rename(columns={"age_days": "day"}, inplace=True)

    cols_to_evaluate = [
        "abx_any_cumcount",
        "abx_max_count_ever",
        "abx_any_last_t_dmonths",
        "abx_any_last_dur_days",
        "geo_location_name",
        "diet_milk",
        "diet_weaning",
        "delivery_mode",
        "div_alpha_faith_pd",
        "div_alpha_observed_features",
        "div_alpha_shannon",
    ]
    ft_df = ft_df[["day", "host_id"] + cols_to_evaluate].copy()
    ft_df = ft_df.assign(
        max_abx_w_microbiome=lambda df: df.groupby("host_id")[
            "abx_any_cumcount"
        ].transform("max"),
    )
    # add additional information to inferred scores
    abx_scores_flat = abx_scores_flat.merge(ft_df, on=["host_id", "day"], how="left")

    return abx_scores_flat


def _create_subplot(
    x_axis,
    y_axis,
    data,
    title,
    ylabel,
    xlabel,
    step_size=1.0,
    n=None,
    result_df=None,
    nb_subplots=2,
    boxplot_color="skyblue",
):
    """Creates boxplot and barplot"""
    # don't change original data
    data_c = data.copy()

    # below try needed since different versions of python are used for modelling
    # vs. eval
    try:
        height_ratios = [1] + (nb_subplots - 1) * [0.5]
        fig, axs = plt.subplots(
            nb_subplots,
            1,
            figsize=(10, 6),
            height_ratios=height_ratios,
            sharex=True,
            dpi=400,
        )
    except:
        fig, axs = plt.subplots(nb_subplots, 1, figsize=(10, 6), sharex=True, dpi=400)

    # axs[0] is the boxplot
    # category used to have consistent x-axis
    min_x = data_c[x_axis].min()
    if min_x > 0:
        min_x = 0  # start at zero for consistency
    max_x = data_c[x_axis].max()
    range_x = list(np.arange(min_x, max_x + step_size, step_size))
    data_c[f"{x_axis}_cat"] = pd.Categorical(data_c[x_axis], categories=range_x)
    sns.boxplot(
        x=f"{x_axis}_cat", y=y_axis, data=data_c, ax=axs[0], color=boxplot_color
    )

    axs[0].set_title(title)
    y_max = data_c[y_axis].max()
    y_min = data_c[y_axis].min()
    if y_min > -1:
        y_min = -1
    else:
        y_min = y_min * 1.1
    if result_df is not None:
        axs[0].set_ylim(y_min, 1.7 * y_max)
    else:
        axs[0].set_ylim(y_min, 1.1 * y_max)
    if n is not None:
        zero_index = np.where(np.array(range_x) == 0.0)[0][0]
        axs[0].axvline(zero_index - 0.5, color="darkred")

    # add horizontal line at zero if boxplot color is purple
    if boxplot_color == "purple":
        axs[0].axhline(0, color="grey")
        axs[0].set_ylabel(axs[0].get_ylabel(), fontsize=8)

    # axs[1] is the barplot
    grouped_counts = (
        data_c.groupby(f"{x_axis}_cat")[y_axis].count().reset_index(name="counts")
    )
    sns.barplot(
        x=f"{x_axis}_cat", y="counts", data=grouped_counts, color="peachpuff", ax=axs[1]
    )

    if result_df is not None:
        # select only x-axis values that are in range_x
        result_df = result_df[result_df.index.isin(range_x)].copy()
        # Add a star above the boxplots if the p-value < 0.10
        unpaired_color = "sandybrown"
        paired_color = "darkgreen"
        dic_tests = {"unpaired": [1.2, unpaired_color], "paired": [1.1, paired_color]}

        for test, (y_shift, color) in dic_tests.items():
            for t1, p_val in zip(result_df.index, result_df[f"P-value {test}"]):
                if p_val < 0.05:
                    sign = "**"
                elif p_val < 0.1:
                    sign = "*"

                if p_val < 0.1:
                    max_y = data_c[y_axis].max()
                    # correct scaling of x star location with regards to step
                    # size
                    x_star_loc = t1 * (1 / step_size) + zero_index
                    axs[0].text(
                        x_star_loc,
                        y_shift * max_y,
                        sign,
                        color=color,
                        ha="center",
                        fontsize=22 + step_size * 2,
                    )

        # add a count barplot in ax[1] for paired and unpaired
        result_df.reset_index(inplace=True)
        # add a count barplot for paired and unpaired
        for k, v in {
            "unpaired": ["none", unpaired_color],
            "paired": ["none", paired_color],
        }.items():
            df_p = result_df[["t1", f"# samples {k}"]].copy()
            sns.barplot(
                x=df_p["t1"].astype(float),
                y=df_p[f"# samples {k}"],
                ax=axs[1],
                facecolor=v[0],
                edgecolor=v[1],
            )
        axs[1].axvline(zero_index - 0.5, color="darkred")
        axs[1].set_ylabel("Number of samples")
        axs[1].set_xlabel(f"Months since {n}. abx exposure")
        axs[1].tick_params(axis="x", labelsize=min(10 * 22 / len(range_x), 10))

        # Create a custom legend
        custom_lines = [
            Line2D([0], [0], color=unpaired_color, lw=3),
            Line2D([0], [0], color=paired_color, lw=3),
        ]
        custom_cross = [
            Line2D(
                [0],
                [0],
                color=unpaired_color,
                marker="*",
                markersize=12,
                linestyle="None",
            ),
            Line2D(
                [0],
                [0],
                color=paired_color,
                marker="*",
                markersize=12,
                linestyle="None",
            ),
        ]
        legend_txt = ["unpaired to -1.0", "paired to -1.0"]

        axs[0].legend(custom_cross, legend_txt, loc="upper right")
        axs[1].legend(custom_lines, legend_txt)

    if n is not None:
        axs[1].axvline(zero_index - 0.5, color="darkred")

    axs[1].set_ylabel(ylabel)
    axs[1].set_xlabel(xlabel)

    plt.tight_layout()
    return fig, axs


def _filter_samples_by_max_abx_w_microbiome(df, score_col):
    """
    Select only samples from hosts with at least score_col suffix abx exposures
    with microbial samples
    """

    # select only hosts with at least x-th abx exposures with microbial samples
    nth_exposure_nb = float(score_col.split("_")[1])
    df_f = df[df["max_abx_w_microbiome"] >= nth_exposure_nb].copy()

    return df_f


def _plot_score_over_age(
    df: pd.DataFrame,
    y_axis: str,
    flag: str,
    path_to_save: str,
    abx_age_values: pd.Series = None,
) -> str:
    x_axis = "month_bin"

    title = flag
    ylabel = f"# samples w {y_axis}"
    xlabel = f"age in {x_axis}"

    if "noabx" not in flag:
        # select only hosts with at least x-th abx exposures with microbial samples
        df = _filter_samples_by_max_abx_w_microbiome(df, y_axis)
        # filter abx_age also by same hosts
        hosts_relevant = df.host_id.unique().tolist()

        if abx_age_values is not None:
            abx_age_values_f = abx_age_values[
                abx_age_values.index.isin(hosts_relevant)
            ].copy()
            nb_subplots = 3
        else:
            nb_subplots = 2
    else:
        nb_subplots = 2

    # plot
    fig, axs = _create_subplot(
        x_axis, y_axis, df, title, ylabel, xlabel, nb_subplots=nb_subplots
    )

    # display age at abx exposure boxplot
    if ("noabx" not in flag) and (abx_age_values is not None):
        # print(abx_age_values_f.describe())

        # # below line is to verify if axes align
        ax2 = axs[2].twiny()
        sns.swarmplot(
            data=abx_age_values_f, ax=ax2, color="gray", alpha=0.5, size=4, orient="h"
        )
        color_lines = "steelblue"
        sns.boxplot(
            data=abx_age_values_f,
            ax=ax2,
            orient="h",
            showfliers=True,
            color="white",
            saturation=1.0,
            boxprops=dict(facecolor="white", edgecolor=color_lines),
            whiskerprops=dict(color=color_lines),
            capprops=dict(color=color_lines),
            medianprops=dict(color=color_lines),
            linewidth=1,
            width=0.3,
        )
        # abx_age_values_f.plot.box(vert=False, ax=ax2, showfliers=False)
        ax2.set_xlim(-0.5, 24.5)
        ax2.set_ylim(axs[2].get_ylim())
        # ax2 only needs to be visible if we want to verify the axes align
        ax2.axis("off")

        axs[2].set_yticklabels([])
        axs[2].set_ylabel(f"Age at {title} \nexposure")

        axs[2].set_xlabel(axs[1].get_xlabel(), labelpad=10)

    path_to_plot = f"{path_to_save}score_over_age_{flag}_{y_axis}.pdf"
    plt.savefig(path_to_plot)
    plt.close()
    return path_to_plot


def _get_abx_info(path_to_abx_ts: str, limit_months: float = None) -> pd.DataFrame:
    abx_df = pd.read_csv(path_to_abx_ts, sep="\t", index_col=0)
    cols_to_keep = ["abx_start_age_months", "abx_type", "abx_reason"]
    abx_df = abx_df[cols_to_keep].reset_index()
    abx_df.sort_values(["host_id", "abx_start_age_months"], inplace=True)

    # HOTFIX: one host namely has abx_start_date given with 7.6 instead of 7.5,
    # this was already wrongly written in the raw supp. data from original
    # authors
    abx_df.loc[abx_df["host_id"] == "E014403", "abx_start_age_months"] = 7.5

    if limit_months is not None:
        abx_df = abx_df[abx_df["abx_start_age_months"] <= limit_months].copy()

    return abx_df


def _get_step_n_indicator(max_resolution: bool):
    """
    Given which resolution is used, determine step size and last bin indicator.

    Args:
        max_resolution (bool): Whether maximally possible resolution of 0.5 is
        used as step_size or 1.0.

    Returns:
        (float, float): Group step size and indicator for smallest bin.
    """
    if max_resolution:
        group_step = 0.5
        last_bin_indicator = -0.5
    else:
        group_step = 1
        last_bin_indicator = -1.0
    return group_step, last_bin_indicator


def _group_samples_prior_to_cutoff(
    abx_nth_samples: pd.DataFrame,
    col_w_time_since_cutoff: str,
    cutoff_uniqueness: list,
    min_samples: float,
    group_step: float,
    last_bin_indicator: float,
):
    """
    Groups samples in the abx_nth_samples DataFrame prior to a specified cutoff.

    Args:
        abx_nth_samples (pd.DataFrame): DataFrame containing sample data.
        col_w_time_since_cutoff (str): Column name representing time since cutoff.
        cutoff_uniqueness (list): List of columns defining cutoff uniqueness.
        min_samples (float): Minimum bin to include in grouping.
        group_step (float): Step size for grouping samples.
        last_bin_indicator (float): Value to indicate the last bin after
        grouping is performed.

    Returns:
        pd.DataFrame: DataFrame with samples grouped prior to the cutoff.
    """
    # col_w_time_since_cutoff = "diff_age_nth_abx"
    range_to_group = list(np.arange(min_samples, 0.0, group_step))
    # select samples to group + to keep
    scores_to_keep = abx_nth_samples[
        ~abx_nth_samples[col_w_time_since_cutoff].isin(range_to_group)
    ].copy()
    scores_to_group = abx_nth_samples[
        abx_nth_samples[col_w_time_since_cutoff].isin(range_to_group)
    ].copy()
    assert (
        scores_to_keep.shape[0] + scores_to_group.shape[0] == abx_nth_samples.shape[0]
    )
    # in scores_to_group select last sample prior to abx exposure per cutoff_uniqueness
    sort_by = cutoff_uniqueness + [col_w_time_since_cutoff]
    scores_to_group.sort_values(by=sort_by, inplace=True)
    selected_samples = scores_to_group.loc[
        scores_to_group.groupby(cutoff_uniqueness)[col_w_time_since_cutoff].idxmax()
    ]
    # replace all values from range_to_group with last_bin_indicator
    for i in range_to_group:
        selected_samples[col_w_time_since_cutoff] = selected_samples[
            col_w_time_since_cutoff
        ].replace(i, last_bin_indicator)

    # append both groups and resort
    abx_nth_samples = pd.concat([scores_to_keep, selected_samples])
    return abx_nth_samples


def _select_samples_around_nth_abx_exposure(
    md_df,
    abx_df,
    n=1,
    min_samples=-3.0,
    max_samples=12.0,
    group_samples=False,
    score_var="score",
    max_resolution=False,
):
    """
    Get observed samples around n-th abx exposure (n=1 is first abx exposure, n=2 is
    second etc.)

    Args:
        md_df (pd.DataFrame): Contains relevant metadata per host.
        abx_df (pd.DataFrame): Contains start month of abx exposure per host.
        n (int, optional): n-th antibiotics exposure to evaluate. Defaults to 1.
        min_samples (float, optional): Minimum months before n-th abx exposure.
        max_samples (float, optional): Maximum months after n-th abx exposure.
        group_samples (bool, optional): If True, group samples from min_samples to
        -1 to one bucket.
        score_var (str, optional): Column name of score value to be used.
        max_resolution (bool, optional): If True, use max. possible 0.5 months
        resolution.

    Returns:
        pd.DataFrame: Dataframe with observed samples around n-th abx exposure.
    """
    # indexing starts at zero
    n = n - 1
    # calculate age at n-th abx exposure for all hosts
    abx_nth_age = abx_df.groupby("host_id").nth(n)
    abx_nth_age = abx_nth_age.rename(columns={"abx_start_age_months": "age_nth_abx"})

    # add this column to all_samples
    all_samples = pd.merge(md_df, abx_nth_age, on="host_id", how="left")

    # calculate time of samples since n-th abx exposure
    all_samples = all_samples.assign(
        diff_age_nth_abx=all_samples["month5_bin"] - all_samples["age_nth_abx"]
    )

    # only select samples that are up to min_samples months prior to n-th abx
    # exposure and max_samples after
    abx_nth_samples = all_samples.loc[
        np.logical_and(
            all_samples["diff_age_nth_abx"] >= min_samples,
            all_samples["diff_age_nth_abx"] <= max_samples,
        ),
        :,
    ]
    abx_nth_samples = abx_nth_samples.copy()
    # bin time since n-th abx exposure
    if max_resolution:
        step_size = 0.5
    else:
        step_size = 1.0
    bins = np.arange(min_samples, max_samples + step_size, step_size)
    # labels = [f"{int(bins[i])} to {int(bins[i+1])}" for i in range(len(bins)-1)]
    # '-3 to -2', '-2 to -1', '-1 to 0', '0 to 1', '1 to 2', '2 to 3', '3 to 4'
    # Use the left edges of the bins as float labels
    labels = bins[:-1]
    abx_nth_samples.loc[:, "diff_age_nth_abx"] = pd.cut(
        abx_nth_samples["diff_age_nth_abx"], bins=bins, labels=labels, right=False
    )

    # select only samples before and after nth abx exposure
    abx_nth_samples = abx_nth_samples.loc[
        np.logical_and(
            ~abx_nth_samples["diff_age_nth_abx"].isna(),
            # really only samples around this n-th exposure
            np.logical_and(
                abx_nth_samples["abx_any_cumcount"] <= (n + 1),
                abx_nth_samples["abx_any_cumcount"] >= n,
            ),
        ),
        :,
    ]

    # remove samples with no observed features
    abx_nth_samples = abx_nth_samples.dropna(subset=[score_var])

    # if there are multiple scores per diff_age_nth_abx bin per host - take last
    # avoids having multiple scores per host per bin
    abx_nth_samples = (
        abx_nth_samples.groupby(["host_id", "diff_age_nth_abx"], observed=True)
        .last()
        .reset_index()
        .copy()
    )
    # select last sample prior to abx exposure in range_to_group
    group_step, last_bin_indicator = _get_step_n_indicator(max_resolution)
    if group_samples:
        abx_nth_samples = _group_samples_prior_to_cutoff(
            abx_nth_samples,
            "diff_age_nth_abx",
            ["host_id"],
            min_samples,
            group_step,
            last_bin_indicator,
        )
        abx_nth_samples.sort_values(["abx", "host_id", "day"], inplace=True)
    return abx_nth_samples


def _get_ordinal_suffix(n):
    return (
        "st"
        if n % 10 == 1 and n != 11
        else "nd"
        if n % 10 == 2 and n != 12
        else "rd"
        if n % 10 == 3 and n != 13
        else "th"
    )


def perform_significance_tests(
    df,
    t0,
    t1_values,
    metric_to_evaluate="diff_metric",
    x_axis="diff_age_nth_abx",
    uniqueness_var_ls=None,
):
    results = []

    if uniqueness_var_ls is not None:
        # add another columns to identify unique samples for paired tests
        unique_vars = ["host_id"] + uniqueness_var_ls
    else:
        unique_vars = ["host_id"]

    # Filter the DataFrame for t0
    cols_to_keep = unique_vars + [metric_to_evaluate]
    df_t0 = df.loc[(df[x_axis] == t0), cols_to_keep]
    df_t0.rename(columns={metric_to_evaluate: "t0"}, inplace=True)

    t1_values.remove(t0)  # no comparison to itself
    for t1 in t1_values:
        # Filter the DataFrame for each t1
        df_t1 = df.loc[(df[x_axis] == t1), cols_to_keep]
        df_t1.rename(columns={metric_to_evaluate: "t1"}, inplace=True)

        # perform the mann-whitney u-test (unpaired/independent)
        t0_unpaired = df_t0["t0"].dropna()
        t1_unpaired = df_t1["t1"].dropna()
        # note: this includes counting potential doubles in case there are
        # multiple samples at t1 that can be compared to t0
        n_unpaired_samples = t1_unpaired.shape[0]  # (df_t0_t1["t1"].notnull()).sum()
        if (t0_unpaired.shape[0] > 0) and (t1_unpaired.shape[0] > 0):
            h_stat_unpair, p_val_unpair = mannwhitneyu(
                t0_unpaired, t1_unpaired, alternative="less", method="exact"
            )
        else:
            h_stat_unpair, p_val_unpair = np.nan, np.nan

        # Perform the wilcoxon test (paired)
        df_t0_t1 = pd.merge(df_t0, df_t1, on=unique_vars, how="inner")
        df_t0_t1.dropna(inplace=True)
        n_paired_samples = df_t0_t1.shape[0]
        if n_paired_samples > 0:
            h_stat_pair, p_val_pair = wilcoxon(
                df_t0_t1["t0"], df_t0_t1["t1"], alternative="less", method="exact"
            )

        else:
            h_stat_pair, p_val_pair = np.nan, np.nan
        results.append(
            (
                t1,
                h_stat_unpair,
                p_val_unpair,
                n_unpaired_samples,
                h_stat_pair,
                p_val_pair,
                n_paired_samples,
            )
        )

    # Create a DataFrame from the results
    result_df = pd.DataFrame(
        results,
        columns=[
            "t1",
            "H-statistic unpaired",
            "P-value unpaired",
            "# samples unpaired",
            "H-statistic paired",
            "P-value paired",
            "# samples paired",
        ],
    )
    result_df.set_index("t1", inplace=True)

    return result_df


def _plot_score_after_nth_abx_exposure(
    data: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    n: int,
    path_to_save: str = None,
    flag: str = "",
    tag: str = "",
    min_samples: float = -3.0,
    max_samples: float = 12.0,
    grouped_samples: bool = False,
    max_resolution: bool = False,
    uniqueness_var_ls: list = None,
    boxplot_color: str = "skyblue",
) -> str:
    suff = _get_ordinal_suffix(n)

    # perform paired/unpaired significance tests
    if max_resolution:
        step_size = 0.5
        t1_reference = -0.5
        end_t1 = max_samples + 0.5
    else:
        step_size = 1
        t1_reference = -1.0
        end_t1 = max_samples

    if grouped_samples:
        start_t1 = t1_reference
    else:
        start_t1 = min_samples

    t1_values = list(np.arange(start_t1, end_t1, step_size))

    significance_df = perform_significance_tests(
        data,
        t1_reference,
        t1_values,
        y_axis,
        x_axis=x_axis,
        uniqueness_var_ls=uniqueness_var_ls,
    )

    title = f"{y_axis} before/after {n}{suff} abx exposure: {tag}"
    ylabel = "# samples"
    xlabel = f"Months since {n}{suff} abx exposure"
    if grouped_samples:
        xlabel += f"\n\n(Here {t1_reference} is last sample prior to abx since {min_samples} months)"
    fig, _ = _create_subplot(
        x_axis,
        y_axis,
        data,
        title,
        ylabel,
        xlabel,
        step_size,
        n,
        significance_df,
        boxplot_color=boxplot_color,
    )

    if path_to_save is not None:
        # if path_to_save is not a path make it a directory
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        path_to_plot = f"{path_to_save}score_after_abx{n}{suff}_{flag}.pdf"
        plt.savefig(path_to_plot)
        return path_to_plot


def display_scatterplot_w_scores(
    dic_to_plot, hide_ylabel_thickmarks=True, sharey=False, path_to_output=None, flag=""
):
    """
    dic_to_plot: dictionary with label and as values: one score_col str and two
    dataframes(md + abx)
    hide_ylabel_thickmarks: hiding thickmarks of y-axis for slides
    """
    n_subplots = len(dic_to_plot)
    if hide_ylabel_thickmarks:
        plt.rcParams.update({"font.size": 6.5})
        fig, axs = plt.subplots(
            1, n_subplots, figsize=(8, 6), sharex=True, sharey=sharey, dpi=400
        )
        markersize = 8
    else:
        plt.rcParams.update({"font.size": 6})
        fig, axs = plt.subplots(
            1, n_subplots, figsize=(9, 10), sharex=True, sharey=sharey, dpi=400
        )
        markersize = 10

    i = 0

    # Create a custom colormap that goes from green to red
    cmap = LinearSegmentedColormap.from_list("green_to_red", ["green", "yellow", "red"])

    ls_score_cols = []
    # Calculate global min and max scores for legend
    global_min_score = float("inf")
    global_max_score = float("-inf")

    for title, v in dic_to_plot.items():
        score_col = v[0]
        ls_score_cols.append(score_col)
        df = v[1]
        abx = v[2]

        # remove all scores with value NaN
        df = df.dropna(subset=[score_col])

        # store min and max scores for legend reference
        global_min_score = min(global_min_score, df[score_col].min())
        global_max_score = max(global_max_score, df[score_col].max())

        # filter if needed
        if abx is not None:
            # remove scores for samples where it is not relevant (e.g. only 1
            # abx exposure, score_2 and score_3 re not relevant)
            nth_exposure_nb = float(score_col.split("_")[1])
            cond_relevant = df["max_abx_w_microbiome"] >= nth_exposure_nb
            df.loc[~cond_relevant, score_col] = np.nan

            # filter abx events by filtered df
            abx_filtered = abx.copy()
            hosts_not_relevant = df[~cond_relevant].host_id.unique().tolist()
            abx_filtered.loc[
                abx_filtered.host_id.isin(hosts_not_relevant), "abx_start_age_months"
            ] = np.nan

        # PLOT: samples first
        scatter1 = sns.scatterplot(
            x="month5_bin",
            y="host_id",
            hue=score_col,
            data=df,
            ax=axs[i],
            s=markersize,
            palette=cmap,
        )
        # abx events second if available
        if abx is not None:
            sns.scatterplot(
                x="abx_start_age_months",
                y="host_id",
                data=abx_filtered,
                ax=axs[i],
                s=markersize * 1.5,
                marker="x",
                color="darkred",
                label="abx event",
            )

        axs[i].set_title(f"Hosts {title} ({df.host_id.nunique()}) - {score_col}")
        axs[i].set_xlabel("Age [months]")
        axs[i].set_ylabel("Host ID")
        axs[i].margins(y=0.005)
        if i != 0:
            axs[i].set_ylabel("")
        if i != len(dic_to_plot) - 1:
            axs[i].get_legend().remove()
        if hide_ylabel_thickmarks:
            axs[i].set_yticklabels([])
        i += 1

    # Create a colorbar legend
    norm = plt.Normalize(global_min_score, global_max_score)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Remove the legend created by seaborn
    scatter1.get_legend().remove()

    # Add the colorbar legend to the right of the subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label="Inferred scores")

    plt.suptitle("Inferred scores over time", fontsize=10, y=1.0)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if path_to_output is not None:
        filename = os.path.join(
            path_to_output,
            f"overall_distribution_samples_t{hide_ylabel_thickmarks}_{flag}.pdf",
        )
        plt.savefig(filename, dpi=400, bbox_inches="tight")
    plt.show()


def plot_trajectory(
    df,
    abx_events,
    host_id,
    score_cols=["score"],
    jitter=False,
    path_to_output=None,
    flag="",
):
    host_data = df[df["host_id"] == host_id]

    plt.figure(figsize=(10, 6))
    for score_col in score_cols:
        if jitter:
            jitter_amount = 0.11
            jittered_y = host_data[score_col] + np.random.normal(
                0, jitter_amount, size=host_data.shape[0]
            )
            sns.lineplot(
                x="month5_bin",
                y=jittered_y,
                data=host_data,
                marker="o",
                label=score_col,
                alpha=0.7,
            )
        else:
            sns.lineplot(
                x="month5_bin",
                y=score_col,
                data=host_data,
                marker="o",
                label=score_col,
                alpha=0.7,
            )

    # Plot ABX events
    if abx_events is not None:
        host_abx_events = abx_events[abx_events["host_id"] == host_id]
        for idx, event in host_abx_events.iterrows():
            plt.axvline(
                x=event["abx_start_age_months"],
                color="darkred",
                # linestyle="-",
                label="abx event" if idx == 0 else None,
            )

    plt.title(f"Trajectory Over Time for Host ID: {host_id}")
    plt.xlabel("Age [months]")
    plt.ylabel("Score")
    plt.grid(True)

    # Add legend manually
    handles, labels = plt.gca().get_legend_handles_labels()
    if "abx event" not in labels:
        handles.append(plt.Line2D([0], [0], color="red", linestyle="--"))
        labels.append("abx event")
    plt.legend(handles=handles, labels=labels)

    if path_to_output is not None:
        filename = os.path.join(
            path_to_output,
            f"indv_trajectory_{host_id}_{flag}.pdf",
        )
        plt.savefig(filename, dpi=400, bbox_inches="tight")
    plt.show()


def _get_age_at_1st_2nd_3rd_abx_exposure(abx_df):
    """
    Retrieve age at 1st, 2nd and 3rd abx exposure for each host in abx_df
    """
    i_dic = {1: "1st", 2: "2nd", 3: "3rd"}
    abx_age_at_all = pd.DataFrame(abx_df["host_id"].unique(), columns=["host_id"])

    for i, i_label in i_dic.items():
        # indexing starts at zero
        i -= 1
        abx_age_at_i = abx_df.groupby("host_id").nth(i)
        # FIX: needed for different py version between modelling and eval
        if "host_id" not in abx_age_at_i.columns:
            abx_age_at_i = abx_age_at_i.reset_index()
        new_col_i = f"age_{i_label}_abx"
        abx_age_at_i = abx_age_at_i.rename(columns={"abx_start_age_months": new_col_i})
        abx_age_at_all = pd.merge(
            abx_age_at_all,
            abx_age_at_i[["host_id", new_col_i]],
            on="host_id",
            how="left",
        )
    abx_age_at_all.set_index("host_id", inplace=True)

    return abx_age_at_all


def _filter_hosts_w_microbiome_samples_prior_to_abx(abx_scores_flat, abx_age_at_all):
    """
    Filter abx_scores_flat by hosts that have at least 1 microbiome sample prior
    to 1st abx exposure
    """
    first_sample = abx_scores_flat[["host_id", "month5_bin"]].groupby("host_id").min()
    first_sample.rename(columns={"month5_bin": "first_microbiome_sample"}, inplace=True)

    m_first_vs_abx = pd.merge(
        first_sample, abx_age_at_all[["age_1st_abx"]], on="host_id", how="left"
    )

    # no microbiome sample prior to first abx exposure
    hosts_to_exclude = m_first_vs_abx[
        m_first_vs_abx["age_1st_abx"] < m_first_vs_abx["first_microbiome_sample"]
    ].index
    print(
        f"Number of hosts with 1st abx exposure prior to 1st microbiome sample: \
            {len(hosts_to_exclude)}"
    )

    abx_scores_flat = abx_scores_flat[
        ~abx_scores_flat["host_id"].isin(hosts_to_exclude)
    ].copy()

    print(
        f"Number of hosts w microbiome sample prior to 1st abx exposure: \
            {abx_scores_flat.host_id.nunique()}"
    )

    return abx_scores_flat


def get_scores_n_abx_info(
    scores_path,
    ft_name,
    limit_months=None,
    abx_ts_name=None,
    no_filter=True,
    path_to_data="../data/original_data/",
):
    """Processes scores and abx info for evaluation"""
    # get train & val scores
    scores_train = _get_all_scores(scores_path, "train", limit_months=limit_months)
    scores_val = _get_all_scores(scores_path, "val", limit_months=limit_months)

    # get noabx samples per split
    noabx_train = scores_train[~scores_train["abx"]].copy()
    noabx_val = scores_val[~scores_val["abx"]].copy()

    # select correct noabx scores
    noabx_train.drop(columns=["score_2", "score_3"], inplace=True)
    noabx_val.drop(columns=["score_2", "score_3"], inplace=True)

    # merge all abx scores into one group: train + val
    # since none of the abx samples were used for training
    abx_scores_flat = scores_train[scores_train["abx"]].copy()
    abx_scores_flat_val = scores_val[scores_val["abx"]].copy()
    abx_scores_flat = pd.concat([abx_scores_flat, abx_scores_flat_val])

    # add more metadata from ft
    # - to abx
    abx_scores_flat = _add_md_from_ft(
        abx_scores_flat, ft_name, path_to_data=path_to_data
    )
    # - to noabx
    noabx_train = _add_md_from_ft(noabx_train, ft_name, path_to_data=path_to_data)
    noabx_val = _add_md_from_ft(noabx_val, ft_name, path_to_data=path_to_data)

    # drop all rows with no observations available from abx
    abx_scores_flat = abx_scores_flat.dropna(subset=["score_1"]).copy()

    if abx_ts_name is not None:
        # get start of each abx course per host
        abx_df = _get_abx_info(
            f"{path_to_data}{abx_ts_name}.tsv", limit_months=limit_months
        )

        # get age at n-th abx exposures
        abx_age_at_all = _get_age_at_1st_2nd_3rd_abx_exposure(abx_df)

        # filter hosts by at least 1 microbiome sample prior to 1st abx exposure
        if not no_filter:
            abx_scores_flat = _filter_hosts_w_microbiome_samples_prior_to_abx(
                abx_scores_flat, abx_age_at_all
            )
    else:
        abx_df = None
        abx_age_at_all = None

    return noabx_train, noabx_val, abx_scores_flat, abx_df, abx_age_at_all


def plot_time_between_abx_exposures(
    abx_age_at_all, n0_label="1st", n1_label="2nd", path_to_save=None
):
    """
    Visualize time between n0_label and n1_label abx exposure
    """
    col_n0 = f"age_{n0_label}_abx"
    col_n1 = f"age_{n1_label}_abx"
    # add this column to all_samples
    both_age = abx_age_at_all.copy()
    both_age["diff_age"] = both_age[col_n1] - both_age[col_n0]

    fig, ax = plt.subplots(dpi=400, figsize=(4, 5))
    both_age[["diff_age"]].boxplot(ax=ax)
    ax.set_title(f"Time between {n0_label} and {n1_label} abx exposure")

    if path_to_save is not None:
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        path_to_plot = f"{path_to_save}time_between_{n0_label}_{n1_label}.pdf"
        plt.savefig(path_to_plot)


def get_cutoff_value_sample_sizes(
    abx_scores_flat, abx_df, n, min_samples, max_samples, group_samples
):
    """
    Get abx score cutoff values and sizes for n-th abx exposure
    """
    score_col = f"score_{n}"
    scores_abx_nth_samples = _select_samples_around_nth_abx_exposure(
        abx_scores_flat,
        abx_df,
        n=n,
        min_samples=min_samples,
        max_samples=max_samples,
        group_samples=group_samples,
        score_var=score_col,
    )
    cutoff_values = scores_abx_nth_samples.loc[
        scores_abx_nth_samples["diff_age_nth_abx"] == 0.0, "month_bin"
    ].values

    # dictionary mapping each cutoff_month to its required sample size
    value_counts = pd.Series(cutoff_values).value_counts()
    sample_sizes = value_counts.to_dict()
    return cutoff_values, sample_sizes


def _calculate_mean_diversity_per_cov(noabx, metric, cov_groups):
    noabx_mean_metric = noabx.groupby(cov_groups)[metric].mean().reset_index()
    return noabx_mean_metric.rename(columns={metric: f"mean_{metric}"})


def calculate_matched_metric_n_diff(metric, abx_scores_flat, noabx, cov_groups):
    """
    Calculates the difference between the observed metric and the mean metric
    matched by covariate groups.

    Args:
        metric (str): The name of the metric to calculate and compare.
        abx_scores_flat (pd.DataFrame): DataFrame containing observed metrics
        and covariate information.
        noabx (pd.DataFrame): DataFrame containing data for the 'no antibiotic'
        group - to be used for calculating the matched means of the metric.
        cov_groups (list of str): List of column names to group by for matching.

    Returns:
        pd.DataFrame: The updated 'abx_scores_flat' DataFrame with the
        calculated matched means and differences included.

    """
    # also replace nan in cov_group columns with "unknown" -> better matching
    abx_scores_flat[cov_groups] = abx_scores_flat[cov_groups].fillna("unknown")
    noabx.loc[:, cov_groups] = noabx[cov_groups].fillna("unknown")

    # get average diversity metric for noabx infants grouped by covariates
    noabx_mean_metric = _calculate_mean_diversity_per_cov(noabx, metric, cov_groups)

    # match to mean_div from no_abx
    abx_scores_flat = pd.merge(
        abx_scores_flat, noabx_mean_metric, on=cov_groups, how="left"
    )

    # calculate difference between mean_div and observed div
    # ignoring samples where mean metric is nan (no comparable group exists in noabx)
    col_for_diff_to_matched = f"diff_2_matched_{metric}"
    abx_scores_flat[col_for_diff_to_matched] = (
        abx_scores_flat[f"mean_{metric}"] - abx_scores_flat[metric]
    )

    print(
        f"Number of samples disregarded because of lacking reference "
        f"in noabx: {(abx_scores_flat[col_for_diff_to_matched].isna()).sum()}"
    )
    return abx_scores_flat
