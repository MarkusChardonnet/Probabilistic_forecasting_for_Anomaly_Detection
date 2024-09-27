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
    Get all split scores for all multi-step predictions from path_to_scores. If
    limit_months is set, only returns scores up to this months limit.
    """
    scores = []
    for i in range(1, 4):
        scores.append(pd.read_csv(f"{path_to_scores}{split}_ad_scores_{i}_coord-0.csv"))

    scores_list = [_transform_scores(x) for x in scores]

    scores_all = scores_list[0].copy()
    scores_all = scores_all.join(scores_list[1][["score"]], rsuffix="_2", how="left")
    scores_all = scores_all.join(scores_list[2][["score"]], rsuffix="_3", how="left")
    scores_all.rename(columns={"score": "score_1"}, inplace=True)

    scores_all = _add_month_bins(scores_all)

    if limit_months is not None:
        scores_all = scores_all[scores_all["month5_bin"] <= limit_months].copy()

    return scores_all


def _create_subplot(
    x_axis, y_axis, data, title, ylabel, xlabel, n=None, result_df=None, nb_subplots=2
):
    """Creates boxplot and barplot"""

    height_ratios = [1] + (nb_subplots - 1) * [0.5]
    fig, axs = plt.subplots(
        nb_subplots,
        1,
        figsize=(10, 6),
        height_ratios=height_ratios,
        sharex=True,
        dpi=400,
    )

    # axs[0] is the boxplot
    # category used to have consistent x-axis
    min_x = data[x_axis].min()
    if min_x > 0:
        min_x = 0  # start at zero for consistency
    max_x = data[x_axis].max()
    range_x = np.arange(min_x, max_x + 1)
    data[f"{x_axis}_cat"] = pd.Categorical(data[x_axis], categories=range_x)
    sns.boxplot(x=f"{x_axis}_cat", y=y_axis, data=data, ax=axs[0], color="skyblue")

    axs[0].set_title(title)
    y_max = data[y_axis].max()
    if result_df is not None:
        axs[0].set_ylim(-1, 1.7 * y_max)
    else:
        axs[0].set_ylim(-1, 1.1 * y_max)
    if n is not None:
        zero_index = np.where(range_x == 0)[0][0]
        axs[0].axvline(zero_index, color="darkred")

    # axs[1] is the barplot
    grouped_counts = data.groupby(x_axis)[y_axis].count().reset_index(name="counts")
    sns.barplot(x=x_axis, y="counts", data=grouped_counts, color="peachpuff", ax=axs[1])

    if result_df is not None:
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
                    max_y = data[y_axis].max()
                    axs[0].text(
                        t1 + zero_index,
                        y_shift * max_y,
                        sign,
                        color=color,
                        ha="center",
                        fontsize=25,
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
        axs[1].axvline(zero_index, color="darkred")
        axs[1].set_ylabel("Number of samples")
        axs[1].set_xlabel(f"Months since {n}. abx exposure")

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
        axs[1].axvline(zero_index, color="darkred")

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
    return path_to_plot


def _get_abx_info(path_to_abx_ts: str, limit_months: float = None) -> pd.DataFrame:
    abx_df = pd.read_csv(path_to_abx_ts, sep="\t", index_col=0)
    cols_to_keep = ["abx_start_age_months", "abx_type", "abx_reason"]
    abx_df = abx_df[cols_to_keep].reset_index()
    abx_df.sort_values(["host_id", "abx_start_age_months"], inplace=True)

    if limit_months is not None:
        abx_df = abx_df[abx_df["abx_start_age_months"] <= limit_months].copy()

    return abx_df


def _select_samples_around_nth_abx_exposure(
    md_df,
    abx_df,
    n=1,
    min_samples=-3.0,
    max_samples=12.0,
    group_samples=False,
    score_var="score",
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
    # round to full months for simplicity. note: added 0.01 since lots of 0.5
    # would otw be rounded down leading to uneven sample distribution
    all_samples["diff_age_nth_abx"] = all_samples["diff_age_nth_abx"] + 0.01
    all_samples["diff_age_nth_abx"] = all_samples["diff_age_nth_abx"].round(0)

    # select only samples before and after nth abx exposure
    abx_nth_samples = all_samples.loc[
        np.logical_and(
            ~all_samples["diff_age_nth_abx"].isna(),
            # really only samples around this n-th exposure
            np.logical_and(
                all_samples["abx_any_cumcount"] <= (n + 1),
                all_samples["abx_any_cumcount"] >= n,
            ),
        ),
        :,
    ]

    # only select samples that are up to 3 months prior to n-th abx exposure and
    # 12 months after
    abx_nth_samples = abx_nth_samples.loc[
        np.logical_and(
            abx_nth_samples["diff_age_nth_abx"] >= min_samples,
            abx_nth_samples["diff_age_nth_abx"] <= max_samples,
        ),
        :,
    ]
    # fix -0.0 artifact
    abx_nth_samples["diff_age_nth_abx"] = abx_nth_samples["diff_age_nth_abx"].replace(
        {-0.0: 0.0}
    )
    # remove samples with no observed features
    abx_nth_samples = abx_nth_samples.dropna(subset=[score_var])

    # select last sample prior to abx exposure in range_to_group
    if group_samples:
        range_to_group = [float(x) for x in range(int(min_samples), 0, 1)]
        # select samples to group + to keep
        scores_to_keep = abx_nth_samples[
            ~abx_nth_samples["diff_age_nth_abx"].isin(range_to_group)
        ].copy()
        scores_to_group = abx_nth_samples[
            abx_nth_samples["diff_age_nth_abx"].isin(range_to_group)
        ].copy()
        assert (
            scores_to_keep.shape[0] + scores_to_group.shape[0]
            == abx_nth_samples.shape[0]
        )

        # in scores_to_group select last sample prior to abx expsosure per host_id
        scores_to_group.sort_values(by=["host_id", "diff_age_nth_abx"], inplace=True)
        selected_samples = scores_to_group.loc[
            scores_to_group.groupby("host_id")["diff_age_nth_abx"].idxmax()
        ]
        # replace all values from range_to_group with -1
        for i in range_to_group:
            selected_samples["diff_age_nth_abx"] = selected_samples[
                "diff_age_nth_abx"
            ].replace(i, -1.0)

        # append both groups and resort
        abx_nth_samples = pd.concat([scores_to_keep, selected_samples])
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


def perform_significance_tests(df, t0, t1_values, metric_to_evaluate="diff_metric"):
    results = []

    # Filter the DataFrame for t0
    df_t0 = df.loc[(df["diff_age_nth_abx"] == t0), ["host_id", metric_to_evaluate]]
    df_t0.rename(columns={metric_to_evaluate: "t0"}, inplace=True)

    t1_values.remove(t0)  # no comparison to itself
    for t1 in t1_values:
        # Filter the DataFrame for each t1
        df_t1 = df.loc[(df["diff_age_nth_abx"] == t1), ["host_id", metric_to_evaluate]]
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
        df_t0_t1 = pd.merge(df_t0, df_t1, on="host_id", how="inner")
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
) -> str:
    suff = _get_ordinal_suffix(n)

    # perform paired/unpaired significance tests
    if grouped_samples:
        t1_values = [x for x in range(int(-1.0), int(max_samples + 1))]
    else:
        t1_values = [x for x in range(int(min_samples), int(max_samples + 1))]
    significance_df = perform_significance_tests(data, -1.0, t1_values, y_axis)

    title = f"Score before/after {n}{suff} abx exposure: {tag}"
    ylabel = f"# samples w {y_axis}"
    xlabel = f"Months since {n}{suff} abx exposure"
    if grouped_samples:
        xlabel += f"\n\n(-1 is last sample prior to abx in {min_samples} to -1.0 range)"
    fig, _ = _create_subplot(
        x_axis, y_axis, data, title, ylabel, xlabel, n, significance_df
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
        if i != 2:
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
            f"overall_distribution_samples_t{hide_ylabel_thickmarks}_{flag}.png",
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
                color="red",
                linestyle="--",
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
            f"indv_trajectory_{host_id}_{flag}.png",
        )
        plt.savefig(filename, dpi=400, bbox_inches="tight")
    plt.show()


def get_age_at_1st_2nd_3rd_abx_exposure(abx_df):
    """
    Retrieve age at 1st, 2nd and 3rd abx exposure for each host in abx_df
    """
    i_dic = {1: "1st", 2: "2nd", 3: "3rd"}
    abx_age_at_all = pd.DataFrame(abx_df["host_id"].unique(), columns=["host_id"])

    for i, i_label in i_dic.items():
        # indexing starts at zero
        i -= 1
        abx_age_at_i = abx_df.groupby("host_id").nth(i)
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
