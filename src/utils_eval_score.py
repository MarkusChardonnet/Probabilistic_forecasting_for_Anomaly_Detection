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


def _transform_scores(scores_wide: pd.DataFrame) -> pd.DataFrame:
    """Transform scores_wide from wide to long"""
    scores = scores_wide.melt(
        id_vars=["host_id", "abx"], var_name="day", value_name="score"
    )
    scores["day"] = scores["day"].str.extract(r"ad_score_day-(\d+)")[0].astype(int)
    scores.sort_values(["abx", "host_id", "day"], inplace=True)

    return scores


def _add_month_bins(scores: pd.DataFrame, days_per_month=30.437) -> pd.DataFrame:
    """Add month bins to scores"""
    scores["month_bin"] = (scores["day"] / days_per_month).round().astype(int)
    scores["month5_bin"] = (scores["day"] / days_per_month * 2).round() / 2
    return scores


def _get_all_scores(path_to_scores, split="train"):
    """Get all split scores for all multi-step predictions from path_to_scores"""
    scores = []
    for i in range(1, 4):
        scores.append(pd.read_csv(f"{path_to_scores}{split}_ad_scores_{i}_coord-0.csv"))
    scores_t = [_transform_scores(x) for x in scores]
    scores_all = scores_t[0].copy()
    scores_all = scores_all.join(scores_t[1][["score"]], rsuffix="_2", how="left")
    scores_all = scores_all.join(scores_t[2][["score"]], rsuffix="_3", how="left")
    scores_all.rename(columns={"score": "score_1"}, inplace=True)

    scores_all = _add_month_bins(scores_all)
    return scores_all


def _create_subplot(
    x_axis, y_axis, data, title, ylabel, xlabel, n=None, result_df=None
):
    """Creates boxplot and barplot"""
    try:
        fig, axs = plt.subplots(
            2, 1, figsize=(10, 6), height_ratios=[1, 0.5], sharex=True, dpi=400
        )
    except:
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=400)

    # axs[0] is the boxplot
    # category used to have consistent x-axis
    min_x = data[x_axis].min()
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


def _plot_score_over_age(
    df: pd.DataFrame, y_axis: str, flag: str, path_to_save: str
) -> str:
    x_axis = "month_bin"

    title = flag
    ylabel = f"# samples w {y_axis}"
    xlabel = f"age in {x_axis}"

    fig, _ = _create_subplot(x_axis, y_axis, df, title, ylabel, xlabel)

    path_to_plot = f"{path_to_save}score_over_age_{flag}_{y_axis}.pdf"
    plt.savefig(path_to_plot)
    return path_to_plot


def _get_abx_info(path_to_abx_ts: str) -> pd.DataFrame:
    abx_df = pd.read_csv(path_to_abx_ts, sep="\t", index_col=0)
    cols_to_keep = ["abx_start_age_months", "abx_type", "abx_reason"]
    abx_df = abx_df[cols_to_keep].reset_index()
    abx_df.sort_values(["host_id", "abx_start_age_months"], inplace=True)
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
            np.logical_and(all_samples["abx_any_cumcount"] <= (n + 1),
                           all_samples["abx_any_cumcount"] >= n)
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


def display_scatterplot_w_scores(dic_to_plot, hide_ylabel_thickmarks=True):
    """
    dic_to_plot: dictionary with label and as values: one score_col str and two
    dataframes(md + abx)
    hide_ylabel_thickmarks: hiding thickmarks of y-axis for slides
    """
    n_subplots = len(dic_to_plot)
    if hide_ylabel_thickmarks:
        plt.rcParams.update({"font.size": 6.5})
        fig, axs = plt.subplots(1, n_subplots, figsize=(8, 6), sharex=True, dpi=400)
        markersize = 8
    else:
        plt.rcParams.update({"font.size": 6})
        fig, axs = plt.subplots(1, n_subplots, figsize=(9, 10), sharex=True, dpi=400)
        markersize = 10

    i = 0

    # Create a custom colormap that goes from green to red
    cmap = LinearSegmentedColormap.from_list("green_to_red", ["green", "yellow", "red"])

    ls_score_cols = []
    for title, v in dic_to_plot.items():
        score_col = v[0]
        ls_score_cols.append(score_col)
        df = v[1]
        abx = v[2]
        # samples
        scatter1 = sns.scatterplot(
            x="month5_bin",
            y="host_id",
            hue=score_col,
            data=df,
            ax=axs[i],
            s=markersize,
            palette=cmap,
        )
        if abx is not None:
            # abx events
            sns.scatterplot(
                x="abx_start_age_months",
                y="host_id",
                data=abx,
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
    # # ! note all scores have same range (min-max) here - so one legend is enough
    # TODO: verify it this remains true
    norm = plt.Normalize(df[score_col].min(), df[score_col].max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Remove the legend created by seaborn
    scatter1.get_legend().remove()

    # Add the colorbar legend
    axs[2].figure.colorbar(sm, ax=axs[2], label="Inferred scores")

    plt.suptitle("Inferred scores over time", fontsize=10, y=1.0)
    plt.tight_layout()
    # filename = os.path.join(
    #     path_to_output,
    #     f"overall_distribution_samples_t{hide_ylabel_thickmarks}.png",
    # )
    # plt.savefig(filename, dpi=400, bbox_inches="tight")
    plt.show()


def plot_trajectory(df, abx_events, host_id, score_cols=["score"], jitter=False):
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

    plt.show()
