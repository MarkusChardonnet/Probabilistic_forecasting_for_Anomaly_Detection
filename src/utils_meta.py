# Functions dispaying metadata in plots
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from src.utils_color_maps import all_color_maps

plt.rcParams.update({"font.family": "DejaVu Sans"})
PLT_STYLE = "tableau-colorblind10"
plt.style.use(PLT_STYLE)


def display_diet_information(
    df, diet_var, age_var, row_label, title="", x_axis="", path_to_save=None
):
    if diet_var == "diet_weaning":
        group_order = ["no", "yes", "unknown"]
    elif diet_var == "diet_milk":
        group_order = ["bd", "mixed", "fd", "unknown"]
    color_dict = all_color_maps[diet_var]
    test_df = df[[age_var, diet_var]].copy()
    test_df[diet_var] = test_df[diet_var].fillna("unknown")

    df_grouped = test_df.groupby([age_var, diet_var]).size().unstack(fill_value=0)
    df_grouped = df_grouped.sort_index()
    # ensure columns are always sorted the same way
    df_grouped = df_grouped.reindex(columns=group_order)
    # ls_colors = [color_dict[i] for i in df_grouped.columns]

    # plot
    fig, ax = plt.subplots(figsize=(9, 6), dpi=400)
    df_grouped.plot.bar(
        ax=ax,
        stacked=True,
        figsize=(10, 5),
        color=[color_dict[col] for col in df_grouped.columns],
    )
    range_x = df_grouped.index.tolist()
    zero_index = np.where(np.array(range_x) == 0.0)[0][0]
    ax.axvline(zero_index - 0.5, color="darkred")

    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
    ax.set_ylabel(f"# {row_label}", fontsize=10)
    ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if path_to_save is not None:
        suffix = re.search(r"since (\d+\w+) abx", x_axis).group(1)
        filename = os.path.join(path_to_save, f"{diet_var}_after_abx{suffix}.pdf")
        print(filename)
        plt.savefig(filename, dpi=400, bbox_inches="tight", format="pdf")

    return fig, ax
