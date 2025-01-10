# Functions dispaying metadata in plots
import matplotlib.pyplot as plt
from src.utils_color_maps import all_color_maps

plt.rcParams.update({"font.family": "DejaVu Sans"})
PLT_STYLE = "tableau-colorblind10"
plt.style.use(PLT_STYLE)


def display_diet_information(df, diet_var, age_var, row_label, title="", x_axis=""):
    if diet_var == "diet_weaning":
        group_order = ["no", "yes", "unknown"]
    elif diet_var == "diet_milk":
        group_order = ["bd", "mixed", "fd", "unknown"]
    color_dict = all_color_maps[diet_var]
    test_df = df[[age_var, diet_var]].copy()
    # test_df[diet_var] = test_df[diet_var].astype(str)
    test_df[diet_var] = test_df[diet_var].fillna("unknown")

    df_grouped = test_df.groupby([age_var, diet_var]).size().unstack(fill_value=0)
    df_grouped = df_grouped.sort_index()
    # ensure columns are always sorted the same way
    df_grouped = df_grouped.reindex(columns=group_order)
    # ls_colors = [color_dict[i] for i in df_grouped.columns]
    ax = df_grouped.plot.bar(
        stacked=True,
        figsize=(10, 5),
        color=[color_dict[col] for col in df_grouped.columns],
    )
    ax.set_xlabel(x_axis)
    ax.set_ylabel(f"# {row_label}")
    ax.set_title(title)
    return ax
