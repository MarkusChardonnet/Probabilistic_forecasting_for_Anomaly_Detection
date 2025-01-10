import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

plt.rcParams.update({"font.family": "DejaVu Sans"})
PLT_STYLE = "tableau-colorblind10"
plt.style.use(PLT_STYLE)


def extract_color(style_name: str, color_index: int) -> str:
    """
    Extracts a specific color from a given style in matplotlib.

    Parameters:
    style_name (str): Name of the style in matplotlib
    color_index (int): Index of the color to be extracted

    Returns:
    str: The extracted color
    """
    plt.style.use(style_name)
    color_cycle = plt.style.library[style_name]["axes.prop_cycle"]

    # Extract the color at the specified index
    color = color_cycle.by_key()["color"][color_index]

    return color


def assign_cmap_colors(map_dic, cmap_to_use, reverse=False):
    if reverse:
        cmap_to_use += "_r"
    cmap = get_cmap(cmap_to_use)
    keys = list(map_dic.keys())
    for i, key in enumerate(keys):
        map_dic[key] = cmap(i)
    return map_dic


map_study_name = {
    "vatanen19": "#d7191c"
    # : '#ffff9f',
}

map_diet_weaning = {
    "no": None,
    "yes": None,
    "finished": None,
    "unknown": None,
}
map_diet_weaning = assign_cmap_colors(map_diet_weaning, "Set3", reverse=True)

map_diet_milk = {
    "bd": None,
    "mixed": None,
    "fd": None,
    "unknown": None,
}
map_diet_milk = assign_cmap_colors(map_diet_milk, "Set3")
map_diet_milk["unknown"] = map_diet_weaning["unknown"]

map_delivery_mode = {
    "cesarean": None,
    "vaginal": None,
    "cesarean_emergency": None,
    "unknown": None,
}
map_delivery_mode = assign_cmap_colors(map_delivery_mode, "Set2", reverse=True)

map_abx_ever = {
    True: extract_color(PLT_STYLE, 0),
    False: extract_color(PLT_STYLE, 5),
    "unknown": extract_color(PLT_STYLE, 3),
}

map_abx_7d_prior = {
    True: extract_color(PLT_STYLE, 0),
    False: extract_color(PLT_STYLE, 5),
    "unknown": extract_color(PLT_STYLE, 3),
}

map_study_subcohort = {
    "abx": extract_color(PLT_STYLE, 0),
    "t1d": extract_color(PLT_STYLE, 5),
    "karelia": extract_color(PLT_STYLE, 3),
}


def generate_reds(n):
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_red", [(1, 0.5, 0.5), (0.5, 0, 0), (0.2, 0, 0)], N=n
    )
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


map_abx_any_cumcount = {0.0: "blue"}
map_abx_any_cumcount.update(
    {i: color for i, color in enumerate(generate_reds(16), start=1)}
)

all_color_maps = {
    "study_name": map_study_name,
    "diet_milk": map_diet_milk,
    "diet_weaning": map_diet_weaning,
    "delivery_mode": map_delivery_mode,
    "abx_ever": map_abx_ever,
    "abx_7d_prior": map_abx_7d_prior,
    "study_subcohort": map_study_subcohort,
    "abx_any_cumcount": map_abx_any_cumcount,
}
