"""
Shifted copies of the antibiotic exposure table and the feature tables for the
sensitivity analysis of the recording resolution of the exposure ages.

All recorded exposure ages are shifted by a fixed number of months and the
exposure-derived columns of the feature tables are recomputed with the rules of
the data-processing pipeline:

- ``abx_any_cumcount``: number of exposures whose start age is at most the
  sample age rounded to the nearest half month (``age_months_rounded05``);
- ``abx_any_last_t_dmonths``: rounded sample age minus the start age of the
  last counted exposure (NaN before the first exposure);
- ``abx_any_last_dur_days``: duration of that exposure, the longest one if
  several share the start age (NaN before the first exposure).

``abx_any_cumdur_days`` and ``abx_max_count_ever`` are copied unchanged. The
outputs are written next to the inputs with the suffix ``_shift_<token>``
(``m05`` for -0.5 months, ``p05`` for +0.5 months). Run from src/:

    uv run python make_shifted_exposure_tables.py --shift_months=-0.5
"""

import os

import pandas as pd
from absl import app, flags

EXPOSURE_TABLE = "ts_vat19_abx_v20240806"
FEATURE_TABLES = [
    "ft_vat19_anomaly_v20240806_entero_family",
    "ft_vat19_anomaly_v20240806_entero_genus",
]
RECOMPUTED_COLUMNS = [
    "abx_any_cumcount",
    "abx_any_last_t_dmonths",
    "abx_any_last_dur_days",
]

FLAGS = flags.FLAGS
flags.DEFINE_float(
    "shift_months", None, "shift of all recorded exposure ages in months"
)
flags.DEFINE_string(
    "original_data_path", "../data/original_data/", "folder of the tables"
)


def shift_token(shift_months):
    """Token of a shift for file and dataset names, e.g. -0.5 -> 'm05'."""
    sign = "m" if shift_months < 0 else "p"
    return sign + f"{abs(shift_months):g}".replace(".", "")


def shift_exposure_table(abx, shift_months):
    """Copy of the exposure table with all start ages shifted by `shift_months`."""
    shifted = abx.copy()
    shifted["abx_start_age_months"] = shifted["abx_start_age_months"] + shift_months
    return shifted


def recompute_exposure_columns(ft, abx):
    """
    Copy of the feature table `ft` with the columns `RECOMPUTED_COLUMNS` derived
    from the exposure rows `abx` (columns host_id, abx_start_age_months,
    abx_duration_days): per sample, the exposures of its host with a start age
    at most the rounded sample age are counted, and the one with the largest
    start age (the longest if several share it) is the last exposure.
    """
    samples = ft[["host_id", "age_months_rounded05"]].copy()
    samples["_sample"] = range(len(ft))
    merged = samples.merge(
        abx[["host_id", "abx_start_age_months", "abx_duration_days"]],
        on="host_id",
        how="inner",
    )
    merged = merged[merged["abx_start_age_months"] <= merged["age_months_rounded05"]]
    merged = merged.sort_values(["_sample", "abx_start_age_months", "abx_duration_days"])
    positions = pd.RangeIndex(len(ft))
    counts = merged.groupby("_sample").size().reindex(positions, fill_value=0)
    last_rows = merged.groupby("_sample", sort=False).tail(1).set_index("_sample")
    last_start = last_rows["abx_start_age_months"].reindex(positions)
    last_duration = last_rows["abx_duration_days"].reindex(positions)

    out = ft.copy()
    out["abx_any_cumcount"] = counts.to_numpy().astype(float)
    out["abx_any_last_t_dmonths"] = (
        ft["age_months_rounded05"].to_numpy() - last_start.to_numpy()
    )
    out["abx_any_last_dur_days"] = last_duration.to_numpy()
    return out


def write_shifted_tables(
    original_data_path,
    shift_months,
    feature_tables=FEATURE_TABLES,
    exposure_table=EXPOSURE_TABLE,
):
    """
    Writes the shifted exposure table and the feature tables with recomputed
    exposure columns next to the originals; returns the written paths.
    """
    token = shift_token(shift_months)
    abx = pd.read_csv(
        os.path.join(original_data_path, f"{exposure_table}.tsv"), sep="\t", index_col=0
    )
    shifted_abx = shift_exposure_table(abx, shift_months)
    abx_path = os.path.join(original_data_path, f"{exposure_table}_shift_{token}.tsv")
    shifted_abx.to_csv(abx_path, sep="\t")
    written = [abx_path]

    abx_rows = shifted_abx.reset_index()
    for name in feature_tables:
        ft = pd.read_csv(
            os.path.join(original_data_path, f"{name}.tsv"), sep="\t", index_col=0
        )
        ft_path = os.path.join(original_data_path, f"{name}_shift_{token}.tsv")
        recompute_exposure_columns(ft, abx_rows).to_csv(ft_path, sep="\t")
        written.append(ft_path)
    return written


def main(argv):
    del argv
    for path in write_shifted_tables(FLAGS.original_data_path, FLAGS.shift_months):
        print("Wrote", path)


if __name__ == "__main__":
    flags.mark_flag_as_required("shift_months")
    app.run(main)
