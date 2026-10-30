import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from make_shifted_exposure_tables import (  # noqa: E402
    EXPOSURE_TABLE,
    FEATURE_TABLES,
    RECOMPUTED_COLUMNS,
    recompute_exposure_columns,
    shift_exposure_table,
    shift_token,
    write_shifted_tables,
)

ORIGINAL_DATA = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "original_data")
)
PUBLISHED_TABLES_EXIST = all(
    os.path.exists(os.path.join(ORIGINAL_DATA, f"{name}.tsv"))
    for name in [EXPOSURE_TABLE] + FEATURE_TABLES
)

# exposure table as read from disk: host_id is the index
ABX = pd.DataFrame(
    {
        "host_id": ["H1", "H1", "H3"],
        "abx_start_age_months": [6.0, 8.5, 0.0],
        "abx_duration_days": [7.0, 10.0, 5.0],
    }
).set_index("host_id")
# feature table: the three recomputed columns hold placeholders (9.0)
FT = pd.DataFrame(
    {
        "host_id": ["H1", "H1", "H1", "H1", "H1", "H2", "H3"],
        "age_months_rounded05": [3.0, 5.5, 6.0, 6.5, 9.0, 2.0, 1.0],
        "abx_any_cumcount": 9.0,
        "abx_any_last_t_dmonths": 9.0,
        "abx_any_last_dur_days": 9.0,
        "abx_any_cumdur_days": 1.0,
    },
    index=[f"S{i}" for i in range(7)],
)
NAN = np.nan


class ShiftTokenTest(unittest.TestCase):
    def test_shift_token(self):
        self.assertEqual(shift_token(-0.5), "m05")
        self.assertEqual(shift_token(0.5), "p05")
        self.assertEqual(shift_token(1.0), "p1")
        self.assertEqual(shift_token(-1.5), "m15")


class ShiftExposureTableTest(unittest.TestCase):
    def test_shift_exposure_table_adds_shift(self):
        abx = ABX.reset_index()
        shifted = shift_exposure_table(abx, 0.5)
        self.assertEqual(list(shifted["abx_start_age_months"]), [6.5, 9.0, 0.5])
        self.assertEqual(list(abx["abx_start_age_months"]), [6.0, 8.5, 0.0])
        pd.testing.assert_series_equal(
            shifted["abx_duration_days"], abx["abx_duration_days"]
        )
        pd.testing.assert_index_equal(shifted.index, abx.index)


class RecomputeExposureColumnsTest(unittest.TestCase):
    def test_recompute_counts_last_exposure(self):
        out = recompute_exposure_columns(FT, ABX.reset_index())
        np.testing.assert_array_equal(
            out["abx_any_cumcount"].to_numpy(), [0, 0, 1, 1, 2, 0, 1]
        )
        np.testing.assert_array_equal(
            out["abx_any_last_t_dmonths"].to_numpy(),
            [NAN, NAN, 0.0, 0.5, 0.5, NAN, 1.0],
        )
        np.testing.assert_array_equal(
            out["abx_any_last_dur_days"].to_numpy(),
            [NAN, NAN, 7.0, 7.0, 10.0, NAN, 5.0],
        )
        np.testing.assert_array_equal(out["abx_any_cumdur_days"].to_numpy(), 1.0)
        pd.testing.assert_index_equal(out.index, FT.index)
        self.assertTrue((FT["abx_any_cumcount"] == 9.0).all())

    def test_recompute_with_shifted_table(self):
        plus = recompute_exposure_columns(FT, shift_exposure_table(ABX.reset_index(), 0.5))
        self.assertEqual(plus.loc["S2", "abx_any_cumcount"], 0)
        self.assertTrue(np.isnan(plus.loc["S2", "abx_any_last_t_dmonths"]))
        self.assertEqual(plus.loc["S3", "abx_any_cumcount"], 1)
        self.assertEqual(plus.loc["S3", "abx_any_last_t_dmonths"], 0.0)

        minus = recompute_exposure_columns(FT, shift_exposure_table(ABX.reset_index(), -0.5))
        self.assertEqual(minus.loc["S1", "abx_any_cumcount"], 1)
        self.assertEqual(minus.loc["S1", "abx_any_last_t_dmonths"], 0.0)
        self.assertEqual(minus.loc["S1", "abx_any_last_dur_days"], 7.0)
        # an exposure shifted to -0.5 months still counts for every sample
        self.assertEqual(minus.loc["S6", "abx_any_cumcount"], 1)
        self.assertEqual(minus.loc["S6", "abx_any_last_t_dmonths"], 1.5)

    @unittest.skipUnless(PUBLISHED_TABLES_EXIST, "published tables not available")
    def test_shift_zero_reproduces_published_columns(self):
        abx = pd.read_csv(
            os.path.join(ORIGINAL_DATA, f"{EXPOSURE_TABLE}.tsv"), sep="\t", index_col=0
        ).reset_index()
        for name in FEATURE_TABLES:
            ft = pd.read_csv(
                os.path.join(ORIGINAL_DATA, f"{name}.tsv"), sep="\t", index_col=0
            )
            out = recompute_exposure_columns(ft, abx)
            for col in RECOMPUTED_COLUMNS:
                pd.testing.assert_series_equal(out[col], ft[col], check_dtype=False)


class WriteShiftedTablesTest(unittest.TestCase):
    def test_write_shifted_tables_names_and_originals_untouched(self):
        name = FEATURE_TABLES[0]
        with tempfile.TemporaryDirectory() as tmp:
            abx_path = os.path.join(tmp, f"{EXPOSURE_TABLE}.tsv")
            ft_path = os.path.join(tmp, f"{name}.tsv")
            ABX.to_csv(abx_path, sep="\t")
            FT.to_csv(ft_path, sep="\t")
            with open(abx_path, "rb") as f:
                abx_bytes = f.read()
            with open(ft_path, "rb") as f:
                ft_bytes = f.read()

            paths = write_shifted_tables(tmp, 0.5, feature_tables=[name])

            self.assertEqual(
                [os.path.basename(p) for p in paths],
                [f"{EXPOSURE_TABLE}_shift_p05.tsv", f"{name}_shift_p05.tsv"],
            )
            for p in paths:
                self.assertTrue(os.path.exists(p))
            abx_out = pd.read_csv(paths[0], sep="\t", index_col=0)
            self.assertEqual(list(abx_out["abx_start_age_months"]), [6.5, 9.0, 0.5])
            self.assertEqual(abx_out.index.name, "host_id")
            ft_out = pd.read_csv(paths[1], sep="\t", index_col=0)
            pd.testing.assert_index_equal(ft_out.index, FT.index)
            self.assertEqual(ft_out.loc["S2", "abx_any_cumcount"], 0)
            self.assertEqual(ft_out.loc["S3", "abx_any_cumcount"], 1)
            with open(abx_path, "rb") as f:
                self.assertEqual(f.read(), abx_bytes)
            with open(ft_path, "rb") as f:
                self.assertEqual(f.read(), ft_bytes)


if __name__ == "__main__":
    unittest.main()
