import os
import tempfile
import unittest

import pandas as pd

from src.utils_eval_score import _add_md_from_ft, _read_table


class TestUtilsEvalScore(unittest.TestCase):
    def test_read_table_infers_object_strings(self):
        """_read_table should keep legacy object dtype for string columns,
        with missing values represented as NaN, instead of pandas>=3's new
        str dtype."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "tiny.tsv")
            with open(path, "w") as fh:
                fh.write("label\tcount\n")
                fh.write("a\t1\n")
                fh.write("\t2\n")

            df = _read_table(path, sep="\t")

        self.assertEqual(df["label"].dtype, object)
        self.assertEqual(df["count"].dtype, "int64")
        self.assertTrue(pd.isna(df["label"].iloc[1]))

    def test_missing_metadata_label_is_none_after_groupby_last(self):
        """Mirrors the groupby(...).last() pattern in
        _select_samples_around_nth_abx_exposure: a metadata column that is
        missing for every row of a host should yield None (object dtype),
        which the notebooks test with `is None`, not NaN (str dtype)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ft_path = os.path.join(tmp_dir, "ft.tsv")
            cols = [
                "sample_id",
                "age_days",
                "host_id",
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
            rows = [
                ["s1", 10, "h1", 0, 0, 0, 0, "CH", "mixed", "yes", "vaginal", 1.0, 10, 2.0],
                ["s2", 20, "h1", 0, 0, 0, 0, "CH", "mixed", "yes", "vaginal", 1.0, 10, 2.0],
                ["s3", 15, "h2", 0, 0, 0, 0, "CH", "", "no", "c-section", 1.0, 10, 2.0],
                ["s4", 25, "h2", 0, 0, 0, 0, "CH", "", "no", "c-section", 1.0, 10, 2.0],
            ]
            with open(ft_path, "w") as fh:
                fh.write("\t".join(cols) + "\n")
                for row in rows:
                    fh.write("\t".join(str(value) for value in row) + "\n")

            scores = pd.DataFrame(
                {
                    "host_id": ["h1", "h1", "h2", "h2"],
                    "day": [10, 20, 15, 25],
                    "score_1": [0.1, 0.2, 0.3, 0.4],
                }
            )

            out = _add_md_from_ft(scores, "ft", path_to_data=f"{tmp_dir}/")

        last_diet_milk = out.groupby("host_id")["diet_milk"].last()
        self.assertIsNone(last_diet_milk["h2"])
        self.assertEqual(last_diet_milk["h1"], "mixed")


if __name__ == "__main__":
    unittest.main()
