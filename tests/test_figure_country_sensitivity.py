import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_country_sensitivity import (  # noqa: E402
    COLUMN_LABELS,
    compose_country_sensitivity_figure,
    save_figure,
)
from tests.test_figure_exposure_shift import synthetic_variants  # noqa: E402


class TestFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        without_country, with_country = synthetic_variants()[:2]
        cls.fig = compose_country_sensitivity_figure(without_country[1:], with_country[1:])

    def test_three_exposure_rows_times_two_model_columns(self):
        self.assertEqual(len(self.fig.axes), 6)
        self.assertEqual([ax.get_title() for ax in self.fig.axes[:2]], list(COLUMN_LABELS))
        self.assertTrue(all(ax.get_title() == "" for ax in self.fig.axes[2:]))

    def test_save_figure_writes_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "supp_country_sensitivity.pdf")
            save_figure(self.fig, path)
            self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main()
