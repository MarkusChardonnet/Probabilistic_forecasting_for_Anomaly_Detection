import os
import sys
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from figure_exposure_shift import select_exposure  # noqa: E402
from figure_exposure_shift_flows import (  # noqa: E402
    FATES,
    NODES,
    band_centre,
    compose_flow_figure,
    fate,
    flow_matrix,
    flows_to_long,
    node_name,
    save_figure,
)
from tests.test_figure_exposure_shift import synthetic_variant  # noqa: E402


def _selection(host_days_bins):
    return pd.DataFrame(
        [dict(host_id=h, day=d, diff_age_nth_abx=b) for h, d, b in host_days_bins]
    )


RECORDED = _selection([("A", 10, -1.0), ("A", 40, 0.0), ("B", 20, 0.0), ("C", 50, 1.0)])
SHIFTED = _selection([("A", 10, 0.0), ("A", 40, 1.0), ("C", 50, 1.0), ("D", 70, 2.0)])


class TestFate(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(fate(0.0, 0.0), "same bin")
        self.assertEqual(fate(0.0, 1.0), "neighbouring bin")
        self.assertEqual(fate(2.0, 1.0), "neighbouring bin")
        self.assertEqual(fate(-1.0, 0.0), "crosses the cut-off")
        self.assertEqual(fate(0.0, -1.0), "crosses the cut-off")
        self.assertEqual(fate(0.0, "not used"), "leaves the selection")
        self.assertEqual(fate("not used", -1.0), "enters the selection")
        self.assertEqual(sorted(set(FATES)), sorted(FATES))

    def test_node_name(self):
        self.assertEqual(node_name(-1.0), "-1")
        self.assertEqual(node_name(3.0), "3")
        self.assertEqual(node_name("not used"), "not used")


class TestFlowMatrix(unittest.TestCase):
    def test_counts_by_sample_key(self):
        M = flow_matrix(RECORDED, SHIFTED)
        self.assertEqual(list(M.index), NODES)
        self.assertEqual(list(M.columns), NODES)
        self.assertEqual(M.loc[-1.0, 0.0], 1)  # A day 10 crosses
        self.assertEqual(M.loc[0.0, 1.0], 1)  # A day 40 moves up
        self.assertEqual(M.loc[0.0, "not used"], 1)  # B leaves
        self.assertEqual(M.loc[1.0, 1.0], 1)  # C stays
        self.assertEqual(M.loc["not used", 2.0], 1)  # D enters
        self.assertEqual(M.to_numpy().sum(), 5)
        self.assertEqual(M.loc["not used", "not used"], 0)

    def test_marginals_match_the_selections(self):
        _, rec_scores, rec_abx = ("", *synthetic_variant(0.0, seed=0))
        _, sh_scores, sh_abx = ("", *synthetic_variant(-0.5, seed=0))
        for n in (1, 2, 3):
            R = select_exposure(rec_scores, rec_abx, n)
            S = select_exposure(sh_scores, sh_abx, n)
            M = flow_matrix(R, S)
            self.assertEqual(M.drop(index="not used").to_numpy().sum(), len(R))
            self.assertEqual(M.drop(columns="not used").to_numpy().sum(), len(S))
            retained = len(
                set(zip(R["host_id"], R["day"])) & set(zip(S["host_id"], S["day"]))
            )
            self.assertEqual(M["not used"].sum(), len(R) - retained)
            self.assertEqual(M.loc["not used"].sum(), len(S) - retained)
            self.assertGreater(len(R) - retained, 0)

    def test_flows_to_long(self):
        M = flow_matrix(RECORDED, SHIFTED)
        long = flows_to_long({2: M})
        self.assertEqual(list(long.columns), ["exposure", "bin_recorded", "bin_shifted", "n"])
        self.assertEqual(len(long), 5)
        self.assertEqual(long["n"].sum(), 5)
        self.assertIn("not used", long["bin_shifted"].tolist())
        self.assertEqual(long["bin_recorded"].iloc[0], "-1")


class TestBandCentre(unittest.TestCase):
    def test_endpoints_and_midpoint(self):
        self.assertAlmostEqual(band_centre(0.0, 3.0, 9.0), 3.0, places=6)
        self.assertAlmostEqual(band_centre(1.0, 3.0, 9.0), 9.0, places=6)
        self.assertAlmostEqual(band_centre(0.5, 0.0, 10.0), 5.0, places=6)


class TestFigure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, rec_scores, rec_abx = ("", *synthetic_variant(0.0, seed=1))
        _, sh_scores, sh_abx = ("", *synthetic_variant(-0.5, seed=1))
        cls.matrices = {
            n: flow_matrix(
                select_exposure(rec_scores, rec_abx, n), select_exposure(sh_scores, sh_abx, n)
            )
            for n in (1, 2, 3)
        }
        cls.fig = compose_flow_figure(cls.matrices, -0.5)

    def test_three_panels_with_titles(self):
        self.assertEqual(len(self.fig.axes), 3)
        titles = [ax.get_title(loc="left") for ax in self.fig.axes]
        for n, title in zip(("1st", "2nd", "3rd"), titles):
            self.assertIn(f"{n} exposure", title)

    def test_one_band_per_nonzero_cell(self):
        for ax, n in zip(self.fig.axes, (1, 2, 3)):
            bands = [p for p in ax.patches if isinstance(p, PathPatch)]
            self.assertEqual(len(bands), int((self.matrices[n].to_numpy() > 0).sum()))

    def test_node_totals_are_labelled(self):
        M = self.matrices[2]
        texts = [t.get_text() for t in self.fig.axes[1].texts]
        self.assertIn(f"bin −1  {int(M.loc[-1.0].sum())}", texts)
        self.assertIn(f"{int(M[0.0].sum())}  bin 0", texts)
        self.assertTrue(any(t.startswith("enter") for t in texts))
        self.assertTrue(any(t.endswith("leave") for t in texts))
        self.assertTrue(any("dated −15 d" in t for t in texts))

    def test_legend_lists_the_fates(self):
        legend = self.fig.legends[0]
        self.assertEqual([t.get_text() for t in legend.get_texts()], FATES)

    def test_save_figure_writes_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "flows.pdf")
            save_figure(self.fig, output)
            with open(output, "rb") as f:
                self.assertEqual(f.read(4), b"%PDF")


if __name__ == "__main__":
    unittest.main()
