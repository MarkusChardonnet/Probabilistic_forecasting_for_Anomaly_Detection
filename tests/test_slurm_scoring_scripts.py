import os
import unittest

slurm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "slurm"))


def _step_line(script, ad_params):
    """Line index of the first Microbial_AD_eval.py step of `script` with `ad_params`."""
    with open(os.path.join(slurm_dir, script)) as f:
        lines = f.read().splitlines()
    return next(
        i
        for i, line in enumerate(lines)
        if "Microbial_AD_eval.py" in line and f"--ad_params={ad_params} " in line
    )


class TestScoringStepOrder(unittest.TestCase):
    def test_one_step_scores_precede_the_quick_evaluation(self):
        # the quick evaluation reads the one-step scores (no cut-off) of the reliability evaluation
        for script in ["score_faith_pd.sbatch", "score_shannon.sbatch", "score_country.sbatch"]:
            with self.subTest(script=script):
                self.assertLess(
                    _step_line(script, "param_list_AD_microbial_novel_alpha_div2_reliability_eval"),
                    _step_line(script, "param_list_AD_microbial_novel_alpha_div2_ev"),
                )


if __name__ == "__main__":
    unittest.main()
