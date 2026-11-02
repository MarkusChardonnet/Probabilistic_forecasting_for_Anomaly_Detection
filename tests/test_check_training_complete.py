import json
import os
import sys
import tempfile
import unittest

import pandas as pd

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

from check_training_complete import trained_epochs  # noqa: E402


def _write_model(folder, model_id, configured_epochs, trained):
    os.makedirs(os.path.join(folder, f"id-{model_id}"))
    pd.DataFrame(
        {"id": [model_id], "description": [json.dumps({"epochs": configured_epochs})]}
    ).to_csv(os.path.join(folder, "model_overview.csv"))
    pd.DataFrame({"epoch": range(1, trained + 1)}).to_csv(
        os.path.join(folder, f"id-{model_id}", f"metric_id-{model_id}.csv")
    )


class TestTrainedEpochs(unittest.TestCase):
    def test_returns_trained_and_configured_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_model(folder, 57, configured_epochs=6, trained=6)
            self.assertEqual(trained_epochs(folder, 57), (6, 6))

    def test_incomplete_training_has_fewer_trained_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            _write_model(folder, 1, configured_epochs=6, trained=4)
            self.assertEqual(trained_epochs(folder, 1), (4, 6))


if __name__ == "__main__":
    unittest.main()
