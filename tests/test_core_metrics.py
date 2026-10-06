import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_results.py"
SPEC = importlib.util.spec_from_file_location("reproduce_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CoreMetricTests(unittest.TestCase):
    def test_empirical_rank_uses_calibration_reference_and_right_ties(self):
        result = MODULE.empirical_rank(
            np.array([1.0, 2.0, 2.0, 4.0]), np.array([0.0, 2.0, 5.0])
        )
        np.testing.assert_allclose(result, np.array([0.2, 0.8, 1.0]))

    def test_sample_f1_loss_special_cases(self):
        y_true = np.array([[0, 0], [1, 0], [1, 1]])
        y_pred = np.array([[0, 0], [0, 1], [1, 0]])
        np.testing.assert_allclose(
            MODULE.sample_f1_loss(y_true, y_pred),
            np.array([0.0, 1.0, 1.0 / 3.0]),
        )

    def test_dense_area_for_constant_risk(self):
        errors = np.full(20, 0.25)
        risk = np.linspace(0.0, 1.0, len(errors))
        result = MODULE.dense_area_summary(errors, risk)
        self.assertTrue(np.isclose(result["dense_paurc"], 0.125))
        self.assertTrue(np.isclose(result["normalized_dense_paurc"], 0.25))

    def test_rank_fusion_is_component_mean(self):
        import pandas as pd

        calibration = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]})
        result = MODULE.rank_fusion(calibration, calibration, ["a", "b"])
        np.testing.assert_allclose(result, np.array([5.0 / 6.0, 5.0 / 6.0]))


if __name__ == "__main__":
    unittest.main()
