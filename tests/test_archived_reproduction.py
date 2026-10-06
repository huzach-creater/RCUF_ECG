import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ArchivedReproductionTests(unittest.TestCase):
    def test_archived_predictions_reproduce_paper_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "reproduce_results.py"),
                    "--root",
                    str(ROOT),
                    "--output-dir",
                    str(output),
                    "--bootstrap",
                    "10",
                ],
                check=True,
            )
            audit = pd.read_csv(output / "reproduction_audit.csv")
            self.assertTrue(audit["matches_rounding"].all())


if __name__ == "__main__":
    unittest.main()
