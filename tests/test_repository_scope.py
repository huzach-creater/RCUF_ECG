import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryScopeTests(unittest.TestCase):
    def test_experiment_sources_have_no_plotting_dependencies(self):
        forbidden = ("matplotlib", "seaborn", "plotly", "savefig(", "plt.")
        checked = list((ROOT / "scripts").glob("*.py")) + list((ROOT / "src").rglob("*.py"))
        for path in checked:
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} found in {path.relative_to(ROOT)}")

    def test_removed_experiment_families_are_not_present(self):
        excluded_tokens = ("noise_stress", "noise_lite", "lead_dropout", "lead_corruption")
        relative_paths = [str(path.relative_to(ROOT)).lower() for path in ROOT.rglob("*") if path.is_file()]
        for token in excluded_tokens:
            self.assertFalse(any(token in path for path in relative_paths), token)


if __name__ == "__main__":
    unittest.main()
