"""Unit + smoke tests. Run with:  python -m unittest discover -s tests -v"""

from __future__ import annotations

import shutil
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import benchmark  # noqa: E402
from src.metrics import classification_metrics, holm_correction, mcnemar_one_sided  # noqa: E402
from src.models import HFTextClassifier  # noqa: E402
from src.perturbations import AVAILABLE, perturb_text  # noqa: E402

TEXT = "You worthless idiot, nobody wants your stupid opinion @user http://x.co"


class TestPerturbations(unittest.TestCase):
    def test_deterministic(self):
        for kind in AVAILABLE:
            a = perturb_text(TEXT, kind, 0.5, 4, 42, "id1")
            b = perturb_text(TEXT, kind, 0.5, 4, 42, "id1")
            self.assertEqual(a, b, kind)

    def test_none_is_identity(self):
        self.assertEqual(perturb_text(TEXT, "none", 0.5, 4, 42, "x").text, TEXT)

    def test_every_perturbation_changes_text(self):
        for kind in AVAILABLE[1:]:
            res = perturb_text(TEXT, kind, 0.5, 4, 42, "x")
            self.assertNotEqual(res.text, TEXT, kind)
            self.assertGreater(res.n_perturbed, 0)

    def test_mentions_and_urls_untouched(self):
        for kind in AVAILABLE[1:]:
            out = perturb_text(TEXT, kind, 1.0, 4, 42, "x").text
            self.assertIn("@user", out)
            self.assertIn("http://x.co", out)

    def test_word_rate_one_perturbs_all_eligible(self):
        res = perturb_text(TEXT, "leetspeak", 1.0, 4, 42, "x")
        self.assertEqual(res.n_perturbed, res.n_eligible)


class TestMetrics(unittest.TestCase):
    def test_classification_metrics(self):
        m = classification_metrics([1, 1, 0, 0], [1, 0, 1, 0], [0.9, 0.2, 0.7, 0.1])
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["fpr"], 0.5)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"]), (1, 1, 1, 1))

    def test_mcnemar_direction(self):
        clean = [True] * 20
        pert = [False] * 15 + [True] * 5
        res = mcnemar_one_sided(clean, pert)
        self.assertEqual((res["b_lost"], res["c_gained"]), (15, 0))
        self.assertLess(res["p_value"], 0.001)
        self.assertEqual(mcnemar_one_sided(clean, clean)["p_value"], 1.0)

    def test_holm(self):
        self.assertEqual(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


class TestHFWrapper(unittest.TestCase):
    """Checks the transformer wrapper logic with a fake `transformers` module (no download)."""

    def setUp(self):
        class FakePipe:
            model = types.SimpleNamespace(config=types.SimpleNamespace(
                id2label={0: "NOT-HATE", 1: "HATE"}, _commit_hash="abc123"))

            def __call__(self, texts, **kw):
                assert kw["truncation"] is True and kw["top_k"] is None
                return [[{"label": "HATE", "score": 0.8 if "idiot" in t else 0.1},
                         {"label": "NOT-HATE", "score": 0.2 if "idiot" in t else 0.9}] for t in texts]

        self.fake_tf = types.ModuleType("transformers")
        self.fake_tf.pipeline = lambda *a, **k: FakePipe()
        self.fake_tf.set_seed = lambda s: None
        self.fake_torch = types.ModuleType("torch")
        self.fake_torch.set_num_threads = lambda n: None
        self._saved = {k: sys.modules.get(k) for k in ("transformers", "torch")}
        sys.modules["transformers"], sys.modules["torch"] = self.fake_tf, self.fake_torch

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_scores_and_label_check(self):
        det = HFTextClassifier("fake", "fake/model", positive_labels=["HATE"], num_threads=1)
        np.testing.assert_allclose(det.predict_proba(["you idiot", "nice day"]), [0.8, 0.1])
        with self.assertRaises(ValueError):
            HFTextClassifier("fake", "fake/model", positive_labels=["TOXIC"])


class TestSmokeRun(unittest.TestCase):
    def test_end_to_end_on_sample(self):
        run_dir = benchmark.main(["--config", "configs/config.yaml"])
        try:
            for f in ("metrics.csv", "hypothesis_tests.csv", "predictions.csv", "run_metadata.json", "benchmark.log"):
                self.assertTrue((run_dir / f).exists(), f)
            metrics = pd.read_csv(run_dir / "metrics.csv")
            self.assertEqual(len(metrics), 2 * len(AVAILABLE))  # 2 enabled models x 5 conditions
            self.assertEqual(int(metrics["errors"].sum()), 0)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
