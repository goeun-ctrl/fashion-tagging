"""CPU synthetic diagnostics; never build EVA or execute CUDA operations."""
import random
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from . import monitoring as m, trainer


class MonitoringTests(unittest.TestCase):
    cfg = {"max_epochs": 30, "patience": 5}

    def row(self, epoch, macro=0.5, train=0.6, loss=0.3):
        return {"epoch": epoch, "holdout_macro": macro, "holdout_bce": loss,
                "diagnostic_train_macro": train, "diagnostic_train_bce": 0.2}

    def test_patience_counts_ties_and_resets_only_on_strict_improvement(self):
        history, best, epoch = [], -1., 0
        for i, score in enumerate([0.5, 0.5, 0.6, 0.59, 0.58, 0.57, 0.56, 0.55], 1):
            row, best, epoch = m.monitor_epoch(history, self.row(i, score), best, epoch, self.cfg)
            history.append(row)
        self.assertEqual(history[1]["best_epoch"], 1)
        self.assertEqual(history[2]["epochs_without_improvement"], 0)
        self.assertFalse(history[-2]["training_complete"])
        self.assertTrue(history[-1]["early_stopping_triggered"])
        self.assertEqual((best, epoch, history[-1]["epochs_without_improvement"]), (0.6, 3, 5))

    def test_max_epoch_stops_even_with_improvement(self):
        history = [{**self.row(i), "train_holdout_macro_gap": 0.1} for i in range(1, 30)]
        row, _, best_epoch = m.monitor_epoch(history, self.row(30, 0.8), 0.7, 29, self.cfg)
        self.assertEqual(best_epoch, 30)
        self.assertTrue(row["training_complete"])
        self.assertFalse(row["early_stopping_triggered"])
        self.assertEqual(row["stop_reason"], "max_epochs")

    def test_warning_trends_do_not_stop_training(self):
        history, best, best_epoch = [], -1., 0
        for i in range(1, 5):
            row, best, best_epoch = m.monitor_epoch(history,
                self.row(i, 0.7 - i * 0.01, 0.7 + i * 0.01, 0.3 + i * 0.01), best, best_epoch, self.cfg)
            history.append(row)
        self.assertEqual(len(history[-1]["overfitting_warnings"].split("|")), 3)
        self.assertFalse(history[-1]["training_complete"])
        self.assertEqual(history[-2]["overfitting_warnings"], "")

    def test_large_static_gap_and_broken_trends_do_not_warn(self):
        history = [{**self.row(i, 0.1, 0.99), "train_holdout_macro_gap": 0.89} for i in range(1, 5)]
        self.assertEqual(m.overfitting_warnings(history), [])
        history[-1]["holdout_bce"] = 0.2
        self.assertEqual(m.overfitting_warnings(history), [])

    def test_holdout_loss_warning_without_train_improvement(self):
        history = [{**self.row(i, loss=0.2 + i * 0.1), "train_holdout_macro_gap": 0.1} for i in range(1, 5)]
        self.assertEqual(m.overfitting_warnings(history), ["holdout_bce_worsening"])

    def test_subset_repeatability_disjointness_and_rng_independence(self):
        names = [f"train_{i}.jpg" for i in range(100)]
        targets = np.zeros((100, 27), dtype=np.int8)
        groups = {"train_inner": (Path("train"), names, targets),
                  "holdout": (None, ["hold.jpg"], None), "val": (None, ["val.jpg"], None)}
        state = np.random.get_state()
        first, meta = m.diagnostic_subset(groups, 20)
        second, meta2 = m.diagnostic_subset(groups, 20)
        after = np.random.get_state()
        self.assertEqual(first[1], second[1])
        self.assertEqual(meta, meta2)
        self.assertEqual(len(set(first[1])), 20)
        self.assertFalse(set(first[1]) & {"hold.jpg", "val.jpg"})
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        groups["val"] = (None, [names[0]], None)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            m.diagnostic_subset(groups, 20)

    def test_nonfinite_metrics_fail_before_best_update(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            m.monitor_epoch([], self.row(1, float("nan")), -1, 0, self.cfg)

    def test_diagnostic_restores_rng_and_mode_even_on_failure(self):
        model = Mock(training=False)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()

        def fail(*args):
            random.random()
            np.random.random()
            torch.rand(1)
            raise RuntimeError("synthetic metric failure")

        with patch.object(trainer.baseline, "measure", side_effect=fail), patch.object(torch.cuda, "is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "synthetic metric failure"):
                trainer.measure_diagnostic(model, None, self.cfg)
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng)
        model.train.assert_called_once_with(False)

    def test_existing_adamw_decay_and_lr_groups_preserved(self):
        # Synthetic parameters only; no forward, backward or optimizer step.
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2)
        model.head = torch.nn.Linear(2, 27)
        model.get_classifier = lambda: model.head
        optimizer = trainer.baseline.optimizer(model)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [5e-5, 1e-3])
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0.05, 0.05])
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]},
                         {id(p) for p in model.parameters()})


if __name__ == "__main__":
    unittest.main()
