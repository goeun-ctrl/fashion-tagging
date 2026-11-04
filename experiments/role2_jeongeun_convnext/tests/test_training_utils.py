import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_utils import accumulation_window_samples, improved_macro, lr_multiplier, validate_partition


class TrainingContractTests(unittest.TestCase):
    def test_unequal_microbatch_gradients_match_combined_batch(self):
        # A mean loss from 3 samples and one from 2 must contribute 3/5 and 2/5.
        # The former implementation's division by accumulation would give 1/2 each.
        gradients = [1.0, 3.0, 5.0, 7.0, 11.0]
        n = accumulation_window_samples(5, 3, 2, 0)
        accumulated = (sum(gradients[:3]) / 3) * 3 / n + (sum(gradients[3:]) / 2) * 2 / n
        self.assertAlmostEqual(accumulated, sum(gradients) / 5)
        self.assertEqual(accumulation_window_samples(11, 4, 2, 2), 3)

    def test_schedule_warms_exact_count_and_reaches_zero(self):
        factors = [lr_multiplier(i, 100, 5) for i in range(100)]
        self.assertEqual(factors[:5], [0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertTrue(all(a >= b for a, b in zip(factors[5:], factors[6:])))
        self.assertAlmostEqual(factors[-1], 0.0)
        self.assertEqual(lr_multiplier(100, 100, 5), 0)

    def test_partition_rejects_leakage_duplicates_and_missing_rows(self):
        self.assertEqual(validate_partition(["a", "b"], ["a"], ["b"])["all"], 2)
        for names, inner, hold in [(["a", "b"], ["a"], ["a", "b"]),
                                   (["a", "b"], ["a", "a"], ["b"]),
                                   (["a", "b", "c"], ["a"], ["b"]),
                                   (["a", "a"], ["a"], ["b"])]:
            with self.assertRaises(ValueError):
                validate_partition(names, inner, hold)

    def test_best_checkpoint_requires_strict_finite_improvement(self):
        self.assertFalse(improved_macro(0.5, 0.5))
        self.assertTrue(improved_macro(0.5001, 0.5))
        with self.assertRaises(ValueError):
            improved_macro(math.nan, 0.5)


if __name__ == "__main__":
    unittest.main()
