"""Unit tests for redesign B pieces (CPU, no data needed):  python tests/test_segb.py"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fashion_tagging.segb.data import group_targets, part_targets, frac  # noqa: E402
from fashion_tagging.segb.losses import group_weights, region_ce, seg_ce, seg_dice, tag_loss  # noqa: E402
from fashion_tagging.segb.model import class_probs, ratio_topk, valid_cells  # noqa: E402
from fashion_tagging.segb.spec import CLASS_TO_GROUP, GROUP_SIZES, GROUPS, N_SEG  # noqa: E402


class T(unittest.TestCase):
    def test_spec(self):
        self.assertEqual(N_SEG, 31)
        self.assertEqual(CLASS_TO_GROUP[1], (0, 2))
        self.assertEqual(CLASS_TO_GROUP[23], (2, 4))

    def test_ratio_topk_ignores_padding_and_uses_min_k(self):
        p = torch.zeros(1, 27, 4, 4)
        p[0, 0, 0, 0] = 1.0                       # one hot cell
        p[0, 0, 3, 3] = 1.0                       # padded cell (outside the real 8x8 px = 2x2 cells)
        valid = valid_cells(torch.tensor([[8, 8]]), 4, 4)
        self.assertEqual(int(valid.sum()), 4)
        r = ratio_topk(p, valid, torch.full((27,), 0.5), min_k=1)
        self.assertAlmostEqual(float(r[0, 0]), 0.5)          # k = round(0.5 * 4) = 2 -> (1 + 0) / 2
        r = ratio_topk(p, valid, torch.full((27,), 0.0), min_k=4)
        self.assertAlmostEqual(float(r[0, 0]), 0.25)         # k = min_k 4 -> 1 / 4

    def test_group_targets_soft_and_ignore(self):
        g = np.zeros((4, 4, 8), np.uint8)
        g[0, :, :2] = 2                            # top covers half of the first cell
        g[3, :, 4:] = 255                          # unknown second cell in accessories
        t, v = group_targets(g)
        self.assertEqual(t.shape, (N_SEG, 1, 2))
        self.assertAlmostEqual(float(t[2, 0, 0]), 0.5)       # group 0, class index 2 (top)
        self.assertAlmostEqual(float(t[0, 0, 0]), 0.5)       # none
        self.assertTrue(v[0, 0, 0] and not v[3, 0, 1])

    def test_part_and_frac(self):
        parts = np.zeros((4, 4), np.uint8); parts[:2] = 1
        self.assertAlmostEqual(float(part_targets(parts)[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(frac(np.ones((2, 2), bool))[0, 0]), 0.25)   # padded to 4x4

    def test_losses_ignore_invalid_cells(self):
        seg = torch.randn(2, N_SEG, 3, 3, requires_grad=True)
        _, logp = class_probs(seg)
        t = torch.zeros(2, N_SEG, 3, 3)
        s = 0
        for n in GROUP_SIZES:
            t[:, s] = 1.0; s += n
        v = torch.zeros(2, 4, 3, 3, dtype=torch.bool); v[0] = True
        gw = group_weights(torch.ones(27))
        l1 = seg_ce(logp, t, v, gw, torch.ones(2))
        t2 = t.clone(); t2[1] = 0; t2[1, 1] = 1                 # change only the invalid image
        self.assertAlmostEqual(float(l1), float(seg_ce(logp, t2, v, gw, torch.ones(2))), places=6)
        l1.backward()
        self.assertTrue(torch.isfinite(seg.grad).all())
        self.assertEqual(float(seg_dice(logp, t, v, torch.ones(2))), 0.0)   # no class present -> 0

    def test_region_ce(self):
        """region = soft target >= thr, >= min_cells; mean logits over garment classes; only `use` images."""
        h = w = 8
        seg = torch.zeros(2, N_SEG, h, w, requires_grad=True)
        t = torch.zeros(2, N_SEG, h, w)
        sh, top = 1 + GROUPS[0].index(0), 1 + GROUPS[0].index(1)        # shirt / top channels (group 0 starts at 0)
        t[:, sh, :4, :] = 1.0                                           # 32-cell shirt region
        t[:, top, 4:5, :2] = 1.0                                        # 2-cell top region (< min_cells)
        t[:, top, 6:, :] = 0.4                                          # boundary-like cells (< thr)
        cw = torch.ones(27)
        use = torch.tensor([True, False])
        l = region_ce(seg, t, use, [0], cw, 0.5, 16)
        n = GROUP_SIZES[0] - 1
        self.assertAlmostEqual(float(l), float(torch.log(torch.tensor(float(n)))), places=5)   # uniform logits
        with torch.no_grad():                                           # shirt logits high in half the region,
            seg[0, sh, :2, :] = 10.0                                    # top high in the other half -> split
            seg[0, top, 2:4, :] = 10.0
        split = float(region_ce(seg, t, use, [0], cw, 0.5, 16))
        with torch.no_grad():
            seg[0, top, 2:4, :] = 0.0; seg[0, sh, 2:4, :] = 10.0        # whole region says shirt
        self.assertLess(float(region_ce(seg, t, use, [0], cw, 0.5, 16)), split)
        t2 = t.clone(); t2[1, sh] = 0; t2[1, top, :, :] = 1.0           # change only the unused image
        self.assertAlmostEqual(float(region_ce(seg, t2, use, [0], cw, 0.5, 16)),
                               float(region_ce(seg, t, use, [0], cw, 0.5, 16)), places=6)
        self.assertEqual(float(region_ce(seg, t, torch.tensor([False, False]), [0], cw)), 0.0)
        region_ce(seg, t, use, [0, 1], cw).backward()
        self.assertTrue(torch.isfinite(seg.grad).all())
        self.assertEqual(float(seg.grad[1].abs().sum()), 0.0)           # no gradient into the unused image

    def test_tag_loss_unknown_and_teacher(self):
        p = torch.full((2, 27), 0.3)
        y = torch.full((2, 27), -1.0); y[0, 0] = 1
        soft = torch.full((2, 27), float("nan"))
        l = tag_loss(p, y, soft, torch.ones(27), 0.5)
        self.assertAlmostEqual(float(l), -np.log(0.3), places=4)          # only the one known label counts
        soft[1] = 0.3
        l2 = tag_loss(p, y, soft, torch.ones(27), 0.5)
        self.assertTrue(float(l2) > 0)


if __name__ == "__main__":
    unittest.main()
