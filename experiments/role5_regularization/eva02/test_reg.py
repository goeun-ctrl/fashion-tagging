"""CPU checks for reg.py. Run from the repo root:
    python -m unittest experiments.role5_regularization.eva02.test_reg -v
"""
import unittest

import torch

from experiments.role5_regularization.eva02 import reg
from experiments.role3_eva02 import run as baseline
import eva

SIZE = 112  # any multiple of 14; small for CPU
LR_B, LR_H, WD = eva.CFG["lr_backbone"], eva.CFG["lr_head"], eva.CFG["weight_decay"]


def opts(**kw):
    return {**reg.OFF, **kw}


def groups(opt):
    return {id(p): (g["lr"], g["weight_decay"]) for g in opt.param_groups for p in g["params"]}


class OffIsRole3(unittest.TestCase):
    def test_architecture_and_optimizer_identical(self):
        torch.manual_seed(0)
        a = eva.architecture(SIZE)
        torch.manual_seed(0)
        b = reg.architecture(SIZE, 0.0)
        self.assertEqual(repr(a), repr(b))
        for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters(), strict=True):
            self.assertEqual(n1, n2)
            self.assertTrue(torch.equal(p1, p2))
        ref, new = baseline.optimizer(a), reg.optimizer(b, opts(), LR_B, LR_H, WD)
        self.assertEqual([(g["lr"], g["weight_decay"], len(g["params"])) for g in ref.param_groups],
                         [(g["lr"], g["weight_decay"], len(g["params"])) for g in new.param_groups])
        self.assertTrue(reg.is_off(opts(), LR_B))

    def test_one_step_identical(self):
        x, y = torch.randn(2, 3, SIZE, SIZE), torch.randint(2, (2, 27)).float()
        out = []
        for build in (lambda m: baseline.optimizer(m), lambda m: reg.optimizer(m, opts(), LR_B, LR_H, WD)):
            torch.manual_seed(0)
            m = eva.architecture(SIZE)
            o = build(m)
            torch.nn.functional.binary_cross_entropy_with_logits(m(x), y).backward()
            o.step()
            out.append([p.detach().clone() for p in m.parameters()])
        for p, q in zip(*out, strict=True):
            self.assertTrue(torch.equal(p, q))


class Options(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.m = eva.architecture(SIZE)

    def test_drop_path(self):
        m = reg.architecture(SIZE, 0.1)
        rates = [getattr(b.drop_path1, "drop_prob", 0.0) for b in m.blocks]   # rate 0 -> Identity
        self.assertAlmostEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[-1], 0.1)
        self.assertTrue(all(a < b for a, b in zip(rates, rates[1:])))
        self.assertEqual(eva.architecture(SIZE).blocks[-1].drop_path1.__class__.__name__, "Identity")

    def test_wd_exclude(self):
        g = groups(reg.optimizer(self.m, opts(wd_exclude=True), LR_B, LR_H, WD))
        head = {id(p) for p in self.m.head.parameters()}
        for name, p in self.m.named_parameters():
            lr, wd = g[id(p)]
            self.assertEqual(lr, LR_H if id(p) in head else LR_B, name)
            no = p.ndim <= 1 or name in ("pos_embed", "cls_token")
            self.assertEqual(wd, 0.0 if no else WD, name)
        self.assertEqual(g[id(self.m.pos_embed)][1], 0.0)
        self.assertEqual(g[id(self.m.blocks[0].attn.qkv.weight)][1], WD)

    def test_layer_decay(self):
        o = reg.optimizer(self.m, opts(wd_exclude=True, layer_decay=0.75), 1e-4, LR_H, WD)
        g = groups(o)
        depth = len(self.m.blocks)
        self.assertEqual(g[id(self.m.head.weight)][0], LR_H)
        self.assertAlmostEqual(g[id(self.m.blocks[-1].attn.qkv.weight)][0], 1e-4 * 0.75)
        self.assertAlmostEqual(g[id(self.m.patch_embed.proj.weight)][0], 1e-4 * 0.75 ** (depth + 1))
        self.assertEqual(g[id(self.m.pos_embed)][1], 0.0)
        self.assertEqual(len(g), sum(1 for _ in self.m.parameters()))
        self.assertAlmostEqual(g[id(self.m.fc_norm.weight)][0], 1e-4)        # post-pool norm: top layer
        self.assertEqual(reg.lr_report(o), (1e-4, LR_H))

    def test_ema_follows_successful_steps_only(self):
        o = reg.optimizer(self.m, opts(), LR_B, LR_H, WD)
        from fashion_tagging.training.optim import build_schedulers
        s = reg.EmaScheduler(build_schedulers([o], "cosine", 2, 10)[0], self.m, 0.9995)
        before = [p.clone() for p in s.ema.module.parameters()]
        with torch.no_grad():
            for p in self.m.parameters():
                p.add_(1.0)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, s.ema.module.parameters())))  # no step yet
        s.step()
        self.assertEqual(s.last_epoch, 1)
        self.assertFalse(torch.equal(before[0], next(s.ema.module.parameters())))
        s2 = build_schedulers([o], "cosine", 2, 10)[0]
        s2.load_state_dict(s.state_dict())
        self.assertEqual(s2.last_epoch, 1)


if __name__ == "__main__":
    unittest.main()
