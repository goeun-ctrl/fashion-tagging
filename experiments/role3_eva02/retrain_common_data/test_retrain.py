"""Synthetic CPU tests. No EVA construction, pretrained tensors, real images or CUDA calls."""
import contextlib
import csv
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

from . import batches, cli, package as pkg, preflight as pf, trainer
import data_check
import eva
from data_check import write_json
from preprocess import preprocess


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


class SyntheticRelease:
    def __init__(self, root, mode):
        self.root = root
        self.inputs = pf.Inputs(root / "package", root / "team.json", root / "train", root / "val",
                                root / "external", root / "mock.weights")
        self.inputs.weights.write_bytes(b"synthetic fixture, no model tensors")
        self.package = self.inputs.package
        self.package.mkdir()
        (self.package / "epoch_plans").mkdir()
        self.cats = [f"synthetic_{i}" for i in range(17)] + sorted(pkg.GENERAL | pkg.MEDIUM | {"leg warmer", "cape"})
        assert len(self.cats) == 27
        self.a, self.h, self.v, self.e = "a.png", "h.png", "v.png", "e.png"
        self.inputs.external_root.mkdir()
        self.array = np.zeros((4, 7, 3), dtype=np.uint8)
        self.array[:, :3] = [180, 60, 24]
        Image.fromarray(self.array).save(self.inputs.external_root / self.e)
        for directory, ids in ((self.inputs.train_root, [self.a, self.h]), (self.inputs.val_root, [self.v])):
            (directory / "images").mkdir(parents=True)
            (directory / "categories.txt").write_text("\n".join(self.cats), encoding="utf-8")
            rows = []
            for name in ids:
                color = {self.a: 13, self.h: 39, self.v: 112}[name]
                Image.new("RGB", (7, 4), color).save(directory / "images" / name)
                rows.append({"file_name": name, "labels": json.dumps([self.cats[0]]),
                             **{c: int(i == 0) for i, c in enumerate(self.cats)}})
            write_csv(directory / "labels.csv", rows)
            np.savez(directory / "labels.npz", filenames=ids, categories=self.cats,
                     targets=[[r[c] for c in self.cats] for r in rows])
        self.y = np.zeros((2, 27), dtype=np.int8)
        self.y[0, 0] = self.y[1, self.cats.index("leg warmer")] = 1
        if mode == "partial":
            self.y[1, :3] = -1
        ids = [self.a, self.e]
        self.rows = []
        for name, source, path, weight in ((self.a, "train_inner", self.inputs.train_root / "images", 1.75),
                                            (self.e, "external", self.inputs.external_root, 2.5)):
            self.rows.append(dict(image_id=name, relative_path=name, source=source, approved=1,
                                  image_sha256=pkg.sha(path / name), sampling_weight=weight, data_version="test"))
        self.manifest = self.package / pkg.MANIFEST
        write_csv(self.manifest, self.rows)
        np.savez(self.package / pkg.TARGETS, image_ids=ids, targets=self.y, data_version="test", categories=self.cats)
        hashes = {name: pkg.sha(self.package / name) for name in (pkg.MANIFEST, pkg.TARGETS)}
        policy = {"data_version": "test", "targets_sha256": hashes[pkg.TARGETS], "flip_p": 0.5,
                  "replay": "pillow-enhance-v1", "pillow_version": Image.__version__,
                  "jitter_p": [0.35, 0.2, 0.0], "ranges": pkg.LIMITS.tolist(), "hue": 0,
                  "focus": ["leg warmer"], "general": sorted(pkg.GENERAL), "medium_weight": sorted(pkg.MEDIUM)}
        write_json(self.package / pkg.POLICY, policy)
        write_csv(self.package / pkg.COUNTS, [{"class": c, "positive": n} for c, n in
                                             zip(self.cats, (self.y == 1).sum(0))])
        hashes.update({name: pkg.sha(self.package / name) for name in (pkg.POLICY, pkg.COUNTS)})
        self.lock = dict(schema="fashion-common-v1", approved=True, seed=42, replacement=True, epochs=2, draws=5,
                         data_version="test", categories=self.cats, label_mode=mode, sha256=hashes)
        for split in ("train", "val"):
            hashes[split + "/labels.npz"] = pkg.sha(root / split / "labels.npz")
        for epoch in (1, 2):
            indices = [1, 0, 1, 0, 1] if epoch == 1 else [0, 1, 0, 1, 0]
            factors = [[1.1, 0.91, 0.96] if i == 1 else [1., 1., 1.] for i in indices]
            path = self.package / pkg.plan_name(epoch)
            np.savez(path, epoch=epoch, label_mode=mode, data_version="test", manifest_sha256=hashes[pkg.MANIFEST],
                     targets_sha256=hashes[pkg.TARGETS], policy_sha256=hashes[pkg.POLICY],
                     indices=indices, flip=[1, 0, 1, 0, 1], factors=factors, order=[[2, 0, 1]] * 5)
            hashes[pkg.plan_name(epoch)] = pkg.sha(path)
        self.sync()
        self.fixed = self.cats, {"train_inner": [self.a], "holdout": [self.h]}

    def sync(self):
        write_json(self.inputs.shared_lock, self.lock)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.base = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.f = SyntheticRelease(self.base, "complete")
        for module, key, value in ((pkg, "DRAWS", 5), (pkg, "EPOCHS", 2), (pf, "VAL_COUNT", 1),
                                   (pf, "RUN_ROOT", self.base / "private" / "retrain")):
            self.stack.enter_context(patch.object(module, key, value))
        self.stack.enter_context(patch.dict(pf.CFG, weight_sha256=pkg.sha(self.f.inputs.weights)))
        # Only substitute fixed metadata. Exercise the production split reader and every package check.
        self.stack.enter_context(patch.object(pf, "team_splits", return_value=self.f.fixed))

    def audit(self):
        return pf.preflight(self.f.inputs, "complete", "synthetic")

    def test_baseline_default_split_path_result_and_missing_image_error(self):
        default = self.base / "MAT4091_fashion_train" / "train"
        shutil.copytree(self.f.inputs.train_root, default)
        with patch.object(data_check, "WORK", self.base):
            root, cats, names, y = data_check.source("train")
            self.assertEqual((root, cats, names), (default, self.f.cats, [self.f.a, self.f.h]))
            np.testing.assert_array_equal(y, data_check.source("train", self.f.inputs.train_root)[3])
            (default / "images" / self.f.h).unlink()
            with self.assertRaises(ValueError):
                data_check.source("train")

    def test_baseline_default_weight_lookup_and_digest_gate(self):
        weight = self.f.inputs.weights
        with patch.object(eva, "WEIGHTS", weight), patch.dict(eva.CFG, weight_sha256=pkg.sha(weight)), \
                patch.object(eva, "load_file", side_effect=RuntimeError("stop before model load")) as reader:
            with self.assertRaisesRegex(RuntimeError, "stop before model load"):
                trainer.baseline.from_pretrained(378)
            reader.assert_called_once_with(str(weight))
            weight.write_bytes(b"incorrect synthetic bytes")
            with self.assertRaisesRegex(ValueError, "wrong pretrained artifact"):
                trainer.baseline.from_pretrained(378)
            self.assertEqual(reader.call_count, 1)

    def test_non_native_npz_targets_reach_torch_with_unknowns_intact(self):
        targets = self.f.y.astype(np.dtype("i2").newbyteorder("S"))
        targets[1, 0] = -1
        pkg.check_labels(targets, "partial")
        ds = batches.PlannedDataset([], targets, ([], [], [], []))
        np.testing.assert_array_equal(ds.targets.numpy(), targets)

    def test_synthetic_release_all_checks_and_class_order(self):
        report, data = self.audit()
        self.assertEqual(report["status"], "PASS", report)
        self.assertEqual(report["approved_external"], 1)
        self.assertEqual(data["manifest_targets_label_mode"][1].shape, (2, 27))
        self.f.lock["categories"] = self.f.cats[::-1]
        self.f.sync()
        self.assertEqual(self.audit()[0]["checks"]["manifest_targets_label_mode"]["status"], "FAIL")

    def test_epoch_order_and_replacement(self):
        _, data = self.audit()
        ps = data["weights_counts_augmentation"][0]
        lists = [pkg.load_plan(self.f.package, self.f.lock, i, ps)[0].tolist() for i in (1, 2)]
        self.assertEqual(lists, [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])

    def test_exactly_one_flip_and_ordered_jitter(self):
        _, data = self.audit()
        plan = pkg.load_plan(self.f.package, self.f.lock, 1, data["weights_counts_augmentation"][0])
        ds = batches.PlannedDataset(data["image_hashes_duplicates_leakage"], self.f.y, plan)
        reference = ImageOps.mirror(Image.fromarray(self.f.array))
        reference = ImageEnhance.Color(reference).enhance(0.96)
        reference = ImageEnhance.Brightness(reference).enhance(1.1)
        reference = ImageEnhance.Contrast(reference).enhance(0.91)
        expected = preprocess(reference, 378, "pad", pf.MEAN, pf.STD)
        with patch("random.random", side_effect=AssertionError("unexpected RNG")):
            x, y = ds[0]
        np.testing.assert_array_equal(x.numpy(), expected)
        np.testing.assert_array_equal(y.numpy(), self.f.y[1])
        dl = batches.loader(ds, 0)
        self.assertIsInstance(dl.sampler, torch.utils.data.SequentialSampler)
        self.assertEqual(list(dl.sampler), list(range(5)))

    def test_holdout_and_val_are_deterministic(self):
        groups = self.audit()[1]["original_splits_labels"]
        with patch("random.random", side_effect=AssertionError("evaluation used RNG")):
            for split in ("holdout", "val"):
                ds = batches.eval_dataset(groups[split])
                self.assertFalse(ds.train)
                torch.testing.assert_close(ds[0][0], ds[0][0], rtol=0, atol=0)

    def test_hash_blocks_training_dispatch(self):
        with self.f.manifest.open("ab") as f:
            f.write(b"\n")
        inp = self.f.inputs
        # Exercise the public gate, never call a training or model operation.
        cli_args = ["train", "--package", str(inp.package), "--shared-lock", str(inp.shared_lock),
                    "--train-root", str(inp.train_root), "--val-root", str(inp.val_root),
                    "--external-root", str(inp.external_root), "--weights", str(inp.weights), "--run-id", "synthetic"]
        with patch.object(trainer, "execute") as server, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(cli_args), 1)
            server.assert_not_called()

    def test_old_plan_invalid_after_approved_label_edit(self):
        hashes = self.f.lock["sha256"]
        y = self.f.y.copy()
        y[1, 0] = 1
        np.savez(self.f.package / pkg.TARGETS, image_ids=[self.f.a, self.f.e], targets=y,
                 data_version="test", categories=self.f.cats)
        hashes[pkg.TARGETS] = pkg.sha(self.f.package / pkg.TARGETS)
        policy_file = self.f.package / pkg.POLICY
        policy = json.loads(policy_file.read_bytes())
        policy["targets_sha256"] = hashes[pkg.TARGETS]
        write_json(policy_file, policy)
        hashes[pkg.POLICY] = pkg.sha(policy_file)
        write_csv(self.f.package / pkg.COUNTS, [{"class": c, "positive": n} for c, n in
                                               zip(self.f.cats, (y == 1).sum(0))])
        hashes[pkg.COUNTS] = pkg.sha(self.f.package / pkg.COUNTS)
        self.f.sync()
        report, _ = self.audit()
        self.assertEqual(report["checks"]["weights_counts_augmentation"]["status"], "PASS", report)
        self.assertEqual(report["checks"]["epoch_plans"]["status"], "FAIL", report)

    def test_complete_release_rejects_unreviewed_external_target(self):
        self.f.y[1, 0] = -1
        path = self.f.package / pkg.TARGETS
        np.savez(path, image_ids=[self.f.a, self.f.e], targets=self.f.y,
                 data_version="test", categories=self.f.cats)
        self.f.lock["sha256"][pkg.TARGETS] = pkg.sha(path)
        self.f.sync()
        result, data = self.audit()
        self.assertEqual(result["checks"]["manifest_targets_label_mode"]["status"], "FAIL")
        self.assertEqual(result["label_mode"], "complete")
        self.assertIsNone(data)

    def test_missing_plan_is_blocked(self):
        (self.f.package / pkg.plan_name(2)).unlink()
        report, data = self.audit()
        self.assertEqual(report["status"], "BLOCKED", report)
        self.assertEqual(report["checks"]["epoch_plans"]["status"], "BLOCKED")
        self.assertIsNone(data)

    def test_missing_original_image_is_blocked(self):
        (self.f.inputs.train_root / "images" / self.f.a).unlink()
        result, data = self.audit()
        self.assertEqual(result["checks"]["original_splits_labels"]["status"], "BLOCKED")
        self.assertIsNone(data)

    def test_rejects_duplicate_external_bytes_even_under_new_id(self):
        path = self.f.inputs.external_root / self.f.e
        path.write_bytes((self.f.inputs.val_root / "images" / self.f.v).read_bytes())
        self.f.rows[1]["image_sha256"] = pkg.sha(path)
        write_csv(self.f.manifest, self.f.rows)
        self.f.lock["sha256"][pkg.MANIFEST] = pkg.sha(self.f.manifest)
        self.f.sync()
        self.assertEqual(self.audit()[0]["checks"]["image_hashes_duplicates_leakage"]["status"], "FAIL")

    def test_baseline_path_and_checkpoint_protection(self):
        for bad in ("../baseline", r"..\baseline", str(self.base / "baseline")):
            with self.assertRaises(ValueError):
                pf.run_directory(bad)
        root = pf.run_directory("baseline")
        self.assertEqual(root.parent, pf.RUN_ROOT)
        with trainer.exclusive_run(root, fresh=True):
            torch.save({"config": {"model": "baseline"}}, root / "last.pt")
        with self.assertRaises(FileExistsError), trainer.exclusive_run(root, fresh=True):
            pass
        with self.assertRaisesRegex(ValueError, "different experiment"):
            trainer.read_checkpoint(root, "last.pt", {"experiment": "role3_eva02_common_v1"})

    def test_peer_signature_detects_difference(self):
        cfg = {"label_mode": "complete", "shared": self.f.lock}
        a, b = self.base / "eva.json", self.base / "peer.json"
        write_json(a, cfg)
        write_json(b, cfg)
        self.assertEqual(cli.compare(a, b)["status"], "PASS")
        self.f.lock["sha256"][pkg.plan_name(2)] = "0" * 64
        write_json(b, cfg)
        result = cli.compare(a, b)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["different_file_hashes"], [pkg.plan_name(2)])

    def test_peer_configs_cannot_agree_on_an_invalid_mode(self):
        self.f.lock["label_mode"] = "unknown"
        cfg = self.base / "bad-config.json"
        write_json(cfg, {"label_mode": "unknown", "shared": self.f.lock})
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["compare", "--eva-config", str(cfg), "--peer-config", str(cfg)]), 1)

    def test_unfinished_evaluation_does_not_publish_results(self):
        cfg = {"max_epochs": 30, "patience": 5}
        trainer.baseline.save(self.base / "last.pt", {"epoch": 1, "best_epoch": 1, "history": [{}], "config": cfg})
        before = set(self.base.iterdir())
        with self.assertRaisesRegex(ValueError, "training must finish"):
            trainer.evaluate(self.base, None, cfg, ("val",))
        self.assertEqual(before, set(self.base.iterdir()))

    def test_server_flow_with_synthetic_draws_and_stubbed_gpu(self):
        audit, data = self.audit()
        self.assertEqual(audit["status"], "PASS", audit)
        root = data["private_output"]
        model = Mock()
        model.cuda.return_value = model
        optimizer = Mock(param_groups=[{"lr": 5e-5}, {"lr": 1e-3}])
        scheduler, scaler = Mock(), Mock()
        for component in (optimizer, scheduler, scaler):
            component.state_dict.return_value = {}
        seen_plans, evaluated = [], []
        steps = iter((1, 0, 1))  # No update in epoch two; retry the same plan on resume.

        def epoch_stub(model, dl, opt, sched, amp, mode, accum):
            self.assertEqual((opt, sched, amp, mode, accum), (optimizer, scheduler, scaler, "complete", 2))
            ds = dl.dataset
            seen_plans.append(ds.indices.tolist())
            y = torch.stack([ds[i][1] for i in dl.sampler])
            model.state_dict.return_value = {"marker": len(seen_plans)}
            return {"optimizer_steps": next(steps)}, (y == 1).sum(0).tolist(), (y != -1).sum(0).tolist()

        def fixed_logits(model, dl, device, use_amp):
            evaluated.append(dl.dataset.names)
            return np.zeros((len(dl.dataset), 27), np.float32), 0.

        # Only replace GPU/model operations. Use the real plan reader, holdout metrics,
        # checkpoint selection/restore and all file writers in the private temp directory.
        with contextlib.ExitStack() as stack, contextlib.redirect_stdout(io.StringIO()):
            stub = stack.enter_context
            stub(patch.dict(trainer.CFG, max_epochs=2, workers=0))
            stub(patch.object(trainer, "set_seed"))
            pretrained = stub(patch.object(trainer, "from_pretrained", return_value=model))
            stub(patch.object(trainer, "architecture", return_value=model))
            stub(patch.object(trainer.baseline, "optimizer", return_value=optimizer))
            stub(patch.object(trainer, "build_schedulers", return_value=[scheduler]))
            stub(patch.object(torch.amp, "GradScaler", return_value=scaler))
            stub(patch.object(torch.cuda, "is_available", return_value=True))
            stub(patch.object(torch.cuda, "get_device_name", return_value="synthetic"))
            stub(patch.object(torch.backends.cudnn, "version", return_value=None))
            stub(patch.object(torch.cuda, "get_rng_state", return_value=torch.zeros(1)))
            stub(patch.object(torch.cuda, "set_rng_state"))
            stub(patch.object(trainer, "train_epoch", side_effect=epoch_stub))
            stub(patch.object(trainer.baseline, "predict", side_effect=fixed_logits))
            args = SimpleNamespace(command="train", resume=False)
            with self.assertRaisesRegex(ValueError, "zero optimizer steps"):
                trainer.execute(args, self.f.inputs, audit, data)
            self.assertEqual(evaluated, [[self.f.h]])
            self.assertFalse((root / "metrics_val.json").exists())
            args.resume = True
            trainer.execute(args, self.f.inputs, audit, data)
            pretrained.assert_called_once_with(378, self.f.inputs.weights)
        model.assert_not_called()
        self.assertEqual(seen_plans, [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0]])
        self.assertEqual(evaluated, [[self.f.h], [self.f.h], [self.f.h], [self.f.v]])
        model.load_state_dict.assert_called_with({"marker": 1}, strict=True)
        cfg = json.loads((root / "resolved_config.json").read_bytes())
        self.assertEqual(trainer.read_checkpoint(root, "best.pt", cfg)["epoch"], 1)
        self.assertEqual(trainer.read_checkpoint(root, "last.pt", cfg)["epoch"], 2)
        self.assertEqual(len(pkg.csv_rows((root / "training_history.csv").read_bytes())), 2)
        self.assertEqual(len(pkg.csv_rows((root / "actual_class_exposure.csv").read_bytes())), 2 * 27)
        self.assertEqual(json.loads((root / "data_audit.json").read_bytes())["status"], "PASS")
        self.assertEqual(json.loads((root / "environment.json").read_bytes())["gpu"], "synthetic")
        for split, name in (("holdout", self.f.h), ("val", self.f.v)):
            self.assertEqual(json.loads((root / f"metrics_{split}.json").read_bytes())["best_epoch"], 1)
            rows = pkg.csv_rows((root / f"per_class_{split}.csv").read_bytes())
            self.assertEqual([r["class"] for r in rows], self.f.cats)
            with np.load(root / f"preds_{split}.npz", allow_pickle=False) as preds:
                self.assertEqual(preds["filenames"].tolist(), [name])

    def test_interrupted_best_publication_recovers_from_last(self):
        cfg = {"experiment": "synthetic"}
        state = {"config": cfg, "epoch": 1, "best_epoch": 1, "best_score": 0.,
                 "model": {"synthetic": torch.ones(2)}, "history": [{"epoch": 1}],
                 "exposure": [{"epoch": 1, "positive": 2}]}
        trainer.baseline.save(self.base / "last.pt", state)
        trainer.restore_progress(self.base, cfg)
        self.assertEqual(pkg.sha(self.base / "last.pt"), pkg.sha(self.base / "best.pt"))
        best = trainer.read_checkpoint(self.base, "best.pt", cfg)
        torch.testing.assert_close(best["model"]["synthetic"], state["model"]["synthetic"])
        self.assertTrue((self.base / "actual_class_exposure.csv").is_file())

    def test_partial_release_requires_explicit_mode(self):
        other = self.base / "partial"
        other.mkdir()
        f = SyntheticRelease(other, "partial")
        self.assertEqual(pf.preflight(f.inputs, "complete", "test")[0]["status"], "FAIL")
        result, _ = pf.preflight(f.inputs, "partial", "test")
        self.assertEqual(result["status"], "PASS", result)


class TensorTests(unittest.TestCase):
    def test_exif_rgb_padding_and_378(self):
        image = Image.new("L", (8, 4), 255)
        image.getexif()[274] = 6
        x = preprocess(image, 378, "pad", pf.MEAN, pf.STD)
        self.assertEqual((x.dtype, x.shape), (np.float32, (3, 378, 378)))
        np.testing.assert_allclose(x[:, 0, 0], -np.array(pf.MEAN) / pf.STD, rtol=1e-6)
        np.testing.assert_allclose(x[:, 0, 189], (1 - np.array(pf.MEAN)) / pf.STD, rtol=1e-6)

    def test_complete_rejects_unknown(self):
        y = np.zeros((1, 27), dtype=np.int8)
        y[0, 3] = -1
        with self.assertRaises(ValueError):
            pkg.check_labels(y, "complete")
        with self.assertRaises(ValueError):
            batches.window_size([(None, torch.tensor(y))], "complete")
        pkg.check_labels(y, "partial")

    def test_masked_value_and_unknown_gradient(self):
        x = torch.tensor([[0.4, -0.7, 1.6]], requires_grad=True)
        y = torch.tensor([[1., -1., 0.]])
        loss = batches.loss_sum(x, y, "partial")
        expected = np.logaddexp(0, -0.4) + np.logaddexp(0, 1.6)
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertEqual(x.grad[0, 1].item(), 0.)
        self.assertGreater(x.grad[0, 2].item(), 0.)

    def test_window_gradient_equals_one_bce_for_both_modes(self):
        for mode in ("complete", "partial"):
            x = torch.linspace(-2, 2, 81).reshape(3, 27).requires_grad_()
            y = (x.detach() > 0).float()
            if mode == "partial":
                y[:2, :24] = -1
            window = [(x[:2], y[:2]), (x[2:], y[2:])]
            n = batches.window_size(window, mode)
            for xb, yb in window:
                (batches.loss_sum(xb, yb, mode) / n).backward()
            other = x.detach().clone().requires_grad_()
            mask = y != -1
            torch.nn.BCEWithLogitsLoss()(other[mask], y[mask]).backward()
            torch.testing.assert_close(x.grad, other.grad)

    def test_entirely_unknown_window_skips_step(self):
        opt, sched, scaler = Mock(), Mock(), Mock()
        row, positive, known = batches.train_epoch(Mock(), [(None, -torch.ones(2, 27))],
                                                   opt, sched, scaler, "partial", 2)
        self.assertEqual((row["empty_windows"], row["optimizer_steps"], row["train_bce"]), (1, 0, None))
        self.assertEqual(sum(positive) + sum(known), 0)
        scaler.step.assert_not_called()
        sched.step.assert_not_called()

    def test_max_weight_and_jitter_probability(self):
        cats = ["leg warmer", "cape", *sorted(pkg.GENERAL | pkg.MEDIUM), "shoe"]
        w, p = pkg.policy_vectors(cats, [1500, 999] + [1000] * (len(cats)-2),
                                  ["leg warmer", "cape"], pkg.GENERAL, pkg.MEDIUM)
        y = np.zeros((5, len(cats)), dtype=int)
        y[0, :] = 1
        y[1, [1, cats.index("watch")]] = 1
        y[2, cats.index(pkg.HEADBAND)] = 1
        y[3, -1] = 1
        y[4, :2] = -1
        np.testing.assert_equal(pkg.max_for_image(y, w, 1.), [2.5, 1.75, 1.3, 1., 1.])
        np.testing.assert_equal(pkg.max_for_image(y, p, 0.), [.35, .35, .2, 0., 0.])

    def test_scheduler_only_advances_after_optimizer_step(self):
        for scales, advances in (([128, 64], False), ([64, 64], True)):
            scaler, sched = Mock(), Mock()
            scaler.get_scale.side_effect = scales
            self.assertEqual(trainer.baseline.update(scaler, Mock(), sched), advances)
            self.assertEqual(sched.step.called, advances)

    def test_cpu_adamw_update_from_accumulated_synthetic_logits(self):
        logits = torch.zeros((3, 27), requires_grad=True)
        labels = torch.ones_like(logits)
        labels[0, :] = -1
        window = [(logits[:2], labels[:2]), (logits[2:], labels[2:])]
        opt = torch.optim.AdamW([logits], lr=1e-3)
        sched, = trainer.build_schedulers([opt], "cosine", 1, 2)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        n = batches.window_size(window, "partial")
        for x, y in window:
            scaler.scale(batches.loss_sum(x, y, "partial") / n).backward()
        previous = sched.last_epoch
        self.assertTrue(trainer.baseline.update(scaler, opt, sched))
        self.assertTrue(torch.all(logits[0] == 0) and torch.all(logits[1:] > 0))
        self.assertEqual(sched.last_epoch, previous + 1)


if __name__ == "__main__":
    unittest.main()
