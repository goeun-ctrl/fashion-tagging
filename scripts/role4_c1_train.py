"""Role4-only training loop derived from Release commit 183aedc5.
C1 model, preprocessing, sampler, accumulation, optimizer, EMA and schedule come from role4_c1_reference.
Changes: explicit shared initialization, reproducibility records, exclusive output, holdout-only evaluation.
"""
from role4_c1_reference import *  # pinned upstream primitives
import hashlib
from role4_c1 import audit_data, digest, environment, state_digest

def train(a):
    audit = audit_data(vars(a))
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing accidental CPU full training")
    set_seed(a.seed)
    device = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not a.no_amp
    out = os.path.join(a.out_dir, a.exp_id)
    os.makedirs(out, exist_ok=False)

    cats, names, targets = load_split(a.train_dir, "train")
    assert len(cats) == 27
    index = {n: i for i, n in enumerate(names)}
    inner, hold = read_names(os.path.join(a.splits_dir, "train_inner.txt")), read_names(os.path.join(a.splits_dir, "holdout.txt"))
    assert not set(inner) & set(hold) and set(inner) | set(hold) == set(names), "fixed split mismatch"
    if a.limit_train:
        inner = inner[:a.limit_train]
    if a.limit_eval:
        hold = hold[:a.limit_eval]
    t_in, t_ho = targets[[index[n] for n in inner]], targets[[index[n] for n in hold]]
    soft = load_teacher(a, inner) if a.distill_npz else None
    mean, std = timm_mean_std(a.model)

    levels = short_levels(a)
    t0 = time.perf_counter()
    por_in, por_ho = (oriented_portrait(r, n) for r, n in
                              ((a.train_dir, inner), (a.train_dir, hold)))
    print(f"levels {levels}, test {a.test_short}; portrait share {por_in.mean():.2f} "
          f"(header scan {time.perf_counter() - t0:.0f}s)", flush=True)

    def loader(root, n, t, por, train, bs):
        sampler = BucketBatchSampler(por, levels if train else [a.test_short], bs, a.seed, train)
        ds = HiResDataset(root, n, t, mean, std, train, a.max_long, a.flip, soft=soft if train else None)
        return DataLoader(ds, batch_sampler=sampler,
                          num_workers=a.workers, pin_memory=device.type == "cuda", collate_fn=pad_collate,
                          worker_init_fn=seed_worker, persistent_workers=False,
                          generator=torch.Generator().manual_seed(a.seed + (0 if train else 10000)))

    train_loader = loader(a.train_dir, inner, t_in, por_in, True, a.batch_size)
    hold_loader = loader(a.train_dir, hold, t_ho, por_ho, False, a.eval_batch_size)

    # channels_last: ConvNeXt's 7x7 depthwise conv backward is ~20x slower in NCHW at these sizes (A5000, AMP:
    # 2.18 s vs 0.093 s per 4-image step at ~800x1216). Same math, memory layout only.
    initial = torch.load(a.initial_state, map_location="cpu", weights_only=False)
    smoke = a.no_pretrained and a.device == 'cpu' and 0 < a.limit_train <= 4 and 0 < a.limit_eval <= 4
    assert (initial["pretrained"] or smoke) and initial["seed"] == a.seed and initial["model_name"] == a.model
    assert initial["model_sha256"] == state_digest(initial["model"])
    model = MaskedPoolConvNeXt(a.model, len(cats), False, a.drop_path)
    model.load_state_dict(initial["model"], strict=True)
    assert state_digest(model.state_dict()) == initial["model_sha256"]
    # Reset after model construction: training randomness must not depend on pretrained I/O.
    set_seed(a.seed)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= PARAM_LIMIT, f"{n_params:,} parameters exceed the 50M limit"
    head_ids = {id(p) for p in model.head_parameters()}
    opt = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": a.lr_backbone, "name": "backbone"},
        {"params": model.head_parameters(), "lr": a.lr_head, "name": "head"},
    ], weight_decay=a.weight_decay)
    sampler = train_loader.batch_sampler
    per_epoch = [math.ceil(len(sampler.plan(e)) / a.accum) for e in range(1, a.epochs + 1)]
    steps_per_epoch, total = round(sum(per_epoch) / len(per_epoch)), sum(per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, total, a.warmup_iters, a.warmup_ratio, a.milestones, a.gamma))
    loss_fn = build_loss(a.loss, a.asl_gamma_neg, a.asl_gamma_pos, a.asl_clip)
    ema = None
    if a.ema:
        from timm.utils import ModelEmaV3
        ema = ModelEmaV3(model, decay=a.ema, use_warmup=True)
    eval_model = ema.module if ema else model
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update(params_total=n_params, steps_per_epoch=steps_per_epoch, total_updates=total,
               torch=str(torch.__version__), timm=str(timm.__version__), device=str(device),
               train=len(inner), holdout=len(hold), val=1158,
               initial_file_sha256=digest(a.initial_state), initial_model_sha256=initial["model_sha256"],
               initial_head_sha256=initial["head_sha256"], initial_source=initial["source"],
               environment=environment(), data_audit=audit,
               loader_rng="independent torch.Generator(seed train; seed+10000 eval), workers reset each epoch",
               sampler_plan_sha256=[hashlib.sha256(json.dumps(sampler.plan(e)).encode()).hexdigest()
                                    for e in range(1, a.epochs + 1)])
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    print(f"[{a.exp_id}] params={n_params:,} train={len(inner)} holdout={len(hold)} updates={total} "
          f"short=[{a.short_min},{a.short_max}] test_short={a.test_short} loss={a.loss} ema={a.ema}", flush=True)

    history, best, best_epoch, step, skipped = [], -1.0, 0, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        run_loss, seen, t0 = 0.0, 0, time.perf_counter()
        sampler.set_epoch(epoch)
        n_batches = len(sampler)
        for bi, batch in enumerate(train_loader):
            x, y, sizes = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y, sizes = y.to(device, non_blocking=True), sizes.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, sizes)
            loss = loss_fn(logits.float(), y)
            if soft is not None:
                loss = (1 - a.distill_alpha) * loss + a.distill_alpha * soft_bce(logits, batch[4].to(device))
            assert torch.isfinite(loss), "non-finite training loss"
            scaler.scale(loss / a.accum).backward()
            run_loss += loss.item() * len(x)
            seen += len(x)
            if (bi + 1) % a.accum == 0 or bi + 1 == n_batches:
                scaler.unscale_(opt)
                if a.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if use_amp and scaler.get_scale() < before:
                    skipped += 1                                   # overflow: no update, schedule waits
                    continue
                sched.step()
                step += 1
                if ema:
                    ema.update(model, step=step)
        train_sec = time.perf_counter() - t0
        lg_ho, _ = predict(eval_model, hold_loader, device, use_amp)
        m = compute_metrics(t_ho, binarize(sigmoid(lg_ho), a.threshold))
        row = dict(epoch=epoch, train_loss=run_loss / max(1, seen), holdout_bce=bce(lg_ho, t_ho),
                   holdout_micro=m["micro_f1"], holdout_macro=m["macro_f1"], holdout_image=m["image_f1"],
                   lr_backbone=opt.param_groups[0]["lr"], lr_head=opt.param_groups[1]["lr"],
                   updates=step, amp_skipped=skipped, train_sec=round(train_sec, 1))
        history.append(row)
        state = {"model": eval_model.state_dict(), "epoch": epoch, "config": cfg, "categories": cats}
        # Full continuation state is separate from EMA inference checkpoints.
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                        scheduler=sched.state_dict(), scaler=scaler.state_dict(), epoch=epoch, step=step,
                        ema=ema.state_dict() if ema else None, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                        python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                        train_loader_rng=train_loader.generator.get_state(),
                        hold_loader_rng=hold_loader.generator.get_state(), config=cfg),
                   os.path.join(out, "training_state.pt"))
        torch.save(state, os.path.join(out, "last.pt"))
        if m["macro_f1"] > best:
            best, best_epoch = m["macro_f1"], epoch
            torch.save(state, os.path.join(out, "best.pt"))
        with open(os.path.join(out, "history.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
        print(f"epoch {epoch}: loss {row['train_loss']:.4f} holdout macro {m['macro_f1']:.4f} micro {m['micro_f1']:.4f} "
              f"bce {row['holdout_bce']:.4f} best {best_epoch} ({train_sec / 60:.1f} min)", flush=True)
        if a.patience and epoch - best_epoch >= a.patience:
            print("early stop", flush=True)
            break

    eval_model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)["model"])
    summary = {"exp_id": a.exp_id, "best_epoch": best_epoch, "epochs_run": len(history), "params_total": n_params,
               "train_minutes": round((time.perf_counter() - t_start) / 60, 1), "amp_skipped": skipped}
    for split, loader, y, n in (("holdout", hold_loader, t_ho, hold),):
        lg, fwd = predict(eval_model, loader, device, use_amp)
        save_predictions(os.path.join(out, f"preds_{split}.npz"), n, y, lg, cats)
        mm = compute_metrics(y, binarize(sigmoid(lg), a.threshold))
        summary[split] = {k: mm[k] for k in ("micro_f1", "macro_f1", "image_f1")}
        summary[split]["bce"] = bce(lg, y)
        summary[f"{split}_forward_ms_per_image"] = 1000 * fwd / len(n)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)


