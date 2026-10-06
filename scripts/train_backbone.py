import argparse
import json

import numpy as np
import pandas as pd
import torch

from experiment_utils import (compute_multilabel_metrics, device_auto, ensure,
                              load_yaml, make_loader, make_model,
                              pos_weight_from_y, predict, seed_all, train_epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--out-dir", default="outputs_ptbxl")
    ap.add_argument("--checkpoint-name", default=None)
    ap.add_argument("--prefix", default="resnet")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    seed_all(args.seed)
    out = ensure(args.out_dir)
    cfg = load_yaml(args.config)
    mcfg = load_yaml(args.model_config)
    requested_device = str(cfg.get("device", "auto")).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("configs requested CUDA, but torch.cuda.is_available() is False")
    device = torch.device("cuda" if requested_device == "cuda" else device_auto())
    train_d = np.load(args.train_data, allow_pickle=True)
    train_indices = None
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"
    prefer_wfdb = bool(cfg.get("prefer_wfdb_streaming", False))
    train_loader = make_loader(args.train_data, batch_size=batch_size, shuffle=True, indices=train_indices, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=pin_memory)
    val_loader = make_loader(args.val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=pin_memory)
    model = make_model(mcfg).to(device)
    posw = pos_weight_from_y(train_d["y"] if train_indices is None else train_d["y"][train_indices], float(cfg.get("pos_weight_clip", 20))).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=float(cfg.get("scheduler_factor", 0.5)), patience=int(cfg.get("scheduler_patience", 2)), min_lr=float(cfg.get("min_lr", 1e-5)))
    best_auc, best_epoch, best_state = -1, 0, None
    rows = []
    patience = int(cfg.get("early_stop_patience", 8)); bad = 0
    accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    use_amp = bool(cfg.get("amp", False))
    for epoch in range(1, int(cfg.get("epochs", 8)) + 1):
        tr_loss = train_epoch(
            model,
            train_loader,
            opt,
            loss_fn,
            device,
            float(cfg.get("gradient_clip_norm", 5)),
            accumulation_steps=accumulation_steps,
            amp=use_amp,
        )
        yv, lv, pv, _ = predict(model, val_loader, device)
        val_loss = float(loss_fn(torch.tensor(lv, dtype=torch.float32, device=device), torch.tensor(yv, dtype=torch.float32, device=device)).detach().cpu())
        mv = compute_multilabel_metrics(yv, pv, (pv >= 0.5).astype(int), ["NORM", "MI", "STTC", "CD", "HYP"])
        lr = opt.param_groups[0]["lr"]
        rows.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, "val_macro_auc": mv["macro_auc"], "val_micro_auc": mv["micro_auc"], "val_macro_f1_at_0p5": mv["macro_f1"], "val_micro_f1_at_0p5": mv["micro_f1"], "learning_rate": lr})
        scheduler.step(mv["macro_auc"] or 0)
        if (mv["macro_auc"] or -1) > best_auc:
            best_auc, best_epoch, best_state = mv["macro_auc"], epoch, {k: v.cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(rows[-1])
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    ckpt = {"model_state_dict": model.state_dict(), "model_config": mcfg, "train_config": cfg, "label_names": ["NORM", "MI", "STTC", "CD", "HYP"], "best_epoch": best_epoch, "best_val_macro_auc": best_auc, "seed": args.seed}
    checkpoint_name = args.checkpoint_name or f"ptbxl_resnet1d_seed{args.seed}.pt"
    torch.save(ckpt, out / "checkpoints" / checkpoint_name)
    log_name = f"ptbxl_resnet_{args.prefix}_train_log_seed{args.seed}.csv" if args.prefix else f"ptbxl_resnet_train_log_seed{args.seed}.csv"
    pd.DataFrame(rows).to_csv(out / "metrics" / log_name, index=False)
    report = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_size": mcfg.get("model_size"),
        "base_channels": mcfg.get("base_channels"),
        "embedding_dim": mcfg.get("embedding_dim"),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "amp": use_amp,
        "num_workers": num_workers,
        "prefer_wfdb_streaming": prefer_wfdb,
        "training_records": int(len(train_d["y"])),
        "epochs_run": len(rows),
        "best_epoch": best_epoch,
        "best_val_macro_auc": best_auc,
        "checkpoint": str(out / "checkpoints" / checkpoint_name),
        "train_log": str(out / "metrics" / log_name),
    }
    title = "# PTB-XL P2G ResNet GPU Training" if args.prefix == "gpu" else "# PTB-XL P2R ResNet Training"
    report_name = "phase_p2g_resnet_gpu_training_report.md" if args.prefix == "gpu" else "phase_p2r_resnet_training_report.md"
    (out / "reports" / report_name).write_text(title + "\n\n" + json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
