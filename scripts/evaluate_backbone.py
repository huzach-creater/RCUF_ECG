import argparse
import json

import numpy as np
import torch

from experiment_utils import (LABEL_ORDER, compute_multilabel_metrics,
                              device_auto, ensure, make_loader, make_model,
                              predict, thresholds_from_val, write_pred_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--cal-data")
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--out-dir", default="outputs_ptbxl")
    ap.add_argument("--prefix", default="resnet")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = ensure(args.out_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = device_auto()
    model = make_model(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    batch_size = int(ckpt.get("train_config", {}).get("batch_size", 64))
    num_workers = int(ckpt.get("train_config", {}).get("num_workers", 0))
    pin_memory = device.type == "cuda"
    prefer_wfdb = bool(ckpt.get("train_config", {}).get("prefer_wfdb_streaming", False))
    val_loader = make_loader(args.val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=pin_memory)
    cal_loader = make_loader(args.cal_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=pin_memory) if args.cal_data else None
    test_loader = make_loader(args.test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefer_wfdb=prefer_wfdb, pin_memory=pin_memory)
    yv, lv, pv, mvmeta = predict(model, val_loader, device)
    cal_pack = predict(model, cal_loader, device) if cal_loader is not None else None
    yt, lt, pt, mtmeta = predict(model, test_loader, device)
    th = thresholds_from_val(yv, pv)
    th_arr = np.array([th[l] for l in LABEL_ORDER])
    predv05, predt05 = (pv >= 0.5).astype(int), (pt >= 0.5).astype(int)
    predvopt, predtopt = (pv >= th_arr).astype(int), (pt >= th_arr).astype(int)
    metrics = {
        "checkpoint": args.checkpoint,
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_macro_auc": ckpt.get("best_val_macro_auc"),
        "val_metrics_threshold_0p5": compute_multilabel_metrics(yv, pv, predv05, LABEL_ORDER),
        "val_metrics_threshold_opt": compute_multilabel_metrics(yv, pv, predvopt, LABEL_ORDER),
        "test_metrics_threshold_0p5": compute_multilabel_metrics(yt, pt, predt05, LABEL_ORDER),
        "test_metrics_threshold_opt": compute_multilabel_metrics(yt, pt, predtopt, LABEL_ORDER),
        "label_thresholds": th,
    }
    if cal_pack is not None:
        yc, lc, pc, mcmeta = cal_pack
        predc05, predcopt = (pc >= 0.5).astype(int), (pc >= th_arr).astype(int)
        metrics["cal_metrics_threshold_0p5"] = compute_multilabel_metrics(yc, pc, predc05, LABEL_ORDER)
        metrics["cal_metrics_threshold_opt"] = compute_multilabel_metrics(yc, pc, predcopt, LABEL_ORDER)
    suffix = f"{args.prefix}_seed{args.seed}" if args.prefix else f"seed{args.seed}"
    (out / f"calibration/ptbxl_label_thresholds_{suffix}.json").write_text(json.dumps({"source": "val", "thresholds": th}, indent=2), encoding="utf-8")
    write_pred_csv(out / f"predictions/ptbxl_val_predictions_{suffix}.csv", mvmeta, yv, lv, pv, predv05, predvopt)
    if cal_pack is not None:
        write_pred_csv(out / f"predictions/ptbxl_cal_predictions_{suffix}.csv", mcmeta, yc, lc, pc, predc05, predcopt)
    write_pred_csv(out / f"predictions/ptbxl_test_predictions_{suffix}.csv", mtmeta, yt, lt, pt, predt05, predtopt)
    metric_name = f"ptbxl_resnet_{args.prefix}_baseline_metrics_seed{args.seed}.json" if args.prefix else f"ptbxl_resnet_baseline_metrics_seed{args.seed}.json"
    (out / "metrics" / metric_name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    tm = metrics["test_metrics_threshold_opt"]
    p2g_mode = args.prefix == "gpu"
    status = "PASS" if (tm["macro_auc"] or 0) >= (0.80 if p2g_mode else 0.75) and tm["macro_f1"] >= (0.50 if p2g_mode else 0.45) else "FAIL"
    title = "# PTB-XL P2G ResNet GPU Baseline Report" if p2g_mode else "# PTB-XL P2R ResNet Baseline Report"
    report = [title, "", f"PASS/FAIL: {status}", f"checkpoint path: {args.checkpoint}", f"threshold source: validation only", f"best epoch: {ckpt.get('best_epoch')}", f"best val Macro-AUC: {ckpt.get('best_val_macro_auc')}", f"test Macro-AUC: {tm['macro_auc']}", f"test Micro-AUC: {tm['micro_auc']}", f"test Macro-AUPRC: {tm['macro_auprc']}", f"test Macro-F1: {tm['macro_f1']}", f"test Micro-F1: {tm['micro_f1']}", f"per-class AUC: {tm['per_class_auc']}", f"per-class F1: {tm['per_class_f1']}", f"thresholds: {th}"]
    report_name = "phase_p2g_resnet_gpu_baseline_report.md" if p2g_mode else "phase_p2r_resnet_baseline_report.md"
    (out / "reports" / report_name).write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": status, "test": tm}, indent=2))


if __name__ == "__main__":
    main()
