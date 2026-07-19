from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from doar.analysis import analyze_image
from doar.dataset import build_manifest
from doar.extract import extract_features
from doar.experiments import run_feature_experiment
from doar.qa import answer
from doar.models import evaluate_model, train_model
from doar.config import load_config, resolve, save_resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="DOAR v3 research pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze-image")
    analyze.add_argument("--image", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--emotion-checkpoint")
    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--output", required=True)
    def _add_leakage_args(parser):
        parser.add_argument("--subject-key", default="subject_id",
                            help="Manifest column for child/subject grouping")
        parser.add_argument("--allow-leakage-override", action="store_true",
                            help="Proceed despite leakage (requires --override-justification)")
        parser.add_argument("--override-justification", default=None,
                            help="Written reason logged to the leakage override audit")

    extract = commands.add_parser("extract-features")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--output", required=True)
    _add_leakage_args(extract)
    train = commands.add_parser("train")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=42)
    _add_leakage_args(train)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=("valid", "test"), default="valid")
    evaluate.add_argument("--unlock-test", action="store_true")
    evaluate.add_argument("--confirm-final-evaluation", action="store_true")
    evaluate.add_argument("--initiated-by")
    feature_model = commands.add_parser("train-feature-model")
    feature_model.add_argument("--features", required=True)
    feature_model.add_argument("--output", required=True)
    feature_model.add_argument("--models", default="")
    feature_model.add_argument("--seeds", default="42,123,2026")
    compare = commands.add_parser("compare-models")
    compare.add_argument("--features", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--models", default="")
    compare.add_argument("--seeds", default="42,123,2026")
    qa = commands.add_parser("qa")
    qa.add_argument("--analysis", required=True)
    qa.add_argument("--question", required=True)
    qa.add_argument("--language", choices=("en", "ar"), default="en")
    image_train = commands.add_parser("train-image-model")
    image_train.add_argument("--config")
    image_train.add_argument("--dataset")
    image_train.add_argument("--model")
    image_train.add_argument("--output")
    image_train.add_argument("--seed", type=int, default=42)
    image_train.add_argument("--epochs", type=int, default=30)
    image_train.add_argument("--batch-size", type=int, default=16)
    image_train.add_argument("--image-size", type=int, default=224)
    image_train.add_argument("--device", default="auto")
    image_train.add_argument("--augmentation", default="conservative")
    image_train.add_argument("--validation-split", default=None)
    image_train.add_argument("--pretrained-weights", default=None)
    image_train.add_argument("--workers", type=int, default=None)
    image_train.add_argument("--freeze-epochs", type=int, default=None)
    image_train.add_argument("--class-weighting", default=None,
                             choices=["true", "false"])
    image_train.add_argument("--early-stopping-patience", type=int, default=None)
    image_train.add_argument("--optimizer", default=None)
    image_train.add_argument("--head-learning-rate", type=float, default=None)
    image_train.add_argument("--backbone-learning-rate", type=float, default=None)
    image_train.add_argument("--scheduler", default=None)
    image_train.add_argument("--calibration", default=None)
    image_train.add_argument("--grad-accum-steps", type=int, default=None)
    image_train.add_argument("--resume")
    _add_leakage_args(image_train)
    image_predict = commands.add_parser("predict-image")
    image_predict.add_argument("--image", required=True)
    image_predict.add_argument("--checkpoint", required=True)
    image_predict.add_argument("--device", default="auto")
    ingest = commands.add_parser("ingest-psychology-pdf")
    ingest.add_argument("--pdf", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--source-id", default=None)
    def _add_test_guard_args(parser):
        parser.add_argument("--unlock-test", action="store_true")
        parser.add_argument("--confirm-final-evaluation", action="store_true")
        parser.add_argument("--initiated-by", default=None)

    export_probs = commands.add_parser("export-probabilities")
    export_probs.add_argument("--model", required=True, help=".joblib bundle or .pt/.pth checkpoint")
    export_probs.add_argument("--features", default=None, help="Features CSV (sklearn models)")
    export_probs.add_argument("--embeddings", default=None, help="Embeddings .npz (fusion models)")
    export_probs.add_argument("--manifest", default=None, help="Manifest CSV (deep checkpoints)")
    export_probs.add_argument("--device", default="auto")
    export_probs.add_argument("--output", required=True, help="Output export .json path")
    export_probs.add_argument("--splits", default="train,valid", help="Comma splits to export")
    _add_test_guard_args(export_probs)

    late_fusion = commands.add_parser("train-late-fusion")
    late_fusion.add_argument("--base", nargs="+", required=True,
                             help="Two or more base-model probability export .json files")
    late_fusion.add_argument("--output", required=True)
    late_fusion.add_argument("--method", default="validation_weighted_late_fusion",
                             choices=["equal_late_fusion", "validation_weighted_late_fusion",
                                      "logistic_probability_meta"])
    late_fusion.add_argument("--calibrated", action="store_true")

    apply_late = commands.add_parser("apply-late-fusion")
    apply_late.add_argument("--model", required=True, help="late_fusion_model.json")
    apply_late.add_argument("--base", nargs="+", required=True)
    apply_late.add_argument("--split", default="valid")
    apply_late.add_argument("--output", required=True)
    _add_test_guard_args(apply_late)
    thesis_cmd = commands.add_parser("generate-thesis-outputs")
    thesis_cmd.add_argument("--output", required=True, help="Output root containing experiment artifacts")

    ablation = commands.add_parser("run-ablation")
    ablation.add_argument("--features", required=True)
    ablation.add_argument("--output", required=True)
    ablation.add_argument("--seeds", default="42,123,2026")

    review_agree = commands.add_parser("review-agreement")
    review_agree.add_argument("--master", required=True, help="review-master CSV")
    review_agree.add_argument("--output", required=True)
    review_agree.add_argument("--include-synthetic", action="store_true")

    explain_feat = commands.add_parser("explain-features")
    explain_feat.add_argument("--model", required=True, help="Objective-feature .joblib model")
    explain_feat.add_argument("--features", required=True)
    explain_feat.add_argument("--output", required=True)
    explain_feat.add_argument("--n-repeats", type=int, default=5)

    explain_cam = commands.add_parser("explain-gradcam")
    explain_cam.add_argument("--image", required=True)
    explain_cam.add_argument("--checkpoint", required=True)
    explain_cam.add_argument("--output", required=True)
    explain_cam.add_argument("--device", default="auto")

    cal_fusion = commands.add_parser("calibrate-fusion")
    cal_fusion.add_argument("--bundle", required=True, help="Fusion .joblib bundle")
    cal_fusion.add_argument("--features", required=True)
    cal_fusion.add_argument("--embeddings", required=True)
    cal_fusion.add_argument("--output", required=True)
    emb_compare = commands.add_parser("compare-embeddings")
    emb_compare.add_argument("--features", required=True)
    emb_compare.add_argument("--generic", required=True, help="Generic embeddings .npz")
    emb_compare.add_argument("--finetuned", required=True, help="Fine-tuned embeddings .npz")
    emb_compare.add_argument("--output", required=True)
    emb_compare.add_argument("--seed", type=int, default=42)
    deep_compare = commands.add_parser("compare-deep-models")
    deep_compare.add_argument("--dataset", required=True)
    deep_compare.add_argument("--output", required=True)
    deep_compare.add_argument("--models", default=None,
                              help="Comma list (default: small_cnn,mobilenet_v3_small,resnet18,efficientnet_b0)")
    deep_compare.add_argument("--seeds", default="42,123,2026")
    deep_compare.add_argument("--batch-size", type=int, default=4, help="6 GB-safe default")
    deep_compare.add_argument("--epochs", type=int, default=30)
    deep_compare.add_argument("--grad-accum-steps", type=int, default=1)
    deep_compare.add_argument("--image-size", type=int, default=224)
    deep_compare.add_argument("--device", default="auto")
    deep_compare.add_argument("--calibration", default=None)
    _add_leakage_args(deep_compare)
    eval_preds = commands.add_parser("evaluate-predictions")
    eval_preds.add_argument("--export", required=True, help="A probability export JSON")
    eval_preds.add_argument("--split", default="valid")
    eval_preds.add_argument("--output", required=True)
    _add_test_guard_args(eval_preds)
    gpu_smoke = commands.add_parser("gpu-smoke")
    gpu_smoke.add_argument("--output", default=None)
    gpu_smoke.add_argument("--device", default="auto")
    gpu_smoke.add_argument("--batch-size", type=int, default=4)
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--dataset", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--device", default="auto")
    embedding = commands.add_parser("extract-embeddings")
    embedding.add_argument("--config")
    embedding.add_argument("--manifest")
    embedding.add_argument("--output")
    embedding.add_argument("--backbone")
    embedding.add_argument("--device", default="auto")
    embedding.add_argument("--batch-size", type=int, default=16)
    embedding.add_argument("--force", action="store_true")
    _add_leakage_args(embedding)
    fusion = commands.add_parser("train-fusion-model")
    fusion.add_argument("--config")
    fusion.add_argument("--features")
    fusion.add_argument("--embeddings")
    fusion.add_argument("--output")
    fusion.add_argument(
        "--methods", default="early_scaled_concat,pca_early_fusion,mlp_early_fusion"
    )
    fusion.add_argument("--seeds", default="42,123,2026")
    validate = commands.add_parser("validate-dataset")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output", required=True)
    readiness = commands.add_parser("check-training-readiness")
    readiness.add_argument("--dataset")
    readiness.add_argument("--output", required=True)
    readiness.add_argument("--manifest")
    readiness.add_argument("--features")
    readiness.add_argument("--embeddings")
    args = parser.parse_args()
    supplied = {
        token[2:].replace("-", "_") for token in sys.argv[1:]
        if token.startswith("--")
    }

    def _gate(source: str, resolved_output=None) -> None:
        """Enforce the leakage gate before any extraction/training. Blocks unless
        clean or explicitly overridden with a written, audit-logged justification.

        `resolved_output` must be passed for config-driven commands where
        args.output may be None (A1). Falls back to args.output for direct CLI."""
        from doar.leakage import enforce_leakage_gate
        base = resolved_output if resolved_output is not None else getattr(args, "output", None)
        if base is None:
            raise ValueError("Leakage gate requires an output directory (config or --output).")
        gate_out = Path(base) / "leakage_gate"
        report = enforce_leakage_gate(
            source, gate_out, subject_key=getattr(args, "subject_key", "subject_id"),
            allow_override=getattr(args, "allow_leakage_override", False),
            override_justification=getattr(args, "override_justification", None),
            initiated_by=getattr(args, "initiated_by", None) or "cli",
        )
        print(json.dumps({"leakage_gate": report["gate"], "status": report["status"],
                          "report": str(gate_out / "leakage_report.json")}, indent=2))

    if args.command == "analyze-image":
        result = analyze_image(args.image, args.output, args.emotion_checkpoint)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "build-manifest":
        print(json.dumps(build_manifest(args.dataset, args.output), indent=2))
    elif args.command == "extract-features":
        _gate(args.manifest)
        print(json.dumps(extract_features(args.manifest, args.output), indent=2))
    elif args.command == "train":
        _gate(args.manifest)
        print(json.dumps(train_model(args.manifest, args.output, args.seed), indent=2))
    elif args.command == "evaluate":
        print(json.dumps(evaluate_model(
            args.manifest, args.checkpoint, args.output, args.split, args.unlock_test,
            args.confirm_final_evaluation, args.initiated_by,
        ), indent=2))
    elif args.command in ("train-feature-model", "compare-models"):
        models = [value.strip() for value in args.models.split(",") if value.strip()] or None
        seeds = tuple(int(value) for value in args.seeds.split(","))
        print(json.dumps(run_feature_experiment(
            args.features, args.output, models=models, seeds=seeds
        ), indent=2))
    elif args.command == "qa":
        analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
        judges_path = Path(args.analysis).with_name("judges.json")
        judges = json.loads(judges_path.read_text(encoding="utf-8"))
        print(json.dumps(answer(args.question, analysis, judges, args.language),
                         ensure_ascii=False, indent=2))
    elif args.command == "train-image-model":
        from doar.deep.trainers import train_image_model
        config = load_config(args.config, {
            "data": {"dataset", "validation_split"},
            "training": {"model", "pretrained_weights", "seed", "image_size", "batch_size",
                         "epochs", "device", "workers", "augmentation", "freeze_epochs",
                         "class_weighting", "early_stopping_patience", "grad_accum_steps"},
            "optimization": {"optimizer", "head_learning_rate", "backbone_learning_rate",
                             "scheduler"},
            "output": {"directory", "calibration"},
        })
        from doar.config import assert_all_config_consumed, save_run_metadata
        # Every accepted config field is mapped -> consumed (Item 2). If a field
        # is accepted by load_config but missing here, assert_all_config_consumed
        # fails, so no setting is ever silently ignored.
        mapping = {
            "dataset": ("data", "dataset"), "validation_split": ("data", "validation_split"),
            "model": ("training", "model"), "pretrained_weights": ("training", "pretrained_weights"),
            "seed": ("training", "seed"), "image_size": ("training", "image_size"),
            "batch_size": ("training", "batch_size"), "epochs": ("training", "epochs"),
            "device": ("training", "device"), "workers": ("training", "workers"),
            "augmentation": ("training", "augmentation"),
            "freeze_epochs": ("training", "freeze_epochs"),
            "class_weighting": ("training", "class_weighting"),
            "early_stopping_patience": ("training", "early_stopping_patience"),
            "optimizer": ("optimization", "optimizer"),
            "head_learning_rate": ("optimization", "head_learning_rate"),
            "backbone_learning_rate": ("optimization", "backbone_learning_rate"),
            "scheduler": ("optimization", "scheduler"),
            "grad_accum_steps": ("training", "grad_accum_steps"),
            "output": ("output", "directory"), "calibration": ("output", "calibration"),
        }
        assert_all_config_consumed(config, mapping)
        values = resolve(vars(args), config, mapping, supplied)
        if not all(values.get(name) for name in ("dataset", "model", "output")):
            raise ValueError("train-image-model requires dataset, model, and output via CLI or config")
        _gate(values["dataset"], values["output"])
        config_hash = save_run_metadata(values["output"], args.command, vars(args), values)

        def _opt(name, cast, default):
            v = values.get(name)
            return cast(v) if v is not None else default
        cw = values.get("class_weighting")
        class_weighting = (str(cw).lower() == "true") if cw is not None else True

        print(json.dumps(train_image_model(
            values["dataset"], values["model"], values["output"],
            seed=_opt("seed", int, 42), epochs=_opt("epochs", int, 30),
            batch_size=_opt("batch_size", int, 16), image_size=_opt("image_size", int, 224),
            device=values["device"] or "auto", workers=_opt("workers", int, 0),
            augmentation=values["augmentation"] or "conservative",
            patience=_opt("early_stopping_patience", int, 7),
            freeze_epochs=_opt("freeze_epochs", int, 3),
            class_weighting=class_weighting,
            optimizer_name=values.get("optimizer") or "adamw",
            head_learning_rate=_opt("head_learning_rate", float, 3e-4),
            backbone_learning_rate=_opt("backbone_learning_rate", float, 1e-4),
            scheduler_name=values.get("scheduler") or "reduce_on_plateau",
            calibration=values.get("calibration"),
            grad_accum_steps=_opt("grad_accum_steps", int, 1),
            pretrained_weights=values.get("pretrained_weights") or "DEFAULT",
            resume=args.resume, configuration_hash=config_hash,
        ), indent=2))
    elif args.command == "predict-image":
        from doar.deep.inference import predict_image
        print(json.dumps(predict_image(
            args.image, args.checkpoint, args.device
        ), ensure_ascii=False, indent=2))
    elif args.command == "ingest-psychology-pdf":
        from doar.psychology_ingest import ingest_pdf
        meta = {"source_id": args.source_id} if args.source_id else None
        draft = ingest_pdf(args.pdf, args.output, meta)
        print(json.dumps({"draft": True, "rule_count": draft["rule_count"],
                          "activation_blocked": draft["activation_blocked"],
                          "output": args.output}, ensure_ascii=False, indent=2))
    elif args.command == "export-probabilities":
        from doar.probability_export import export_probabilities
        from doar.test_guard import require_test_access
        req_splits = [s.strip() for s in args.splits.split(",")]
        if "test" in req_splits:
            require_test_access(
                "test", unlock_test=args.unlock_test,
                confirm_final_evaluation=args.confirm_final_evaluation,
                initiated_by=args.initiated_by, command="export-probabilities",
                audit_dir=Path(args.output).parent, model=args.model, splits=req_splits)
        print(json.dumps(export_probabilities(
            args.model, args.features, args.embeddings, args.output,
            splits=req_splits, manifest=args.manifest, device=args.device,
        ), indent=2))
    elif args.command == "train-late-fusion":
        from doar.fusion.late import train_late_fusion
        print(json.dumps(train_late_fusion(
            args.base, args.output, args.method, args.calibrated
        ), ensure_ascii=False, indent=2, default=float))
    elif args.command == "apply-late-fusion":
        import numpy as _np
        from doar.fusion.late import load_late_fusion, apply_late_fusion
        from doar.test_guard import require_test_access
        if args.split == "test":
            require_test_access(
                "test", unlock_test=args.unlock_test,
                confirm_final_evaluation=args.confirm_final_evaluation,
                initiated_by=args.initiated_by, command="apply-late-fusion",
                audit_dir=args.output, model=args.model)
        model_meta = load_late_fusion(args.model)
        ids, fused = apply_late_fusion(model_meta, args.base, args.split)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        order = model_meta["class_order"]
        rows = [{"sample_id": sid, **{order[j]: float(fused[i][j]) for j in range(len(order))}}
                for i, sid in enumerate(ids)]
        (out / f"fused_{args.split}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({"split": args.split, "count": len(ids),
                          "output": str(out / f"fused_{args.split}.json")}, indent=2))
    elif args.command == "generate-thesis-outputs":
        from doar.thesis import generate_thesis_outputs
        print(json.dumps(generate_thesis_outputs(args.output), indent=2))
    elif args.command == "run-ablation":
        from doar.ablation import run_feature_ablation
        seeds = tuple(int(s) for s in args.seeds.split(","))
        print(json.dumps(run_feature_ablation(args.features, args.output, seeds), indent=2))
    elif args.command == "review-agreement":
        from doar.review import compute_agreement
        result = compute_agreement(args.master, exclude_synthetic=not args.include_synthetic)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "agreement.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.command == "explain-features":
        import numpy as _np
        import joblib
        from doar.explain.feature_importance import permutation_importance
        from doar.fusion.embedding_comparison import load_features_indexed
        from doar.dataset import CLASSES
        bundle = joblib.load(args.model)
        model = bundle["model"]
        feat = load_features_indexed(args.features)["valid"]
        names = bundle.get("feature_names") or []
        ids = sorted(feat)
        X = _np.vstack([feat[i][0] for i in ids])
        y = _np.array([feat[i][1] for i in ids])
        result = permutation_importance(model.predict_proba, X, y,
                                        names or [f"f{i}" for i in range(X.shape[1])],
                                        n_repeats=args.n_repeats)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "feature_importance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        import csv as _csv
        with open(out / "feature_importance.csv", "w", newline="", encoding="utf-8") as h:
            w = _csv.writer(h); w.writerow(["feature", "importance_mean", "importance_std"])
            for r in result["importances"]:
                w.writerow([r["feature"], r["importance_mean"], r["importance_std"]])
        print(json.dumps({"top": result["importances"][:5], "disclaimer": result["disclaimer"]}, indent=2))
    elif args.command == "explain-gradcam":
        from doar.explain.gradcam import generate_gradcam
        print(json.dumps(generate_gradcam(
            args.image, args.checkpoint, args.output, args.device), indent=2))
    elif args.command == "calibrate-fusion":
        from doar.fusion.calibrate import calibrate_fusion_bundle
        print(json.dumps(calibrate_fusion_bundle(
            args.bundle, args.features, args.embeddings, args.output), indent=2))
    elif args.command == "compare-embeddings":
        from doar.fusion.embedding_comparison import run_embedding_comparison
        print(json.dumps(run_embedding_comparison(
            args.features, args.generic, args.finetuned, args.output, seed=args.seed,
        ), indent=2))
    elif args.command == "compare-deep-models":
        from doar.deep.compare import run_deep_comparison, DEFAULT_MODELS
        _gate(args.dataset, args.output)
        models = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS
        seeds = tuple(int(s) for s in args.seeds.split(","))
        print(json.dumps(run_deep_comparison(
            args.dataset, args.output, models=models, seeds=seeds,
            batch_size=args.batch_size, image_size=args.image_size, device=args.device,
            epochs=args.epochs, grad_accum_steps=args.grad_accum_steps,
            calibration=args.calibration,
        ), indent=2))
    elif args.command == "evaluate-predictions":
        import numpy as _np
        from doar.evaluation import (load_probability_export, compute_metrics,
                                     write_metrics_csv, CLASS_ORDER)
        from doar.test_guard import require_test_access
        if args.split == "test":
            require_test_access(
                "test", unlock_test=args.unlock_test,
                confirm_final_evaluation=args.confirm_final_evaluation,
                initiated_by=args.initiated_by, command="evaluate-predictions",
                audit_dir=args.output, export=args.export)
        exp = load_probability_export(args.export)
        rows = [r for r in exp["predictions"] if r["split"] == args.split]
        if not rows:
            raise ValueError(f"No predictions for split {args.split!r}")
        order = exp["class_order"]
        y_true = _np.array([order.index(r["true_label"]) for r in rows])
        proba = _np.array([[r["probabilities"][c] for c in order] for r in rows])
        y_pred = proba.argmax(1)
        metrics = compute_metrics(y_true, y_pred, proba, class_names=order)
        out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_metrics_csv(metrics, out / "per_class_metrics.csv")
        print(json.dumps({"split": args.split, "macro_f1": metrics["macro_f1"],
                          "accuracy": metrics["accuracy"]}, indent=2))
    elif args.command == "gpu-smoke":
        from doar.gpu_smoke import run_gpu_smoke
        print(json.dumps(run_gpu_smoke(args.output, args.device, args.batch_size), indent=2))
    elif args.command == "calibrate":
        from doar.deep.calibration import calibrate_checkpoint
        print(json.dumps(calibrate_checkpoint(
            args.checkpoint, args.dataset, args.output, args.device
        ), ensure_ascii=False, indent=2))
    elif args.command == "extract-embeddings":
        from doar.deep.embeddings import extract_embeddings
        config = load_config(args.config, {
            "input": {"manifest"}, "embedding": {"backbone", "device", "batch_size"},
            "output": {"directory"},
        })
        values = resolve(vars(args), config, {
            "manifest": ("input", "manifest"), "backbone": ("embedding", "backbone"),
            "device": ("embedding", "device"), "batch_size": ("embedding", "batch_size"),
            "output": ("output", "directory"),
        }, supplied)
        if not all(values.get(name) for name in ("manifest", "backbone", "output")):
            raise ValueError("extract-embeddings requires manifest, backbone, and output")
        _gate(values["manifest"], values["output"])
        config_hash = save_resolved(values["output"], args.command, values)
        print(json.dumps(extract_embeddings(
            values["manifest"], values["output"], values["backbone"], values["device"],
            int(values["batch_size"]), args.force,
        ), indent=2))
    elif args.command == "train-fusion-model":
        from doar.fusion.trainer import train_primary_fusion
        config = load_config(args.config, {
            "inputs": {"features", "embeddings"},
            "experiment": {"methods", "seeds", "selection_split", "primary_metric",
                           "psychologist_rules_used", "concern_profiles_used"},
            "output": {"directory", "calibration"},
        })
        values = resolve(vars(args), config, {
            "features": ("inputs", "features"), "embeddings": ("inputs", "embeddings"),
            "output": ("output", "directory"), "methods": ("experiment", "methods"),
            "seeds": ("experiment", "seeds"),
        }, supplied)
        if not all(values.get(name) for name in ("features", "embeddings", "output")):
            raise ValueError("train-fusion-model requires features, embeddings, and output")
        methods = values["methods"] if isinstance(values["methods"], list) else values["methods"].split(",")
        seeds = values["seeds"] if isinstance(values["seeds"], list) else values["seeds"].split(",")
        config_hash = save_resolved(values["output"], args.command, values)
        print(json.dumps(train_primary_fusion(
            values["features"], values["embeddings"], values["output"],
            [str(item).strip() for item in methods if str(item).strip()],
            tuple(int(item) for item in seeds), configuration_hash=config_hash,
        ), indent=2))
    elif args.command == "validate-dataset":
        from doar.readiness import validate_dataset
        print(json.dumps(validate_dataset(args.dataset, args.output), indent=2))
    elif args.command == "check-training-readiness":
        from doar.readiness import check_training_readiness
        print(json.dumps(check_training_readiness(
            args.dataset, args.output, args.manifest, args.features, args.embeddings
        ), indent=2))


if __name__ == "__main__":
    main()
