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
    late_fusion = commands.add_parser("train-late-fusion")
    late_fusion.add_argument("--base", nargs="+", required=True,
                             help="Two or more exported base-model probability .npz files")
    late_fusion.add_argument("--output", required=True)
    late_fusion.add_argument("--method", default="validation_weighted_late_fusion",
                             choices=["equal_late_fusion", "validation_weighted_late_fusion",
                                      "logistic_probability_meta"])
    late_fusion.add_argument("--calibrated", action="store_true")
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

    def _gate(source: str) -> None:
        """Enforce the leakage gate before any extraction/training. Blocks unless
        clean or explicitly overridden with a written, audit-logged justification."""
        from doar.leakage import enforce_leakage_gate
        gate_out = Path(args.output) / "leakage_gate"
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
                         "class_weighting", "early_stopping_patience"},
            "optimization": {"optimizer", "head_learning_rate", "backbone_learning_rate",
                             "scheduler"},
            "output": {"directory", "calibration"},
        })
        values = resolve(vars(args), config, {
            "dataset": ("data", "dataset"), "model": ("training", "model"),
            "output": ("output", "directory"), "seed": ("training", "seed"),
            "epochs": ("training", "epochs"), "batch_size": ("training", "batch_size"),
            "image_size": ("training", "image_size"), "device": ("training", "device"),
            "augmentation": ("training", "augmentation"),
        }, supplied)
        if not all(values.get(name) for name in ("dataset", "model", "output")):
            raise ValueError("train-image-model requires dataset, model, and output via CLI or config")
        _gate(values["dataset"])
        config_hash = save_resolved(values["output"], args.command, values)
        print(json.dumps(train_image_model(
            values["dataset"], values["model"], values["output"], seed=int(values["seed"]),
            epochs=int(values["epochs"]), batch_size=int(values["batch_size"]),
            image_size=int(values["image_size"]), device=values["device"],
            augmentation=values["augmentation"], resume=args.resume,
            configuration_hash=config_hash,
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
    elif args.command == "train-late-fusion":
        from doar.fusion.late import train_late_fusion
        print(json.dumps(train_late_fusion(
            args.base, args.output, args.method, args.calibrated
        ), ensure_ascii=False, indent=2, default=float))
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
        _gate(values["manifest"])
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
