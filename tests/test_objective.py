from __future__ import annotations

import sys
import tempfile
import unittest
import csv
import json
import types
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doar.analysis import analyze_image


class ObjectiveTests(unittest.TestCase):
    def analyze(self, image: Image.Image):
        temp = tempfile.TemporaryDirectory()
        path = Path(temp.name) / "image.png"
        image.save(path)
        result = analyze_image(path, Path(temp.name) / "out")
        return temp, result

    def test_blank_page(self):
        temp, result = self.analyze(Image.new("RGB", (200, 200), "white"))
        self.addCleanup(temp.cleanup)
        self.assertLess(result.composition["foreground_coverage"], 0.01)
        self.assertEqual(result.composition["placement"], "unavailable")
        self.assertEqual(result.colour["colour_diversity"], 0)

    def test_centered_black_object(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((60, 60, 140, 140), fill="black")
        temp, result = self.analyze(image)
        self.addCleanup(temp.cleanup)
        self.assertEqual(result.composition["placement"], "middle_center")
        self.assertAlmostEqual(
            result.composition["foreground_coverage"] + result.composition["empty_space_ratio"],
            1.0,
        )
        self.assertEqual(result.colour["dominant_colour"], "dark")

    def test_top_left_object(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((10, 10, 50, 50), fill="red")
        temp, result = self.analyze(image)
        self.addCleanup(temp.cleanup)
        self.assertEqual(result.composition["placement"], "top_left")
        self.assertEqual(result.colour["dominant_colour"], "red")
        top = next(item for item in result.rule_evaluations
                   if item["rule_id"] == "PSY_AR_PLACE_TOP_017")
        self.assertEqual(top["status"], "weak_support")
        self.assertTrue(top["references"])
        self.assertIn("not supported", top["professional_reasoning"])

    def test_symbol_rules_do_not_activate_without_detector(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).ellipse((60, 60, 140, 140), fill="black")
        temp, result = self.analyze(image)
        self.addCleanup(temp.cleanup)
        fox = next(item for item in result.rule_evaluations
                   if item["rule_id"] == "PSY_AR_ANIMAL_FOX_005")
        self.assertEqual(fox["status"], "not_evaluated")
        self.assertEqual(result.concerns, [])

    def test_tiny_noise_not_dominant(self):
        pixels = np.full((200, 200, 3), 255, dtype=np.uint8)
        pixels[100, 100] = [255, 0, 0]
        temp, result = self.analyze(Image.fromarray(pixels))
        self.addCleanup(temp.cleanup)
        self.assertEqual(result.colour["dominant_colour"], "none_or_neutral")

    def test_off_white_blank_page(self):
        temp, result = self.analyze(Image.new("RGB", (200, 200), (242, 238, 225)))
        self.addCleanup(temp.cleanup)
        self.assertLess(result.composition["foreground_coverage"], 0.01)
        self.assertEqual(result.composition["placement"], "unavailable")

    def test_faint_pencil_preserved(self):
        image = Image.new("RGB", (200, 200), (250, 248, 240))
        ImageDraw.Draw(image).line((30, 100, 170, 100), fill=(180, 178, 172), width=3)
        temp, result = self.analyze(image)
        self.addCleanup(temp.cleanup)
        self.assertGreater(result.composition["foreground_coverage"], 0.001)
        self.assertLess(result.composition["foreground_coverage"], 0.1)

    def test_bbox_rules_cite_bbox_evidence(self):
        image = Image.new("RGB", (200, 200), "white")
        ImageDraw.Draw(image).rectangle((50, 50, 150, 150), fill="black")
        temp, result = self.analyze(image)
        self.addCleanup(temp.cleanup)
        evidence_ids = {item.evidence_id for item in result.evidence}
        self.assertIn("ev_bbox_coverage", evidence_ids)
        size_rules = [item for item in result.rule_evaluations
                      if item["rule_id"].startswith("PSY_AR_SIZE_")]
        for rule in size_rules:
            if rule["matched_evidence_ids"]:
                self.assertEqual(rule["matched_evidence_ids"], ["ev_bbox_coverage"])

    def test_extract_features_cache(self):
        from doar.extract import extract_features
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (80, 60), "white").save(image_path)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image_id", "path", "split", "class", "readable"]
                )
                writer.writeheader()
                writer.writerow({
                    "image_id": "case1", "path": str(image_path), "split": "train",
                    "class": "Happy", "readable": "True",
                })
            output = root / "features"
            summary = extract_features(manifest, output)
            self.assertEqual(summary["processed"], 1)
            self.assertGreater(summary["feature_count"], 40)
            self.assertTrue((output / "features.csv").exists())
            schema = json.loads((output / "feature_schema.json").read_text())
            self.assertIn("segmentation.bounding_box_coverage", schema)

    def test_case_outputs_judges_reports_and_grounded_qa(self):
        from doar.qa import answer
        image = Image.new("RGB", (120, 120), "white")
        ImageDraw.Draw(image).rectangle((35, 35, 85, 85), fill="blue")
        temp, result = self.analyze(image)
        self.addCleanup(temp.cleanup)
        output = Path(temp.name) / "out"
        judges = json.loads((output / "judges.json").read_text(encoding="utf-8"))
        self.assertEqual(judges["rule_judge"]["status"], "pass")
        self.assertEqual(judges["safety_judge"]["status"], "pass")
        self.assertTrue((output / "reports" / "professional_ar.html").exists())
        response = answer("What colours were detected?", result.to_dict(), judges)
        self.assertEqual(response["evidence_ids"], ["ev_dominant_colour"])
        unavailable = answer("Is there a person?", result.to_dict(), judges)
        self.assertIn("Unavailable", unavailable["answer"])
        saved = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["emotion"]["status"], "unavailable")
        self.assertFalse(Path(saved["artifacts"]["foreground_mask"]).is_absolute())
        report = (output / "reports" / "professional_en.html").read_text(encoding="utf-8")
        self.assertIn(
            "The required visual condition was not observed. No psychological interpretation was produced.",
            report,
        )
        self.assertIn("required module or evidence is missing", report)

    def test_deep_registry_is_complete_without_importing_torch(self):
        from doar.deep import MODEL_NAMES
        for name in (
            "small_cnn", "mobilenet_v3_small", "mobilenet_v3_large", "resnet18",
            "resnet50", "efficientnet_b0", "convnext_tiny", "vit_b_16",
        ):
            self.assertIn(name, MODEL_NAMES)

    def test_unified_checkpoint_loader_rejects_malformed_checkpoint(self):
        from doar.emotion import predict
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            checkpoint = Path(directory) / "bad.xyz"
            Image.new("RGB", (20, 20), "white").save(image)
            checkpoint.write_text("not a model")
            result = predict(image, checkpoint)
            self.assertEqual(result["status"], "failed")
            self.assertIn("Unsupported checkpoint type", result["reason"])

    def test_primary_fusion_alignment_excludes_clinical_inputs(self):
        from doar.fusion import PRIMARY_METHODS
        from doar.fusion.trainer import _load
        self.assertNotIn("rules", PRIMARY_METHODS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.csv"
            with features.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["image_id", "path", "split", "class", "quality.width"]
                )
                writer.writeheader()
                writer.writerow({
                    "image_id": "a", "path": "x", "split": "train",
                    "class": "Happy", "quality.width": "100",
                })
            embeddings = root / "embeddings.npz"
            np.savez_compressed(
                embeddings, embeddings=np.asarray([[1.0, 2.0]], np.float32),
                image_ids=np.asarray(["a"]), splits=np.asarray(["train"]),
                labels=np.asarray(["Happy"]),
            )
            names, joined = _load(features, embeddings)
            self.assertEqual(names, ["quality.width"])
            self.assertEqual(joined[0][0], "a")

    def test_temperature_scaling_and_ensemble_uncertainty(self):
        from doar.deep.calibration import fit_temperature
        from doar.uncertainty import summarize_ensemble
        logits = np.asarray([[4.0, 1.0, 0.0, -1.0], [3.0, 1.0, 0.0, -1.0]])
        labels = np.asarray([0, 1])
        calibration = fit_temperature(logits, labels)
        self.assertEqual(calibration["fit_split"], "valid")
        self.assertGreater(calibration["temperature"], 0)
        uncertainty = summarize_ensemble(
            [[.8, .1, .05, .05], [.2, .7, .05, .05]], calibrated=False
        )
        self.assertIn("uncalibrated_confidence", uncertainty["warnings"])
        self.assertGreater(uncertainty["disagreement"], 0)

    def test_dataset_and_training_readiness_commands(self):
        from doar.readiness import check_training_readiness, validate_dataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            for split in ("train", "valid", "test"):
                for label in ("Angry", "Fear", "Happy", "Sad"):
                    folder = root / split / label
                    folder.mkdir(parents=True)
                    Image.new("RGB", (16, 12), "white").save(folder / f"{split}_{label}.png")
            output = Path(directory) / "readiness"
            report = validate_dataset(root, output)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["totals"], {"train": 4, "valid": 4, "test": 4})
            self.assertEqual(report["decoded_test_images"], 0)
            hardware = check_training_readiness(root, output / "hardware")
            self.assertTrue(hardware["checks"]["valid_split_name"])
            self.assertTrue(hardware["checks"]["test_locked"])

    def test_fusion_bundle_requires_analysis_context(self):
        from doar.emotion import predict
        payload = {
            "checkpoint_type": "doar_fusion_bundle_v1",
            "classes": ("Angry", "Fear", "Happy", "Sad"),
        }
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            checkpoint = Path(directory) / "fusion.joblib"
            Image.new("RGB", (20, 20), "white").save(image)
            checkpoint.write_bytes(b"placeholder")
            fake_joblib = types.SimpleNamespace(load=lambda _: payload)
            with mock.patch.dict(sys.modules, {"joblib": fake_joblib}):
                result = predict(image, checkpoint)
            self.assertEqual(result["status"], "failed")
            self.assertIn("requires objective analysis context", result["reason"])

    def test_toml_config_validation_resolution_and_hash(self):
        from doar.config import load_config, resolve, save_resolved
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                '[training]\nmodel="resnet18"\nseed=123\n[output]\ndirectory="out"\n',
                encoding="utf-8",
            )
            parsed = load_config(config, {
                "training": {"model", "seed"}, "output": {"directory"},
            })
            values = resolve(
                {"model": None, "seed": 42, "output": None}, parsed,
                {"model": ("training", "model"), "seed": ("training", "seed"),
                 "output": ("output", "directory")},
                {"seed"},
            )
            self.assertEqual(values["model"], "resnet18")
            self.assertEqual(values["seed"], 42)
            digest = save_resolved(root / "run", "train-image-model", values)
            self.assertEqual(len(digest), 64)
            self.assertTrue((root / "run" / "resolved_config.json").exists())

    def test_fusion_families_use_distinct_input_types(self):
        from doar.fusion import PRIMARY_METHODS, PROBABILITY_METHODS
        from doar.fusion.probability import (
            equal_late_fusion, probability_meta_features,
            validate_oof_folds, validation_weighted_late_fusion,
        )
        self.assertTrue(set(PRIMARY_METHODS).isdisjoint(PROBABILITY_METHODS))
        objective = np.asarray([[.7, .1, .1, .1], [.1, .7, .1, .1]])
        deep = np.asarray([[.6, .2, .1, .1], [.2, .6, .1, .1]])
        equal = equal_late_fusion(objective, deep)
        weighted = validation_weighted_late_fusion(objective, deep, np.asarray([0, 1]))
        self.assertEqual(equal["weights"], [.5, .5])
        self.assertEqual(weighted["fit_split"], "valid")
        self.assertEqual(probability_meta_features(objective, deep).shape, (2, 8))
        validate_oof_folds(["a", "b", "c", "d"], [0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
