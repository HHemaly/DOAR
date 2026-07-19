# Training guide

Run feature extraction once, then compare objective-feature models across three
seeds using the physical `train` and `valid` folders:

```powershell
python main.py extract-features `
  --manifest "outputs\dataset\manifest.csv" `
  --output "outputs\features\v3_1"

python main.py compare-models `
  --features "outputs\features\v3_1\features.csv" `
  --output "outputs\experiments\objective_features" `
  --seeds "42,123,2026"
```

The leaderboard ranks models by mean validation macro F1, then balanced
accuracy, calibration error, variance, and training time. It does not access
the test split.

The legacy histogram/statistics classifier is named **whole-image statistical
baseline**. It is not the objective-feature experiment family.

## Deep image models

```powershell
python main.py train-image-model `
  --dataset "C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing" `
  --model resnet18 `
  --output "outputs\experiments\resnet18_seed42" `
  --seed 42 --epochs 50 --batch-size 16 --image-size 224 `
  --device auto --augmentation conservative
```

Resume with `--resume "outputs\experiments\resnet18_seed42\last.pt"`.

```powershell
python main.py predict-image `
  --image "C:\path\drawing.png" `
  --checkpoint "outputs\experiments\resnet18_seed42\best.pt"
```

Supported registry names: `small_cnn`, `mobilenet_v3_small`,
`mobilenet_v3_large`, `resnet18`, `resnet50`, `efficientnet_b0`,
`convnext_tiny`, and `vit_b_16`.

## Embeddings and primary fusion

```powershell
python main.py extract-embeddings `
  --manifest "outputs\dataset\manifest.csv" `
  --backbone resnet18 `
  --output "outputs\embeddings\resnet18" `
  --device auto

python main.py train-fusion-model `
  --features "outputs\features\v3_1\features.csv" `
  --embeddings "outputs\embeddings\resnet18\embeddings.npz" `
  --output "outputs\experiments\primary_fusion_resnet18" `
  --methods "early_scaled_concat,pca_early_fusion,mlp_early_fusion" `
  --seeds "42,123,2026"
```

Primary fusion uses objective numerical features and deep embeddings only. It
deliberately excludes psychologist rules and concern profiles.
