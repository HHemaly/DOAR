from __future__ import annotations


def build_transforms(image_size: int, training: bool, profile: str = "conservative"):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    if not training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize,
        ])
    profiles = {
        "conservative": [
            transforms.RandomAffine(7, translate=(.03, .03), scale=(.95, 1.05)),
            transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.05),
        ],
        "moderate": [
            transforms.RandomAffine(10, translate=(.05, .05), scale=(.9, 1.1)),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.1),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=.1),
        ],
        "strong": [
            transforms.RandomAffine(12, translate=(.08, .08), scale=(.85, 1.15)),
            transforms.ColorJitter(brightness=.25, contrast=.25, saturation=.15),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=.15),
            transforms.RandomErasing(p=.1, scale=(.01, .05)),
        ],
    }
    if profile not in profiles:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        *profiles[profile],
        transforms.ToTensor(),
        normalize,
    ])
