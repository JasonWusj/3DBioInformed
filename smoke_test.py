import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.losses import BiophysicsInformedLoss3D
from src.model import BiophysicsSegModel3D


def main():
    cfg = {
        "data": {
            "num_channels": 4,
            "num_classes": 4,
        },
        "model": {
            "features": [4, 8],
            "density_estimator": {
                "hidden_dim": 16,
                "num_layers": 2,
                "feature_size": [4, 4, 4],
            },
        },
        "loss": {
            "lambda_pde": 1.0,
            "lambda_bc": 1.0,
            "d_range": [0.02, 1.5],
            "rho_range": [0.002, 0.2],
            "sample_parameters": "voxel",
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiophysicsSegModel3D(cfg).to(device)
    criterion = BiophysicsInformedLoss3D(
        lambda_pde=cfg["loss"]["lambda_pde"],
        lambda_bc=cfg["loss"]["lambda_bc"],
        d_range=cfg["loss"]["d_range"],
        rho_range=cfg["loss"]["rho_range"],
        sample_parameters=cfg["loss"]["sample_parameters"],
    ).to(device)

    image = torch.randn(1, 4, 16, 16, 16, device=device)
    target = torch.zeros(1, 4, 16, 16, 16, device=device)
    target[:, 0] = 1.0
    target[:, 2, 5:11, 5:11, 5:11] = 1.0
    target[:, 1, 6:10, 6:10, 6:10] = 1.0
    target[:, 3, 7:9, 7:9, 7:9] = 1.0
    target[:, 0] = 1.0 - target[:, 2]

    logits, u_hat, t_tensor = model(image, return_density=True)
    loss, parts = criterion(logits, target, u_hat, t_tensor)
    loss.backward()

    assert logits.shape == (1, 4, 16, 16, 16), logits.shape
    assert u_hat.shape == (1, 1, 4, 4, 4), u_hat.shape
    assert torch.isfinite(loss), loss
    assert all(value >= 0 for value in parts.values()), parts
    print("smoke_test passed")
    print(parts)


if __name__ == "__main__":
    main()
