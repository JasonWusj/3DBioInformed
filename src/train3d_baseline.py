import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train3d import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        default_config = Path(__file__).resolve().parents[1] / "configs" / "baseline3d_unet.yaml"
        sys.argv.extend(["--config", str(default_config)])
    main()
