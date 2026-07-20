#!/usr/bin/env python3

import subprocess
import sys


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)

    print("\nChecking installed-package consistency...")
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=True,
    )

    print("\nTesting the complete SAM 3 import path...")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "import sam3; "
                "print('PyTorch:', torch.__version__); "
                "print('CUDA build:', torch.version.cuda); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('SAM 3 import: OK')"
            ),
        ],
        text=True,
    )

    if result.returncode != 0:
        raise SystemExit(
            "\nSAM 3 environment check FAILED. "
            "The traceback above identifies the remaining problem."
        )

    print("\nSAM 3 environment check PASSED.")


if __name__ == "__main__":
    main()
