from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

from heavy_lab.hardware import detect_hardware
from heavy_lab.install import resolve_torch_install


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the PyTorch build selected by local hardware detection.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Storage root used for disk/hardware probe")
    parser.add_argument("--dry-run", action="store_true", help="Print the exact pip command without executing it")
    args = parser.parse_args()

    profile = detect_hardware(args.root)
    command = resolve_torch_install(
        device=profile.device,
        cuda_version=profile.cuda_version,
        os_name=profile.os_name,
    )
    print(shlex.join(command))
    if args.dry_run:
        return 0
    subprocess.check_call(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
