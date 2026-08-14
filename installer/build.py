"""Build the Windows installer.

Stages a clean copy of the app into ``payload/``, draws the brand icon,
and compiles ``installer.nsi`` with NSIS. Run from the repo root:

    python installer/build.py

Output: ``installer/dist/Conversation-Analyst-Setup-<version>.exe``

The payload is the app *source* plus the first-run machinery; the heavy
dependencies install on the user's machine on first launch. That keeps the
download a few megabytes and the installer instant, at the cost of a
visible one-time setup -- the same trade every web installer makes.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "installer"
PAYLOAD = HERE / "payload"
DIST = HERE / "dist"

MAKENSIS = Path(r"C:\Program Files (x86)\NSIS\makensis.exe")

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "conversation_analyst.egg-info",
                "convlab.egg-info"}


def version() -> str:
    text = (ROOT / "src" / "conversation_analyst" / "__init__.py").read_text(
        encoding="utf-8"
    )
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def stage() -> None:
    if PAYLOAD.exists():
        shutil.rmtree(PAYLOAD)
    app = PAYLOAD / "app"
    app.mkdir(parents=True)

    def copy_tree(src: Path, dst: Path) -> None:
        shutil.copytree(
            src, dst,
            ignore=lambda d, names: [n for n in names if n in EXCLUDE_DIRS],
        )

    copy_tree(ROOT / "src", app / "src")
    copy_tree(ROOT / "configs", app / "configs")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, app / name)

    shutil.copy2(HERE / "setup.bat", PAYLOAD / "setup.bat")
    shutil.copy2(HERE / "launch.vbs", PAYLOAD / "launch.vbs")


def draw_icon() -> None:
    """The two-bubble brand mark as a multi-size .ico, drawn with PIL."""
    from PIL import Image, ImageDraw

    def frame(size: int) -> Image.Image:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        s = size / 32.0

        def bubble(x0, y0, x1, y1, color, tail_x, tail_y, flip):
            d.rounded_rectangle(
                [x0 * s, y0 * s, x1 * s, y1 * s], radius=4 * s, fill=color
            )
            dx = -3 * s if flip else 3 * s
            d.polygon(
                [(tail_x * s, tail_y * s),
                 (tail_x * s + dx, tail_y * s),
                 (tail_x * s, tail_y * s + 4 * s)],
                fill=color,
            )

        bubble(2, 2, 20, 15, (251, 191, 36, 255), 6, 14, True)    # B, amber
        bubble(12, 12, 30, 25, (45, 212, 191, 255), 26, 24, False)  # A, teal
        return img

    frames = [frame(n) for n in (16, 24, 32, 48, 64, 128, 256)]
    frames[-1].save(
        HERE / "app.ico",
        sizes=[(f.width, f.height) for f in frames],
        append_images=frames[:-1],
    )


def compile_installer() -> Path:
    DIST.mkdir(exist_ok=True)
    out = DIST / f"Conversation-Analyst-Setup-{version()}.exe"
    subprocess.run(
        [str(MAKENSIS), f"/DVERSION={version()}", f"/DOUTFILE={out}",
         str(HERE / "installer.nsi")],
        check=True, cwd=HERE,
    )
    return out


def main() -> int:
    print("staging payload...")
    stage()
    print("drawing icon...")
    draw_icon()
    print("compiling installer...")
    out = compile_installer()
    print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
