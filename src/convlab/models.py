"""Model asset registry.

Weights are downloaded on first use and verified against a pinned SHA-256.
Pinning is not ceremony: an upstream model that silently changes would move
every number this pipeline produces, and a study that ran across a model
change would contain two incomparable halves with nothing in the output to
say so. The digest is recorded in each run's manifest.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(os.environ.get("CONVLAB_MODEL_DIR", "models"))

_USER_AGENT = "convlab/0.1 (+https://github.com/alexmueller07/conversationalist-lab-research)"


@dataclass(frozen=True)
class ModelAsset:
    name: str
    filename: str
    url: str
    sha256: str
    size: int
    purpose: str
    license: str


REGISTRY: dict[str, ModelAsset] = {
    "silero_vad": ModelAsset(
        name="silero_vad",
        filename="silero_vad.onnx",
        url="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        size=2_327_524,
        purpose="Voice activity detection at 32 ms resolution",
        license="MIT",
    ),
    "face_landmarker": ModelAsset(
        name="face_landmarker",
        filename="face_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        ),
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        size=3_758_596,
        purpose="478 face landmarks, 52 blendshapes, head pose matrix",
        license="Apache-2.0",
    ),
    "pose_landmarker": ModelAsset(
        name="pose_landmarker",
        filename="pose_landmarker_full.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        ),
        sha256="4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad",
        size=9_398_198,
        purpose="33 body landmarks for posture and gesture",
        license="Apache-2.0",
    ),
    "hand_landmarker": ModelAsset(
        name="hand_landmarker",
        filename="hand_landmarker.task",
        url=(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        ),
        sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        size=7_819_105,
        purpose="21 hand landmarks for fine gesture description",
        license="Apache-2.0",
    ),
    "yamnet": ModelAsset(
        name="yamnet",
        filename="yamnet.tflite",
        url=(
            "https://storage.googleapis.com/mediapipe-models/audio_classifier/"
            "yamnet/float32/1/yamnet.tflite"
        ),
        sha256="4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf",
        size=4_126_810,
        purpose="AudioSet tagging; supplies the laughter and speech classes",
        license="Apache-2.0",
    ),
}


class ModelError(RuntimeError):
    pass


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def model_path(name: str, model_dir: str | Path | None = None) -> Path:
    asset = REGISTRY[name]
    root = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    return root / asset.filename


def ensure(
    name: str,
    model_dir: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Return a local path to the asset, downloading it if necessary.

    Raises
    ------
    ModelError
        If the download fails or the digest does not match. A mismatched
        digest is never used: silently continuing would produce results that
        cannot be reproduced.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {sorted(REGISTRY)}")
    asset = REGISTRY[name]
    path = model_path(name, model_dir)

    if path.exists():
        if not verify:
            return path
        digest = _sha256(path)
        if digest == asset.sha256:
            return path
        log.warning(
            "%s has digest %s, expected %s; re-downloading",
            path.name, digest[:12], asset.sha256[:12],
        )
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    log.info("downloading %s (%.1f MB) from %s", asset.filename, asset.size / 1e6, asset.url)
    try:
        request = urllib.request.Request(asset.url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"could not download {asset.name} from {asset.url}: {exc}. "
            f"Download it manually to {path} and re-run."
        ) from exc

    digest = _sha256(tmp)
    if verify and digest != asset.sha256:
        tmp.unlink(missing_ok=True)
        raise ModelError(
            f"{asset.name} digest mismatch: got {digest}, expected {asset.sha256}. "
            "The upstream file changed; update the registry deliberately rather "
            "than trusting the new weights."
        )
    os.replace(tmp, path)
    return path


def ensure_all(model_dir: str | Path | None = None) -> dict[str, Path]:
    return {name: ensure(name, model_dir) for name in REGISTRY}


def status(model_dir: str | Path | None = None) -> list[dict[str, object]]:
    """Report which assets are present and valid, for the CLI and manifest."""
    rows: list[dict[str, object]] = []
    for name, asset in REGISTRY.items():
        path = model_path(name, model_dir)
        present = path.exists()
        valid = present and _sha256(path) == asset.sha256
        rows.append(
            {
                "name": name,
                "file": str(path),
                "present": present,
                "valid": valid,
                "size_mb": round(asset.size / 1e6, 1),
                "purpose": asset.purpose,
                "license": asset.license,
                "sha256": asset.sha256,
            }
        )
    return rows
