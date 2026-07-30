"""Fetch the RF-DETR COCO checkpoints.

    python -m tools.fetch_detector_weights            # fetch what is missing
    python -m tools.fetch_detector_weights --check    # verify what is on disk
    python -m tools.fetch_detector_weights --variant small

Both nano and small are fetched by default: they differ by 0.4 ms per frame
here, so having the larger one already on disk makes switching
`headway.detect.VARIANT` a one-line change rather than a download.

Unlike the UFLDv2 weights these come from Google Cloud Storage over plain
HTTPS, so there is no Drive quota page to guard against -- but the size is
still checked, because a truncated 366 MB download otherwise surfaces as an
unpickling error inside the first live frame.

LICENCE: RF-DETR is Apache-2.0 (Roboflow). This is load-bearing, not
bookkeeping -- the obvious alternative detector family, Ultralytics YOLO, is
AGPL-3.0 and cannot ship in a commercial product. headway/detect.py re-checks
the licence field at load time.
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Upstream rfdetr/assets/model_weights.py.
MODELS = {
    "nano": (
        "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "rf-detr-nano.pth", 366287238),
    "small": (
        "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "rf-detr-small.pth", 386045550),
}
WEIGHTS_DIR = Path("/workspace/rio-phase1/weights")


def verify(path, expect_bytes):
    if not os.path.exists(path):
        return False, "missing"
    size = os.path.getsize(path)
    if size != expect_bytes:
        return False, f"wrong size: {size} bytes, expected {expect_bytes}"
    return True, "ok"


def fetch(url, dest, expect_bytes):
    print(f"fetching {url}\n     -> {dest} ({expect_bytes / 1e6:.0f} MB)")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            pct = 100.0 * done / max(expect_bytes, 1)
            print(f"\r     {done / 1e6:7.0f} MB  {pct:5.1f}%", end="", flush=True)
    print()
    tmp.replace(dest)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=sorted(MODELS) + ["all"], default="all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    wanted = sorted(MODELS) if args.variant == "all" else [args.variant]
    rc = 0
    for name in wanted:
        url, fname, size = MODELS[name]
        dest = WEIGHTS_DIR / fname
        ok, msg = verify(dest, size)
        if args.check:
            print(f"{name:6} {dest}: {msg}")
            rc |= 0 if ok else 1
            continue
        if ok and not args.force:
            print(f"{name:6} already present and the right size")
            continue
        if dest.exists():
            print(f"{name:6} {msg} — removing")
            dest.unlink()
        try:
            fetch(url, dest, size)
        except Exception as e:
            print(f"{name:6} FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            rc |= 1
            continue
        ok, msg = verify(dest, size)
        print(f"{name:6} {msg}")
        if not ok:
            dest.unlink(missing_ok=True)
            rc |= 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
