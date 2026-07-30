"""Fetch the UFLDv2 culane_res18 checkpoint.

    python -m tools.fetch_lane_weights            # fetch if missing
    python -m tools.fetch_lane_weights --check    # verify what is on disk
    python -m tools.fetch_lane_weights --force    # re-fetch

The checkpoint is 825 MB, so it is gitignored and fetched instead. Upstream
publishes it on Google Drive (Ultra-Fast-Lane-Detection-v2 README, model zoo),
which is a hostile place to download from unattended: a quota block or an
interstitial returns 200 with an HTML page in the body, and a half-finished
transfer leaves a file that is the wrong size but looks plausible. So the
digest is pinned and checked, and a bad file is deleted rather than left where
headway/lanes.py would try to load it.

Without the checkpoint nothing breaks: headway falls back to the static
trapezoid corridor and logs `corridor_source: static` on every frame.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Upstream model zoo: CULane / ResNet18, F1 75.0.
GDRIVE_ID = "1oEjJraFr-3lxhX_OXduAGFWalWa6Xh3W"
SHA256 = "956616371ee758551c455d28c1c8a9732b39d7fbd85bc70b9869424a5121f967"
SIZE_BYTES = 825264756


def digest(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path):
    """-> (ok, message)."""
    if not os.path.exists(path):
        return False, "missing"
    size = os.path.getsize(path)
    if size != SIZE_BYTES:
        return False, f"wrong size: {size} bytes, expected {SIZE_BYTES}"
    got = digest(path)
    if got != SHA256:
        return False, f"wrong sha256: {got}"
    return True, "ok"


def main():
    from headway import lanes

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=lanes.DEFAULT_WEIGHTS)
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--force", action="store_true", help="re-fetch even if valid")
    args = ap.parse_args()

    out = Path(args.out)
    ok, msg = verify(out)

    if args.check:
        print(f"{out}: {msg}")
        return 0 if ok else 1

    if ok and not args.force:
        print(f"{out}: already present and verified")
        return 0
    if not ok and out.exists():
        print(f"{out}: {msg} — removing")
        out.unlink()

    try:
        import gdown
    except ImportError:
        print("gdown is required to fetch from Google Drive: pip install gdown",
              file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetching culane_res18.pth ({SIZE_BYTES / 1e6:.0f} MB) -> {out}")
    gdown.download(id=GDRIVE_ID, output=str(out), quiet=False)

    ok, msg = verify(out)
    if not ok:
        # Almost always a Drive quota page saved under the .pth name. Leaving it
        # there would turn a clean "weights missing, using the trapezoid" into a
        # confusing unpickling error inside the first live frame.
        print(f"FAILED verification: {msg}", file=sys.stderr)
        if out.exists():
            out.unlink()
        print("Fetch it by hand from the model zoo in the UFLDv2 README, or set "
              "RIO_LANE_WEIGHTS to a copy you already have.", file=sys.stderr)
        return 1

    print(f"{out}: ok ({SHA256[:16]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
