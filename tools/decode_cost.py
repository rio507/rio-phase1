"""What a slower card would cost, measured as milliseconds per output token.

    python -m tools.decode_cost
    python -m tools.decode_cost --tokens 8,16,32,64 --repeats 3

WHY THIS AND NOT A GUESS
------------------------
tools/vram_budget.py answers whether the live pipeline FITS on a smaller card,
by taking the memory away and running the drive in what is left. It cannot
answer whether the pipeline would still be FAST on one, because bandwidth
cannot be taken away the way memory can.

But the shape of the answer is measurable here. A generate is two different
costs with two different bottlenecks:

  prefill   the image and the prompt, all at once. Compute-bound: it scales
            with the card's FLOPs and with how many vision tokens the picture
            turns into.
  decode    one token at a time, each one reading all 16 GB of weights.
            BANDWIDTH-bound, and almost perfectly linear in output length --
            so the slope of (latency vs tokens) is the decode cost per token
            ON THIS CARD, and it is the number that scales when the card
            changes.

Measure the line here, and a card with a third of the bandwidth moves the
slope by about three, while the intercept moves with compute instead. That is
still a projection -- it assumes the smaller card is not also thermally capped
or running a worse kernel -- but it is a projection from two measured numbers
rather than from a feeling about GPUs.

RUN IT WITH THE SERVER STOPPED, or on a card with room for a second copy of
the weights: this loads its own Qwen3-VL-8B rather than borrowing the live
one, so that nothing it measures is contending with a drive.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO = Path(__file__).resolve().parent.parent
CLIP = REPO / "runs" / "road_clip.mp4"


def a_frame(max_side=512):
    """One real road frame at the size the observer actually sends.

    512 px because that is what vision._downscale settles on for the observer
    -- measured there at 360 ms against 432 ms for the full frame, for the same
    sentence. Measuring at 720p would be measuring a call this code does not
    make.
    """
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(CLIP))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frames from {CLIP}")
    h, w = f.shape[:2]
    s = max_side / max(h, w)
    f = cv2.resize(f, (int(w * s), int(h * s)))
    return Image.fromarray(f[:, :, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,16,32,64")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=512)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import vision

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"card: {name}, {total:.0f} GiB")

    print(f"loading {vision.MODEL_ID} (bf16, cuda:0)...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(vision.MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        vision.MODEL_ID, dtype=torch.bfloat16, device_map="cuda:0")
    print(f"   {time.time() - t0:.0f} s, "
          f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GiB of weights")

    img = a_frame(args.max_side)
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe the road ahead in one sentence."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda:0")
    n_in = int(inputs["input_ids"].shape[1])
    print(f"   prompt is {n_in} tokens including the picture at "
          f"{args.max_side} px")

    rows = []
    for n in [int(x) for x in args.tokens.split(",")]:
        ts = []
        for i in range(args.repeats + 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=n, min_new_tokens=n,
                               do_sample=False)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000.0
            if i:                      # the first is warm-up, not data
                ts.append(ms)
        rows.append((n, statistics.median(ts)))
        print(f"   {n:>3} tokens: {statistics.median(ts):8.1f} ms  "
              f"(n={len(ts)}, spread {max(ts) - min(ts):.0f} ms)")

    # Least squares on two or more points: intercept = prefill, slope = decode.
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
             / sum((x - mx) ** 2 for x in xs))
    intercept = my - slope * mx

    print("\n" + "=" * 72)
    print(f"prefill (the picture and the prompt): {intercept:7.1f} ms   "
          f"compute-bound")
    print(f"decode, per output token:             {slope:7.2f} ms   "
          f"bandwidth-bound")
    print("-" * 72)
    # THE REAL CAPS, read off the code rather than imagined. A generate stops
    # at an end token long before these in the ordinary case -- vision.py's own
    # measurement of the observer is 360-432 ms, which at the slope above is
    # ~18-21 tokens -- so each row is that path's WORST case.
    for label, ntok in (
            ("observer, typical sentence (~20 tokens)", 20),
            ("observer, its cap (vision.py:160, 60 tokens)", 60),
            ("visual_qa answer (config.VISUAL_QWEN_MAX_TOKENS, 96)", 96),
            ("/perceive JSON (perceive.MAX_NEW_TOKENS, 130)", 130)):
        here = intercept + slope * ntok
        print(f"{label:<52} {here:7.0f} ms")
    # HOW MUCH OF A TOKEN IS ACTUALLY BANDWIDTH. This is the part that a
    # bandwidth ratio gets wrong if it is applied to the whole slope: at batch
    # 1, HuggingFace's generate spends most of each token in kernel launches
    # and python, not in reading weights.
    try:
        gib = torch.cuda.memory_allocated() / 1024 ** 3
        bw = {"H200": 4800, "H100": 3350, "A100": 2039, "L40S": 864,
              "L40": 864, "A10G": 600, "A10": 600, "L4": 300,
              "RTX 4090": 1008, "RTX 5090": 1792, "RTX 3090": 936}
        here_bw = next((v for k, v in bw.items() if k in name), None)
        if here_bw:
            weights_gb = gib * 1.074
            floor = 1000.0 * weights_gb / here_bw
            print(f"\nof those {slope:.2f} ms, reading {weights_gb:.1f} GB of "
                  f"weights at this card's\n{here_bw} GB/s accounts for "
                  f"{floor:.1f} ms. The remaining {slope - floor:.1f} ms is "
                  f"kernel launches,\npython and attention at batch 1 -- and "
                  f"it does NOT scale with bandwidth.")
            print("\nso a candidate card's decode cost is roughly "
                  f"{slope - floor:.1f} ms + {weights_gb:.1f} GB / its "
                  "bandwidth:")
            for card_name, card_bw in sorted(bw.items(), key=lambda kv: -kv[1]):
                per = (slope - floor) + 1000.0 * weights_gb / card_bw
                print(f"   {card_name:<9} {card_bw:>5} GB/s   "
                      f"{per:5.1f} ms/token   "
                      f"observer ~20 tok: {intercept + per * 20:6.0f} ms")
    except Exception:
        pass
    print("\nThat table is a PROJECTION from two measured numbers, not a "
          "measurement:\nit assumes the smaller card is not also slower in "
          "the part that is not\nbandwidth. Run this tool on the candidate "
          "and the projection becomes a\nmeasurement -- which is why it is "
          "committed rather than pasted into a note.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
