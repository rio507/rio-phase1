"""The Teachers card, in a real browser, against a fixture.

    python -m tools.teacher_card_selftest
    python -m tools.teacher_card_selftest --shot /tmp/teachers.png

WHY A BROWSER AND NOT A UNIT TEST
---------------------------------
Because what is being asserted is what a driver sees, and that is a property of
the stylesheet and the layout engine rather than of the JavaScript. "The card
renders" is not "renderColumn returned a DOM node" -- it is: both columns are
there, in the right order, with the right labels; the reasoning trace does not
push the tally strip off the screen; the actor chip says which track it means;
the two models are told apart by something other than the order they happen to
be in; and on a 390px phone none of it overlaps the picture.

THE FIXTURE IS THE POINT
------------------------
The card is driven from a hand-written /teachers/state payload -- one healthy
reading, one failed service, one unmatched actor -- rather than from a live
panel. Three reasons, and the third is the real one:

  * it runs without 37 GB of weights,
  * it runs identically every time,
  * and it can contain the states a live panel almost never produces. A
    service that is down, a model that named a traffic light, a reading that
    has gone stale: those are the states the card most needs to be honest in
    and the ones a demo will never show you.

Requires playwright, like the other two browser suites
(`pip install playwright && python -m playwright install chromium`).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []

PORTRAIT = (390, 844)
DESKTOP = (1440, 900)


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


def intersects(a, b):
    if not a or not b:
        return False
    if a["width"] <= 0 or a["height"] <= 0 or b["width"] <= 0 or b["height"] <= 0:
        return False
    return not (a["x"] + a["width"] <= b["x"] or b["x"] + b["width"] <= a["x"] or
                a["y"] + a["height"] <= b["y"] or b["y"] + b["height"] <= a["y"])


# ---------------------------------------------------------------------------
# The fixture. Every branch the card has, in one payload.
# ---------------------------------------------------------------------------
LONG_TRACE = (
    "The lead vehicle, a white panel van in the ego lane, has illuminated its "
    "brake lights while the gap is closing at roughly two metres per second. "
    "There is a cyclist on the near-side shoulder approximately fifteen metres "
    "ahead who is not currently in the corridor but whose trajectory would "
    "bring them into it if the van's deceleration forces a lane change. The "
    "correct action is to release the throttle now and prepare to brake, "
    "rather than to change lane, because the near-side is occupied and the "
    "off-side has a vehicle in the mirror's blind spot. "
) * 3

FIXTURE = {
    "enabled": True,
    "session": "fixture",
    "tally": {"keyframes": 47, "readings": 88, "agree": 21,
              "agree_eligible": 39, "matched": 61, "actors": 74,
              "dropped_build": 3, "agreement_pct": 54, "matched_pct": 82},
    "last_build_refusal": "stale_frame",
    "spoken": {"text": "Ease off — that van is slowing.", "at": 0,
               "kind": "warning"},
    "models": {
        # A healthy reading, with a long trace and a matched actor.
        "alpamayo1.5": {
            "reading": {
                "model": "alpamayo1.5", "model_id": "nvidia/Alpamayo-1.5-10B",
                "revision": "7aba829", "precision": "bf16", "ok": True,
                "error": None, "latency_ms": 3120.0, "queue_ms": 210.0,
                "freshness_s": 1.8, "fresh": True, "age_s": 1.4,
                "raw": {}, "scene": "A two-lane urban road, overcast, with a "
                                    "white panel van in the ego lane and a "
                                    "cyclist on the near-side shoulder.",
                "critical_actor": "The white panel van directly ahead, because "
                                  "it has just braked and the gap is closing.",
                "attention": "The van's brake lights, and the cyclist who may "
                             "be squeezed if it stops.",
                "reasoning": LONG_TRACE,
                "thinking": None, "meta_action": "decelerate",
                "trajectory": {"xyz": [[i * 1.3, 0, 0] for i in range(64)],
                               "hz": 10, "horizon_s": 6.4,
                               "frame": "FLU_at_t0",
                               "pixels": [[320 + i, 700 - i * 5]
                                          for i in range(60)]},
                "gpu": {"vram_reserved_mb": 24100.0},
            },
            "association": {
                "matched": True, "track_id": 7, "label": "truck",
                "box": [560, 260, 840, 430], "range_m": 21.4, "score": 0.94,
                "reason": "class+in-corridor+range+size+lead",
                "phrase": "The white panel van directly ahead",
                "wanted_class": "truck", "wanted_side": "ahead",
            },
        },
        # A service that is down, AND an unmatched actor. Both of the states a
        # live demo will never show, and both of the ones the card has to be
        # honest in.
        "cosmos-reason2": {
            "reading": {
                "model": "cosmos-reason2",
                "model_id": "nvidia/Cosmos-Reason2-8B",
                "revision": "a9fae2c", "precision": "fp8", "ok": False,
                "error": "URLError: [Errno 111] Connection refused",
                "latency_ms": 0.0, "queue_ms": 0.0, "freshness_s": 9.4,
                "fresh": False, "age_s": 9.4, "raw": {},
                "scene": "", "critical_actor": "", "attention": "",
                "reasoning": "", "thinking": None, "meta_action": None,
                "trajectory": None, "gpu": {},
            },
            "association": {
                "matched": False, "track_id": None, "label": None,
                "box": None, "range_m": None, "score": 0.0,
                "reason": "names_undetectable_class:traffic light",
                "phrase": "The traffic light ahead",
                "wanted_class": None, "wanted_side": "ahead",
            },
        },
    },
    "services": {
        "alpamayo1.5": {"name": "alpamayo1.5", "url": "http://127.0.0.1:8801",
                        "loaded": True, "model": "nvidia/Alpamayo-1.5-10B",
                        "precision": "bf16", "evicted": 4, "stale_dropped": 2,
                        "failed": 0, "last_error": None},
        "cosmos-reason2": {"name": "cosmos-reason2",
                           "url": "http://127.0.0.1:8802", "loaded": False,
                           "model": None, "precision": None, "evicted": 0,
                           "stale_dropped": 0, "failed": 12,
                           "last_error": "Connection refused"},
    },
}

RENDER = """
(state) => {
  if (!(window.RIO && RIO.teachers)) throw new Error('RIO.teachers is missing');
  // The page's own renderer, not a copy of it.
  const panel = RIO.teachers.create({ sessionId: () => 'fixture' });
  panel._render(state);
  return true;
}
"""

READ = """
() => {
  const box = (el) => {
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const cs = getComputedStyle(el);
    return { x: r.x, y: r.y, width: r.width, height: r.height,
             color: cs.color, display: cs.display, text: el.textContent };
  };
  const cols = Array.from(document.querySelectorAll('#teachcols .teach-col'));
  return {
    card: box(document.getElementById('teachcard')),
    feed: box(document.querySelector('.cam-wrap')),
    tally: box(document.getElementById('teachtally')),
    svc: box(document.getElementById('teachsvc')),
    layers: Array.from(document.querySelectorAll('#teachlayers .teach-layer'))
      .map(b => ({ layer: b.dataset.layer, pressed: b.getAttribute('aria-pressed'),
                   box: box(b) })),
    columns: cols.map(c => ({
      model: c.dataset.model,
      fresh: c.classList.contains('is-fresh'),
      name: (c.querySelector('.teach-name') || {}).textContent,
      nameColor: c.querySelector('.teach-name')
        ? getComputedStyle(c.querySelector('.teach-name')).color : null,
      borderLeft: getComputedStyle(c).borderLeftColor,
      labels: Array.from(c.querySelectorAll('.teach-label')).map(l => l.textContent),
      meta: (c.querySelector('.teach-meta') || {}).textContent,
      actor: (c.querySelector('.teach-actor') || {}).textContent,
      actorMatched: !!c.querySelector('.teach-actor.matched'),
      more: !!c.querySelector('.teach-more'),
      box: box(c),
      textBoxes: Array.from(c.querySelectorAll('.teach-text')).map(box),
    })),
    scrollHeight: document.documentElement.scrollHeight,
  };
}
"""

EXPAND = """
() => {
  const b = document.querySelector('#teachcols .teach-col .teach-more');
  if (!b) return null;
  const field = b.closest('.teach-field');
  const before = field.querySelector('.teach-text').getBoundingClientRect().height;
  b.click();
  const after = field.querySelector('.teach-text').getBoundingClientRect().height;
  return { before: before, after: after, label: b.textContent };
}
"""

LAYER_TOGGLE = """
() => {
  const seen = [];
  if (!(window.RIO && RIO.overlay && RIO.overlay.teacherLayers)) return null;
  const btn = document.querySelector('#teachlayers [data-layer="actor-a"]');
  const before = RIO.overlay.teacherLayers({});
  btn.click();
  const after = RIO.overlay.teacherLayers({});
  return { before: before, after: after,
           pressed: btn.getAttribute('aria-pressed') };
}
"""


def run(page, size, label, shot=None):
    section(f"the card at {label} {size[0]}x{size[1]}")
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.evaluate(RENDER, FIXTURE)
    page.wait_for_timeout(120)
    r = page.evaluate(READ)

    ok(r["card"] and r["card"]["width"] > 100,
       f"the card is on the page ({round((r['card'] or {}).get('width', 0))}px wide)")
    cols = r["columns"]
    ok(len(cols) == 2, f"two columns ({len(cols)})")
    if len(cols) != 2:
        return r
    ok([c["model"] for c in cols] == ["alpamayo1.5", "cosmos-reason2"],
       f"Alpamayo left, Cosmos right, always ({[c['model'] for c in cols]})")
    ok("Alpamayo 1.5" in (cols[0]["name"] or ""),
       f"named for a person to read ({cols[0]['name']!r})")
    ok("Cosmos-Reason2" in (cols[1]["name"] or ""),
       f"...and so is the other ({cols[1]['name']!r})")

    section("the labels — every field the card promises")
    a_labels = [s.strip() for s in cols[0]["labels"]]
    c_labels = [s.strip() for s in cols[1]["labels"]]
    ok("Scene" in a_labels and "Scene" in c_labels,
       f"Scene, on both ({a_labels})")
    ok("Chain-of-Causation" in a_labels,
       "Alpamayo's trace is called a Chain-of-Causation, because that is what "
       "it is")
    ok("Physical reasoning" in c_labels,
       "and Cosmos's is called physical reasoning — each model is labelled "
       "with the kind of trace it actually produces")
    ok(any(l.startswith("Attention") for l in a_labels)
       and any(l.startswith("Attention") for l in c_labels),
       "Attention · next 2 s, on both")
    ok("Referenced actor" in a_labels and "Referenced actor" in c_labels,
       "Referenced actor, on both")

    section("freshness, latency, precision — the three numbers")
    meta = cols[0]["meta"] or ""
    ok("fresh" in meta and "1.8s" in meta, f"freshness ({meta!r})")
    ok("lat" in meta and "3.12s" in meta, "latency")
    ok("prec" in meta and "bf16" in meta, "precision")
    ok(cols[0]["fresh"] and not cols[1]["fresh"],
       "a fresh reading is marked fresh and a 9.4 s old one is not")
    ok("fp8" in (cols[1]["meta"] or ""),
       "the two columns can be at different precisions and each says which")

    section("the two models are told apart by more than their position")
    ok(cols[0]["nameColor"] != cols[1]["nameColor"],
       f"different name colours ({cols[0]['nameColor']} vs {cols[1]['nameColor']})")
    ok(cols[0]["borderLeft"] != cols[1]["borderLeft"],
       "and different column edges")
    for c in cols:
        col = c["nameColor"] or ""
        # None of the safety colours: green #3ddc84, amber #ffb04a, red #ff6d5a.
        ok(col not in ("rgb(61, 220, 132)", "rgb(255, 176, 74)",
                       "rgb(255, 109, 90)"),
           f"{c['model']} is not drawn in a safety colour ({col}) — a shadow "
           f"reading must never read as the car's own judgement")

    section("the referenced actor says which track, or why not")
    ok(cols[0]["actorMatched"] and "#7" in (cols[0]["actor"] or ""),
       f"a match names the track the overlay is ringing ({cols[0]['actor']!r})")
    ok(not cols[1]["actorMatched"]
       and "traffic light" in (cols[1]["actor"] or ""),
       f"and a miss says why ({cols[1]['actor']!r})")

    section("a failed service is visible, not blank")
    ok("Connection refused" in (cols[1]["box"]["text"] or ""),
       "the column carries the error")
    ok("cosmos" in (r["svc"]["text"] or "").lower()
       and "no answer" in (r["svc"]["text"] or "").lower(),
       f"and the service strip says so too ({(r['svc']['text'] or '')[:80]!r})")
    ok("alpamayo1.5: nvidia/Alpamayo-1.5-10B @ bf16" in (r["svc"]["text"] or ""),
       "while the live one names its weights and precision")

    section("the tally strip")
    t = (r["tally"]["text"] or "")
    ok("Keyframes" in t and "47" in t, f"keyframes processed ({t[:70]!r})")
    ok("Agreement" in t and "54%" in t, "agreement on the critical actor")
    ok("Matched tracks" in t and "82%" in t, "matched-track rate")
    ok("Dropped" in t and "3" in t,
       "and the keyframes that were refused before they were sent")

    if shot:
        # BEFORE the expander is clicked, so the picture is the card as a
        # driver first meets it rather than the card with a trace unfolded.
        page.screenshot(path=shot, full_page=False)
        print(f"       wrote {shot}")

    section("a long trace does not eat the card")
    ok(cols[0]["more"], "a 1200-character trace gets a Show all control")
    tallest = max((b["height"] for b in cols[0]["textBoxes"]), default=0)
    ok(tallest < 140,
       f"and is clamped until it is asked for ({round(tallest)}px tall)")
    exp = page.evaluate(EXPAND)
    ok(exp and exp["after"] > exp["before"],
       f"Show all expands it ({round(exp['before'])} -> {round(exp['after'])}px)")
    ok(exp and "less" in (exp["label"] or "").lower(),
       f"and the control changes to say so ({exp['label']!r})")

    return r


def run_layers(page):
    section("the layer switches are wired to the overlay")
    r = page.evaluate(READ)
    got = {l["layer"] for l in r["layers"]}
    ok(got == {"actor-a", "actor-c", "path"},
       f"three switches: the two actor rings and the predicted path ({sorted(got)})")
    pressed = {l["layer"]: l["pressed"] for l in r["layers"]}
    ok(pressed.get("path") == "false",
       "the predicted path is OFF by default — it is a 6.4 s guess through a "
       "flat-road camera model and it should be asked for")
    ok(pressed.get("actor-a") == "true" and pressed.get("actor-c") == "true",
       "the actor rings are on")

    t = page.evaluate(LAYER_TOGGLE)
    ok(t is not None, "the overlay exposes the layer state")
    if t:
        ok(t["before"]["actor-a"] is True and t["after"]["actor-a"] is False,
           f"clicking one turns it off in the overlay ({t['before']} -> "
           f"{t['after']})")
        ok(t["pressed"] == "false", "and aria-pressed follows")


def run_mobile(page):
    section("mobile — the card is below the feed and over nothing")
    page.set_viewport_size({"width": PORTRAIT[0], "height": PORTRAIT[1]})
    page.evaluate("() => document.body.classList.add('driving')")
    page.evaluate(RENDER, FIXTURE)
    page.wait_for_timeout(150)
    r = page.evaluate(READ)
    card, feed = r["card"], r["feed"]
    ok(card and feed, "both the card and the feed are laid out")
    if not (card and feed):
        return
    ok(card["y"] >= feed["y"] + feed["height"] - 1,
       f"the card starts below the picture (card y={round(card['y'])}, feed "
       f"ends {round(feed['y'] + feed['height'])})")
    ok(not intersects(card, feed), "and never overlaps it")
    ok(card["width"] <= PORTRAIT[0],
       f"it fits the width ({round(card['width'])} <= {PORTRAIT[0]})")
    cols = r["columns"]
    if len(cols) == 2:
        ok(abs(cols[0]["box"]["x"] - cols[1]["box"]["x"]) < 2,
           "the two columns stack rather than squeezing side by side")
    for l in r["layers"]:
        ok(l["box"]["height"] >= 40,
           f"the {l['layer']} switch is at least 40px tall on a phone "
           f"({round(l['box']['height'])}px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/")
    ap.add_argument("--shot", default=None)
    args = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed:\n"
              "  pip install playwright && python -m playwright install chromium")
        return 2

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": DESKTOP[0],
                                          "height": DESKTOP[1]})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"could not load {args.url}: {e}")
            browser.close()
            return 2
        page.wait_for_timeout(700)

        run(page, DESKTOP, "desktop", shot=args.shot)
        run_layers(page)
        run_mobile(page)
        section("no page errors")
        ok(not errors, f"the page threw nothing while rendering ({errors[:2]})")
        browser.close()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
