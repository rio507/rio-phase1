"""The citation card, in a real browser: does it show, and can it be clicked?

    python -m tools.sources_card_selftest
    python -m tools.sources_card_selftest --url http://127.0.0.1:8888/

This one is not a design check. OpenAI's web search terms require that where
web results — or information drawn from them — are shown to a person, inline
citations are "clearly visible and clickable". RIO's answer is SPOKEN, so this
card is the only surface in the product that can carry them, and a card that
silently fails to render is the same as not having built it.

That cannot be proved from node or from reading the markup, because the things
that break are a hidden container never being unhidden, an anchor rendered
without an href, and a renderer that throws halfway through a list and leaves
half a citation on screen. So this drives the real page and asserts what a
person would be able to see and click:

  * the card is hidden before any answer, and appears when one arrives
  * every citation renders its source, its headline and its publication time
  * every citation has a REAL anchor with a working href, target and rel
  * the link is visibly a link at rest, not only on hover
  * a second answer does not replace the first -- scrollback survives, because
    an answer given four questions ago is still one worth checking
  * a citation with no URL is left out rather than rendered dead

WHAT THIS DOES NOT PROVE, and must not be read as proving: that the requirement
is satisfied for a driver who is listening with the phone face-down or in a
pocket. There is no visible UI in that case and this card changes nothing about
it. See LICENSING.md section 4.

Requires playwright (`pip install playwright && playwright install chromium`).
Exit code is the number of failures.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, BAD = "ok  ", "FAIL"
_fails = []


def ok(name, cond, extra=""):
    if cond:
        print(f"  {OK} {name}")
    else:
        _fails.append(name)
        print(f"  {BAD} {name}{('  ' + str(extra)) if extra else ''}")


# `all([])` is True, so a claim about every member of a collection passes when the
# collection is empty -- and a DOM query that matched nothing returns exactly
# that. See tools/assert_guard.py; tools/news_selftest.py is where this stopped
# being hypothetical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import assert_guard as _guard                                # noqa: E402

ok_all, ok_none, non_empty = _guard.bind(ok, order="name_first")


# Two answers, so the scrollback claim can be tested. The second deliberately
# carries a citation with no URL and one with no publication date.
ANSWER_1 = {
    "question": "Any news around here?",
    "citations": [
        {"source": "Santa Monica Daily Press",
         "headline": "Coastal Cleanup Day returns for 42nd year",
         "url": "https://smdp.example/coastal-cleanup",
         "published": "2026-09-16T09:00:00-07:00", "age_h": 3.4,
         "where": "Santa Monica", "geo_level": "city",
         "source_type": "news_org"},
        {"source": "City of Santa Monica",
         "headline": "City launches $100,000 Small Business Relief Microgrant",
         "url": "https://santamonica.example/microgrant",
         "published": "2026-09-16T10:10:00-07:00", "age_h": 2.2,
         "where": "Santa Monica", "geo_level": "city",
         "source_type": "official"},
    ],
}

ANSWER_2 = {
    "question": "What's in the news today?",
    "citations": [
        {"source": "Associated Press",
         "headline": "UN chief urges global cooperation on AI safety",
         "url": "https://apnews.example/un-ai",
         "published": "2026-09-16T12:00:00Z", "age_h": 1.3,
         "where": None, "geo_level": None, "source_type": "news_org"},
        {"source": "No Link Wire", "headline": "Should never be rendered",
         "url": "", "published": "2026-09-16T12:00:00Z", "age_h": 1.0,
         "where": None, "geo_level": None, "source_type": "unknown"},
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8888/")
    a = ap.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed — cannot prove the card renders.")
        return 1

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(a.url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1200)

        print("\n== before any answer")
        ok("the card is hidden until there is something to cite",
           page.eval_on_selector("#sourceshistory", "el => el.hidden") is True)
        ok("...and so is its label",
           page.eval_on_selector("#sourceslabel", "el => el.hidden") is True)

        print("\n== first answer arrives")
        page.evaluate("p => RIO.ui.sources(p)", ANSWER_1)
        page.wait_for_timeout(200)
        ok("the card becomes visible",
           page.eval_on_selector("#sourceshistory", "el => el.hidden") is False)
        ok("...and is actually on screen, not merely un-hidden",
           page.is_visible("#sourceshistory"))
        ok("the label appears too",
           page.eval_on_selector("#sourceslabel", "el => el.hidden") is False)

        rows = page.eval_on_selector_all(
            "#sourceshistory .src-row",
            """rows => rows.map(r => ({
                 head: (r.querySelector('.src-head')||{}).textContent || '',
                 meta: (r.querySelector('.src-meta')||{}).textContent || '',
                 href: (r.querySelector('a.src-link')||{}).href || '',
                 target: (r.querySelector('a.src-link')||{}).target || '',
                 rel: (r.querySelector('a.src-link')||{}).rel || '',
               }))""")
        ok("both citations rendered", len(rows) == 2, len(rows))
        for i, want in enumerate(ANSWER_1["citations"]):
            if i >= len(rows):
                break
            r = rows[i]
            ok(f"[{i}] headline shown", want["headline"][:24] in r["head"], r["head"])
            ok(f"[{i}] source named", want["source"] in r["meta"], r["meta"])
            ok(f"[{i}] publication time shown",
               "Sep" in r["meta"] and "ago" in r["meta"], r["meta"])
            ok(f"[{i}] geography shown where geographic",
               (want["where"] or "") in r["meta"], r["meta"])
            ok(f"[{i}] the URL is a real clickable anchor",
               r["href"] == want["url"], r["href"])
            ok(f"[{i}] opens in a new tab", r["target"] == "_blank", r["target"])
            ok(f"[{i}] and cannot reach back into the dashboard",
               "noopener" in r["rel"], r["rel"])

        # "Clearly visible" is the wording. A link styled only on hover is not.
        deco = page.eval_on_selector(
            "#sourceshistory a.src-link",
            "el => getComputedStyle(el).textDecorationLine")
        ok("the link is underlined at rest, not only on hover",
           "underline" in (deco or ""), deco)
        colour = page.eval_on_selector(
            "#sourceshistory a.src-link", "el => getComputedStyle(el).color")
        body = page.eval_on_selector(
            "#sourceshistory .src-head", "el => getComputedStyle(el).color")
        ok("...and is a different colour from the text around it",
           colour != body, f"{colour} vs {body}")
        box = page.eval_on_selector(
            "#sourceshistory a.src-link",
            "el => { const b = el.getBoundingClientRect();"
            "        return {w: b.width, h: b.height}; }")
        ok("the link is a real target, not a hairline",
           box["w"] > 40 and box["h"] >= 15, box)

        print("\n== a second answer must not erase the first")
        page.evaluate("p => RIO.ui.sources(p)", ANSWER_2)
        page.wait_for_timeout(200)
        blocks = page.eval_on_selector_all(
            "#sourceshistory .hist-item", "els => els.length")
        ok("both answers are in the scrollback", blocks == 2, blocks)
        qs = page.eval_on_selector_all(
            "#sourceshistory .src-q", "els => els.map(e => e.textContent)")
        ok("each block names the question it answers", len(qs) == 2, qs)
        ok("the newest answer is on top",
           qs and ANSWER_2["question"] in qs[0], qs)
        ok("...and the older question is still readable",
           len(qs) > 1 and ANSWER_1["question"] in qs[1], qs)

        print("\n== a citation with nothing to click")
        all_rows = page.eval_on_selector_all(
            "#sourceshistory .src-row",
            "rows => rows.map(r => !!r.querySelector('a.src-link'))")
        # THE CHECK THAT CARRIES SECTION 4, AND IT PASSED ON AN EMPTY CARD.
        # `all_rows` is one boolean per rendered row. If the selector matched
        # nothing -- the card never un-hidden, the renderer thrown halfway, the
        # block never appended -- the list is empty, all() is True, and the
        # assertion that every citation is clickable reports green over a card
        # with no citations on it at all. That is the exact failure LICENSING.md
        # section 4 promises cannot happen, asserted by a check that could not
        # see it. The row count is asserted immediately below, but on its own
        # RE-QUERY: this line was vacuous by itself, and this line is the one
        # whose name makes the licensing claim.
        #
        # It matters now rather than in principle: stage 2 moves citations to
        # bare [1][2] text markers with no url_citation objects behind them, so
        # the parser feeding this card is about to be rewritten. A check that
        # cannot tell "every row is clickable" from "there are no rows" is no
        # use for reviewing that.
        ok_all("a citation with no URL is left out, not rendered dead",
               all_rows, lambda has_link: has_link, all_rows)
        ok("...and the linkless one did not silently kill the block",
           page.eval_on_selector_all("#sourceshistory .src-row",
                                     "e => e.length") == 3)

        print("\n== the page itself")
        ok("no javascript errors were thrown", not errors, errors[:2])

        # An empty citation list must do nothing at all, not add a blank card.
        page.evaluate("RIO.ui.sources({question: 'x', citations: []})")
        page.wait_for_timeout(100)
        ok("an answer with no citations adds no block",
           page.eval_on_selector_all("#sourceshistory .hist-item",
                                     "els => els.length") == 2)

        browser.close()

    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)')
    for f in _fails:
        print(f"    - {f}")
    print("\n  NOTE: this proves the DASHBOARD surface only. It says nothing")
    print("  about a driver listening with the screen off — see LICENSING §4.")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
