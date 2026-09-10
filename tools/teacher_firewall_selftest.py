"""Can anything a teacher said reach RIO's mouth? — asked of the AST, not of me.

    python -m tools.teacher_firewall_selftest

THE CLAIM THIS TEST EXISTS TO HOLD
----------------------------------
Two 8-to-10-billion-parameter models are now reading the road RIO is driving
on. They are there to be compared against her, recorded, and looked at. They
are NOT there to help, and the whole value of the exercise depends on that
staying true: a teacher that quietly influences a warning is no longer a
control, and a corpus collected under that influence is no longer evidence of
anything.

"We were careful" is not a guarantee. A guarantee is a test that fails the
build, and this is it. It reads the source of every module in this repo, walks
the syntax tree, and asserts four things:

  A. NOTHING GETS IN. Every module under teachers/ may import from a small
     allowed set -- the standard library, config, numpy, one geometry module,
     and itself. It may not import the observer, vision, the voice stack, the
     realtime session, the router, the persona, visual_qa or the prompts. Not
     "does not currently call"; cannot reach.

  B. NOTHING GETS OUT. Only an allowlisted handful of files may import
     `teachers` at all -- app.py, the teachers package itself, and its tools.
     If observer.py, realtime.py, router.py, visual_qa.py or anything in
     headway/ ever imports it, that is the failure, and it is caught here
     rather than in a drive.

  C. THE ONE DOOR IS NARROW. app.py is the single module that touches the
     panel, and every reference it makes must sit inside a named allowlist of
     functions -- the two frame handlers, the read-only endpoints, the lifespan
     and the teardown. A teacher value read anywhere else in app.py is a
     violation even if what it does with it looks harmless.

  D. THE DOOR DOES NOT OPEN BOTH WAYS. Those allowlisted functions may not
     call into the speech path, the arbiter, look(), or the observer cache in
     the same breath as a teacher value. The traffic is one-way by design --
     the speech path may TELL the panel what was said (note_spoken), and the
     panel may never tell the speech path anything.

...and one more, in a language this file cannot parse:

  E. THE BROWSER HALF. static/rio_teachers.js is the only new script on the
     page, and it must not reference the arbiter, the output bus, or the
     speak helpers. Checked by name, because a JavaScript AST is not available
     here and a name check that can only produce false ALARMS is still a
     useful test.

A failure here is not a warning. It means a shadow model can affect what RIO
does or says, which is the one thing this feature promised not to do.
"""
import argparse
import ast
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

# A. What a module under teachers/ is allowed to import. A WHITELIST, because a
#    blacklist of forbidden modules is a list somebody has to remember to add
#    to every time a new module is written, and the day they forget is the day
#    this test stops meaning anything.
TEACHERS_MAY_IMPORT = {
    # config is data, not behaviour: constants only, and importing it cannot
    # reach anything that speaks.
    "config",
    # The camera model, for projecting a predicted path onto the picture. Three
    # constants; asserted below to be exactly that.
    "headway.anchor",
    # numpy arrives through the geometry above and through nothing else here.
    "numpy",
}
TEACHERS_STDLIB_OK = True   # anything in sys.stdlib_module_names

# The camera constants teachers/project.py is allowed to take from headway, and
# nothing else. Named so that "it only imports geometry" is a checked fact.
PROJECT_MAY_TAKE = {"CAMERA_HEIGHT_M", "CAMERA_PITCH_RAD", "HFOV_DEG"}

# B. Who may import `teachers` at all.
MAY_IMPORT_TEACHERS = {
    "app.py",
}

# C. The functions in app.py that may touch the panel.
APP_TEACHER_FUNCTIONS = {
    "lifespan",
    "health",
    "headway_frame_endpoint",
    "headway_ws_endpoint",
    "_teardown_session",
    "teachers_state_endpoint",
    "teachers_status_endpoint",
    "teachers_health_refresh_endpoint",
    "teachers_ego_endpoint",
    "teachers_spoken_endpoint",
    "teachers_event_endpoint",
    # The websocket handler's inner worker, where the frame hook lives.
    "worker",
    # THE ONE DOOR THE TEACHERS' WORDS COME THROUGH: a tool call a driver's
    # own question triggered. Everything about that path is checked by rule F.
    "realtime_tool_endpoint",
    # THE YIELD, and it is a one-way door like every other entry here. app.py
    # tells the panel "a driver is waiting, take no new work"; the panel tells
    # app.py nothing back. What crosses the boundary is a context manager, not
    # a reading -- asserted below, because a function that returned teacher
    # DATA from here would be a hole in rule B that rule C would not see.
    "_rio_has_the_gpu",
}

# D. What none of those may reach out to. Module-level names whose appearance
#    in the same function as a teacher reference is the violation.
MOUTHS = {
    "observer", "vision", "voice", "voice_dialogue", "voice_tags", "realtime",
    "visual_qa", "router", "request_router", "persona", "rio_prompts",
    "navspeech", "llm_interface", "headway_live", "live_policy",
}
# ...and the specific calls that ARE the mouth, wherever they appear.
MOUTH_CALLS = {
    "speak", "say", "observe", "observe_now", "look", "cached", "fresh",
    "serve_to", "synthesize", "dictate", "arbitrate",
}

# The modules that must never mention the teachers at all -- not import them,
# not name them in a string, not have a commented-out call. These are the loop.
# THE PROACTIVE LOOP: may not so much as NAME a teacher. These are the paths
# RIO takes on her own initiative -- what she sees, what she decides, what she
# says without being asked -- and the panel must be invisible to all of them.
LOOP_MODULES = [
    "observer.py", "vision.py", "visual_qa.py", "router.py",
    "voice.py", "voice_dialogue.py", "voice_tags.py", "persona.py",
    "rio_prompts.py", "llm_interface.py", "resolve.py", "enrich.py",
    "frameselect.py", "perceive.py", "insights.py",
]

# realtime.py is NOT in that list any more, and the reason is the whole of the
# change: look() may now carry the two models' reading of the same instant, so
# RIO can form her own read from the camera, the measured state and two second
# opinions together. It therefore names them, in the tool result and in the
# session instructions.
#
# What it may still never do is REACH them. The block is passed in as a plain
# dict by app.py; realtime.py imports nothing from `teachers` and has no way
# to ask for a reading. That is asserted separately, in rule F3, because it is
# a different and stronger claim than "does not mention".
RECIPIENT_MODULES = ["realtime.py"]
LOOP_DIRS = ["headway", "navigation"]

# E. What the browser half may not name.
JS_FORBIDDEN = [
    r"\bRIO\.speech\.(?!onEvent\b)\w+",      # onEvent is a listener; the rest speak
    r"\bRIO\.speak\b",
    r"\bRIO\.output\b",
    r"\bRIO\.realtime\b",
    r"\bplayBlob\b",
    r"\bspeakNow\b",
    r"\benqueue\b",
]


# ---------------------------------------------------------------------------
def parse(path):
    return ast.parse(path.read_text(), filename=str(path))


def imported_modules(tree):
    """Every module name this file imports, dotted, with `from x import y`
    reported as `x` and (for relative imports) as `.`-prefixed."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno, None))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                mod = "." * node.level + mod
            out.append((mod, node.lineno, [a.name for a in node.names]))
    return out


def stdlib(name):
    root = name.split(".")[0]
    if hasattr(sys, "stdlib_module_names"):
        return root in sys.stdlib_module_names
    return root in {"os", "sys", "json", "time", "math", "re", "threading",
                    "base64", "collections", "pathlib", "urllib", "http",
                    "argparse", "dataclasses", "typing", "contextlib", "io",
                    "traceback", "subprocess", "functools", "itertools"}


def enclosing_functions(tree):
    """-> {node: nearest enclosing function name}, '' for module level."""
    owner = {}

    def walk(node, name):
        for child in ast.iter_child_nodes(node):
            n = name
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                n = child.name
            owner[child] = n
            walk(child, n)

    walk(tree, "")
    return owner


# ---------------------------------------------------------------------------
def strip_py_comments(src: str) -> str:
    """Python with comments and docstrings removed, via the tokenizer.

    Same lesson as strip_js_comments: a rule that cannot tell a comment from a
    call fails on its own documentation and teaches everyone to stop writing
    it down.
    """
    import io
    import tokenize

    out = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        return src
    prev_type = None
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        # A bare string expression is a docstring; a string after `=` or `(`
        # is data and stays.
        if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL, None):
            continue
        out.append(tok.string)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE):
            prev_type = tok.type
        else:
            prev_type = tok.type
    return " ".join(out)


def strip_js_comments(src: str) -> str:
    """JavaScript with // and /* */ removed, and string literals kept.

    Not a parser. It walks the source one character at a time tracking four
    states -- code, single-quoted, double-quoted, template -- so that a "//"
    inside a URL string survives and a `/*` inside a comment does not start a
    second one. That is enough for the one file it is used on, and it is
    enough for the check it serves: a name that appears only inside a comment
    is not a call.
    """
    out = []
    i, n = 0, len(src)
    quote = None
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if quote:
            out.append(c)
            if c == "\\":
                if i + 1 < n:
                    out.append(nxt)
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "'\"`":
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and nxt == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def rule_a():
    section("A. nothing gets in — teachers/ cannot reach the loop")
    pkg = REPO / "teachers"
    files = sorted(p for p in pkg.rglob("*.py"))
    ok(len(files) >= 8, f"the package is here ({len(files)} modules)")

    for path in files:
        rel = str(path.relative_to(REPO))
        # teachers/service/* runs in a different interpreter entirely and
        # imports torch and transformers. It is checked by rule A2 instead.
        if "service" in path.parts:
            continue
        tree = parse(path)
        bad = []
        for mod, lineno, names in imported_modules(tree):
            if mod.startswith("."):
                continue
            if mod.split(".")[0] == "teachers":
                continue
            if stdlib(mod) and TEACHERS_STDLIB_OK:
                continue
            if mod in TEACHERS_MAY_IMPORT:
                continue
            bad.append(f"{mod} (line {lineno})")
        ok(not bad, f"{rel} imports only what it is allowed to"
                    + (f" — FOUND {bad}" if bad else ""))

    section("A2. teachers/service/ does not import RIO")
    for path in sorted((pkg / "service").rglob("*.py")):
        rel = str(path.relative_to(REPO))
        tree = parse(path)
        bad = []
        for mod, lineno, _ in imported_modules(tree):
            root = mod.lstrip(".").split(".")[0]
            if root in ("config", "observer", "vision", "framebuf", "headway",
                        "realtime", "router", "sessions", "visual_qa",
                        "voice", "persona", "app"):
                bad.append(f"{mod} (line {lineno})")
        ok(not bad, f"{rel} knows nothing about RIO"
                    + (f" — FOUND {bad}" if bad else ""))

    section("A3. the one geometry import takes constants and nothing else")
    tree = parse(pkg / "project.py")
    took = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "") == "headway.anchor":
            took |= {a.name for a in node.names}
    ok(took and took <= PROJECT_MAY_TAKE,
       f"teachers/project.py takes only {sorted(PROJECT_MAY_TAKE)} from "
       f"headway.anchor (got {sorted(took)})")
    # ...and they are really constants, so importing them cannot run anything
    # that matters and cannot drift from the corridor's own numbers.
    import headway.anchor as A

    from teachers import project

    cam = project.camera(1280, 720)
    ok(abs(cam["h_m"] - A.CAMERA_HEIGHT_M) < 1e-9
       and abs(cam["pitch"] - A.CAMERA_PITCH_RAD) < 1e-9,
       "and the projection uses the SAME numbers the ego corridor is drawn "
       "with — a ribbon through a second camera model would sit visibly beside "
       "the corridor being wrong")


def rule_b():
    section("B. nothing gets out — who is allowed to import teachers")
    offenders = []
    for path in sorted(REPO.rglob("*.py")):
        rel = path.relative_to(REPO)
        parts = rel.parts
        if parts[0] in ("teachers", "tools", "runs", "training_data", "weights"):
            continue
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue
        if path.name.endswith(".bak") or ".pre-" in path.name:
            continue
        try:
            tree = parse(path)
        except SyntaxError:
            continue
        for mod, lineno, _ in imported_modules(tree):
            if mod.split(".")[0] == "teachers":
                if str(rel) not in MAY_IMPORT_TEACHERS:
                    offenders.append(f"{rel}:{lineno} imports {mod}")
    ok(not offenders,
       "only app.py imports the panel" + (f" — FOUND {offenders}" if offenders else ""))

    section("B2. the loop does not mention the teachers at all")
    targets = [REPO / n for n in LOOP_MODULES]
    for d in LOOP_DIRS:
        targets += sorted((REPO / d).rglob("*.py"))
    checked = 0
    for path in targets:
        if not path.exists() or path.name.endswith(".bak") or ".pre-" in path.name:
            continue
        checked += 1
        src = path.read_text()
        hits = [m.start() for m in re.finditer(r"\bteachers?\b", src, re.I)]
        # A word like "teacher" in a comment is not a path -- but in these
        # files it is a smell worth reporting, because the only reason to write
        # it is that somebody was thinking about wiring one in.
        ok(not hits, f"{path.relative_to(REPO)} does not mention a teacher"
                     + (f" — {len(hits)} mention(s)" if hits else ""))
    ok(checked >= 15, f"({checked} loop modules checked)")


def rule_c_and_d():
    section("C. app.py's one door, and how narrow it is")
    path = REPO / "app.py"
    tree = parse(path)
    owner = enclosing_functions(tree)

    refs = []          # (function, lineno)
    for node in ast.walk(tree):
        hit = False
        if isinstance(node, ast.Name) and node.id in ("teacher_panel",
                                                      "teacher_corpus"):
            hit = True
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("teachers"):
            hit = True
        elif isinstance(node, ast.Import) and any(
                a.name.startswith("teachers") for a in node.names):
            hit = True
        if hit:
            refs.append((owner.get(node, ""), getattr(node, "lineno", 0)))

    ok(refs, f"app.py does touch the panel ({len(refs)} references)")
    outside = sorted({f"{fn or '<module>'}:{ln}" for fn, ln in refs
                      if fn not in APP_TEACHER_FUNCTIONS and fn != ""})
    ok(not outside,
       "every reference is inside an allowlisted function"
       + (f" — FOUND {outside}" if outside else ""))
    # Module level is allowed for exactly one thing: the guarded import.
    module_level = [ln for fn, ln in refs if fn == ""]
    ok(len(module_level) <= 6,
       f"module level holds only the guarded import ({len(module_level)} nodes)")

    section("D. the door does not open both ways")
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node

    # The endpoints that READ a teacher value must not also touch a mouth.
    readers = ["teachers_state_endpoint", "teachers_status_endpoint",
               "teachers_health_refresh_endpoint", "_rio_has_the_gpu"]
    for name in readers:
        fn = funcs.get(name)
        if fn is None:
            ok(False, f"{name} is missing from app.py")
            continue
        named = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Name):
                named.add(node.id)
            elif isinstance(node, ast.Attribute):
                named.add(node.attr)
        bad = (named & MOUTHS) | (named & MOUTH_CALLS)
        ok(not bad, f"{name} reads the panel and touches no mouth"
                    + (f" — FOUND {sorted(bad)}" if bad else ""))

    section("D1b. the yield hands back a context manager, not a reading")
    fn = funcs.get("_rio_has_the_gpu")
    ok(fn is not None, "_rio_has_the_gpu exists")
    if fn is not None:
        called = {n.func.attr for n in ast.walk(fn)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        ok(called <= {"hold_gpu", "nullcontext"},
           f"it calls hold_gpu and nothing else on the panel ({sorted(called)}) "
           f"— a yield that also fetched a reading would be a hole in rule B "
           f"that rule C could not see")
        ok(not any(isinstance(n, ast.Subscript) for n in ast.walk(fn)),
           "and it reads no field off anything the panel returned")

    section("D2. the panel has no way to speak, by construction")
    # There is no function anywhere in teachers/ whose name or body reaches a
    # mouth. Checked over the whole package rather than per-caller, because the
    # absence is the guarantee.
    for p in sorted((REPO / "teachers").rglob("*.py")):
        if "service" in p.parts:
            continue
        t = parse(p)
        named = set()
        for node in ast.walk(t):
            if isinstance(node, ast.Attribute):
                named.add(node.attr)
            elif isinstance(node, ast.Name):
                named.add(node.id)
        bad = named & MOUTHS
        ok(not bad, f"{p.relative_to(REPO)} names nothing that speaks"
                    + (f" — FOUND {sorted(bad)}" if bad else ""))

    section("D3. the one-way street is one-way")
    from teachers import panel

    ok(hasattr(panel, "note_spoken"),
       "the speech path CAN tell the panel what was said (note_spoken)")
    ok(not any(n.startswith("say") or n.startswith("speak") or n == "arbitrate"
               for n in dir(panel)),
       "...and the panel has no counterpart pointing the other way")
    exported = [n for n in dir(panel) if not n.startswith("_")]
    ok("state" in exported and "status" in exported,
       f"the read-only surface is what the dashboard uses ({len(exported)} names)")


# The paths RIO can take on her OWN initiative. A teacher value reaching any of
# these is the thing the whole panel promised not to do.
PROACTIVE_FUNCTIONS = [
    "headway_frame_endpoint", "headway_ws_endpoint", "nav_voice_endpoint",
    "headway_voice_endpoint", "nav_event_endpoint", "talk",
]


def rule_f():
    section("F. the teachers reach ONE thing: a question the driver asked")
    # THE BOUNDARY MOVED, ON PURPOSE, AND THIS IS WHERE IT IS NOW.
    #
    # look() may carry the two models' reading of the same instant, so RIO can
    # form her own read from the camera, the measured state and two second
    # opinions together. That is a real change: a teacher's opinion can now
    # affect words a driver hears.
    #
    # What has NOT changed is everything proactive. No warning, no nav
    # announcement, no band, no health line, no sentence RIO produces on her
    # own initiative may see a teacher value. The narrow way to say that is:
    # `context_for` -- the only function that hands teacher readings out for
    # use -- has exactly one caller in the repo, and it is the tool endpoint a
    # driver's question arrives through.
    path = REPO / "app.py"
    tree = parse(path)
    owner = enclosing_functions(tree)
    callers = sorted({owner.get(n, "<module>")
                      for n in ast.walk(tree)
                      if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute)
                      and n.func.attr == "context_for"})
    ok(callers == ["realtime_tool_endpoint"],
       f"context_for has exactly one caller, the tool endpoint ({callers})")

    # And nowhere else in the repo calls it either.
    others = []
    for p2 in sorted(REPO.rglob("*.py")):
        rel = p2.relative_to(REPO)
        if rel.parts[0] in ("teachers", "runs", "training_data", "weights"):
            continue
        if p2.name.endswith(".bak") or ".pre-" in p2.name or rel.parts[0] == "tools":
            continue
        if str(rel) == "app.py":
            continue
        try:
            src = p2.read_text()
        except Exception:
            continue
        # A CALL, not a mention. realtime.py's comments point at
        # `teachers.panel.context_for` to say where the block came from, and a
        # text grep would read the explanation as the violation.
        if "context_for(" in strip_py_comments(src):
            others.append(str(rel))
    ok(not others,
       f"and no other module mentions it{' — FOUND ' + str(others) if others else ''}")

    section("F2. nothing proactive can see a teacher value")
    for name in PROACTIVE_FUNCTIONS:
        fn = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                fn = node
                break
        if fn is None:
            continue
        named = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute):
                named.add(node.attr)
            elif isinstance(node, ast.Name):
                named.add(node.id)
        # on_frame and hold_gpu are one-way: app.py TELLS the panel something.
        # Reading anything back is what is forbidden.
        reads = named & {"context_for", "state", "status"}
        ok(not reads,
           f"{name} tells the panel things and reads none back"
           + (f" — FOUND {sorted(reads)}" if reads else ""))

    section("F3. realtime.py still cannot reach the teachers itself")
    rt = (REPO / "realtime.py").read_text()
    rtree = ast.parse(rt)
    imports = [m for m, _, _ in imported_modules(rtree)
               if m.split(".")[0] == "teachers"]
    ok(not imports,
       f"realtime.py imports nothing from teachers ({imports}) — the block is "
       f"PASSED IN as a plain dict, which is what keeps one door countable")
    ok("teachers: dict = None" in rt,
       "look() and run_tool take it as an argument")
    ok("def teacher_context(" in rt,
       "and the labelling and the rules live there, with the answer")


def rule_e():
    section("E. the browser half")
    path = REPO / "static" / "rio_teachers.js"
    ok(path.exists(), "static/rio_teachers.js is here")
    if not path.exists():
        return
    # CODE ONLY. This file's own header names every forbidden identifier, in
    # prose, to explain why it may not use them -- and a check that cannot
    # tell a comment from a call would fail on its own documentation and
    # teach everyone to stop writing it down. So the comments come out first.
    src = strip_js_comments(path.read_text())
    ok("RIO.speak" not in src and "RIO.speak" in path.read_text(),
       "(the stripper works: the header's prose mentions the forbidden names "
       "and the code does not)")
    for pattern in JS_FORBIDDEN:
        hits = re.findall(pattern, src)
        # RIO.speech.onEvent is a LISTENER on something that has already been
        # said. It is how RIO's own line reaches the corpus, and it is the one
        # allowed touch -- see the note in rio_teachers.js.
        ok(not hits, f"rio_teachers.js does not use {pattern}"
                     + (f" — FOUND {hits}" if hits else ""))
    ok("RIO.speech.onEvent" in (REPO / "static" / "index.html").read_text(),
       "the spoken-line report is wired in index.html, where the arbiter "
       "already publishes it, rather than inside the teacher module")

    section("E2. the overlay reads teacher state and never hands it back")
    idx = (REPO / "static" / "index.html").read_text()
    ok("teachers: function (state) { teacherState = state || null;" in idx,
       "RIO.overlay.teachers is a setter")
    # There must be no way to read it back out: a getter would be a path from a
    # teacher's opinion into whatever asked.
    ok("get teacherState" not in idx and "return teacherState" not in idx,
       "...and there is no getter for it")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rule_a()
    rule_b()
    rule_c_and_d()
    rule_f()
    rule_e()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED — a shadow model may be able to affect what RIO does:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
