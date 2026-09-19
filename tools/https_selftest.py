"""HTTPS for the dashboard: is the phone actually going to be able to say yes?

    python -m tools.https_selftest
    python -m tools.https_selftest --https https://127.0.0.1:8443/

THE POINT OF THIS SUITE IS ONE BOOLEAN. `window.isSecureContext` decides
whether a browser will even OFFER the camera, the microphone and geolocation —
and over plain http on a LAN address it is false, so all three are refused
before any prompt is drawn. Everything else here exists to make that boolean
true on a device that is not localhost.

WHICH DEPLOYMENT THIS IS FOR, because getting it wrong wasted an afternoon.
It is for **the in-car one**: a Jetson or a laptop on a local network with no
proxy in front of it. A RunPod pod is published through an HTTPS proxy and is
ALREADY a secure origin — it needs none of this.

It was built on the belief that a non-secure origin caused the 2026-09-16
drive to fail. The logs disprove that (docs/session_scope.md §2a): the client
addresses were RunPod's proxy range, and the phone logged geolocation TIMEOUT
rather than the PERMISSION_DENIED an insecure origin raises. The real cause was
a 4-second watchdog tearing down the watch while the permission sheet was still
on screen. This suite is still worth having — for the deployment it was
actually right about — and section F pins the one thing that misled: the page
must name the RIGHT url for where it is running.

What is checked, and why each one is a way it silently fails anyway:

  cert        SANs present and iOS-shaped. Apple ignores the Common Name
              entirely, so a certificate with a CN and no SAN is a certificate
              for nothing — it fails AFTER the driver taps through the warning,
              which is the most confusing possible moment.
  serving     TLS actually answers, and the page loads through it.
  secure      isSecureContext is true and the three APIs are present. This is
              the assertion the whole exercise is for.
  websocket   the frame transport survives the terminator. It is a raw byte
              proxy, and an upgrade that does not pass through would take the
              camera feed with it while leaving the page looking fine.
  untouched   plain HTTP still works. Fifteen suites and every curl in this
              repository speak it, and TLS was added for the phone, not
              instead of them.

Needs the dashboard running (`bash boot.sh restart`) and playwright for the
browser half. Exit code is the number of failures.
"""
import argparse
import socket
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
CERT = REPO / "cert" / "rio-cert.pem"

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


def section(t):
    print(f"\n== {t}")


def t_cert():
    section("A. the certificate is one iOS will accept")
    if not CERT.exists():
        ok("a certificate exists", False,
           f"{CERT} — run: python -m tools.make_cert")
        return
    ok("a certificate exists", True)
    txt = subprocess.run(["openssl", "x509", "-in", str(CERT), "-noout",
                          "-text"], capture_output=True, text=True).stdout

    # Apple reads ONLY the SAN. A cert with a CN and no SAN fails after the
    # driver has already tapped through the warning.
    ok("it has Subject Alternative Names", "Subject Alternative Name" in txt)
    ok("...including localhost or an IP",
       "DNS:localhost" in txt or "IP Address:" in txt)
    ok("Extended Key Usage includes serverAuth",
       "TLS Web Server Authentication" in txt, "iOS rejects it without this")
    ok("signed with SHA-256 or better",
       "sha256" in txt.lower() or "sha384" in txt.lower()
       or "sha512" in txt.lower())
    ok("key is EC P-256 or RSA 2048+",
       "prime256v1" in txt or "secp384r1" in txt or "2048 bit" in txt
       or "4096 bit" in txt, "iOS refuses weaker keys")

    dates = subprocess.run(["openssl", "x509", "-in", str(CERT), "-noout",
                            "-dates"], capture_output=True, text=True).stdout
    # 825 days is Apple's ceiling for anything issued after 2019-07-01.
    import datetime
    try:
        nb = [l for l in dates.splitlines() if l.startswith("notBefore=")][0][10:]
        na = [l for l in dates.splitlines() if l.startswith("notAfter=")][0][9:]
        fmt = "%b %d %H:%M:%S %Y %Z"
        span = (datetime.datetime.strptime(na, fmt)
                - datetime.datetime.strptime(nb, fmt)).days
        ok("validity is 825 days or less", span <= 825, f"{span} days")
    except (IndexError, ValueError) as e:
        ok("validity is readable", False, str(e))

    # The private key must not be in git. It is regenerated in seconds; there is
    # nothing to preserve and everything to lose.
    r = subprocess.run(["git", "check-ignore", "cert/rio-key.pem"],
                       capture_output=True, text=True, cwd=str(REPO))
    ok("the private key is gitignored", r.returncode == 0,
       "cert/ must be in .gitignore")


def t_serving(https_url, http_url):
    section("B. TLS answers, and plain HTTP is untouched")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE   # self-signed: the driver taps through

    try:
        with urllib.request.urlopen(https_url + "health", context=ctx,
                                    timeout=15) as r:
            ok("the dashboard answers over TLS", r.status == 200, r.status)
    except Exception as e:
        ok("the dashboard answers over TLS", False, f"{type(e).__name__}: {e}")

    try:
        with urllib.request.urlopen(http_url + "health", timeout=10) as r:
            ok("...and plain HTTP still answers, for every local tool",
               r.status == 200, r.status)
    except Exception as e:
        ok("...and plain HTTP still answers, for every local tool", False,
           f"{type(e).__name__}: {e}")

    # The terminator must present the certificate we actually made.
    host = https_url.split("//", 1)[1].rstrip("/")
    hostname, _, port = host.partition(":")
    try:
        with socket.create_connection((hostname, int(port or 443)), timeout=10) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ss:
                peer = ss.getpeercert(binary_form=True)
                ok("it presents a certificate", bool(peer))
                ok("negotiated TLS 1.2 or better",
                   ss.version() in ("TLSv1.2", "TLSv1.3"), ss.version())
    except Exception as e:
        ok("the TLS handshake completes", False, f"{type(e).__name__}: {e}")


def t_browser(https_url, http_url):
    section("C. the boolean the whole exercise is for")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        ok("playwright is installed", False, "cannot check isSecureContext")
        return

    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--no-sandbox"])
        # ignore_https_errors is the state a phone is in AFTER tapping through
        # the self-signed warning, which is the state that matters.
        p = b.new_page(ignore_https_errors=True)
        errs = []
        p.on("pageerror", lambda e: errs.append(str(e)))
        p.goto(https_url, wait_until="domcontentloaded", timeout=40000)
        p.wait_for_timeout(1500)

        ok("the page loads over https", p.evaluate("location.protocol") == "https:")
        ok("isSecureContext is TRUE", p.evaluate("window.isSecureContext") is True,
           "this is the one that was false on the phone")
        ok("the permission module agrees",
           p.evaluate("RIO.permissions.secure()") is True)
        ok("getUserMedia is offered",
           p.evaluate("!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)")
           is True)
        ok("geolocation is offered",
           p.evaluate("!!navigator.geolocation") is True)
        ok("no javascript errors on the page", not errs, errs[:2])

        section("D. the frame transport survives the terminator")
        res = p.evaluate("""() => new Promise((resolve) => {
          const url = 'wss://' + location.host + '/headway_ws?client_id=httpsselftest';
          let ws; const done=(o)=>{try{ws&&ws.close()}catch(e){};resolve(o)};
          const t=setTimeout(()=>done({ok:false,why:'timeout'}),15000);
          try { ws = new WebSocket(url); } catch(e){ clearTimeout(t); return done({ok:false,why:String(e)}); }
          let opened=false;
          ws.onopen=()=>{opened=true};
          ws.onmessage=(ev)=>{clearTimeout(t); let m=null; try{m=JSON.parse(ev.data)}catch(e){}
            done({ok:true,opened:opened,op:m&&m.op})};
          ws.onerror=()=>{clearTimeout(t); done({ok:false,why:'error',opened:opened})};
        })""")
        ok("the websocket upgrades through the TLS proxy",
           res.get("ok") is True and res.get("opened") is True, res)
        ok("...and a message comes back through it",
           res.get("op") == "ready", res.get("op"))

        section("E. the page tells an insecure visitor what to do")
        p2 = b.new_page()
        p2.goto(http_url, wait_until="domcontentloaded", timeout=30000)
        p2.wait_for_timeout(1200)
        p2.evaluate("""RIO.ui.permissions({camera:'insecure',mic:'insecure',
            geo:'insecure',detail:{},secure:false,asked:true,fix:null,
            canSee:false,canLocate:false})""")
        p2.wait_for_timeout(200)
        note = p2.eval_on_selector("#permnote", "el => el.innerText")
        ok("the note names https, not just 'a secure context'",
           "https://" in note, note[:80])
        ok("...and names the port to use", ":8443" in note, note[:120])
        rows = p2.eval_on_selector_all(".perm-row", "els => els.map(e => e.className)")
        # "all THREE rows" over a list that could be empty: a selector matching
        # nothing -- the permission panel never rendered, which is a real way for
        # this to break -- makes all() True and the sentence a lie in the same
        # breath. ok_all fails on empty and prints the count it actually saw, so
        # "three" is now something the output states rather than something the
        # name claims.
        ok_all("all three rows show the insecure state",
               rows, lambda c: "insecure" in c, rows)

        section("F. the hint names the RIGHT url for the deployment")
        # Two deployments, two different right answers. Naming the wrong one is
        # what sent the first diagnosis of this bug off to build a certificate
        # for a pod that was already behind an HTTPS proxy.
        pod = p2.evaluate(
            "([h,o]) => RIO.ui.secureOriginHint(h,o)",
            ["abc123-8888.proxy.runpod.net",
             "http://abc123-8888.proxy.runpod.net"])
        ok("a RunPod host is told to use the proxy URL",
           "proxy.runpod.net/" in pod and "https://abc123-8888" in pod, pod[:110])
        ok("...and explicitly that no certificate is needed",
           "no certificate is needed" in pod, pod[:110])
        ok("...and is NOT sent to make one",
           "make_cert" not in pod, pod[:140])

        lan = p2.evaluate("([h,o]) => RIO.ui.secureOriginHint(h,o)",
                          ["192.168.1.42", "http://192.168.1.42:8888"])
        ok("a LAN address is sent to the TLS terminator",
           ":8443/" in lan, lan[:110])
        ok("...and told how to make the certificate",
           "make_cert" in lan, lan[:140])
        ok("...and is NOT told to use a proxy it does not have",
           "runpod" not in lan.lower(), lan[:140])
        b.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--https", default="https://127.0.0.1:8443/")
    ap.add_argument("--http", default="http://127.0.0.1:8888/")
    a = ap.parse_args()
    t_cert()
    t_serving(a.https, a.http)
    t_browser(a.https, a.http)
    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
