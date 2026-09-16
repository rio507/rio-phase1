"""A TLS certificate the phone will actually accept, for the dashboard.

    python -m tools.make_cert                       # localhost + detected IPs
    python -m tools.make_cert --add 192.168.1.42    # ...and the address you type
    python -m tools.make_cert --add rio.local --add 10.0.0.5
    python -m tools.make_cert --force               # replace an existing one

WHY THIS EXISTS. Over plain http on a LAN address a browser refuses the camera,
the microphone and geolocation before drawing any prompt — not as a permission
the driver denied, but as a capability that was never offered. That is what the
2026-09-16 drive hit: no geolocation prompt on the phone, place search answering
about somewhere else, and nothing on screen saying why. `localhost` is a secure
context by special dispensation; `http://192.168.1.42:8888` is not. The phone is
never on localhost.

So the dashboard needs TLS, and on a private network there is no certificate
authority to get one from. This makes a self-signed one.

WHAT APPLE REQUIRES, because "openssl req -x509" alone produces a certificate
iOS will not accept even after the driver taps through the warning:

  * SUBJECT ALTERNATIVE NAMES, and only those. iOS and modern Safari ignore the
    Common Name entirely. A cert whose CN is the IP and whose SAN list is empty
    is a cert for nothing.
  * VALIDITY OF 825 DAYS OR LESS for anything issued after 2019-07-01.
  * EXTENDED KEY USAGE containing serverAuth.
  * SHA-256 or better, and EC P-256 or RSA 2048+.

All four are set below. They are the difference between a warning you can tap
through and a page that will not load at all.

THE ADDRESS PROBLEM, stated plainly. A certificate is only valid for the names
in it, and this script cannot know the address the phone will use: the server
sees a container address (172.x), the phone dials the host machine's LAN
address, and those are different. Detected addresses are included, `--add` is
for the one you actually type, and a mismatch shows up as a second warning
rather than a silent failure.

WHAT IT DOES NOT DO. A self-signed certificate is not trusted by anything, so
the browser will warn on first visit. Tapping through is enough to make the
page a secure context, which is all the permission prompts need. For a phone
you use every day, installing `cert/rio-cert.pem` as a trusted profile removes
the warning — see docs/session_scope.md §6.
"""
import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CERT_DIR = REPO / "cert"
CERT = CERT_DIR / "rio-cert.pem"
KEY = CERT_DIR / "rio-key.pem"

# Apple refuses anything longer for certificates issued after 2019-07-01.
DAYS = 825


def local_addresses() -> list:
    """Every address this machine can see itself at. Best effort, never fatal.

    Deliberately includes the container address even though a phone cannot dial
    it: a certificate with too many names costs nothing, and leaving one out is
    the failure that sends somebody back to this script.
    """
    out = set()
    try:
        out.add(socket.gethostbyname(socket.gethostname()))
    except Exception:
        pass
    try:
        # The address the kernel would use to reach the internet, which on a
        # normal machine is the LAN address the phone wants. No packet is sent.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            out.add(s.getsockname()[0])
        finally:
            s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            out.add(info[4][0])
    except Exception:
        pass
    out.discard("")
    return sorted(out)


def is_ip(s: str) -> bool:
    try:
        socket.inet_aton(s)
        return s.count(".") == 3
    except OSError:
        return False


def build_config(names: list, ips: list) -> str:
    alt = []
    for i, n in enumerate(names, 1):
        alt.append(f"DNS.{i} = {n}")
    for i, a in enumerate(ips, 1):
        alt.append(f"IP.{i} = {a}")
    return f"""[req]
distinguished_name = dn
x509_extensions = ext
prompt = no

[dn]
CN = RIO dashboard

[ext]
# Apple ignores the Common Name entirely and reads only these.
subjectAltName = @alt
# serverAuth is required; without it iOS rejects the certificate outright.
extendedKeyUsage = serverAuth
keyUsage = critical, digitalSignature, keyEncipherment
basicConstraints = critical, CA:FALSE
subjectKeyIdentifier = hash

[alt]
{chr(10).join(alt)}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[],
                    help="another hostname or IP the phone will use; repeatable")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing certificate")
    ap.add_argument("--days", type=int, default=DAYS)
    a = ap.parse_args()

    if a.days > DAYS:
        print(f"  !! {a.days} days exceeds Apple's {DAYS}-day limit; iOS will "
              f"refuse this certificate. Using {DAYS}.")
        a.days = DAYS

    if CERT.exists() and not a.force:
        print(f"Certificate already exists: {CERT}")
        print("Use --force to replace it, or point boot.sh at it as-is.")
        show(CERT)
        return 0

    names = ["localhost"]
    ips = ["127.0.0.1"]
    for extra in a.add:
        extra = extra.strip()
        if not extra:
            continue
        (ips if is_ip(extra) else names).append(extra)
    for found in local_addresses():
        if found not in ips:
            ips.append(found)

    CERT_DIR.mkdir(parents=True, exist_ok=True)
    conf = CERT_DIR / "openssl.cnf"
    conf.write_text(build_config(names, ips))

    print("Generating a self-signed certificate for:")
    for n in names:
        print(f"   DNS  {n}")
    for i in ips:
        print(f"   IP   {i}")
    print(f"   valid {a.days} days, EC P-256, SHA-256, serverAuth\n")

    cmd = [
        "openssl", "req", "-x509", "-nodes",
        "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
        "-sha256", "-days", str(a.days),
        "-keyout", str(KEY), "-out", str(CERT),
        "-config", str(conf),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("openssl failed:\n" + (r.stderr or r.stdout))
        return 1

    # The key must not be world-readable. It is a private key sitting in a
    # repository directory, and the .gitignore entry is the other half of that.
    try:
        os.chmod(KEY, 0o600)
    except OSError:
        pass

    print(f"  wrote {CERT}")
    print(f"  wrote {KEY}  (chmod 600)")
    show(CERT)
    print("\nNext:")
    print("   bash boot.sh restart        # picks the certificate up")
    print("   then open https://<this machine's LAN address>:8443/ on the phone")
    print("\nThe phone will warn once — it is a self-signed certificate and")
    print("nothing has vouched for it. Tapping through is enough to make the")
    print("page a secure context, which is all the permission prompts need.")
    return 0


def show(cert: Path):
    r = subprocess.run(["openssl", "x509", "-in", str(cert), "-noout",
                        "-ext", "subjectAltName", "-dates"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("\n" + r.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())
