"""RIO contextual navigation.

    The navigation provider determines WHERE the driver needs to go.
    RIO uses map context and computer vision to describe that ALREADY-KNOWN
    maneuver in a more natural way.

        instead of  "In 500 feet, turn left."
        RIO can say "Turn left by the Shell station."

That sentence is the whole design, and the authority firewall (§2) is what
makes it safe. Route, next road, turn direction, maneuver sequence, maneuver
coordinates, route progress, whether a maneuver has been passed, rerouting,
arrival side and when an instruction becomes time-critical come from the
provider and the deterministic tracker, and from nowhere else. Perception is
allowed to answer exactly one question — "can the driver clearly see this
specific expected landmark right now?" — and its only possible effects are to
improve a sentence or to change nothing at all.

    provider.py / providers/    where the route comes from; the ONLY
                                provider-shaped code (§3, §36)
    model.py                    RIO's canonical route vocabulary
    geo.py                      one projection, shared by every stage
    service.py                  journeys, route generations, the speech table
    speech.py                   deterministic templates — no LLM, ever (§23)
    landmarks.py                what is near the turn, and where the turn sits
                                relative to it — from MAP DATA (§13, §16)
    anchors.py                  the validation gates and the selector (§18-20)
    verify.py                   the camera's single question (§14, §15)
    fixtures.py                 synthetic routes and landmarks; the harness

The route tracker itself is client-side, in static/rio_navcore.js, for the
reason it has always been: an announcement is worth nothing late, and nothing
between "you are four seconds from the turn" and RIO saying so may depend on
the network.
"""
from .provider import NavError, NavigationProvider  # noqa: F401
