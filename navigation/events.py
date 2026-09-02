"""The navigation event vocabulary (§31).

One name per thing that happens, used identically by the server, the browser
tracker and the session log, so a drive can be replayed from the JSONL and the
sequence read as prose. Anchor rejections are events in their own right and are
expected to outnumber acceptances several times over — a drive whose log shows
no NAV_ANCHOR_REJECTED is a drive where the gates are not doing their job.

Two rules about what may be written alongside these names:
  * never an API key, in any field, ever;
  * no persistent precise-location history. Events carry the route-relative
    position (which maneuver, how far to it) rather than a trail of raw
    coordinates. The one place a coordinate is written is the reroute point,
    because a reroute cannot be understood without knowing where it happened.
"""

ROUTE_STARTED = "NAV_ROUTE_STARTED"
ROUTE_FAILED = "NAV_ROUTE_FAILED"
DESTINATION_AMBIGUOUS = "NAV_DESTINATION_AMBIGUOUS"
# A destination RIO set herself, on the driver's word, rather than one typed
# into the panel. Same route, same tracker, same everything downstream — but a
# review of the drive should be able to see which of the two it was, and what
# was actually asked for out loud.
VOICE_DESTINATION = "NAV_VOICE_DESTINATION"
MANEUVER_SELECTED = "NAV_MANEUVER_SELECTED"
EARLY_GUIDANCE = "NAV_EARLY_GUIDANCE"
CONTEXT_ACQUISITION_STARTED = "NAV_CONTEXT_ACQUISITION_STARTED"
ANCHOR_CANDIDATE = "NAV_ANCHOR_CANDIDATE"
ANCHOR_VERIFIED = "NAV_ANCHOR_VERIFIED"
ANCHOR_REJECTED = "NAV_ANCHOR_REJECTED"
CONTEXTUAL_CALL = "NAV_CONTEXTUAL_CALL"
NEAR_TURN = "NAV_NEAR_TURN"
MANEUVER_PASSED = "NAV_MANEUVER_PASSED"
GPS_DEGRADED = "NAV_GPS_DEGRADED"
GPS_STALE = "NAV_GPS_STALE"
GPS_OK = "NAV_GPS_OK"
OFF_ROUTE_CANDIDATE = "NAV_OFF_ROUTE_CANDIDATE"
OFF_ROUTE_CONFIRMED = "NAV_OFF_ROUTE_CONFIRMED"
REROUTE_STARTED = "NAV_REROUTE_STARTED"
REROUTE_COMPLETE = "NAV_REROUTE_COMPLETE"
REROUTE_FAILED = "NAV_REROUTE_FAILED"
SPEECH_EXPIRED = "NAV_SPEECH_EXPIRED"
SPEECH_INVALIDATED = "NAV_SPEECH_INVALIDATED"
SPEECH_SPOKEN = "NAV_SPEECH_SPOKEN"
ARRIVED = "NAV_ARRIVED"

ALL = tuple(v for k, v in sorted(globals().items())
            if k.isupper() and isinstance(v, str) and v.startswith("NAV_"))
