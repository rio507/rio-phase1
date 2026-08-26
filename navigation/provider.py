"""The provider boundary. Everything Google-shaped stops here (§3, §36).

A NavigationProvider answers exactly one question — *what is the route?* — and
answers it in RIO's vocabulary. It does not drive the car through the route,
does not decide when to speak, and does not know that RIO has a camera.

Three implementations are anticipated:

    GoogleProvider   the shipping one (navigation/providers/google.py)
    FixtureProvider  routes from JSON on disk; the whole test harness and the
                     answer to "the API is not enabled yet" — nothing else has
                     to change for RIO to be developed against fixtures
    <embedded>       an offline/onboard engine, later

Only the first is a licensing question, and that is exactly why the boundary is
drawn here: LICENSING.md's review scope is this file's implementations, not the
tracker, the speech planner, the verifier or the arbiter (§36).

WHAT A PROVIDER MAY NOT DO
--------------------------
Return anything a downstream stage has to special-case. If a provider cannot
supply the destination side, it says UNKNOWN and RIO omits the side. If it
cannot supply lane information, that field is None and nothing speaks about
lanes. "Degrade to less information" is always available; "return a different
shape" is not.
"""
import abc
from typing import List, Optional

from .model import CanonicalDestination, CanonicalRoute


class NavError(RuntimeError):
    """A routing request failed in a way the driver needs told about.

    Distinct from "no landmarks found" or "the camera is unavailable", which
    are ordinary outcomes and never surface as errors.
    """


class DestinationCandidate:
    """One possible reading of what the driver asked for.

    Destination resolution is allowed to be ambiguous — "the Getty" is two
    museums — and the honest response to ambiguity is a question, not a
    coin toss (§4). This carries what the clarifier needs to ask one.
    """

    def __init__(self, display_name: str, formatted_address: str,
                 provider_place_id: Optional[str] = None,
                 latitude: Optional[float] = None, longitude: Optional[float] = None,
                 distance_m: Optional[float] = None):
        self.display_name = display_name
        self.formatted_address = formatted_address
        self.provider_place_id = provider_place_id
        self.latitude = latitude
        self.longitude = longitude
        self.distance_m = distance_m

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "formatted_address": self.formatted_address,
            "provider_place_id": self.provider_place_id,
            "lat": self.latitude, "lng": self.longitude,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
        }


class NavigationProvider(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def suggest(self, query: str, lat: Optional[float] = None,
                lng: Optional[float] = None, limit: int = 5,
                session: Optional[str] = None) -> List[DestinationCandidate]:
        """Readings of a free-text destination, best first. Never raises:
        an empty list is a valid answer and the caller asks the driver.

        `session` is RIO's own opaque id for ONE typing session — everything
        the driver types between starting to type and picking something. What a
        provider does with it is the provider's business: Google groups the
        keystrokes and the final lookup into a single billed autocomplete
        session, a provider with no such concept ignores it. The browser never
        sees a provider's token, only RIO's id for the session (§3).
        """

    @abc.abstractmethod
    def destination(self, query: str = "", place_id: str = "",
                    label: str = "", session: Optional[str] = None
                    ) -> Optional[CanonicalDestination]:
        """Pin one candidate to coordinates. None if it cannot be resolved.

        Passing the `session` the suggestions came from is what CLOSES that
        typing session — for Google, it is the call the whole session is billed
        as. A provider must treat the session as spent afterwards.
        """

    @abc.abstractmethod
    def route(self, origin_lat: float, origin_lng: float,
              destination: CanonicalDestination) -> CanonicalRoute:
        """The route. Raises NavError if there is not one to be had.

        The returned route carries no journey identity — `route_id`,
        `journey_id` and `generation_id` are assigned by the service, because
        a reroute's relationship to the route it replaces is RIO's bookkeeping
        and not the provider's.
        """

    def landmarks_near(self, lat: float, lng: float, radius_m: float,
                       types: tuple) -> List[dict]:
        """Places of the given classes near a point.

        Optional. A provider with no place data returns [] and RIO navigates
        with no contextual anchors at all, which is a fully supported mode —
        see §27. Each entry: {place_id, name, types, primary_type, lat, lng}.
        """
        return []
