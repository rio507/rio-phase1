"""The teacher panel — two AV foundation models watching, and never driving.

WHAT THIS IS
------------
Alpamayo 1.5 (nvidia/Alpamayo-1.5-10B) and Cosmos-Reason2-8B, run over the
same 4-frame window and the same ego-motion history RIO's own loop is working
from, once every couple of seconds while a drive is live. Their readings are
shown side by side on the dashboard, associated back to the RF-DETR tracks they
name, and written to a corpus on the volume.

WHAT THIS IS NOT
----------------
It is not part of RIO. Qwen3-VL-8B is still the observer, still the only model
whose sentences may be spoken, and still the only one look() reads from. No
output of either teacher reaches the arbiter, the speech path, look(), or the
observer cache -- not as a warning, not as a hint, not as a tie-break. The two
6.4 s trajectories Alpamayo predicts are drawn on the overlay as a faint ribbon
and are used for nothing.

That is a claim, so it is a test: tools/teacher_firewall_selftest.py walks the
AST of this package and of everything that imports it and asserts the absence
of every such path. It fails the build, not the drive.

WHY IT IS WORTH THE GPU
-----------------------
Because "was RIO right about that junction" currently has no answer except a
human watching the clip back. Two models built for exactly this question,
reading exactly the frames RIO read, with their reasoning traces kept next to
RIO's own deterministic state at the same instant, is an answer -- and a
labelled corpus of the disagreements is the beginning of a better observer.

WHERE THE MODELS LIVE
---------------------
Not here. Each runs in its own Python environment behind a loopback HTTP
service (teachers/service/), because they want transformers, torch and a
Python version RIO's runtime environment does not have and must not be made to
have. This package is the client, the cadence, the association and the record;
it imports nothing heavier than numpy.
"""
