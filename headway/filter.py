"""Kalman filter on the gap [d, ḋ] with innovation gating and cut-in escape.

Design ref: headway_design.md §4 (smoothing/confidence), §5 event handling, §7.9.

Constant-velocity model on the *gap*, not on either vehicle. That is the key
simplification from §3: ḋ is the relative speed, so closing risk never requires
knowing the lead's absolute speed.

---------------------------------------------------------------------------
Q DERIVATION  (spec target: "tracks a 3 m/s² braking lead within ~0.3 s")
---------------------------------------------------------------------------
A constant-velocity model is *wrong* the moment the lead brakes -- that braking
is unmodelled acceleration, and Q is exactly the knob that says how wrong we
expect to be. So Q is sized from the manoeuvre we must not miss.

Model the unmodelled acceleration as white noise of standard deviation σ_a
acting over one step Δt. It perturbs velocity by σ_a·Δt and position by
½σ_a·Δt², and the covariance of that perturbation vector [½Δt², Δt]ᵀ·a is the
standard discrete white-noise-acceleration form:

    Q = σ_a² · [[Δt⁴/4, Δt³/2],
                [Δt³/2, Δt²  ]]

The spec names the manoeuvre directly -- a 3 m/s² braking lead -- so the
derivation starts at σ_a = 3.0 m/s²: the filter is told to expect accelerations
of precisely the magnitude it is required to track.

That fixes Q's *shape* and starting scale. Responsiveness, though, is set by the
ratio Q/R, so σ_a = 3.0 only satisfies the spec if R is right too -- and against
a ramp in ḋ (constant braking) a CV filter never converges to zero error, it
settles to a constant *lag*. The spec's "~0.3 s" is therefore read as that lag:
a 0.9 m/s error under 3 m/s² braking is exactly 0.3 s of lag.

Verified empirically with `python -m headway.filter --tune` (15 Hz, 3 m/s² brake,
R as configured below):

    sigma_a = 3.0  ->  steady-state lag 0.286 s,  ḋ noise 0.294 m/s (1σ)

which meets the ~0.3 s target with the σ_a the derivation already pointed at --
a useful consistency check rather than a fitted constant. Raising σ_a tracks
harder manoeuvres faster at the cost of noisier ḋ (and twitchier TTC); lowering
it smooths ḋ but lags real braking. `--sweep` prints the whole trade curve.

CAVEAT -- R is the assumption, not Q. The lag above holds only for the depth
noise assumed below, which is an engineering estimate, not a measurement. Tuning
Q/R against real clips is literally Stage 0's job (§8), so run_clip.py reports a
measured innovation spread on every clip; feed that back into R_REL_FRAC before
trusting the lag figure on real video.
"""
import argparse

import numpy as np

# --- process noise -----------------------------------------------------------
SIGMA_A = 3.0            # m/s² -- the braking magnitude from the spec's Q target

# --- measurement noise (§4: R = R₀ / depth_conf) -----------------------------
# DA-V2 metric error is largely a constant scale bias per install (§0 Challenge
# 2), which the filter cannot and need not remove -- it cancels in TTC and is
# calibrated out for τ in Stage 2. What R must model is only the *random*
# frame-to-frame flicker (§7.9), which grows with range, hence a relative term
# with an absolute floor.
#
# 0.5% is far tighter than DA-V2's ~10-20% absolute accuracy, and deliberately
# so: absolute bias is systematic (it does not vary frame to frame, so it is not
# measurement noise), and taking the median over several hundred ROI pixels
# averages down the per-pixel component hard. ESTIMATE, NOT MEASUREMENT --
# re-fit from run_clip.py's reported innovation spread on the first real clip.
R_REL_FRAC = 0.005       # 0.5% of range, 1σ
R_ABS_M = 0.10           # metres, 1σ floor for near targets
DEPTH_CONF_FLOOR = 0.05  # never divide by ~0 and blow R up to infinity

# --- initial covariance ------------------------------------------------------
P0_D = 3.0 ** 2          # m²   -- first measurement is decent but not exact
P0_V = 5.0 ** 2          # (m/s)² -- relative speed starts completely unknown

# --- gating (§4) -------------------------------------------------------------
GATE_SIGMA = 3.0

# A RUN of rejected measurements is the world changing, not noise -- cut-in, new
# lead, occlusion ending (§4, §5). How long that run must be is a TIME, not a
# frame count: the old flat "3 frames" is 0.25 s at the Stage 0 harness's 12 Hz
# but 1.5 s at the live loop's 2 fps, so the same constant meant two completely
# different behaviours depending on who was calling. Same frames-vs-time
# mismatch that live_policy.py fixes in its band confirmation, and it is fixed
# the same way here: sized in seconds, so a cut-in takes the same wall time to
# recognise at every cadence.
REJECT_WINDOW_S = 0.5

# Tolerance on the time term only, matching live_policy.CONFIRM_TOL_S and there
# for the same reason: frame arrival jitters either side of the nominal cadence,
# and without it a run landing a hair short of the window costs a whole extra
# frame.
REJECT_WINDOW_TOL_S = 0.1

# Hard floor, whatever the cadence, and it is load-bearing rather than
# decorative: at 2 fps a SINGLE rejected frame already spans 0.5 s and would
# satisfy the time term on its own. One outlier depth sample must never re-seed
# the gap -- telling a cut-in apart from a bad reading is the entire job of this
# gate, and getting it wrong hands the filter a spurious range with full
# confidence.
MIN_CONSEC_REJECTS = 2

MAX_COAST_S = 1.0        # §5: never warn from coasted data older than this

# --- the gap is a distance, and distances are not negative -------------------
# A constant-velocity model extrapolated far enough always walks through zero:
# `predict` is x <- [d + ḋ·Δt, ḋ], and with a closing ḋ and no measurement to
# correct it, d passes 0 and keeps going. Nothing in the maths objects, because
# nothing in the maths knows d is a distance.
#
# It reached the HUD. Observed on a coastal clip: GAP -66.8 M, which is what
# ~13 s of coasting at -5 m/s looks like after the lead drove out of frame at
# the end of the road. The warning path was never fooled -- confidence decays to
# zero at MAX_COAST_S and the state machine declares LOST -- but `distance_m`
# was reported and painted regardless, so the driver was shown a number that
# cannot exist.
#
# Clamped in the STATE rather than at the display, so the whole system agrees
# about what the gap is: TTC, tau and the log all read the same clamped value.
# The velocity is zeroed at the same time, because a gap held at the floor by a
# closing rate would otherwise carry a phantom closing speed forever.
GAP_FLOOR_M = 0.0


class HeadwayFilter:
    """Kalman on [d, ḋ] with a 3σ gate and a cut-in escape hatch."""

    def __init__(self, sigma_a=SIGMA_A, r_rel=R_REL_FRAC, r_abs=R_ABS_M,
                 gate_sigma=GATE_SIGMA, min_consec_rejects=MIN_CONSEC_REJECTS,
                 reject_window_s=REJECT_WINDOW_S):
        self.sigma_a = float(sigma_a)
        self.r_rel = float(r_rel)
        self.r_abs = float(r_abs)
        self.gate_sigma = float(gate_sigma)
        self.min_consec_rejects = int(min_consec_rejects)
        self.reject_window_s = float(reject_window_s)

        self.x = None                # [d, ḋ]
        self.P = None
        self.consec_rejects = 0
        self.reject_span = 0.0       # seconds of the current consecutive-reject run
        self.coast_age = 0.0         # seconds since the last accepted measurement
        self.new_lead = False        # latched for one step after a reset
        self.initialised = False
        self.n_updates = 0
        self.n_clamped = 0           # times the gap floor had to hold the state up

    # -- properties ----------------------------------------------------------
    @property
    def d(self):
        return None if self.x is None else float(self.x[0])

    @property
    def d_dot(self):
        return None if self.x is None else float(self.x[1])

    # -- core ----------------------------------------------------------------
    def initialise(self, z: float) -> None:
        # ḋ starts at 0, not at some guess: relative speed is genuinely unknown
        # from a single range sample, and P0_V says so honestly.
        self.x = np.array([float(z), 0.0], dtype=float)
        self.P = np.diag([P0_D, P0_V]).astype(float)
        self.consec_rejects = 0
        self.reject_span = 0.0
        self.coast_age = 0.0
        self.initialised = True
        self.n_updates = 0

    def _Q(self, dt: float) -> np.ndarray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return (self.sigma_a ** 2) * np.array([[dt4 / 4.0, dt3 / 2.0],
                                               [dt3 / 2.0, dt2]], dtype=float)

    def _R(self, z: float, depth_conf: float) -> float:
        sigma = max(self.r_abs, self.r_rel * abs(float(z)))
        conf = max(float(depth_conf), DEPTH_CONF_FLOOR)
        return (sigma ** 2) / conf

    def predict(self, dt: float) -> None:
        if not self.initialised:
            return
        dt = max(float(dt), 1e-4)
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        self._clamp()

    def _clamp(self) -> bool:
        """Hold the gap at or above GAP_FLOOR_M. -> did it have to?

        The covariance is deliberately left alone. P describes how uncertain the
        estimate is, and the clamp does not make it more certain -- if anything
        a state that had to be corrected by a constraint is less trustworthy,
        and letting P keep growing is the honest version of that.
        """
        if self.x is None or self.x[0] >= GAP_FLOOR_M:
            return False
        self.x[0] = GAP_FLOOR_M
        # A closing rate that has already driven the gap into the floor has
        # nowhere left to close to. Left in, it would re-violate the floor on
        # every subsequent predict and report a gap that is pinned at zero while
        # claiming to be shrinking.
        self.x[1] = max(self.x[1], 0.0)
        self.n_clamped += 1
        return True

    def step(self, z, depth_conf: float, dt: float) -> dict:
        """One filter tick. `z=None` (or NaN) means no measurement -> coast.

        Returns a snapshot dict; run_clip.py drops it straight into the shadow log.
        """
        dt = max(float(dt), 1e-4)
        self.new_lead = False

        have_z = z is not None and np.isfinite(z) and float(z) > 0.0

        if not self.initialised:
            if not have_z:
                return self._snapshot(accepted=False, reason="uninitialised", dt=dt)
            self.initialise(float(z))
            self.new_lead = True
            return self._snapshot(accepted=True, reason="initialised", dt=dt,
                                  z=float(z), depth_conf=depth_conf)

        self.predict(dt)

        if not have_z:
            # §5: coast on prediction alone, but the age is reported so the state
            # machine can declare LOST past MAX_COAST_S.
            self.coast_age += dt
            return self._snapshot(accepted=False, reason="no_measurement", dt=dt)

        z = float(z)
        H = np.array([1.0, 0.0], dtype=float)
        R = self._R(z, depth_conf)
        innovation = z - float(H @ self.x)
        S = float(H @ self.P @ H.T) + R
        sigma = float(np.sqrt(max(S, 1e-12)))

        if abs(innovation) > self.gate_sigma * sigma:
            self.consec_rejects += 1
            self.reject_span += dt
            self.coast_age += dt
            # Both conditions, always: enough consecutive frames that this
            # cannot be one bad sample, AND enough elapsed time that it cannot
            # be a brief burst at a high frame rate.
            if (self.consec_rejects >= self.min_consec_rejects
                    and self.reject_span >= self.reject_window_s - REJECT_WINDOW_TOL_S):
                # Three disagreements in a row is a world change, not noise:
                # cut-in, new lead, or occlusion ending (§4, §5). Trust the
                # measurement over the model and restart -- NEW_LEAD forces the
                # state machine to re-confirm before it may warn again.
                self.initialise(z)
                self.new_lead = True
                return self._snapshot(accepted=True, reason="reset_new_lead", dt=dt,
                                      z=z, depth_conf=depth_conf,
                                      innovation=innovation, sigma=sigma, R=R)
            return self._snapshot(accepted=False, reason="gated", dt=dt, z=z,
                                  depth_conf=depth_conf, innovation=innovation,
                                  sigma=sigma, R=R)

        K = (self.P @ H) / S
        self.x = self.x + K * innovation
        self._clamp()
        I_KH = np.eye(2) - np.outer(K, H)
        # Joseph form: stays positive-definite under the repeated gating and
        # resets this filter does, where the short form can drift negative.
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R

        self.consec_rejects = 0
        self.reject_span = 0.0
        self.coast_age = 0.0
        self.n_updates += 1
        return self._snapshot(accepted=True, reason="updated", dt=dt, z=z,
                              depth_conf=depth_conf, innovation=innovation,
                              sigma=sigma, R=R)

    def _snapshot(self, accepted, reason, dt, z=None, depth_conf=None,
                  innovation=None, sigma=None, R=None) -> dict:
        return {
            "d": self.d,
            "d_dot": self.d_dot,
            "accepted": bool(accepted),
            "reason": reason,
            "z": None if z is None else round(float(z), 3),
            "depth_conf": None if depth_conf is None else round(float(depth_conf), 4),
            "innovation": None if innovation is None else round(float(innovation), 4),
            "sigma": None if sigma is None else round(float(sigma), 4),
            "R": None if R is None else round(float(R), 5),
            "consec_rejects": int(self.consec_rejects),
            "reject_span": round(float(self.reject_span), 4),
            "coast_age": round(float(self.coast_age), 4),
            "n_clamped": int(self.n_clamped),
            "new_lead": bool(self.new_lead),
            "P_dd": None if self.P is None else round(float(self.P[0, 0]), 5),
            "P_vv": None if self.P is None else round(float(self.P[1, 1]), 5),
            "dt": round(float(dt), 5),
        }


# ---------------------------------------------------------------------------
# Q tuning harness -- verifies the "3 m/s² braking lead within ~0.3 s" target
# ---------------------------------------------------------------------------
def tune_report(sigma_a=SIGMA_A, rate_hz=15.0, brake_a=3.0, d0=40.0, seed=0,
                depth_conf=0.8, brake_s=2.0, settle_s=2.0):
    """Simulate a 3 m/s² braking lead and measure how far behind ḋ runs.

    Against a *ramp* in ḋ (which is what constant braking is) a constant-velocity
    filter never converges to zero error -- it settles to a constant lag. So the
    meaningful reading of "tracks a 3 m/s² braking lead within ~0.3 s" is that
    steady-state lag, recovered as  lag = |ḋ_err| / a.  A 0.9 m/s error under
    3 m/s² braking *is* 0.3 s of lag.

    Returns dict with:
      lag_s      steady-state tracking lag in seconds  <- the spec target
      noise_std  1σ jitter of ḋ while the gap is steady (the cost of a big Q)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / rate_hz
    f = HeadwayFilter(sigma_a=sigma_a)
    d_true, v_true = float(d0), 0.0

    # Phase 1 -- steady gap. Lets the filter settle, and the spread of ḋ here is
    # the noise penalty a larger Q buys the responsiveness with.
    steady = []
    for i in range(int(settle_s * rate_hz)):
        z = d_true + rng.normal(0.0, max(R_ABS_M, R_REL_FRAC * d_true))
        snap = f.step(z, depth_conf, dt)
        if i > int(0.5 * rate_hz) and snap["d_dot"] is not None:
            steady.append(snap["d_dot"])
    noise_std = float(np.std(steady)) if steady else float("nan")

    # Phase 2 -- constant braking.
    errs, series = [], []
    n = int(brake_s * rate_hz)
    for i in range(n):
        t = i * dt
        v_true -= brake_a * dt
        d_true = max(d_true + v_true * dt, 1.0)
        z = d_true + rng.normal(0.0, max(R_ABS_M, R_REL_FRAC * d_true))
        snap = f.step(z, depth_conf, dt)
        est = snap["d_dot"] if snap["d_dot"] is not None else 0.0
        series.append((t, v_true, est))
        # Only the back half: the front half still contains the onset transient.
        if t >= brake_s * 0.5:
            errs.append(v_true - est)      # negative est lagging -> positive err

    mean_err = float(np.mean(errs)) if errs else float("nan")
    lag_s = abs(mean_err) / brake_a
    return {"lag_s": lag_s, "mean_err": mean_err, "noise_std": noise_std,
            "sigma_a": sigma_a, "rate_hz": rate_hz, "series": series}


def _main():
    ap = argparse.ArgumentParser(description="Kalman Q tuning check (design §4)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--sigma-a", type=float, default=SIGMA_A)
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--sweep", action="store_true", help="scan sigma_a values")
    args = ap.parse_args()

    if args.sweep:
        print(f"{'sigma_a':>9} {'lag(s)':>9} {'d_dot noise 1s(m/s)':>21}   target lag ~0.30 s")
        for s in (1.0, 3.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 250.0):
            r = tune_report(sigma_a=s, rate_hz=args.rate)
            flag = "  <-- closest" if abs(r["lag_s"] - 0.3) < 0.05 else ""
            print(f"{s:9.1f} {r['lag_s']:9.3f} {r['noise_std']:21.3f}{flag}")
        return

    r = tune_report(sigma_a=args.sigma_a, rate_hz=args.rate)
    print(f"sigma_a={args.sigma_a} rate={args.rate} Hz")
    print(f"  steady-state lag:   {r['lag_s']:.3f} s   (spec target ~0.3 s)")
    print(f"  mean d_dot error:   {r['mean_err']:.3f} m/s under {3.0} m/s² braking")
    print(f"  d_dot noise (1σ):   {r['noise_std']:.3f} m/s on a steady gap")


if __name__ == "__main__":
    _main()
