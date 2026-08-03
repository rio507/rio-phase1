"""drivecycle.py — the tire domain's drive cycles.

The implementation is diag/drivecycle.py. What is here is the binding: this
domain's store, and the two thresholds that decide when the fallback motion
heuristic thinks a drive began and ended.

Exactly one tire monitor requires a drive cycle — the slow leak, where a decline
measured inside a single drive is mostly measuring the drive — and nothing
urgent waits for one. A critically low tire that sat through three drives before
being mentioned would be a design failure, not diagnostic rigour.
"""
from __future__ import annotations

import config

from diag.drivecycle import DriveCycle, DriveCycleTracker as _BaseTracker

from . import store

__all__ = ["DriveCycle", "DriveCycleTracker"]


class DriveCycleTracker(_BaseTracker):
    """One tracker per process, wired to the tire store and tire thresholds."""

    def __init__(self):
        super().__init__(store, start_mph=config.TIRE_DIAG_DRIVE_START_MPH,
                         end_parked_s=config.TIRE_DIAG_DRIVE_END_PARKED_S,
                         id_prefix="drive")
