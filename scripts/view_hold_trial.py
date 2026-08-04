"""Watch one condition run the actual hold-benchmark trial, in the viewer.

This is the SAME protocol hold_benchmark.py scores, step for step -- it reuses
the adapters and the phase structure rather than re-implementing them, so what
you see is what the numbers came from. Phases are announced on stdout as they
happen:

    SETTLE   object resting on its support pillar, hand pre-shaped, everything
             damping out (this is silent and can look like nothing happening)
    CLOSE    device ramps its input 0 -> 1; the contact gate needs 2+ digits
             touching for a continuous 1.0 s before the grasp counts
    HOLD     support pillar yanked away; the clock that produces "hold_med"
             starts here and stops when the object falls 5 cm

Usage (macOS needs mjpython for any MuJoCo viewer):
    mjpython myogloves_dev/scripts/view_hold_trial.py splint_D3
    mjpython myogloves_dev/scripts/view_hold_trial.py glove_dev_box --seed 1003

Condition names are hold_benchmark's adapter keys; run with no arguments to
list them.
"""

from __future__ import annotations

import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from hold_benchmark import (
    DROP_THRESHOLD_M, GATE_MIN_DIGITS, GATE_MIN_SECONDS, T_MAX_DEFAULT,
    WristHold, _digits_in_contact, glove_dev_adapters, settle,
)


def main():
    pool = glove_dev_adapters()
    ap = argparse.ArgumentParser()
    ap.add_argument("condition", nargs="?", help="adapter key; omit to list")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    a = ap.parse_args()

    if not a.condition:
        print("conditions:")
        for k, ad in pool.items():
            print(f"  {k:52s} {ad.name}")
        return
    if a.condition not in pool:
        sys.exit(f"unknown condition {a.condition!r}; run with no arguments to list")

    ad = pool[a.condition]
    rng = np.random.default_rng(a.seed)
    m, d = ad.build(rng)
    dt = m.opt.timestep

    mujoco.mj_forward(m, d)
    if hasattr(ad, "wrist"):
        ad.wrist = WristHold(m, d)

    print(f"\n{ad.name}   seed={a.seed}   pre-shape: {ad.preshape}")
    print("=" * 68)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        def run(steps, fn, label):
            """Step with the viewer synced and paced. fn returns True to stop early."""
            print(label, flush=True)
            for step in range(steps):
                if not viewer.is_running():
                    return None
                if fn(step):
                    return step
                mujoco.mj_step(m, d)
                viewer.sync()
                time.sleep(dt / max(a.speed, 1e-6))
            return None

        # SETTLE -- run headless first. Settling is thousands of quiet steps and
        # watching it in real time is just a stationary hand; the benchmark
        # treats it as setup, not as part of the trial.
        print("SETTLE   (running headless -- this is setup, not the trial)")
        _, ok = settle(m, d, ad.oid, ad.settle_cap)
        viewer.sync()
        if not ok:
            print("SETUP_FAIL: object left its support before the trial began")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
            return
        time.sleep(1.0)  # a beat to look at the start pose

        ramp_steps = max(1, int(ad.closing_seconds / dt))
        gate_needed = int(GATE_MIN_SECONDS / dt)
        state = {"run": 0, "gated": False, "peak": 0}

        def closing(step):
            ad.set_input(m, d, min(step / ramp_steps, 1.0))
            digits = _digits_in_contact(m, d, ad)
            state["peak"] = max(state["peak"], len(digits))
            if len(digits) >= GATE_MIN_DIGITS:
                state["run"] += 1
                if state["run"] >= gate_needed:
                    state["gated"] = True
                    print(f"         grasp gated at t={step*dt:.2f}s "
                          f"({', '.join(sorted(digits))})")
                    return True
            else:
                state["run"] = 0
            return False

        run(ramp_steps + int(6.0 / dt), closing,
            f"CLOSE    ramping input over {ad.closing_seconds}s; "
            f"need {GATE_MIN_DIGITS}+ digits for {GATE_MIN_SECONDS}s")

        if not viewer.is_running():
            return
        if not state["gated"]:
            print(f"NO_GRASP peak digits in contact: {state['peak']}")
        else:
            ad.release_support(m, d)
            release_z = float(d.xpos[ad.oid][2])
            dropped = {"at": None}

            def holding(step):
                ad.set_input(m, d, 1.0)
                if release_z - float(d.xpos[ad.oid][2]) > DROP_THRESHOLD_M:
                    dropped["at"] = step * dt
                    return True
                return False

            run(int(T_MAX_DEFAULT / dt), holding,
                "HOLD     support removed -- clock running")
            if dropped["at"] is not None:
                print(f"SLIPPED  held {dropped['at']:.2f}s")
            elif viewer.is_running():
                print(f"SURVIVED full {T_MAX_DEFAULT}s (censored)")

        print("=" * 68)
        print("trial over -- window stays open, close it when done")
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    main()
