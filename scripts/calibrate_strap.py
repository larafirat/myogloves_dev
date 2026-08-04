"""Fit STRAP_STIFFNESS to published data instead of choosing it.

hold_benchmark.STRAP_STIFFNESS models the glove-to-bone interface as a series
spring (see its comment block). Its value is not a free knob: Zhao et al. 2025
published measured joint-angle similarity for D1's index finger, and this
script fits the single stiffness parameter that best reproduces those three
numbers in simulation.

    published   MCP 41.67%   PIP 74.19%   DIP 59.02%

One parameter, three observations, so the fit can fail -- and a residual that
refuses to come down would be telling us the series-spring model is the wrong
shape, not that the number needs more tuning. That is the point of fitting
rather than picking.

Uses the free-air ROM protocol from posture_benchmark, which is the same
protocol Zhao et al. used (relaxed hand, device-driven, no object).

Usage:
    python myogloves_dev/scripts/calibrate_strap.py
    python myogloves_dev/scripts/calibrate_strap.py --ks 0,0.02,0.05,0.1 --trials 5
"""

from __future__ import annotations

import argparse

import numpy as np

import hold_benchmark as hb
import posture_benchmark as pb

# Zhao et al. 2025, Sec. 3.2 -- index finger, exoskeleton-driven vs unassisted.
PUBLISHED = {"mcp2_flexion": 41.67, "pm2_flexion": 74.19, "md2_flexion": 59.02}
DEVICE = "portOP_D1_underactuated_distal_calibrated"


def similarity_triple(device_key, ref, trials, seed0=3000):
    ad = hb.glove_dev_adapters()[device_key]
    res, n = pb.median_over([pb.rom_trial(ad, seed0 + i) for i in range(trials)])
    if res is None:
        return None, 0
    return {j: 100.0 * res[j] / ref[j] for j in PUBLISHED}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="0,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.25")
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()

    # Healthy reference is the denominator of Eq. 1 and must be measured with
    # the strap OFF -- the healthy hand wears no device. Measured once and
    # reused, so every candidate stiffness is scored against the same baseline.
    hb.STRAP_STIFFNESS = 0.0
    ref, ref_n = pb.median_over(
        [pb.rom_trial(hb.glove_dev_adapters()["healthy_box"], 3000 + i)
         for i in range(a.trials)])
    print(f"healthy reference ({ref_n} trials): "
          + "  ".join(f"{j.split('_')[0]}={ref[j]:.1f}deg" for j in PUBLISHED))
    print(f"\n{'k (N*m/rad)':>12}   {'MCP':>7} {'PIP':>7} {'DIP':>7}   {'RMS err':>8}")
    print(f"{'published':>12}   " + "".join(f"{PUBLISHED[j]:6.1f}% "
          for j in ("mcp2_flexion", "pm2_flexion", "md2_flexion")) + "        --")

    best = None
    for k in [float(v) for v in a.ks.split(",")]:
        hb.STRAP_STIFFNESS = k
        sim, n = similarity_triple(DEVICE, ref, a.trials)
        if sim is None:
            print(f"{k:12.3f}   (no usable trials)")
            continue
        err = np.sqrt(np.mean([(sim[j] - PUBLISHED[j]) ** 2 for j in PUBLISHED]))
        flag = ""
        if best is None or err < best[0]:
            best, flag = (err, k, sim), "  <-- best so far"
        print(f"{k:12.3f}   " + "".join(f"{sim[j]:6.1f}% " for j in
              ("mcp2_flexion", "pm2_flexion", "md2_flexion"))
              + f"{err:8.1f}{flag}")

    if best:
        err, k, sim = best
        print(f"\nBEST FIT  STRAP_STIFFNESS = {k} N*m/rad   (RMS residual {err:.1f} points)")
        for j, lbl in (("mcp2_flexion", "MCP"), ("pm2_flexion", "PIP"), ("md2_flexion", "DIP")):
            print(f"  {lbl}  simulated {sim[j]:6.1f}%   published {PUBLISHED[j]:6.1f}%"
                  f"   delta {sim[j]-PUBLISHED[j]:+6.1f}")


if __name__ == "__main__":
    main()
