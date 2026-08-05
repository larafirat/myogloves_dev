"""Find placements where an UNASSISTED hand can grasp, per object.

Ordering matters and was got backwards for the box. That placement was found
by optimising a device (D4), and the healthy control was only checked
afterwards. The can inherited the same approach and shows the consequence: at
its authored placement the healthy hand grasps 0 of 15 while Tyrone's glove
manages 86.7%, i.e. the position is tuned to one device rather than being a
fair place to test any. The tuna is worse -- nothing grasps there at all.

So: search with the healthy hand FIRST, adopt a placement it can actually use,
then measure devices there.

Ranked by GRASP RATE, not median hold. hold_median is conditioned on having
grasped (summarise() medians over grasped trials only), so ranking by it
rewards placements that fail often and hold well in the few they win -- a trap
this repo already fell into once, documented at hold_benchmark's placesweep.
"""

import argparse
import itertools
import json
import sys

import hold_benchmark as hb

GRIDS = {
    # around each object's current placement
    "box":  dict(xs=[0.05, 0.07, 0.09], ys=[0.08, 0.10, 0.12], zs=[0.200, 0.220, 0.240]),
    "can":  dict(xs=[0.06, 0.08, 0.10], ys=[0.08, 0.10, 0.12], zs=[0.155, 0.170, 0.185]),
    "tuna": dict(xs=[0.03, 0.05, 0.07], ys=[0.04, 0.06, 0.08], zs=[0.145, 0.160, 0.175]),
}
BOTTOM_OFFSET = {"box": -0.044, "can": 0.0, "tuna": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objects", nargs="*", default=["box", "can", "tuna"])
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out", default="myogloves_dev/placement_feasibility.json")
    a = ap.parse_args()

    out = {}
    for obj in a.objects:
        g = GRIDS[obj]
        ad = hb.glove_dev_adapters()[f"healthy_{obj}"]
        ranked = []
        print(f"\n=== {obj}: searching {len(g['xs'])*len(g['ys'])*len(g['zs'])} placements "
              f"with the HEALTHY hand ({a.trials} trials each)", flush=True)
        for z, x, y in itertools.product(g["zs"], g["xs"], g["ys"]):
            ad.pos_override = (x, y, z)
            res = [hb.run_trial(ad, seed=2000 + i, t_max=hb.T_MAX_DEFAULT)
                   for i in range(a.trials)]
            s = hb.summarise(res)
            # Rank by grasp rate, then PENALISE setup failures: at the
            # reference friction the object can slide off its support during
            # settling, which the legacy grip was hiding. A placement that
            # cannot even hold the object still is not a placement.
            ranked.append((s["grasp_rate"] - 0.5 * s["setup_fail"] / max(a.trials, 1),
                           s["hold_median"], x, y, z, s["setup_fail"]))
            print(f"  ({x:.3f},{y:.3f},{z:.3f})  grasp={s['grasp_rate']*100:5.1f}%  "
                  f"hold={s['hold_median']:5.2f}s"
                  + (f"  SETUPFAIL={s['setup_fail']}" if s["setup_fail"] else ""), flush=True)
        ranked.sort(reverse=True)
        out[obj] = [{"grasp": g_, "hold": h, "pos": [x, y, z], "setup_fail": sf}
                    for g_, h, x, y, z, sf in ranked]
        print(f"  >>> best for {obj}: grasp={ranked[0][0]*100:.1f}% "
              f"hold={ranked[0][1]:.2f}s at {ranked[0][2:5]}", flush=True)
        json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
