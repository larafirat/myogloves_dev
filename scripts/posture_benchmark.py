"""Posture benchmark -- part 2 of the evaluation plan, alongside hold_benchmark.py.

hold_benchmark.py answers "can it keep hold of the object?". That is a necessary
metric but not a sufficient one for a rehabilitation device: a glove that
achieves a secure grip by dragging the hand into an anatomically wrong shape is
not a good rehabilitation device, and the hold metric cannot see the difference.
This script measures the shape.

Two metrics, because they answer different questions.

1. ROM SIMILARITY (mode `rom`) -- free-air closing, no object.

   This one is deliberately a REPRODUCTION of a published protocol rather than
   an invented metric, so the numbers can be checked against reality. Zhao et
   al. 2025 (D1's own paper), Sec. 3.2 and Eq. 1:

       Similarity = (joint angle wearing exoskeleton)
                  / (joint angle without exoskeleton) x 100%

   Their protocol maps onto this simulation almost exactly. Subjects were told
   to keep their fingers "completely relaxed during the exoskeleton-assisted
   movement, allowing for purely passive movement", then to "perform natural
   flexion and extension movements" unassisted -- i.e. device-driven passive
   hand vs. muscle-driven active hand, which is precisely the difference
   between the KMatrix adapters and HealthyHandAdapter. Measured in free air
   (their Fig. 9), so the object is decoupled here too.

   Published reference values for D1, index finger: PIP 74.19%, DIP 59.02%,
   MCP 41.67%. If this model of D1 is any good, it should land near those --
   and that is a check no other metric in this repo provides, since every other
   number here is self-referential.

   RESULT OF THAT CHECK (8 trials, corrected five-motor D1, box rig):
       joint      simulated     published
       index MCP     96.2%        41.67%
       index PIP     98.5%        74.19%
       index DIP    461.3%        59.02%
   The simulation overestimates delivered ROM by roughly 1.3x at the PIP and
   2.3x at the MCP, and is off the scale at the DIP. This is not a tuning
   failure, it is a structural limit of the whole K-matrix abstraction every
   device in this repo is built on: a K matrix is an IDEAL torque source
   applied directly to the joint, with no strap compliance, no linkage
   friction, and no slack. Zhao et al. name exactly those losses as the reason
   their own numbers are low -- "the space occupied by the Velcro used to
   secure the exoskeleton on the hand, as well as the limitations of the
   underactuated structure's force transmission, which reduces the effective
   range of motion". None of that exists here.

   The practical consequence is that every device in this repo, including
   Tyrone's, is being simulated at its theoretical best rather than its
   as-worn performance, and by a factor large enough to matter. Since the bias
   applies to all of them it should not reorder the comparison, but any
   absolute claim ("this glove restores N% of hand function") is not supported
   by these numbers. Fixing it properly means adding a transmission-loss term
   between the device and the joint, which no device definition here has.

   SECOND THING THIS MODE CAUGHT -- D4's thumb opposition lands in the wrong
   basin. Measured free-air thumb abduction excursion, against a healthy
   reference of -16.7 deg (negative = opposing, toward the fingers):

       D4 as currently tuned            +28.5 deg   (abducting AWAY)
       D4 with the flexion gate delayed  -4.4 deg   (opposing, but weak)

   The device definition is not at fault -- driven with raw torque and no
   controller, D4's thumb reaches -34.8 deg, correctly opposed. The problem is
   recruitment ORDER. Tracing the joint through the ramp, abduction leads
   correctly at first (-3.8 deg at t=1s), then thumb flexion arrives with
   roughly twice the torque (0.098 and 0.122 N*m at MP and IP, against 0.053
   for abduction) and drags the CMC saddle joint positive, ending at +27 deg.
   The abduct_lead/abduct_full parameters exist precisely to hold flexion back
   while abduction leads, but they were tuned when abduction shared the thumb's
   flexion channel; now that it has its own channel both ramp together, so
   flexion gates in before abduction has won the joint.

   Raising abduct_lead 0.24 -> 0.40 flips the sign, at the cost of MP flexion
   (-38.5 -> -28.1 deg). Both timings are kept as separate devices
   (D4_v2_hybrid_per_finger_calibrated and D4_v2_opposition_first_calibrated)
   and both were measured on both benchmarks. RESOLUTION -- it is not a trade,
   it is CONDITIONAL on who supplies the opposition:

     device must oppose for itself (no OP pre-shape)
       shipped timing      fidelity 60.3%   grasp 53.3%   hold 0.14s
       opposition-first    fidelity 72.3%   grasp 60.0%   hold 0.15s
     opposition borrowed from the OP muscle
       shipped timing      fidelity 68.3%                 hold 1.01s
       opposition-first    fidelity 65.8%                 hold 0.95s

   Standing on its own the opposition-first timing is better on every axis at
   once -- +12 points of posture fidelity, +6.7pp grasp rate, hold unchanged.
   There is nothing to trade away. With the OP pre-shape it is mildly worse on
   both, and the reason is mechanical rather than a tuning accident: OP already
   drives cmc_abduction to its joint LIMIT (settled -32.2 deg against a -28.6
   deg range floor), so there is no abduction left for the device's own
   opposition motor to contribute. Delaying thumb flexion then buys nothing and
   only costs MP range.

   So the honest reading is that the shipped timing is not tuned for the
   device, it is tuned for the pre-shape propping it up. Which is the same
   finding this project keeps arriving at from different directions: the OP
   assumption dominates, and it masks what the devices themselves do.

   Note the asymmetry this exposes between the two benchmarks: the hold
   benchmark cannot see any of this, because a thumb abducting the wrong way
   still contacts the object and still counts.

2. GRASP POSTURE ERROR (mode `grasp`) -- object present, posture sampled at the
   moment the contact gate passes.

   How far the assisted grasp's joint vector sits from the healthy hand's
   grasp of the SAME object at the SAME placement, in RMS degrees. This is the
   metric HealthyHandAdapter's docstring already anticipated ("the joint
   trajectory recorded here is x_ref(t)"). Unlike ROM similarity it is
   object-conditioned: a device can have poor free-air ROM but still arrive at
   a reasonable grasp shape once contact does the shaping for it, and the two
   metrics disagreeing is informative rather than contradictory.

Usage:
    python myogloves_dev/scripts/posture_benchmark.py rom
    python myogloves_dev/scripts/posture_benchmark.py grasp --trials 10
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

from hold_benchmark import (
    GATE_MIN_DIGITS, GATE_MIN_SECONDS, SETTLE_QUIET_STEPS, SETTLE_VEL_EPS,
    WristHold, _digits_in_contact, glove_dev_adapters, settle,
)

# Joints reported, grouped by digit. Names are MyoHand's; the MCP/PIP/DIP
# labels are the ones Zhao et al. use, so the index row is directly comparable
# to their published similarity figures.
DIGIT_JOINTS = {
    "index":  [("MCP", "mcp2_flexion"), ("PIP", "pm2_flexion"), ("DIP", "md2_flexion")],
    "middle": [("MCP", "mcp3_flexion"), ("PIP", "pm3_flexion"), ("DIP", "md3_flexion")],
    "ring":   [("MCP", "mcp4_flexion"), ("PIP", "pm4_flexion"), ("DIP", "md4_flexion")],
    "little": [("MCP", "mcp5_flexion"), ("PIP", "pm5_flexion"), ("DIP", "md5_flexion")],
    "thumb":  [("ABD", "cmc_abduction"), ("CMC", "cmc_flexion"),
               ("MP", "mp_flexion"), ("IP", "ip_flexion")],
}
ALL_JOINTS = [(dig, lbl, jnt) for dig, js in DIGIT_JOINTS.items() for lbl, jnt in js]

CLOSING_SECONDS = 1.5   # same ramp the hold benchmark uses, so postures are comparable
FREE_AIR_SECONDS = 4.0  # ramp + settle time for the free-air excursion to plateau

# A ratio metric needs a denominator worth dividing by. Several joints barely
# move even in the healthy reference -- the index DIP travels ~18 deg and the
# middle/little DIPs under 6 deg, because MyoHand's FDS pulls against FDP at
# the distal joint -- and dividing by those turns a couple of degrees of
# simulation noise into similarities in the hundreds of percent. Joints whose
# reference excursion is below this threshold are reported but NOT scored.
# (For context on whether the reference itself is sane where it does move:
# healthy index MCP ~79 deg and PIP ~88 deg here, against 80 deg and 120 deg
# measured on a human in Zhao et al.'s own Fig. 2. The DIP is where this model
# and their subject genuinely part company, ~18 deg vs ~40 deg, which is
# exactly why the DIP rows are excluded rather than trusted.)
MIN_REF_EXCURSION_DEG = 15.0


def scorable(ref_excursion):
    return abs(ref_excursion) >= MIN_REF_EXCURSION_DEG


# A device that does not drive a joint at all and a device that drives it badly
# are different failures, and averaging them into one number hides both. D3 is
# the case that forces the issue: it is a two-finger device whose thumb is
# deliberately splinted, so scoring its motionless thumb against a healthy
# moving one reads as poor fidelity when it is actually a design choice, and
# dragged its mean to 29.6% -- last place, for doing exactly what it should.
#
# So each device is reported as two numbers over the scorable joints:
#   COVERAGE  how many it drives at all           (breadth of the design)
#   FIDELITY  how naturally it moves THOSE        (quality where it acts)
# Neither alone is a verdict: full coverage with poor fidelity is a glove that
# moves everything wrongly, and high fidelity on two joints is a pinch aid.
MIN_LEVERAGE_MM = 0.5   # tendon moment arm below this is numerical, not drive


def driven_joints(model, adapter):
    """Joint names this device actually actuates, and how it knows.

    Two device families need two answers, and neither can be guessed from the
    resulting motion (that would be circular -- the motion is what is being
    scored):

      K-matrix devices (D1-D4)  a joint is driven if its K row is non-zero for
                                some channel. This is the device's own
                                declaration of what it drives.
      Tyrone's glove            it has no K matrix, it has real routed tendons.
                                A joint is driven if one of the exo tendons has
                                meaningful leverage over it, measured the same
                                way MOMENT_ARMS_MM was. Result: index and
                                middle in full, plus cmc_abduction and
                                mp_flexion -- 9 of 16. Notably its abduction
                                moment arm is 13.7 mm, roughly 4x the FPL's
                                3.9 mm, so the design is weighted hard toward
                                thumb opposition. It does not drive cmc_flexion
                                (0.06 mm), ip_flexion, or the ulnar digits.

    Splinted joints are returned separately: they are neither driven nor
    ignored, they are deliberately immobilised, and they belong in neither
    average.
    """
    from exo_devices import JOINT_NAMES

    splinted = set()
    if hasattr(adapter, "SPLINT_POSE"):
        splinted = set(adapter.SPLINT_POSE)

    device = getattr(adapter, "device", None)
    if device is not None:
        driven = {j for row, j in enumerate(JOINT_NAMES)
                  if np.any(device.K[row, :] != 0.0)}
        return driven - splinted, splinted

    driven = set()
    data = mujoco.MjData(model)
    for tname in ("exo_flex_tendon", "exo_thumb_tendon", "exo_ext_tendon"):
        try:
            tid = model.tendon(tname).id
        except KeyError:
            continue
        for jname in JOINT_NAMES:
            try:
                qadr = model.joint(jname).qposadr[0]
            except KeyError:
                continue
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            l0 = data.ten_length[tid]
            data.qpos[qadr] += 1e-4
            mujoco.mj_forward(model, data)
            if abs((data.ten_length[tid] - l0) / 1e-4 * 1000.0) > MIN_LEVERAGE_MM:
                driven.add(jname)
    return driven - splinted, splinted


def similarity(dev_excursion, ref_excursion):
    """Zhao et al.'s Eq. 1 as a percentage, or NaN where it is not meaningful.

    Opposite-signed motion is rejected rather than reported as a negative
    percentage: a device that drives a joint the WRONG WAY has not achieved
    "-33% of the natural range", it has failed to reproduce the motion at all,
    and letting a negative number into a mean would let a wrong-direction joint
    cancel out a correct one.
    """
    if not scorable(ref_excursion):
        return float("nan")
    if dev_excursion * ref_excursion <= 0.0:
        return 0.0
    return 100.0 * dev_excursion / ref_excursion


def match(dev_excursion, ref_excursion):
    """Symmetric agreement, 0-100%, used for the ACROSS-JOINT aggregate.

    Eq. 1 is kept per-joint because it is the published quantity and the point
    of this mode is to be checkable against published numbers. It is the wrong
    thing to average, though: it is unbounded above, so a joint the device
    over-flexes to 460% of natural does not read as "badly wrong" in a mean,
    it reads as a large bonus that can drag a whole device's score above 100%
    and hide genuine deficits elsewhere. Under- and over-shooting by the same
    factor should cost the same, which is what min/max gives.
    """
    if not scorable(ref_excursion):
        return float("nan")
    if dev_excursion * ref_excursion <= 0.0:
        return 0.0
    lo, hi = sorted((abs(dev_excursion), abs(ref_excursion)))
    return 100.0 * lo / hi


def _joint_qadr(model):
    """qpos addresses for the reported joints, skipping any the model lacks."""
    adr = {}
    for _, _, jnt in ALL_JOINTS:
        try:
            adr[jnt] = model.joint(jnt).qposadr[0]
        except KeyError:
            continue
    return adr


def _decouple_object(model, adapter):
    """Free-air mode: make the object and its support pillar non-colliding.

    Zeroing contype/conaffinity rather than moving the bodies away is the safer
    of the two options -- the graspable object sits on slide joints whose axes
    are expressed in the BODY frame, and for the rotated gelatin box those axes
    do not line up with the world ones (a bug this repo has already been bitten
    by once). Killing the contact bitmasks needs no coordinates at all.

    Once decoupled the object simply falls away, which is harmless but does
    mean hold_benchmark's settle() cannot be reused here -- it treats a falling
    object as SETUP_FAIL. See settle_hand_only below.
    """
    for g in adapter.obj_geoms:
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = 0
    try:
        pg = model.geom("glove_dev_support_pillar_geom").id
        model.geom_contype[pg] = 0
        model.geom_conaffinity[pg] = 0
    except KeyError:
        pass


def settle_hand_only(model, data, obj_dofs, cap_steps):
    """Free-air settle: wait for the HAND to go quiet, ignoring the object.

    hold_benchmark.settle() cannot be used in free-air mode for two reasons,
    both caused by the decoupled object falling: it trips the drop check
    (scored as SETUP_FAIL), and its ever-growing fall velocity means the
    whole-system quiescence test never passes either. Masking the object's own
    DoFs out of the velocity check fixes both, and leaves the hand-side
    criterion identical to the shared one so postures stay comparable.
    """
    mask = np.ones(model.nv, dtype=bool)
    mask[obj_dofs] = False
    quiet = 0
    for s_ in range(cap_steps):
        mujoco.mj_step(model, data)
        if float(np.abs(data.qvel[mask]).max()) < SETTLE_VEL_EPS:
            quiet += 1
            if quiet >= SETTLE_QUIET_STEPS:
                return s_ + 1, True
        else:
            quiet = 0
    return cap_steps, True


def _object_dofs(model, obj_body_id):
    dofs = []
    for j in range(model.njnt):
        if model.jnt_bodyid[j] != obj_body_id:
            continue
        n = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
             mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[model.jnt_type[j]]
        dofs.extend(range(model.jnt_dofadr[j], model.jnt_dofadr[j] + n))
    return dofs


def rom_trial(adapter, seed):
    """Free-air closing. Returns per-joint excursion (deg) from the settled pose."""
    rng = np.random.default_rng(seed)
    m, d = adapter.build(rng)
    _decouple_object(m, adapter)
    dt = m.opt.timestep

    mujoco.mj_forward(m, d)
    if hasattr(adapter, "wrist"):
        adapter.wrist = WristHold(m, d)
    settle_hand_only(m, d, _object_dofs(m, adapter.oid), adapter.settle_cap)

    adr = _joint_qadr(m)
    start = {j: float(d.qpos[a]) for j, a in adr.items()}
    peak = dict(start)

    ramp_steps = max(1, int(CLOSING_SECONDS / dt))
    for step in range(int(FREE_AIR_SECONDS / dt)):
        adapter.set_input(m, d, min(step / ramp_steps, 1.0))
        mujoco.mj_step(m, d)
        for j, a in adr.items():
            q = float(d.qpos[a])
            # Track the largest EXCURSION in either direction, not the largest
            # positive angle: thumb opposition in this model is reached by
            # NEGATIVE cmc_abduction, so a max-only rule would score a
            # correctly opposing thumb as having moved nowhere.
            if abs(q - start[j]) > abs(peak[j] - start[j]):
                peak[j] = q

    return {j: np.degrees(peak[j] - start[j]) for j in adr}


def grasp_trial(adapter, seed):
    """Closing WITH the object. Returns the joint vector (deg) at the moment the
    contact gate passes, plus which digits were in contact. Returns None if the
    device never establishes a gated grasp -- posture is undefined without one,
    and substituting the end-of-ramp pose would quietly mix "grasped badly" in
    with "never grasped", which are different failures."""
    rng = np.random.default_rng(seed)
    m, d = adapter.build(rng)
    dt = m.opt.timestep

    mujoco.mj_forward(m, d)
    if hasattr(adapter, "wrist"):
        adapter.wrist = WristHold(m, d)
    _, ok = settle(m, d, adapter.oid, adapter.settle_cap)
    if not ok:
        return None

    adr = _joint_qadr(m)
    ramp_steps = max(1, int(CLOSING_SECONDS / dt))
    gate_steps_needed = int(GATE_MIN_SECONDS / dt)
    gate_run = 0

    for step in range(ramp_steps + int(6.0 / dt)):
        adapter.set_input(m, d, min(step / ramp_steps, 1.0))
        mujoco.mj_step(m, d)
        digits = _digits_in_contact(m, d, adapter)
        if len(digits) >= GATE_MIN_DIGITS:
            gate_run += 1
            if gate_run >= gate_steps_needed:
                return {"q": {j: np.degrees(float(d.qpos[a])) for j, a in adr.items()},
                        "digits": sorted(digits)}
        else:
            gate_run = 0
    return None


def median_over(trials, key=None):
    """Per-joint median across trials, ignoring failed ones."""
    good = [t for t in trials if t is not None]
    if not good:
        return None, 0
    vals = [t if key is None else t[key] for t in good]
    joints = vals[0].keys()
    return {j: float(np.median([v[j] for v in vals])) for j in joints}, len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rom", "grasp"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--reference", default="healthy_box")
    ap.add_argument("--devices", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pool = glove_dev_adapters()
    # splint_D3 rather than portOP_D3: the splint is the device as it actually
    # ships, and the OP-borrowing variant holds the thumb at the OPPOSITE end
    # of its travel (+40 vs -45 deg at the MP), so the two are not
    # interchangeable for a posture measurement in particular.
    default = ["glove_dev_box"] + [
        f"portOP_{d}" for d in ("D1_underactuated_distal_calibrated",
                                "D2_synergy_cross_finger_calibrated",
                                "D4_v2_hybrid_per_finger_calibrated")] + ["splint_D3"]
    names = [n for n in a.devices.split(",") if n] or default

    run = rom_trial if a.mode == "rom" else grasp_trial
    key = None if a.mode == "rom" else "q"

    ref_ad = pool[a.reference]
    ref, ref_n = median_over([run(ref_ad, 3000 + i) for i in range(a.trials)], key)
    if ref is None:
        raise SystemExit(f"reference {a.reference} produced no usable trials")

    print(f"\nREFERENCE  {ref_ad.name}  ({ref_n}/{a.trials} usable trials)")
    print(f"  {'joint':<10}{'deg':>8}   scored?")
    for dig, lbl, j in ALL_JOINTS:
        if j not in ref:
            continue
        mark = "yes" if scorable(ref[j]) else f"NO  (|ref| < {MIN_REF_EXCURSION_DEG:.0f} deg)"
        print(f"  {dig[:3]+'.'+lbl:<10}{ref[j]:8.1f}   {mark}")

    out = {"mode": a.mode, "reference": ref_ad.name, "reference_deg": ref,
           "min_ref_excursion_deg": MIN_REF_EXCURSION_DEG, "devices": {}}

    for name in names:
        ad = pool[name]
        res, n = median_over([run(ad, 3000 + i) for i in range(a.trials)], key)
        if res is None:
            print(f"\n{ad.name}: no usable trials ({a.trials} attempted)")
            out["devices"][ad.name] = {"usable": 0}
            continue

        if a.mode == "rom":
            probe_m, _ = ad.build(np.random.default_rng(0))
            driven, splinted = driven_joints(probe_m, ad)

            sim = {j: similarity(res[j], ref[j]) for j in res}
            mat = {j: match(res[j], ref[j]) for j in res}
            scorable_joints = [j for j in res if mat[j] == mat[j]]
            on_driven = [mat[j] for j in scorable_joints if j in driven]
            coverage = (100.0 * len([j for j in scorable_joints if j in driven])
                        / len(scorable_joints)) if scorable_joints else float("nan")
            fidelity = float(np.mean(on_driven)) if on_driven else float("nan")

            print(f"\n{ad.name}  ({n}/{a.trials} usable)")
            print(f"  COVERAGE {coverage:5.1f}%  ({len(on_driven)} of "
                  f"{len(scorable_joints)} scorable joints driven)"
                  + (f", {len(splinted)} splinted" if splinted else ""))
            print(f"  FIDELITY {fidelity:5.1f}%  (mean match on the joints it drives)")
            print(f"  {'joint':<10}{'excursion':>12}  {'Eq.1':>8}  {'match':>7}  role")
            for dig, lbl, j in ALL_JOINTS:
                if j not in res:
                    continue
                s = f"{sim[j]:7.1f}%" if sim[j] == sim[j] else "      --"
                mm = f"{mat[j]:6.1f}%" if mat[j] == mat[j] else "     --"
                role = ("splinted" if j in splinted else
                        "driven" if j in driven else "not driven")
                print(f"  {dig[:3]+'.'+lbl:<10}{res[j]:8.1f} deg  {s}  {mm}  {role}")
            out["devices"][ad.name] = {
                "usable": n, "excursion_deg": res, "similarity_pct": sim,
                "match_pct": mat, "driven": sorted(driven), "splinted": sorted(splinted),
                "coverage_pct": coverage if coverage == coverage else None,
                "fidelity_pct": fidelity if fidelity == fidelity else None}
        else:
            probe_m, _ = ad.build(np.random.default_rng(0))
            driven, splinted = driven_joints(probe_m, ad)
            err = {j: res[j] - ref[j] for j in res}
            rms_all = float(np.sqrt(np.mean([e ** 2 for e in err.values()])))
            on_driven = [err[j] for j in err if j in driven]
            rms_driven = (float(np.sqrt(np.mean([e ** 2 for e in on_driven])))
                          if on_driven else float("nan"))
            # Same reason the ROM mode splits coverage from fidelity: an error
            # at a joint the device never drives is the healthy hand moving,
            # not the device being wrong, and pooling the two makes a
            # narrow-but-accurate device look like a wide-but-sloppy one.
            print(f"\n{ad.name}  ({n}/{a.trials} usable)")
            print(f"  RMS posture error: {rms_driven:5.1f} deg on driven joints"
                  f"   ({rms_all:.1f} deg over all)")
            for dig, lbl, j in ALL_JOINTS:
                if j in res:
                    role = ("splinted" if j in splinted else
                            "driven" if j in driven else "not driven")
                    print(f"  {dig[:3]+'.'+lbl:<10}{res[j]:8.1f} deg   {err[j]:+7.1f}   {role}")
            out["devices"][ad.name] = {
                "usable": n, "posture_deg": res, "error_deg": err,
                "driven": sorted(driven), "splinted": sorted(splinted),
                "rms_error_deg": rms_all,
                "rms_error_driven_deg": rms_driven if rms_driven == rms_driven else None}

    if a.mode == "grasp":
        print("\n" + "=" * 62)
        print(f"{'device':<34}{'RMS(driven)':>13}{'RMS(all)':>11}")
        for nm, v in out["devices"].items():
            if v.get("rms_error_driven_deg") is None:
                continue
            print(f"{nm:<34}{v['rms_error_driven_deg']:11.1f} deg"
                  f"{v['rms_error_deg']:8.1f} deg")
        print("=" * 62)

    if a.mode == "rom":
        print("\n" + "=" * 62)
        print(f"{'device':<34}{'COVERAGE':>10}{'FIDELITY':>10}")
        for nm, v in out["devices"].items():
            if v.get("coverage_pct") is None:
                continue
            print(f"{nm:<34}{v['coverage_pct']:9.1f}%{v['fidelity_pct']:9.1f}%")
        print("=" * 62)
        print("COVERAGE = share of scorable joints the device drives at all")
        print("FIDELITY = how closely those joints match the healthy hand's motion")
        print("\nPublished reference for D1 (Zhao et al. 2025, Sec. 3.2, index finger):")
        print("  MCP 41.67%   PIP 74.19%   DIP 59.02%")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
