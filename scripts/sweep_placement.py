"""Object-placement sweep / single-candidate test harness for the exoglove
grasp tests.

This is the tool used to tune where the test object sits for each of the
myohand_glove_dev*.xml variants. It exists as a proper script (rather than
an ad-hoc snippet) because several hard-won correctness requirements are
baked into it, and re-deriving them by hand kept producing wrong numbers:

1. FRESH MODEL PER CANDIDATE, ONE CANDIDATE PER PROCESS.
   An earlier sweep reused one compiled MjModel across candidates and
   mutated model.geom_size (pillar height) between them. That produced
   ghost results that did not reproduce when a candidate was re-tested
   alone -- e.g. a placement reported as holding 2.664s genuinely never
   touches the hand at all. Always invoke this script once per candidate
   (sweep_placement.py sweep does exactly that, via subprocess).

2. STIFFENED CONTACT SOLVER.
   MyoHand's bone geoms ship with a soft contact model
   (solimp=[0.8, 0.8, 0.01] -- constant impedance 0.8, 10mm softening
   width). That is fine for gentle manipulation but is crushed straight
   through by EXO_FLEX (204.8 N) / EXO_THUMB (180 N) at full activation:
   measured penetration was 16.8mm into the gelatin box, 27.6mm into the
   soup can and 32.2mm into the tuna can -- bones passing bodily through
   the object ("transparent hand" in the viewer). Stiffening the solver
   takes that to 0.0mm. This is a contact-model fidelity setting, in the
   same category as the hand-joint damping boost -- NOT an
   EXO_THUMB/EXO_FLEX/EXO_EXT force, lengthrange, tendon or routing value.

3. NO CONTACT EXCLUDES, RING/PINKY PASSIVE BUT COLLIDING.
   The models no longer exclude the palm/metacarpals or the ring/pinky.
   Those excludes were a workaround for (2); with correct contact the ring
   and pinky rest against the object and act as a passive ulnar-side cage
   that stops it squirting out sideways under the pinch. Driving them
   actively (FDP4/FDP5/FDS4/FDS5) was tested and is worse -- see
   run_glove_dev_grasp_can.py's docstring for the numbers -- so ring_ctrl
   defaults to 0.

Usage:
  Single candidate (prints one result line):
    python sweep_placement.py test <model> <obj> <mass> <r> <halfh> \
        <x> <y> <z> [--pillar-top Z] [--ring F] [--hold F] [--window S]

  Sweep (spawns one subprocess per candidate, prints ranked successes):
    python sweep_placement.py sweep <model> <obj> <mass> <r> <halfh> \
        --xs 0.04,0.06,0.09 --ys 0.04,0.08 --zs 0.25 \
        [--bottom-offset -0.044] [--ring F] [--hold F] [--window S]

`--pillar-top` / `--bottom-offset` exist because the pillar must end exactly
at the object's BOTTOM face. For the cans the collision geom is
(0,0,half_h) with size half_h, so the bottom is body_pos.z and no offset is
needed. For the standing gelatin box the 90deg rotation puts the bottom
0.044 below body_pos.z, so pass --bottom-offset -0.044.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import mujoco
import numpy as np

ARM_JOINTS = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
ARM_KP = {"ARTx": 175, "ARTy": 175, "ARTz": 175, "ARRx": 150, "ARRy": 150, "ARRz": 150}
ARM_STIFF_BOOST = 8.0
ARM_KV_RATIO = 0.3
OP_CTRL = 0.5
HAND_DAMPING_BOOST = 25.0
RAMP_SECONDS = 4.0
SETTLE_S = 0.15
RETRACT_OFFSET = np.array([0.5, 0.0, 0.0])
FAIL_ROT_DEG = 45.0
FAIL_DRIFT_M = 0.05
FULL_FORCE_THRESHOLD = 0.95
PRESETTLE_STEPS = 6000
RING_MUSCLES = ["FDP4", "FDP5", "FDS4", "FDS5"]

HAND_BODY_NAMES = [
    "firstmc", "proximal_thumb", "distal_thumb",
    "secondmc", "proxph2", "midph2", "distph2",
    "thirdmc", "proxph3", "midph3", "distph3",
]
CHANNEL_BODIES = {
    "flex": {"secondmc", "proxph2", "midph2", "distph2",
             "thirdmc", "proxph3", "midph3", "distph3"},
    "thumb": {"firstmc", "proximal_thumb", "distal_thumb"},
}


def quat_dev_deg(q0, q):
    return np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q0, q))))))


def run_one(model_path, obj, mass, radius, half_h, x, y, z,
            pillar_top, ring_ctrl, hold_frac, window_s):
    """Run the full retract-and-hold protocol for ONE placement.

    Returns (status, first_touch_s, hold_s). Builds the model fresh; callers
    must not reuse anything across candidates (see module docstring).
    """
    m = mujoco.MjModel.from_xml_path(model_path)
    oid = m.body(obj).id

    # Object mass/inertia override: every YCB body here has a visual mesh geom
    # with no explicit mass, so MuJoCo silently adds its own mesh-volume mass
    # on top of the collision geom's stated mass. Override to the intended
    # collision primitive only (cylinder formulas; the gelatin box is close
    # enough in-plane that the same call signature is reused for it).
    full_h = 2 * half_h
    i_radial = mass * (3 * radius ** 2 + full_h ** 2) / 12
    m.body_mass[oid] = mass
    m.body_inertia[oid] = [i_radial, i_radial, 0.5 * mass * radius ** 2]
    m.body_ipos[oid] = [0, 0, half_h]
    m.body_iquat[oid] = [1, 0, 0, 0]
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == oid:
            m.geom_friction[g] = [2.5, 0.02, 0.002]

    # (2) contact stiffening -- see module docstring.
    for g in range(m.ngeom):
        m.geom_solref[g] = [0.01, 1.0]
        m.geom_solimp[g] = [0.98, 0.999, 0.0005, 0.5, 2.0]

    for j in range(m.njnt):
        jn = m.joint(j).name
        if jn in ARM_JOINTS or jn.startswith("OBJ"):
            continue
        ds = m.jnt_dofadr[j]
        dc = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
              mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[m.jnt_type[j]]
        for d_ in range(ds, ds + dc):
            m.dof_damping[d_] *= HAND_DAMPING_BOOST

    for j in ARM_JOINTS:
        a = m.actuator(f"A_{j}").id
        kp = ARM_KP[j] * ARM_STIFF_BOOST
        m.actuator_gainprm[a][0] = kp
        m.actuator_biasprm[a][1] = -kp
        m.actuator_biasprm[a][2] = -kp * ARM_KV_RATIO

    pillar_mocap = m.body("glove_dev_support_pillar").mocapid[0]
    pillar_geom = m.geom("glove_dev_support_pillar_geom").id
    # Pillar sits on the table (z=0) and its top must meet the object's bottom.
    pillar_half = pillar_top / 2.0
    m.geom_size[pillar_geom] = [0.08, 0.08, pillar_half]

    # Move the object by rewriting body_pos directly, NOT by offsetting the
    # OBJT* slide joints. Real bug this avoids: those joints' axes are
    # expressed in the BODY frame, and the gelatin box carries
    # euler="-1.5708 0 0" -- so a "+y" slide offset actually moved it along
    # world -z, silently scrambling every swept coordinate for any rotated
    # object. (The cans are unrotated, so they were unaffected.) Setting
    # body_pos is rotation-independent and leaves the joints at zero.
    m.body_pos[oid] = [x, y, z]

    d = mujoco.MjData(m)
    pillar_pos = np.array([x, y, pillar_half])
    d.mocap_pos[pillar_mocap] = pillar_pos

    for j in ARM_JOINTS:
        d.ctrl[m.actuator(f"A_{j}").id] = 0.0
    d.ctrl[m.actuator("OP").id] = OP_CTRL
    for n in RING_MUSCLES:
        d.ctrl[m.actuator(n).id] = ring_ctrl

    mujoco.mj_forward(m, d)
    for _ in range(PRESETTLE_STEPS):
        mujoco.mj_step(m, d)

    flex = m.actuator("EXO_FLEX").id
    thumb = m.actuator("EXO_THUMB").id
    obj_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == oid]
    hand_ids = {m.body(n).id for n in HAND_BODY_NAMES}
    chan_ids = {c: {m.body(n).id for n in v} for c, v in CHANNEL_BODIES.items()}

    inc = m.opt.timestep / RAMP_SECONDS
    settle_steps = int(SETTLE_S / m.opt.timestep)
    post_steps = int(window_s / m.opt.timestep)
    thresh = min(FULL_FORCE_THRESHOLD, hold_frac)

    u = {"flex": 0.0, "thumb": 0.0}
    first_touch = settle_start = retract = fail = None
    pos_at_retract = quat_at_retract = None

    step = 0
    max_steps = PRESETTLE_STEPS + post_steps + 40000
    while step < max_steps:
        touched = {"flex": False, "thumb": False}
        for i in range(d.ncon):
            c = d.contact[i]
            other = None
            if c.geom1 in obj_geoms and m.geom_bodyid[c.geom2] in hand_ids:
                other = m.geom_bodyid[c.geom2]
            elif c.geom2 in obj_geoms and m.geom_bodyid[c.geom1] in hand_ids:
                other = m.geom_bodyid[c.geom1]
            if other is None:
                continue
            for ch, bodies in chan_ids.items():
                if other in bodies:
                    touched[ch] = True

        for ch in ("flex", "thumb"):
            u[ch] = min(u[ch] + inc, hold_frac)
        d.ctrl[flex] = u["flex"]
        d.ctrl[thumb] = u["thumb"]
        mujoco.mj_step(m, d)

        if (touched["flex"] or touched["thumb"]) and first_touch is None:
            first_touch = step

        # Retract only on a genuine, currently-live two-channel grip that has
        # also finished ramping -- a historical "has touched" latch is not the
        # same as "is touching", and retracting mid-ramp pulls the support
        # before the grip has developed real force.
        full = (touched["flex"] and touched["thumb"]
                and u["flex"] >= thresh and u["thumb"] >= thresh)
        if full and retract is None:
            if settle_start is None:
                settle_start = step
            if step - settle_start >= settle_steps:
                d.mocap_pos[pillar_mocap] = pillar_pos + RETRACT_OFFSET
                retract = step
                pos_at_retract = d.xpos[oid].copy()
                quat_at_retract = d.xquat[oid].copy()
        elif not full and retract is None:
            settle_start = None

        if retract is not None and fail is None:
            if (quat_dev_deg(quat_at_retract, d.xquat[oid]) > FAIL_ROT_DEG
                    or np.linalg.norm(d.xpos[oid] - pos_at_retract) > FAIL_DRIFT_M):
                fail = step

        if retract is not None and step - retract >= post_steps:
            break
        step += 1

    dt = m.opt.timestep
    if first_touch is None:
        return "never_touched", None, None
    if retract is None:
        return "never_retracted", first_touch * dt, None
    if fail is None:
        return "held_full", first_touch * dt, (step - retract) * dt
    return "failed", first_touch * dt, (fail - retract) * dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "sweep"])
    ap.add_argument("model")
    ap.add_argument("obj")
    ap.add_argument("mass", type=float)
    ap.add_argument("radius", type=float)
    ap.add_argument("half_h", type=float)
    ap.add_argument("x", type=float, nargs="?")
    ap.add_argument("y", type=float, nargs="?")
    ap.add_argument("z", type=float, nargs="?")
    ap.add_argument("--pillar-top", type=float, default=None)
    ap.add_argument("--bottom-offset", type=float, default=0.0,
                    help="object bottom relative to body_pos.z (e.g. -0.044 "
                         "for the standing gelatin box)")
    ap.add_argument("--ring", type=float, default=0.0)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--xs")
    ap.add_argument("--ys")
    ap.add_argument("--zs")
    a = ap.parse_args()

    if a.mode == "test":
        top = a.pillar_top if a.pillar_top is not None else a.z + a.bottom_offset
        status, ft, hold = run_one(a.model, a.obj, a.mass, a.radius, a.half_h,
                                   a.x, a.y, a.z, top, a.ring, a.hold, a.window)
        ft_s = f"{ft:.3f}" if ft is not None else "-"
        hd_s = f"{hold:.3f}" if hold is not None else "-"
        print(f"({a.x},{a.y},{a.z}) {status} first_touch={ft_s} hold={hd_s}")
        return

    xs = [float(v) for v in a.xs.split(",")]
    ys = [float(v) for v in a.ys.split(",")]
    zs = [float(v) for v in a.zs.split(",")]
    results = []
    for z in zs:
        for x in xs:
            for y in ys:
                # One SUBPROCESS per candidate -- see docstring point (1).
                cmd = [sys.executable, __file__, "test", a.model, a.obj,
                       str(a.mass), str(a.radius), str(a.half_h),
                       str(x), str(y), str(z),
                       "--bottom-offset", str(a.bottom_offset),
                       "--ring", str(a.ring), "--hold", str(a.hold),
                       "--window", str(a.window)]
                out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
                print(out, flush=True)
                if " held_full " in out or " failed " in out:
                    try:
                        results.append((float(out.rsplit("hold=", 1)[1]), x, y, z,
                                        "held_full" in out))
                    except (IndexError, ValueError):
                        pass
    print("=" * 60)
    if not results:
        print("NO WORKING PLACEMENTS FOUND")
        return
    results.sort(reverse=True)
    print("RANKED:")
    for hold, x, y, z, fullwin in results[:12]:
        print(f"  ({x},{y},{z}) hold={hold:.3f}s{'  [FULL WINDOW]' if fullwin else ''}")


if __name__ == "__main__":
    main()
