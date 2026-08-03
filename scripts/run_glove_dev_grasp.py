"""Grasp-hold test for myohand_glove_dev.xml (the muscle-tendon EXO_THUMB/
EXO_FLEX/EXO_EXT exoglove design), mirroring the retract-and-hold protocol used
for the other exoskeleton devices in this project (run_grasp_baseline.py):
ramp the exo actuators up, wait for a settled grip, retract the temporary
support pillar, then measure how long the object stays gripped before it
drifts/rotates past a threshold.

Usage:
    python run_glove_dev_grasp.py                 open-loop ramp (default)
    python run_glove_dev_grasp.py --controller     closed-loop: per-channel
                                                    ease-off-on-contact, like
                                                    the other devices' GraspController
    python run_glove_dev_grasp.py --view            add either flag to also
                                                     open the interactive viewer

Real differences from the other devices' test, called out rather than
papered over:
- Default mode is a simple OPEN-LOOP ramp (ctrl 0->1 over ramp_seconds, then
  held at 1.0) -- no contact-triggered ease-off. --controller adds a minimal
  closed-loop version (per-channel: stop ramping and drop to HOLD_FRACTION
  the instant that channel's bodies touch the object), mirroring the other
  devices' GraspController but with only 2 channels (flex, thumb) since
  that's all this design's muscle-tendon routing supports.
- Only EXO_FLEX (index+middle flexion, FDP2/FDP3 routing) and EXO_THUMB (FPL
  routing) are driven. EXO_EXT is a finger-EXTENSION path (opens the hand) --
  driving it during a grasp would fight the other two, so it's left at 0, same
  as the other devices' hold reflex never re-opens a locked channel.
- No ring/little finger channel exists in this design at all (unlike our own
  D4_v2 devices' shared ulnar channel) -- this glove only assists index,
  middle, and thumb.
- The 6-DoF position-servo arm mount (ARTx/y/z, ARRx/y/z) is actively
  stabilized: gains boosted 8x + velocity damping added (see
  view_glove_dev_arm_pose.py) so it holds a deliberate, non-swaying pose
  instead of sagging under gravity to an arbitrary orientation.
- The native "OP" (opponens pollicis) muscle is driven as a baseline
  pre-shape, alongside the exo tendons -- a modeling assumption that the
  wearer voluntarily positions their own thumb into opposition while the
  exoglove assists grip FORCE on top of that. Without it, the thumb and
  fingers approach the object from the same side (confirmed both
  numerically and visually) and no controller tuning can produce a real
  pinch; with it, isolated-channel reach converges to the same height only
  2.4cm apart, vs 4-8cm before.
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = "myogloves_dev/models/myohand_glove_dev.xml"

ARM_JOINTS = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
# Stabilize the arm mount instead of letting it sag under gravity to an
# arbitrary orientation: boost the authored position-actuator gains and add
# velocity damping (see view_glove_dev_arm_pose.py -- kv=0 originally, so
# boosting kp alone made it ring/sway for several seconds on startup before
# settling). ctrl=0 for all 6 -> holds at the authored "textbook" neutral
# pose (already deliberately oriented for tabletop reach), just no longer
# losing that pose to gravity.
ARM_AUTHORED_KP = {"ARTx": 175, "ARTy": 175, "ARTz": 175, "ARRx": 150, "ARRy": 150, "ARRz": 150}
ARM_STIFF_BOOST = 8.0
ARM_KV_RATIO = 0.3

# Baseline thumb pre-shape: without this, the thumb has no reason to oppose
# the fingers -- it just ends up draped alongside them on the same side of
# the object (confirmed numerically and visually). OP is a native muscle,
# not part of the exoglove's own EXO_THUMB/EXO_FLEX/EXO_EXT tendons.
OP_CTRL = 0.5

RAMP_SECONDS = 4.0  # was 2.0 (too fast) then 6.0 (per request, slowed down);
                     # nudged back down slightly per follow-up request for a
                     # bit faster closing while still being gradual/visible
SETTLE_SECONDS = 1.6  # for open-loop; --controller uses SETTLE_SECONDS_CONTROLLER instead
# Both channels only stay in simultaneous contact for ~0.2-0.4s windows before
# one drops out (a real seesaw instability -- pressing with one nudges the
# object out of the other's reach), confirmed by requiring CURRENTLY-touching
# (not just ever-touched) contact through the settle window. 1.6s demanded
# far more stability than exists; shortened to fit the real contact windows.
SETTLE_SECONDS_CONTROLLER = 0.15
RETRACT_OFFSET = np.array([0.5, 0.0, 0.0])
POST_RETRACT_SECONDS = 3.0
FAIL_ROT_DEG = 45.0
FAIL_DRIFT_M = 0.05
HOLD_FRACTION = 1.0  # No ease-off at all once touched: traced the actual failure
                      # (option 1/2 investigation) and found the object was in
                      # literal FREE FALL after retract (Z-velocity reaching
                      # -2m/s, matching g*t almost exactly, while X/Y barely
                      # moved) -- the side-pinch was never generating enough
                      # squeeze/friction force to support the object's weight
                      # at HOLD_FRACTION=0.7, regardless of contact-normal
                      # opposition quality. Removed the ease-off reflex
                      # entirely (channels keep ramping to full ctrl=1.0
                      # rather than backing off at HOLD_FRACTION once
                      # touched) so the exoglove's own already-defined muscle
                      # force gets fully and continuously applied -- this
                      # changes OUR controller's hold behavior, not any
                      # EXO_THUMB/EXO_FLEX/EXO_EXT force/lengthrange/routing
                      # value in the exoglove itself.

HAND_BODY_NAMES = [
    "firstmc", "proximal_thumb", "distal_thumb",
    "secondmc", "proxph2", "midph2", "distph2",
    "thirdmc", "proxph3", "midph3", "distph3",
]

# This design only has two grasp-relevant channels (EXO_FLEX drives index+middle
# TOGETHER via one shared tendon/actuator -- they can't be eased off
# independently, unlike our own devices' per-digit K-matrix channels).
CHANNEL_BODIES = {
    "flex": {"secondmc", "proxph2", "midph2", "distph2", "thirdmc", "proxph3", "midph3", "distph3"},
    "thumb": {"firstmc", "proximal_thumb", "distal_thumb"},
}


def quat_angle_deviation(q0, q):
    dot = min(1.0, abs(float(np.dot(q0, q))))
    return np.degrees(2 * np.arccos(dot))


def main():
    interactive = "--view" in sys.argv
    use_controller = "--controller" in sys.argv

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Real bug found while troubleshooting: the object's actual compiled mass
    # was 0.27kg (weight 2.65N), not the intended 0.097kg (~0.95N) --
    # body_009_gelatin_box.xml's VISUAL mesh geom has no explicit mass, so
    # MuJoCo auto-computed its own inertial contribution from mesh volume and
    # silently added it on top of the collision geom's explicit mass="0.097".
    # We've been asking the grip to lift a 2.7x-too-heavy object the whole
    # time. Overriding body_mass/inertia/ipos to match ONLY the intended
    # collision box (a property of our own test prop, not the exoglove).
    obj_body_id_early = model.body("009_gelatin_box").id
    box_mass = 0.097
    hx, hy, hz = 0.036, 0.044, 0.014
    model.body_mass[obj_body_id_early] = box_mass
    model.body_inertia[obj_body_id_early] = [
        box_mass * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12,
        box_mass * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12,
        box_mass * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12,
    ]
    model.body_ipos[obj_body_id_early] = [0, 0, hz]
    model.body_iquat[obj_body_id_early] = [1, 0, 0, 0]

    # Test object's own friction (unset in the vendored YCB asset -- was
    # silently defaulting to MuJoCo's global 1/0.005/0.0001), bumped to match
    # the friction used for the grasp object in this project's other
    # exoskeleton devices. A property of our own test prop, not the exoglove.
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == obj_body_id_early:
            model.geom_friction[gid] = [2.5, 0.02, 0.002]

    # CONTACT STIFFENING -- the fix for fingers visibly passing *through* the
    # object. MyoHand's own bone geoms ship with a deliberately soft contact
    # model (solimp=[0.8, 0.8, 0.01]: constant impedance 0.8 with a 10mm
    # softening width), fine for gentle manipulation but simply crushed
    # through by EXO_FLEX (204.8 N) / EXO_THUMB (180 N) at full activation.
    # Measured penetration into this box reached 16.8mm (27.6mm on the soup
    # can, 32.2mm on the tuna can) -- bones passing bodily through the
    # object, which is what read as a "transparent hand" in the viewer.
    #
    # solref time constant 0.01s is 5x the 0.002s timestep, comfortably clear
    # of the ~2*dt instability threshold. Applied to every geom so the
    # pillar/table stay consistent.
    #
    # This is a contact-model fidelity setting, same category as the
    # hand-joint damping boost below -- NOT an EXO_THUMB/EXO_FLEX/EXO_EXT
    # force, lengthrange, tendon or routing value.
    #
    # This also removed the need for the <contact><exclude> block that used
    # to be in myohand_glove_dev.xml: with near-rigid contact the ring and
    # pinky no longer sink into the object, so instead of being deleted from
    # the physics they rest against it and act as a passive ulnar-side cage.
    # (Driving them actively was tested and is worse -- see
    # run_glove_dev_grasp_can.py's docstring for the numbers.)
    for gid in range(model.ngeom):
        model.geom_solref[gid] = [0.01, 1.0]
        model.geom_solimp[gid] = [0.98, 0.999, 0.0005, 0.5, 2.0]

    # Extra passive damping on the hand/wrist joints (not the arm mount,
    # handled separately below): even after the arm settles and the object
    # comes to rest (qpos stable), individual finger DOFs -- especially
    # undriven ones like index MCP abduction -- kept showing persistent qvel
    # noise (contact jitter) that never fully damped out even after 20000
    # steps, despite qpos staying essentially fixed. Their native damping is
    # very low (e.g. 0.05). A uniform damping boost is a physically
    # reasonable "soft tissue" assumption, not an exoglove value.
    #
    # Boosted from 8x to 25x after moving the object closer to the hand's
    # rest pose (per request) reintroduced a real startup kick: OP is a
    # muscle with its own activation dynamics, so the thumb builds up real
    # velocity approaching the (now much closer) object and overshoots into
    # it before contact forces can stop it (confirmed: zero overlap at step
    # 0, growing to -13mm by step 500, corrected with a violent ~1 rad/s
    # angular-velocity spike around step 1000-2500). More damping doesn't
    # prevent that initial contact-driven kick (peak transient stays ~1.6
    # rad/s from 8x up through 25x) but drains it out MUCH faster and more
    # completely afterward: final residual velocity drops from 0.08 (8x) to
    # 0.003 (25x), stable and flat from step 3000 onward. (Tried ramping OP's
    # activation up slowly instead of stepping it directly to OP_CTRL --
    # that made the spike WORSE, not better, by giving the thumb more time
    # to build velocity before contact; reverted.)
    HAND_DAMPING_BOOST = 25.0
    for jid in range(model.njnt):
        jname = model.joint(jid).name
        if jname in ARM_JOINTS or jname.startswith("OBJ"):
            continue
        dof_start = model.jnt_dofadr[jid]
        dof_count = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
                     mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[model.jnt_type[jid]]
        for d in range(dof_start, dof_start + dof_count):
            model.dof_damping[d] *= HAND_DAMPING_BOOST

    # Stabilize the arm (boosted gain + damping, holds the authored neutral
    # pose instead of sagging) and pre-shape the thumb into opposition via the
    # native OP muscle -- both explained above. Neither touches the
    # exoglove's own EXO_THUMB/EXO_FLEX/EXO_EXT tendons.
    for j in ARM_JOINTS:
        aid = model.actuator(f"A_{j}").id
        kp = ARM_AUTHORED_KP[j] * ARM_STIFF_BOOST
        model.actuator_gainprm[aid][0] = kp
        model.actuator_biasprm[aid][1] = -kp
        model.actuator_biasprm[aid][2] = -kp * ARM_KV_RATIO
        data.ctrl[aid] = 0.0
    op_id = model.actuator("OP").id
    data.ctrl[op_id] = OP_CTRL
    mujoco.mj_forward(model, data)
    # Let the arm settle and the thumb pre-shape into opposition BEFORE the
    # exo ramp starts (and BEFORE the viewer opens), so the grasp attempt
    # begins from a genuinely at-rest pose. 3000 steps was not enough: the
    # object (resting against the OP-preshaped thumb) has a real settling
    # transient with a real angular-velocity spike (~0.5 rad/s around step
    # 500) that was still only half-damped out at step 3000 (~0.11 rad/s
    # residual) -- opening the viewer there showed this ongoing motion as a
    # "jump". Confirmed by tracing velocity every 500 steps: doesn't reach
    # near-zero until ~step 5000. Extended to 6000 for margin.
    for _ in range(6000):
        mujoco.mj_step(model, data)

    exo_flex_id = model.actuator("EXO_FLEX").id
    exo_thumb_id = model.actuator("EXO_THUMB").id

    obj_body_id = model.body("009_gelatin_box").id
    # body_009_gelatin_box.xml's collision geom has no name -- select by body id.
    obj_geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == obj_body_id]

    hand_body_ids = {model.body(n).id for n in HAND_BODY_NAMES}
    channel_body_ids = {
        ch: {model.body(n).id for n in names} for ch, names in CHANNEL_BODIES.items()
    }

    pillar_body = model.body("glove_dev_support_pillar")
    pillar_mocap_id = pillar_body.mocapid[0]
    original_pillar_pos = data.mocap_pos[pillar_mocap_id].copy()

    ramp_step = model.opt.timestep / RAMP_SECONDS
    settle_seconds = SETTLE_SECONDS_CONTROLLER if use_controller else SETTLE_SECONDS
    settle_steps = int(settle_seconds / model.opt.timestep)
    post_retract_steps = int(POST_RETRACT_SECONDS / model.opt.timestep)

    u = {"flex": 0.0, "thumb": 0.0}
    locked = {"flex": False, "thumb": False}
    first_touch_step = None
    settle_start_step = None
    retract_step = None
    pos_at_retract = None
    quat_at_retract = None
    fail_step = None

    max_steps = 20000

    viewer_ctx = None
    if interactive:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)

    step = 0
    while step < max_steps:
        # Per-channel contact check (only meaningful with --controller, but
        # cheap enough to always compute so first_touch_step stays comparable
        # across both modes).
        touched = {"flex": False, "thumb": False}
        for i in range(data.ncon):
            c = data.contact[i]
            other = None
            if c.geom1 in obj_geom_ids and model.geom_bodyid[c.geom2] in hand_body_ids:
                other = model.geom_bodyid[c.geom2]
            elif c.geom2 in obj_geom_ids and model.geom_bodyid[c.geom1] in hand_body_ids:
                other = model.geom_bodyid[c.geom1]
            if other is None:
                continue
            for ch, bodies in channel_body_ids.items():
                if other in bodies:
                    touched[ch] = True
        hand_touch = touched["flex"] or touched["thumb"]

        if use_controller:
            # "locked" still marks the first genuine touch (used below to
            # decide when to retract), but no longer caps u at a reduced
            # HOLD_FRACTION -- it keeps ramping smoothly to full ctrl=1.0
            # regardless, since capping it at 0.7 was the direct cause of the
            # object free-falling once support was removed (see HOLD_FRACTION
            # comment above).
            for ch in ("flex", "thumb"):
                if touched[ch] and not locked[ch]:
                    locked[ch] = True
                u[ch] = min(u[ch] + ramp_step, 1.0)
        else:
            for ch in ("flex", "thumb"):
                u[ch] = min(u[ch] + ramp_step, 1.0)

        data.ctrl[exo_flex_id] = u["flex"]
        data.ctrl[exo_thumb_id] = u["thumb"]
        mujoco.mj_step(model, data)

        if hand_touch and first_touch_step is None:
            first_touch_step = step

        # Real bug found while troubleshooting: "locked" only records that a
        # channel touched at SOME point, not that it's still touching -- at
        # the actual retract moment, direct contact-force inspection showed
        # only the thumb (and the pillar) were in contact; the finger had
        # touched once early on then drifted back out, so the "full grasp"
        # trigger fired on a stale one-time latch while only a single contact
        # point was ever really engaged. Now requires BOTH channels touching
        # CONTINUOUSLY through the whole settle window, resetting the timer
        # the instant either one drops -- so retraction only happens with a
        # genuine, currently-live two-point grip, not a historical one.
        #
        # Also requires both channels to be near FULL ramped force (u>=0.95),
        # not just touching -- contact can happen early in the 4s ramp (e.g.
        # at u~0.4), and retracting only 0.15s after first touch pulled the
        # platform before the grip had actually developed real force. Now it
        # only retracts once the grasp has genuinely finished closing.
        FULL_FORCE_THRESHOLD = 0.95
        full_grasp_now = (touched["flex"] and touched["thumb"]
                           and u["flex"] >= FULL_FORCE_THRESHOLD
                           and u["thumb"] >= FULL_FORCE_THRESHOLD) if use_controller else hand_touch
        if full_grasp_now and retract_step is None:
            if settle_start_step is None:
                settle_start_step = step
                print(f"{'full grasp' if use_controller else 'first contact'} "
                      f"at t={step * model.opt.timestep:.3f}s; settling...")
            if step - settle_start_step >= settle_steps:
                data.mocap_pos[pillar_mocap_id] = original_pillar_pos + RETRACT_OFFSET
                retract_step = step
                pos_at_retract = data.xpos[obj_body_id].copy()
                quat_at_retract = data.xquat[obj_body_id].copy()
                print(f"pillar retracted at t={step * model.opt.timestep:.3f}s")
        elif use_controller and not full_grasp_now and retract_step is None:
            if settle_start_step is not None:
                print(f"lost contact at t={step * model.opt.timestep:.3f}s "
                      f"(touched={touched}); resetting settle timer")
            settle_start_step = None

        if retract_step is not None and fail_step is None:
            rot_dev = quat_angle_deviation(quat_at_retract, data.xquat[obj_body_id])
            drift = np.linalg.norm(data.xpos[obj_body_id] - pos_at_retract)
            if rot_dev > FAIL_ROT_DEG or drift > FAIL_DRIFT_M:
                fail_step = step
                print(f"grip failed at t={step * model.opt.timestep:.3f}s "
                      f"(rot_dev={rot_dev:.1f}deg drift={drift:.3f}m)")

        if viewer_ctx is not None:
            viewer_ctx.sync()
            time.sleep(0.002)
            if not viewer_ctx.is_running():
                break

        # In interactive mode, keep the window open (still stepping physics
        # so the object's post-failure motion is visible) once the measured
        # window ends, rather than auto-closing after ~1s -- previously the
        # window closed right after the test finished, too fast to actually
        # look at.
        if retract_step is not None and step - retract_step >= post_retract_steps and viewer_ctx is None:
            break

        step += 1

    if viewer_ctx is not None:
        print("Test finished -- close the viewer window when you're done looking.")
        while viewer_ctx.is_running():
            mujoco.mj_step(model, data)
            viewer_ctx.sync()
            time.sleep(0.002)
        viewer_ctx.close()

    print("=" * 60)
    if first_touch_step is None:
        print("RESULT: never made contact with the object")
        return
    print(f"first contact: t={first_touch_step * model.opt.timestep:.3f}s")
    if retract_step is None:
        print("RESULT: contact made but grip never settled long enough to retract")
        return
    print(f"retracted at: t={retract_step * model.opt.timestep:.3f}s")
    if fail_step is None:
        hold_s = (step - retract_step) * model.opt.timestep
        print(f"RESULT: held for the full {hold_s:.3f}s post-retract window (never failed)")
    else:
        hold_s = (fail_step - retract_step) * model.opt.timestep
        print(f"RESULT: held for {hold_s:.3f}s before drift/rotation exceeded threshold")


if __name__ == "__main__":
    main()
