"""HOLDING BENCHMARK -- object grasp-and-hold, comparable across devices.

Implements the holding benchmark spec: initialise -> close the grasp ->
contact gate -> remove support -> measure time-to-failure, repeated over K
randomised trials and swept over object mass/shape.

WHY THIS EXISTS
---------------
Device numbers produced before this harness are NOT comparable to each
other. Two concrete reasons, both measured rather than assumed:

1. Contact model. MyoHand's bone geoms ship with solimp=[0.8, 0.8, 0.01]
   (constant impedance 0.8, 10mm softening width). Exoskeleton-scale forces
   crush straight through that: measured finger penetration was 16.8mm into
   the gelatin box, 27.6mm into a soup can and 32.2mm into a tuna can. A
   "grasp" measured that way is partly the fingers passing through the
   object. Correcting it changed one device's hold from 0.830s to
   surviving the entire measurement window. Every device is therefore run
   here with contact parameters PINNED to the same values (CONTACT_SOLREF /
   CONTACT_SOLIMP). Without this, a device comparison measures contact
   softness, not device design.

2. Contact excludes. The exoglove env carries 66 <contact><exclude> pairs;
   the glove_dev models now carry none. Excludes delete contacts from the
   physics entirely, so two setups with different exclude lists are not the
   same experiment. This harness reports each model's exclude count in its
   output so the asymmetry is visible rather than silent.

METHODOLOGY NOTES
-----------------
- One freshly compiled MjModel per trial. Never reuse a model whose fields
  were mutated for a previous trial: doing that produced ghost results that
  did not reproduce when the same condition was re-run in isolation.
- Determinism is not robustness. A single seeded run reproduces exactly
  because MuJoCo is deterministic, which says nothing about sensitivity to
  initial conditions. Hence K randomised trials and mean/median +- SD.
- Right-censoring is tracked explicitly. Trials surviving T_MAX have no
  observed failure time, so a plain mean over them is biased downward and
  meaningless once most trials survive. The headline metric is therefore
  SURVIVAL RATE (fraction reaching T_MAX); hold-time stats are reported
  alongside with the censored count stated.
- Outcomes are three-way, not pass/fail: NO_GRASP (contact gate never
  satisfied -- "couldn't grasp") is kept distinct from a grasp that formed
  and then slipped ("couldn't hold"). Collapsing them hides which failure
  mode a device has.

KNOWN ASYMMETRY (stated, not silently averaged away)
----------------------------------------------------
The glove_dev device is run with its native OP (opponens pollicis) thumb
pre-shape, a documented modelling assumption that the wearer voluntarily
opposes their own thumb. These are genuinely different device concepts and
the pre-shape is part of one of them -- it is configurable per adapter (see
Adapter.preshape) and recorded in the results so any comparison can account
for it.

Where each device's opposition comes from, once the primary sources were
checked (an earlier version of this note said "D1/D2/D3 have zero thumb rows
entirely", which was wrong for D1 and D2):
  D1  nowhere. It drives all five digits but has NO abduction DoF at all --
      by design, it is a flexion/extension TRAINING device, not a grasp aid.
      Under the opposition gate it forms an opposed grasp in 0 of 15 trials.
  D2  thumb flexion only, no abduction actuator in the source device.
  D3  a rigid C-bar splint holding the thumb as a passive post. Modelled.
  D4  a dedicated thumb-opposition tendon on its own motor -- the only
      device here with actively actuated opposition.

Usage:
    python hold_benchmark.py pilot                 # short run, picks T_max
    python hold_benchmark.py run --devices ... --trials 20
    python hold_benchmark.py masssweep --device ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- constants

# Pinned contact model -- see module docstring point (1). solref time
# constant 0.01s is 5x the 0.002s timestep, comfortably clear of the ~2*dt
# threshold where the solver destabilises.
CONTACT_SOLREF = [0.01, 1.0]
CONTACT_SOLIMP = [0.98, 0.999, 0.0005, 0.5, 2.0]

CLOSING_SECONDS = 1.5      # ramp u 0->1 over this window, then hold

# Pre-ramp settle. The spec asks for "settle to static equilibrium so the
# object isn't already moving", but the two environments differ in a way
# that matters and cannot be papered over with one constant:
#
#   glove_dev env: the object rests against the OP-preshaped thumb, and that
#     pre-shape has a real activation transient that is not fully damped out
#     until ~5000 steps. Starting the ramp earlier shows up as a visible
#     startup jump.
#   exoglove env (D1-D4): the object is at rest on its pillar within ~10
#     steps, but the hand has NO static equilibrium when undriven -- it
#     drifts under gravity, its pinky contacts the object around step 300,
#     and it has knocked the object clean off the pillar by step ~1200
#     (traced directly). The original D1-D4 test sidesteps this by driving
#     from step 0 and never settling.
#
# So settling is adaptive: step until the OBJECT is at rest and stays at
# rest, with a per-adapter cap. The cap is what differs, and it is a
# property of the environment rather than of the device being measured.
SETTLE_VEL_EPS = 1e-3      # m/s, object linear speed considered "at rest"
SETTLE_QUIET_STEPS = 50    # consecutive quiet steps required
GATE_MIN_DIGITS = 2        # >= N distinct digits in contact...
GATE_MIN_SECONDS = 1.0     # ...continuously for >= this long
# ...AND at least one pair of those digits must actually OPPOSE each other.
#
# Digit count alone was not a grasp test. Measured at the gate moment, seed
# 1000, mean contact normal per digit oriented away from the object:
#     glove_dev   index vs thumb  -0.991   opposing
#     splint_D3   index vs thumb  -1.000   opposing
#     D1          index vs little +1.000   pushing the SAME way
#     D4          index vs little +1.000   pushing the SAME way
# D1 and D4 were being certified by two fingers shoving the object in one
# direction, with nothing on the far side. The gate opens the hold clock, so
# those two started their clock in a configuration that could not hold and
# were then scored on how fast it failed -- a property of the metric, not of
# the device.
#
# The separation between real and fake grasps here is enormous (-0.99 against
# +1.00), so the threshold is not a sensitive tuning knob; anything in the
# broad middle gives the same answer. -0.5 is roughly "more than 120 degrees
# apart".
GATE_REQUIRE_OPPOSITION = True
GATE_OPPOSITION_DOT = -0.5
DROP_THRESHOLD_M = 0.05    # object falling this far RELATIVE TO THE HAND counts
                           # as failure -- see HAND_REF_BODY below
# Measured in the HAND's frame, not the world's. This was a real defect, and it
# inverted the answer on at least one object.
#
# The arm is a position servo with finite stiffness. While the support pillar
# is in place it carries part of the load; removing it transfers that to the
# arm, which then deflects until its springs rebalance. Traced on the box with
# the healthy hand: the object descends 47.4 mm in world coordinates and was
# scored as dropped, but the PALM descends 39.3 mm over the same interval, so
# the object moves only 8.1 mm relative to the hand -- and that number stops
# growing (8.0, 8.2, 8.2, 8.1 mm at 0.4/0.6/0.8/1.0 s). The grasp is stable.
# The benchmark was scoring arm compliance as grasp failure.
#
# Grip force was ruled out before landing here: capping the healthy hand's
# hold-phase activation from 1.0 down to 0.12 moved the can's median hold by
# 0.1 s and the box's not at all, so the failures were never a squeeze problem.
# Nor could they be a friction problem -- at the gate the digits apply ~193 N
# of normal force to a 3.4 N can with mu=2.5, a friction capacity two orders of
# magnitude above the load.
#
# capitate is the reference: a central carpal bone, so it tracks the palm
# rather than any one finger, and it does not move when the digits close.
HAND_REF_BODY = "capitate"
T_MAX_DEFAULT = 5.0
# Soft-tissue damping boost: DISABLED (1.0 = no boost), deliberately.
#
# It was introduced as a glove_dev-specific fix for a startup transient from
# the OP pre-shape, and 25x works there because that device drives MuJoCo
# muscle actuators rated 180-205 N. The K-matrix devices apply ~0.13 N*m --
# three orders of magnitude smaller -- and 25x damping simply swamps them:
# measured, D4 goes from 5 digits of contact to ZERO with the boost applied.
# A "shared" constant that helps one device and disables another is not a
# fair comparison, and glove_dev is unaffected by removing it (it still
# reaches 3-4 digits and holds), so it is off for everyone.
HAND_DAMPING_BOOST = 1.0

# Object friction. This repo had been running at [2.5, 0.02, 0.002] with no
# recorded justification -- 2.5x the sliding, 4x the torsional and 20x the
# rolling friction of the value MyoAssist uses for the SAME object.
#
# myoMPL (Tan et al., MyoAssist 0.1, ICORR 2025) manipulates YCB
# 009_gelatin_box at 97 g, which is exactly the box here -- its
# myoarm_bionic_bimanual.xml includes the same body_009_gelatin_box.xml. Its
# friction is [1.0, 0.005, 0.0001] sliding/torsional/rolling, stated in the
# paper and confirmed in the model file. Running grippier than the reference
# environment makes every grasp here easier than the published benchmark and
# is not a difference any device earned.
MYOASSIST_FRICTION = [1.0, 0.005, 0.0001]   # myoMPL default, the comparable setting
LEGACY_FRICTION = [2.5, 0.02, 0.002]        # what this repo used before the audit
OBJECT_FRICTION = list(MYOASSIST_FRICTION)

# Randomisation per trial (small perturbations, per the spec).
# NOTE myoMPL randomises considerably harder: mass +-50 g (not +-2%), object
# dimensions +-0-5% width / 0-10% depth / 0-5% height, and friction +-0.1 /
# +-0.001 / +-0.00002. Its baseline drops from ~25% to ~3% success once those
# are switched on, so the gap between our randomisation and theirs is large
# enough to matter for any cross-paper comparison.
POS_JITTER_M = 0.003
MASS_JITTER_FRAC = 0.02

# --- Transmission loss: devices as WORN, not as idealised -------------------
#
# Every device here delivers torque straight to the joint. Real ones do not:
# force reaches the finger through fabric, straps, Velcro and (for the rigid
# designs) linkage play, all of which stretch. posture_benchmark.py measured
# how much that matters by reproducing Zhao et al.'s published joint-angle
# similarity protocol, and the gap is not subtle -- simulated 96/99/461% at
# the index MCP/PIP/DIP against their measured 42/74/59%. Zhao et al. name the
# cause themselves: "the space occupied by the Velcro used to secure the
# exoskeleton on the hand, as well as the limitations of the underactuated
# structure's force transmission, which reduces the effective range of
# motion".
#
# MODEL: series elasticity. The strap is a spring between the actuator and the
# bone, anchored at the pose the device was DONNED in (captured lazily at the
# first driven step, i.e. after settling). The further the joint travels from
# there, the more of the commanded torque goes into stretching the strap
# instead of moving the finger:
#
#     tau_delivered = tau_commanded - STRAP_STIFFNESS * (q - q_donned)
#
# A scalar efficiency (tau * eta) was considered first and rejected: with a
# pure torque source in free air, scaling the torque changes how FAST a joint
# travels but not how far it ends up, so it cannot reproduce a range-of-motion
# deficit at all. Series compliance can, because it introduces an equilibrium
# at roughly tau/k -- which is exactly the mechanism the paper describes.
#
# Applied to EVERY device including glove_dev. This does not touch the
# exoglove's own force ratings, lengthranges or tendon routing -- it is a model
# of the glove-to-skin interface, which his design has as much as any other
# (it is a soft glove). Excluding him would hand him an advantage the hardware
# does not have.
#
# NOT applied to the healthy-hand control, which wears no device.
#
# STRAP_STIFFNESS is CALIBRATED against Zhao et al.'s published numbers rather
# than chosen -- one parameter fitted to three published observations, so it
# can be wrong. See calibrate_strap.py for the fit and its residual.
# CALIBRATION RESULT -- and it is a NEGATIVE result worth reading before using
# this. Fitting the single stiffness against Zhao et al.'s three published index
# similarities (calibrate_strap.py, 5 trials per candidate):
#
#     k       MCP     PIP     DIP    RMS err
#   0.000    85.5%   69.6%  288.9%    135.1
#   0.035    69.0%   58.2%  197.9%     82.3
#   0.100    28.5%   23.2%   78.6%     32.4   <- best
#   0.250    11.6%    9.3%   31.5%     44.2
#   published 41.7%   74.2%   59.0%
#
# The residual bottoms out at 32.4 points and will not go lower, because the
# mismatch is not one of SCALE, which is all a single spring can change. It is
# one of SHAPE. Published, the PIP gets the most range (74.2%) and the MCP the
# least (41.7%). Simulated, the MCP beats the PIP at every stiffness tried.
# No value of k reorders them.
#
# That points somewhere specific, and not at this file. D1's calibrated K is
# proximal-biased (1.000 / 0.908 / 0.618) because MyoHand's FDP moment arm
# shrinks from MCP to DIP, which is what flipped Zhao et al.'s distal-biased
# FORCE measurements (4.1 / 5.2 / 5.6 N) into proximal-biased TORQUE. The
# published similarity is distal-favouring, consistent with their force data
# and not with our torque conversion. So either MyoHand's FDP moment arms are
# not representative of this device's real geometry, or the device does not
# transmit along the FDP path that every calibrated K in this repo assumes.
# Resolving that is a moment-arm question, not a strap question.
#
# Consequence for use: the strap model is physically right in KIND (ideal
# transmission is certainly wrong) but its one fitted number carries a large
# residual, so switching it on trades a known bias for a poorly-constrained
# one. It therefore defaults OFF and is opt-in per run (--strap), and both
# settings are reported rather than one being quietly adopted.
# SECOND REASON IT DEFAULTS OFF, found by running it. At the best-fit k=0.1,
# 15 trials on the box:
#     HEALTHY        60.0% grasp  1.45s   (unchanged -- wears no device)
#     D3 + splint   100.0% grasp  1.45s   (barely touched: 1.63 -> 1.45s)
#     D4             13.3% grasp  0.88s   (was 93.3%)
#     Tyrone          0.0% grasp          (was 93.3%)
#     D1, D2          0.0% grasp
# D3 survives almost untouched because its opposition comes from a splint,
# which is modelled as a rigid joint constraint rather than as actuator torque,
# so the strap never touches it. Every other device's opposition is torque and
# gets attenuated. That is not D3 winning -- it is the one device whose key
# mechanism is still idealised being compared against five that no longer are.
# The same asymmetry was flagged when the splint was added ("every actively
# driven thumb here competes against a perfect post"); this quantifies it.
#
# Making that comparison fair needs compliance in the splint too, and there is
# no published number to set it from, so it is not invented here. Until then
# the as-worn table measures the modelling asymmetry more than it measures the
# devices, which is why 0.0 remains the default and both are always reported.
STRAP_STIFFNESS = 0.0   # N*m/rad; 0 = ideal. Best fit to published data: 0.1


class StrapTransmission:
    """Series-elastic glove-to-bone interface. See STRAP_STIFFNESS above.

    Anchors on first use rather than at build time: the donned pose is where
    the hand sits once settled, not the model's authored keyframe, and the
    strap is by definition unstretched at the moment the device starts pulling.
    """

    def __init__(self, model, joint_names, stiffness):
        self.k = float(stiffness)
        self.rows = []
        for jname in joint_names:
            try:
                j = model.joint(jname)
            except KeyError:
                continue
            self.rows.append((j.qposadr[0], j.dofadr[0]))
        self.q_donned = None

    def apply(self, data):
        if self.k <= 0.0 or not self.rows:
            return
        if self.q_donned is None:
            self.q_donned = np.array([float(data.qpos[q]) for q, _ in self.rows])
        for (qadr, dofadr), q0 in zip(self.rows, self.q_donned):
            data.qfrc_applied[dofadr] -= self.k * (float(data.qpos[qadr]) - q0)


def settle(model, data, obj_body_id, cap_steps):
    """Step until the object is at rest (and stays at rest), or until cap.

    Returns (steps_used, ok). ok=False means the object left its support
    during settling -- a SETUP failure, which must not be silently scored as
    a device failure.
    """
    z0 = float(data.xpos[obj_body_id][2])
    quiet = 0
    for s_ in range(cap_steps):
        mujoco.mj_step(model, data)
        if z0 - float(data.xpos[obj_body_id][2]) > DROP_THRESHOLD_M:
            return s_ + 1, False
        # Quiescence of the WHOLE system, not just the object: the object can
        # read as "at rest" while the hand is still mid-transient (the
        # glove_dev OP pre-shape does exactly this), and starting the ramp
        # there measures a moving hand.
        if float(np.abs(data.qvel).max()) < SETTLE_VEL_EPS:
            quiet += 1
            if quiet >= SETTLE_QUIET_STEPS:
                return s_ + 1, True
        else:
            quiet = 0
    return cap_steps, True


class Outcome:
    NO_GRASP = "NO_GRASP"      # contact gate never satisfied
    SETUP_FAIL = "SETUP_FAIL"  # object left its support before the test began
    SLIPPED = "SLIPPED"        # gate passed, then failed before T_max
    SURVIVED = "SURVIVED"      # gate passed, still held at T_max (censored)


# ------------------------------------------------------------- shared setup

def pin_contacts(model):
    for g in range(model.ngeom):
        model.geom_solref[g] = CONTACT_SOLREF
        model.geom_solimp[g] = CONTACT_SOLIMP


def boost_hand_damping(model, skip_joints, skip_bodies=()):
    """Soft-tissue damping boost on the HAND joints only.

    skip_bodies must include the graspable object's body id. Matching on
    name alone is not enough and was a real bug: in the exoglove env the
    object's free joint is UNNAMED (""), so a startswith("OBJ") guard
    silently let it through and damped the object itself 25x, which stopped
    it responding to the hand at all.
    """
    skip_bodies = set(skip_bodies)
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in skip_joints or name.startswith("OBJ"):
            continue
        if model.jnt_bodyid[j] in skip_bodies:
            continue
        adr = model.jnt_dofadr[j]
        n = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
             mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}[model.jnt_type[j]]
        for d in range(adr, adr + n):
            model.dof_damping[d] *= HAND_DAMPING_BOOST


# ------------------------------------------------------------------ adapters

class Adapter:
    wears_device = True   # False only for the healthy control (no glove -> no strap)

    """Uniform interface so the benchmark loop is device-agnostic.

    Subclasses must provide a compiled model+data with the object resting on
    a support, a way to set the device input level u in [0,1], and the set of
    bodies that count as 'digits' for the contact gate.
    """

    name = "?"
    preshape = None
    settle_cap = 6000   # env-specific, see SETTLE_VEL_EPS comment
    drive = "open_loop_ramp"   # or "own_controller"
    # World z of the object's BOTTOM face relative to its body_pos.z. The
    # support pillar's top must meet it exactly. 0 for the cans (collision
    # geom is (0,0,half_h) with size half_h); -0.044 for the gelatin box,
    # whose 90deg rotation puts the bottom below body_pos.z.
    bottom_offset = 0.0
    pos_override = None        # (x,y,z) to place the object at, or None
    # Closing-window length. Per-environment, because it is bounded by how
    # long that environment can hold the object before the grasp forms, not
    # by anything about the device. See KMatrixAdapter for the exoglove env.
    closing_seconds = CLOSING_SECONDS  # human-readable note on any native-muscle pre-shape used

    def build(self, rng, mass_override=None):
        raise NotImplementedError

    def set_input(self, model, data, u):
        raise NotImplementedError

    def release_support(self, model, data):
        raise NotImplementedError


class GloveDevAdapter(Adapter):
    """Tyrone's exoglove: MuJoCo muscle actuators driven through data.ctrl."""

    ARM_JOINTS = ["ARTx", "ARTy", "ARTz", "ARRx", "ARRy", "ARRz"]
    ARM_KP = {"ARTx": 175, "ARTy": 175, "ARTz": 175, "ARRx": 150, "ARRy": 150, "ARRz": 150}
    DIGIT_BODIES = {
        "thumb": {"proximal_thumb", "distal_thumb"},
        "index": {"proxph2", "midph2", "distph2"},
        "middle": {"proxph3", "midph3", "distph3"},
        "ring": {"proxph4", "midph4", "distph4"},
        "little": {"proxph5", "midph5", "distph5"},
        # Gerez et al. 2020's telescopic 6th-digit appendage, present only in
        # the D4 extra_thumb variants. It is a SEPARATE body, so omitting it
        # here silently benchmarked those variants without counting the very
        # feature that defines them.
        "extra_thumb": {"extra_thumb", "extra_thumb_v2", "extra_thumb_v2_shaft_body"},
    }
    preshape = "native OP (opponens pollicis) at 0.5"
    settle_cap = 6000   # OP activation transient needs ~5000 steps

    def __init__(self, model_path, obj_name, obj_radius, obj_half_h, obj_mass, label,
                 bottom_offset=0.0):
        self.bottom_offset = bottom_offset
        self.model_path = model_path
        self.obj_name = obj_name
        self.radius = obj_radius
        self.half_h = obj_half_h
        self.nominal_mass = obj_mass
        self.name = label

    def build(self, rng, mass_override=None):
        m = mujoco.MjModel.from_xml_path(self.model_path)
        oid = m.body(self.obj_name).id

        mass = mass_override if mass_override is not None else self.nominal_mass
        mass *= 1.0 + rng.uniform(-MASS_JITTER_FRAC, MASS_JITTER_FRAC)
        # Override mass/inertia to the intended COLLISION primitive only: the
        # YCB visual mesh geoms carry no explicit mass, so MuJoCo silently
        # adds their mesh-volume mass on top of the collision geom's stated
        # mass (measured: 0.27kg for a box declared 0.097kg).
        fh = 2 * self.half_h
        ir = mass * (3 * self.radius ** 2 + fh ** 2) / 12
        m.body_mass[oid] = mass
        m.body_inertia[oid] = [ir, ir, 0.5 * mass * self.radius ** 2]
        m.body_ipos[oid] = [0, 0, self.half_h]
        m.body_iquat[oid] = [1, 0, 0, 0]
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == oid:
                m.geom_friction[g] = OBJECT_FRICTION

        # Placement override (device-neutral placement studies). Set body_pos
        # directly rather than offsetting the OBJT* slides: those joint axes
        # are in the BODY frame, so for the rotated gelatin box a "+y" offset
        # actually moves it along world -z.
        if self.pos_override is not None:
            m.body_pos[oid] = np.array(self.pos_override, dtype=float)
        # Nominal (un-jittered) placement, kept for positioning the support
        # pillar below. The pillar must NOT follow the jitter: the jitter
        # models placing the object slightly differently on a FIXED support,
        # so moving the support with it would cancel out part of the very
        # randomisation it exists to provide. This was inconsistent -- the
        # override path re-derived the pillar from the jittered position while
        # the default path left the authored pillar alone -- which made a
        # pos_override set to the model's own authored placement produce
        # measurably different results from not setting it at all.
        nominal_pos = m.body_pos[oid].copy()
        # Trial randomisation: jitter the object's resting placement.
        m.body_pos[oid] = m.body_pos[oid] + rng.uniform(-POS_JITTER_M, POS_JITTER_M, 3)

        pin_contacts(m)
        boost_hand_damping(m, set(self.ARM_JOINTS), skip_bodies={oid})
        for j in self.ARM_JOINTS:
            a = m.actuator(f"A_{j}").id
            kp = self.ARM_KP[j] * 8.0
            m.actuator_gainprm[a][0] = kp
            m.actuator_biasprm[a][1] = -kp
            m.actuator_biasprm[a][2] = -kp * 0.3

        d = mujoco.MjData(m)
        for j in self.ARM_JOINTS:
            d.ctrl[m.actuator(f"A_{j}").id] = 0.0
        d.ctrl[m.actuator("OP").id] = 0.5  # documented pre-shape, see docstring

        self.oid = oid
        self.flex = m.actuator("EXO_FLEX").id
        self.thumb = m.actuator("EXO_THUMB").id
        self.pillar_mocap = m.body("glove_dev_support_pillar").mocapid[0]
        if self.pos_override is not None:
            top = float(nominal_pos[2]) + self.bottom_offset
            half = top / 2.0
            m.geom_size[m.geom("glove_dev_support_pillar_geom").id] = [0.08, 0.08, half]
            d.mocap_pos[self.pillar_mocap] = [float(nominal_pos[0]),
                                              float(nominal_pos[1]), half]
        self.obj_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == oid]
        self.digit_ids = {
            dig: {m.body(b).id for b in bs if _has_body(m, b)}
            for dig, bs in self.DIGIT_BODIES.items()
        }
        self.mass_used = mass
        self._build_strap(m)
        return m, d

    def _build_strap(self, model):
        """Series-elastic transmission on the joints this device drives.

        Called at the end of every build(). Subclasses that populate .device
        call it again afterwards, because driven_joints() reads .device and the
        base build() runs before the subclass has set it. The healthy-hand
        control sets wears_device=False -- it has no glove, so it has no strap
        losses either, and giving it any would quietly handicap the one
        condition that exists to say what is physically achievable here.
        """
        self.strap = None
        if STRAP_STIFFNESS > 0.0 and self.wears_device:
            driven, _ = driven_joints(model, self)
            self.strap = StrapTransmission(model, sorted(driven), STRAP_STIFFNESS)

    def set_input(self, model, data, u):
        data.ctrl[self.flex] = u
        data.ctrl[self.thumb] = u
        if getattr(self, "strap", None) is not None:
            self.strap.apply(data)

    def release_support(self, model, data):
        data.mocap_pos[self.pillar_mocap] = data.mocap_pos[self.pillar_mocap] + np.array([0.5, 0.0, 0.0])


class HealthyHandAdapter(GloveDevAdapter):
    """Healthy MyoHand reference: no device at all, driving the hand's OWN
    flexor muscles at full strength.

    Serves two purposes.

    (1) FEASIBILITY CONTROL. Device results are uninterpretable without it:
        a device scoring 0% could mean the device is inadequate, or it could
        mean the object placement is simply unreachable for any hand. Running
        an unimpaired hand through the identical protocol separates those.

    (2) POSTURE REFERENCE. The joint trajectory recorded here is x_ref(t),
        the healthy closing posture for this object, against which a device's
        posture error can later be measured. Note the reference is only
        comparable if generated with the SAME object placement and initial
        hand pose as the device trials -- which it is, since this reuses the
        glove_dev adapter's build() unchanged and only swaps what is driven.
    """

    # Long finger flexors (deep + superficial), thumb flexor, and opponens.
    HEALTHY_MUSCLES = ["FDP2", "FDP3", "FDP4", "FDP5",
                       "FDS2", "FDS3", "FDS4", "FDS5",
                       "FPL", "OP"]
    preshape = "n/a -- healthy hand, native muscles driven directly"
    drive = "native_muscles"
    wears_device = False

    def build(self, rng, mass_override=None):
        m, d = super().build(rng, mass_override=mass_override)
        self.healthy_ids = [m.actuator(n).id for n in self.HEALTHY_MUSCLES
                            if _has_actuator(m, n)]
        # The exo tendons are present in the model but must stay silent: this
        # is the unassisted hand.
        d.ctrl[self.flex] = 0.0
        d.ctrl[self.thumb] = 0.0
        return m, d

    def set_input(self, model, data, u):
        data.ctrl[self.flex] = 0.0
        data.ctrl[self.thumb] = 0.0
        for aid in self.healthy_ids:
            data.ctrl[aid] = u


def _has_actuator(model, name):
    try:
        model.actuator(name)
        return True
    except Exception:
        return False


class KMatrixAdapter(Adapter):
    """D1-D4 (and the bare_msk no-device baseline): torque devices applying
    tau = K @ (u * tau_max) to the hand joints via qfrc_applied."""

    DIGIT_BODIES = GloveDevAdapter.DIGIT_BODIES
    preshape = "none (device drives its own opposition where it has a thumb row)"
    # NO pre-settle at all. This env has no undriven equilibrium: the hand
    # sags under gravity, its pinky contacts the object within ~300 steps and
    # has knocked it off the pillar by ~1200 (traced directly). Even a 0.6s
    # settle was enough to lose the object before the controller engaged.
    # The original D1-D4 protocol drives from step 0 for exactly this reason,
    # and the object is already at rest on its pillar at t=0 (verified: only
    # start_pillar contacts, 0.00mm penetration), so nothing is gained by
    # settling anyway.
    # With WristHold active this env is stable undriven (object z held at
    # 1.387 across 3000 steps, vs falling by ~1200 before), so it can now use
    # the SAME settle and the SAME spec-compliant ramp as the glove_dev
    # family. The earlier step-input deviation is no longer needed and has
    # been removed -- closing dynamics are measured for both families again.
    settle_cap = 2000
    closing_seconds = CLOSING_SECONDS

    # D1-D4 are driven by the control policy they were designed with
    # (GraspController, including its thumb-sequencing row_gate) rather than
    # the bare open-loop ramp. Measured reason: under a plain ramp these
    # devices reach only ONE digit of contact, whereas their own controller
    # reaches five. Driving them open-loop would measure them well below
    # their real capability. The cost is that control policy is now part of
    # what is measured, which is recorded per device in the results.
    drive = "own_controller"

    def __init__(self, device_name, model_path, label=None):
        self.device_name = device_name
        self.model_path = model_path
        self.name = label or device_name

    def build(self, rng, mass_override=None):
        from exo_devices import DEVICES, ExoApplicator

        m = mujoco.MjModel.from_xml_path(self.model_path)
        oid = m.body("grasp_object").id
        ogeom = m.geom("grasp_object_geom").id

        mass = mass_override if mass_override is not None else float(m.body_mass[oid])
        mass *= 1.0 + rng.uniform(-MASS_JITTER_FRAC, MASS_JITTER_FRAC)
        m.body_mass[oid] = mass
        if self.pos_override is not None:
            m.body_pos[oid] = np.array(self.pos_override, dtype=float)
        m.body_pos[oid] = m.body_pos[oid] + rng.uniform(-POS_JITTER_M, POS_JITTER_M, 3)

        pin_contacts(m)
        boost_hand_damping(m, set(), skip_bodies={oid})

        d = mujoco.MjData(m)

        self.device = None if self.device_name == "bare_msk" else DEVICES[self.device_name]
        if self.device is not None and not self.device.tau_max_is_calibrated:
            self.device.tau_max[:] = 0.03
        # Use the project's own tested applicator rather than re-deriving the
        # joint->dof mapping here; it already handles passive coupling and
        # row gating consistently with the other D1-D4 scripts.
        self.applicator = ExoApplicator(m)
        self.wrist = None   # created after mj_forward in run_trial
        self.controller = None
        if self.device is not None:
            from grasp_controller import GraspController
            self.controller = GraspController(
                m, self.device, ogeom, **dict(self.device.controller_overrides))
        self.oid = oid
        self.obj_geoms = [ogeom]
        pillar = m.body("start_pillar")
        self.pillar_mocap = pillar.mocapid[0]
        self.pillar_home = m.body_pos[pillar.id].copy()
        self.digit_ids = {
            dig: {m.body(b).id for b in bs if _has_body(m, b)}
            for dig, bs in self.DIGIT_BODIES.items()
        }
        self.mass_used = mass
        return m, d

    def set_input(self, model, data, u):
        # u (the harness ramp level) is deliberately ignored here: this
        # device family supplies its own closing profile via GraspController.
        if self.device is not None:
            u_dev = self.controller.step(model, data)
            self.applicator.apply(data, self.device, u=u_dev,
                                  row_gate=self.controller.row_gate)
        if self.wrist is not None:
            self.wrist.apply(model, data)

    def release_support(self, model, data):
        data.mocap_pos[self.pillar_mocap] = self.pillar_home + np.array([0.5, 0.0, 0.0])


# Wrist joints. The exoglove env has no arm and nothing holding the wrist,
# so it collapses under gravity (measured over 1200 undriven steps:
# pro_sup +0.306 rad, deviation -0.175, flexion +0.118) and swings the hand
# into the object, knocking it off its pillar. The glove_dev env never shows
# this because its 6-DoF position-servo arm holds the wrist steady. A freely
# flopping wrist is not a property of any device under test -- it is a
# missing boundary condition -- so it is held here, mirroring the arm
# stabilisation already applied on the other side.
WRIST_JOINTS = ["pro_sup", "deviation", "flexion"]
WRIST_KP = 20.0
WRIST_KV = 2.0


class WristHold:
    """PD hold on the wrist joints at their initial pose.

    Applies to dofs that are NOT in exo_devices.JOINT_NAMES, so it never
    clobbers the torques ExoApplicator writes for the device itself.
    """

    def __init__(self, model, data):
        self.dofs, self.targets = [], []
        for name in WRIST_JOINTS:
            if not _has_joint(model, name):
                continue
            j = model.joint(name)
            self.dofs.append(j.dofadr[0])
            self.targets.append(float(data.qpos[j.qposadr[0]]))
        self.qadr = [model.joint(n).qposadr[0] for n in WRIST_JOINTS
                     if _has_joint(model, n)]

    def apply(self, model, data):
        for dof, qa, tgt in zip(self.dofs, self.qadr, self.targets):
            err = tgt - float(data.qpos[qa])
            data.qfrc_applied[dof] += WRIST_KP * err - WRIST_KV * float(data.qvel[dof])


class HealthyExogloveEnvAdapter(KMatrixAdapter):
    """Healthy hand IN THE D1-D4 ENVIRONMENT, driving native flexors.

    This control was missing and its absence mattered. "bare_msk" is not a
    healthy hand -- it applies no device torque AND no muscle activation, so
    it is a LIMP hand and is guaranteed to fail. Without a genuinely healthy
    condition in this environment, D1-D4's 0% could equally mean "these
    devices are inadequate" or "nothing can grasp at this object placement".
    The glove_dev environment has its own healthy control (HealthyHandAdapter)
    but that is a different model with a different object in a different
    coordinate frame, so it says nothing about this one.
    """

    HEALTHY_MUSCLES = HealthyHandAdapter.HEALTHY_MUSCLES
    preshape = "n/a -- healthy hand, native muscles driven directly"
    drive = "native_muscles"
    wears_device = False

    def build(self, rng, mass_override=None):
        m, d = super().build(rng, mass_override=mass_override)
        self.healthy_ids = [m.actuator(n).id for n in self.HEALTHY_MUSCLES
                            if _has_actuator(m, n)]
        self.device = None          # no exoskeleton torque at all
        self.controller = None
        return m, d

    def set_input(self, model, data, u):
        for aid in self.healthy_ids:
            data.ctrl[aid] = u
        if self.wrist is not None:
            self.wrist.apply(model, data)


class KMatrixInGloveDevAdapter(GloveDevAdapter):
    """D1-D4 devices ported INTO the glove_dev environment.

    Why: the exoglove environment turned out to be unusable -- it has no arm
    and nothing holding the wrist, so the hand only ever reached its object
    by sagging into it under gravity, and once the wrist is held at its
    authored pose nothing can grasp there at all (healthy hand included,
    across 48 tested placements). The glove_dev env has a stabilised 6-DoF
    servo arm and placements validated against a healthy-hand control, so
    porting the devices to it gives one environment where every device can be
    measured on equal terms.

    What carries over unchanged: the arm stabilisation, the pinned contact
    model, the object mass/inertia correction, the placement, and the
    protocol. What differs per device is only the actuation.

    extra_thumb / extra_thumb_v2 do not exist in this model (they are bodies
    added to the exoglove env's own copy of myohand_body.xml). Their K rows
    are zero for every device ported here, which is asserted at build time
    rather than assumed -- a device relying on them would be silently
    weakened otherwise.
    """

    drive = "own_controller"

    def __init__(self, device_name, model_path, obj_name, r, half_h, mass,
                 label, bottom_offset=0.0, op_preshape=0.0):
        super().__init__(model_path, obj_name, r, half_h, mass, label,
                         bottom_offset=bottom_offset)
        self.device_name = device_name
        self.op_preshape = op_preshape
        self.preshape = (f"native OP at {op_preshape}" if op_preshape
                         else "none (device drives its own opposition)")

    def build(self, rng, mass_override=None):
        from exo_devices import DEVICES, JOINT_NAMES
        from grasp_controller import GraspController

        m, d = super().build(rng, mass_override=mass_override)
        # The exo tendons belong to the OTHER device -- silence them.
        d.ctrl[self.flex] = 0.0
        d.ctrl[self.thumb] = 0.0
        d.ctrl[m.actuator("OP").id] = self.op_preshape

        self.device = DEVICES[self.device_name]
        if not self.device.tau_max_is_calibrated:
            self.device.tau_max[:] = 0.03

        # Map device joint rows onto this model, tolerating absent joints but
        # refusing to silently drop a row that actually carries torque.
        probe = self.device.torque(np.ones(self.device.n_inputs))
        self.rows = []
        for idx, jname in enumerate(JOINT_NAMES):
            if _has_joint(m, jname):
                self.rows.append((idx, m.joint(jname).dofadr[0]))
            elif abs(float(probe[idx])) > 1e-9:
                raise RuntimeError(
                    f"{self.device_name} drives '{jname}' with "
                    f"{probe[idx]:.4f} N*m but that joint is absent from "
                    f"{self.model_path}; porting it would understate the device")
        obj_geom = self.obj_geoms[0]
        self.controller = GraspController(m, self.device, obj_geom,
                                          **dict(self.device.controller_overrides))
        self._build_strap(m)   # .device now exists, so driven_joints() is meaningful
        return m, d

    def set_input(self, model, data, u):
        data.ctrl[self.flex] = 0.0
        data.ctrl[self.thumb] = 0.0
        u_dev = self.controller.step(model, data)
        tau = self.device.torque(u_dev)
        if self.controller.row_gate is not None:
            tau = tau * self.controller.row_gate
        data.qfrc_applied[:] = 0.0
        for idx, dof in self.rows:
            data.qfrc_applied[dof] = float(tau[idx])
        if getattr(self, "strap", None) is not None:
            self.strap.apply(data)


class SplintedThumbAdapter(KMatrixInGloveDevAdapter):
    """D3 (Thimabut et al. 2022) WITH the rigid thumb splint it ships with.

    Closes a modelling gap that was documented but never fixed. D3 is a
    two-fingered device: it drives index and middle, and the thumb is not
    actuated at all -- it is immobilised by a C-bar splint at 50 deg MCP
    flexion (plus a latex glove for friction). The thumb is the surface the
    fingers grasp AGAINST. Simulating D3 without it meant simulating a device
    with no opposing digit, which is why it could not pinch anything on its own
    and had to borrow the OP muscle pre-shape to score at all -- an assist the
    real device does not need and does not have.

    A splint is not a muscle, and the difference is not cosmetic:
      * It is PASSIVE. No activation, so op_preshape is 0 here. The patient
        this device is for cannot oppose their own thumb; that is the point.
      * It is RIGID. It cannot be pushed out of the way by grasp forces, and
        equally it cannot adapt to the object. Modelled as a stiff PD hold on
        all four thumb DoFs rather than a muscle that can be overpowered.
      * It holds a DIFFERENT posture. The OP pre-shape leaves the thumb MP and
        IP resting at their EXTENSION limits (+40 and +21 deg measured); the
        splint flexes the MP to -45 deg. Those are opposite ends of the joint.

    Posture, and how much of it comes from the paper:
      mp_flexion   -- the paper's own 50 deg MCP flexion, clamped to this
                      model's -45 deg limit (MyoHand's thumb MP simply does not
                      travel to 50). Note flexion is NEGATIVE at this joint;
                      see the sign convention note in exo_devices.py.
      ip_flexion   -- 0, thumb held straight. A C-bar is a rigid arc; it does
                      not flex the IP. Not stated in the paper.
      cmc_*        -- held at the pose the OP pre-shape settles to, this repo's
                      existing reference for "thumb in opposition". The paper
                      gives no CMC angle, so this is an explicit assumption:
                      the splint's whole purpose is to park the thumb in
                      opposition, and this is the model's version of that.

    RESULT (15 trials, box, median hold):
        D3 with no thumb at all      6.7% grasp   0.11s   <- the old gap
        D3 borrowing the OP muscle  93.3% grasp   0.84s
        D3 with its actual splint  100.0% grasp   1.59s +- 0.06
    Modelling the real opposition mechanism is worth more than an order of
    magnitude, and turns the device that looked weakest into the strongest in
    the set. Read it with one caveat, though: this splint is an ideal rigid
    constraint, with no strap compliance and no soft tissue between the orthosis
    and the bone, so some of the margin is the rigidity rather than the design.
    Every actively-driven thumb here is competing against a perfect post.
    """

    # Measured once from the settled OP=0.5 pre-shape in this model; re-measure
    # if the hand model or the pre-shape level changes.
    SPLINT_POSE = {
        "cmc_abduction": -0.562,   # -32.2 deg, OP-settled opposition
        "cmc_flexion": 0.714,      # +40.9 deg, OP-settled opposition
        "mp_flexion": -0.785,      # -45.0 deg = the paper's 50 deg, at this model's limit
        "ip_flexion": 0.0,         # straight; the C-bar does not flex the IP
    }
    # The splint is written into the MODEL as joint stiffness about a spring
    # reference, not applied as a per-step force from set_input(). set_input is
    # only called during the closing ramp and the hold, whereas settle() runs
    # thousands of steps before either -- a per-step hold would let the thumb
    # sag out of the splint under gravity before the trial even started, and
    # the pose set in build() would be gone by the time it mattered. A passive
    # elastic constraint that is simply always present is also a better
    # description of what an orthosis is.
    SPLINT_STIFFNESS = 40.0   # stiff: an orthosis, not a muscle
    SPLINT_DAMPING = 4.0

    preshape = "rigid C-bar thumb splint (passive); no muscle pre-shape"

    def __init__(self, device_name, model_path, obj_name, r, half_h, mass,
                 label, bottom_offset=0.0):
        super().__init__(device_name, model_path, obj_name, r, half_h, mass,
                         label, bottom_offset=bottom_offset, op_preshape=0.0)
        # The parent sets self.preshape from op_preshape, which would report
        # this condition as having no pre-shape at all. It has one -- just a
        # passive orthotic rather than a muscle.
        self.preshape = self.__class__.preshape

    def build(self, rng, mass_override=None):
        m, d = super().build(rng, mass_override=mass_override)
        for jname, target in self.SPLINT_POSE.items():
            j = m.joint(jname)
            lo, hi = m.jnt_range[j.id]
            target = float(np.clip(target, lo, hi))
            m.jnt_stiffness[j.id] = self.SPLINT_STIFFNESS
            m.qpos_spring[j.qposadr[0]] = target
            m.dof_damping[j.dofadr[0]] = self.SPLINT_DAMPING
            # Start IN the splint rather than driving to it: the patient is
            # strapped in before the trial begins, not during it.
            d.qpos[j.qposadr[0]] = target
        mujoco.mj_forward(m, d)
        return m, d


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


def _has_body(model, name):
    try:
        model.body(name)
        return True
    except Exception:
        return False


def _has_joint(model, name):
    try:
        model.joint(name)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ one trial

def run_trial(adapter, seed, t_max, mass_override=None):
    rng = np.random.default_rng(seed)
    m, d = adapter.build(rng, mass_override=mass_override)
    dt = m.opt.timestep

    mujoco.mj_forward(m, d)
    if hasattr(adapter, "wrist"):
        adapter.wrist = WristHold(m, d)
    _, ok = settle(m, d, adapter.oid, adapter.settle_cap)
    if not ok:
        return {"outcome": Outcome.SETUP_FAIL, "hold_s": 0.0, "censored": False,
                "max_digits": 0, "mass": adapter.mass_used}

    ramp_steps = max(1, int(adapter.closing_seconds / dt))
    gate_steps_needed = int(GATE_MIN_SECONDS / dt)
    gate_run = 0
    gate_passed_step = None
    max_digits = 0

    # --- close the grasp, and watch for the contact gate -------------------
    # Cap the pre-release phase generously: the gate needs the ramp to finish
    # plus time to hold contact, and some devices reach contact late.
    for step in range(ramp_steps + int(6.0 / dt)):
        u = min(step / ramp_steps, 1.0)
        adapter.set_input(m, d, u)
        mujoco.mj_step(m, d)

        digits, ok_now = _gate_satisfied(m, d, adapter)
        max_digits = max(max_digits, len(digits))
        if ok_now:
            gate_run += 1
            if gate_run >= gate_steps_needed:
                gate_passed_step = step
                break
        else:
            gate_run = 0

    if gate_passed_step is None:
        return {"outcome": Outcome.NO_GRASP, "hold_s": 0.0, "censored": False,
                "max_digits": max_digits, "mass": adapter.mass_used}

    # --- remove support; t=0 for the hold clock ----------------------------
    adapter.release_support(m, d)
    href = _hand_ref(m)
    def _rel_z():
        return float(d.xpos[adapter.oid][2]) - (float(d.xpos[href][2]) if href is not None else 0.0)
    release_z = _rel_z()
    hold_steps = int(t_max / dt)
    for step in range(hold_steps):
        adapter.set_input(m, d, 1.0)
        mujoco.mj_step(m, d)
        if release_z - _rel_z() > DROP_THRESHOLD_M:
            return {"outcome": Outcome.SLIPPED, "hold_s": step * dt, "censored": False,
                    "max_digits": max_digits, "mass": adapter.mass_used}

    return {"outcome": Outcome.SURVIVED, "hold_s": t_max, "censored": True,
            "max_digits": max_digits, "mass": adapter.mass_used}


def _hand_ref(model):
    """Body the drop test is measured against. None falls back to world."""
    try:
        return model.body(HAND_REF_BODY).id
    except KeyError:
        return None


def _digit_contact_normals(model, data, adapter):
    """digit -> unit mean contact normal, oriented to point AWAY from the object.

    MuJoCo's contact normal (frame rows 0-2) points from geom1 toward geom2, so
    the sign has to be flipped when the object is geom2. Getting that backwards
    would invert every opposition test, so it is done once, here.
    """
    acc = {}
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in adapter.obj_geoms:
            other, sign = model.geom_bodyid[c.geom2], 1.0
        elif c.geom2 in adapter.obj_geoms:
            other, sign = model.geom_bodyid[c.geom1], -1.0
        else:
            continue
        n = np.array(c.frame[:3], dtype=float) * sign
        for dig, ids in adapter.digit_ids.items():
            if other in ids:
                acc.setdefault(dig, []).append(n)
    out = {}
    for dig, ns in acc.items():
        v = np.mean(ns, axis=0)
        norm = float(np.linalg.norm(v))
        out[dig] = v / norm if norm > 1e-9 else v
    return out


def _digits_in_contact(model, data, adapter):
    return set(_digit_contact_normals(model, data, adapter))


def _gate_satisfied(model, data, adapter):
    """(digits_in_contact, gate_is_satisfied) for this instant.

    Satisfied = enough distinct digits AND, unless disabled, at least one pair
    of them opposing. See GATE_REQUIRE_OPPOSITION for why the second clause
    exists.
    """
    normals = _digit_contact_normals(model, data, adapter)
    digits = set(normals)
    if len(digits) < GATE_MIN_DIGITS:
        return digits, False
    if not GATE_REQUIRE_OPPOSITION:
        return digits, True
    keys = sorted(normals)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if float(np.dot(normals[a], normals[b])) < GATE_OPPOSITION_DOT:
                return digits, True
    return digits, False


# ---------------------------------------------------------------- aggregation

def summarise(results):
    n = len(results)
    setup_fail = sum(r["outcome"] == Outcome.SETUP_FAIL for r in results)
    no_grasp = sum(r["outcome"] == Outcome.NO_GRASP for r in results)
    survived = sum(r["outcome"] == Outcome.SURVIVED for r in results)
    slipped = sum(r["outcome"] == Outcome.SLIPPED for r in results)
    grasped = [r for r in results if r["outcome"] in (Outcome.SLIPPED, Outcome.SURVIVED)]
    times = [r["hold_s"] for r in grasped]
    out = {
        "n": n,
        "grasp_rate": (n - no_grasp - setup_fail) / n if n else 0.0,
        "setup_fail": setup_fail,
        "survival_rate": survived / n if n else 0.0,   # headline metric
        "no_grasp": no_grasp, "slipped": slipped, "survived": survived,
        "censored": survived,
    }
    if times:
        out["hold_mean"] = statistics.fmean(times)
        out["hold_median"] = statistics.median(times)
        out["hold_sd"] = statistics.stdev(times) if len(times) > 1 else 0.0
    else:
        out["hold_mean"] = out["hold_median"] = out["hold_sd"] = 0.0
    return out


def fmt_summary(label, s, extra=""):
    star = " *censored" if s["censored"] else ""
    return (f"{label:<34} surv={s['survival_rate']*100:5.1f}%  "
            f"grasp={s['grasp_rate']*100:5.1f}%  "
            f"hold_med={s['hold_median']:5.2f}s  sd={s['hold_sd']:4.2f}  "
            f"[NG={s['no_grasp']} SL={s['slipped']} SV={s['survived']}"
            f"{' SETUPFAIL=' + str(s['setup_fail']) if s.get('setup_fail') else ''}]{star}{extra}")


# ---------------------------------------------------------------------- CLI

# The three graspable objects, and everything that differs between them. Until
# now only the box had ported devices, so every cross-device conclusion in this
# repo rested on one object at one placement -- which is a single-object result
# reported as a device comparison. Note the objects differ in MASS as well as
# shape (0.097 / 0.171 / 0.349 kg), because they are the real YCB items; a
# device doing worse on the can is not necessarily losing to its geometry.
OBJECTS = {
    "box":  ("myohand_glove_dev.xml",      "009_gelatin_box",     0.036, 0.014, 0.097, -0.044),
    "tuna": ("myohand_glove_dev_tuna.xml", "007_tuna_fish_can",   0.042, 0.016, 0.171,  0.000),
    "can":  ("myohand_glove_dev_can.xml",  "005_tomato_soup_can", 0.033, 0.050, 0.349,  0.000),
}

PORTED_DEVICES = (
    "D1_underactuated_distal_calibrated",
    "D2_synergy_cross_finger_calibrated",
    "D3_uniform_single_dof_calibrated",
    "D4_v2_hybrid_per_finger_calibrated",
    "D4_v2_opposition_first_calibrated",
)


def glove_dev_adapters():
    """Every condition, over every object.

    Key naming: the box keys stay UNSUFFIXED ("portOP_D1_...", "splint_D3",
    "healthy_box") so existing runs, scripts and saved result files keep
    working unchanged; the other objects take a "_can" / "_tuna" suffix.
    """
    base = "myogloves_dev/models"
    pool = {}
    for obj, (xml, body, r, half_h, mass, bottom) in OBJECTS.items():
        path = f"{base}/{xml}"
        sfx = "" if obj == "box" else f"_{obj}"
        common = dict(bottom_offset=bottom)

        pool[f"glove_dev_{obj}"] = GloveDevAdapter(
            path, body, r, half_h, mass, f"glove_dev/{obj}", **common)
        pool[f"healthy_{obj}"] = HealthyHandAdapter(
            path, body, r, half_h, mass, f"HEALTHY/{obj}", **common)
        pool[f"splint_D3{sfx}"] = SplintedThumbAdapter(
            "D3_uniform_single_dof_calibrated", path, body, r, half_h, mass,
            f"SPLINT/{obj}/D3_uniform_single_dof", **common)

        for d in PORTED_DEVICES:
            short = d.replace("_calibrated", "")
            pool[f"port_{d}{sfx}"] = KMatrixInGloveDevAdapter(
                d, path, body, r, half_h, mass, f"PORT/{obj}/{short}", **common)
            pool[f"portOP_{d}{sfx}"] = KMatrixInGloveDevAdapter(
                d, path, body, r, half_h, mass, f"PORT+OP/{obj}/{short}",
                op_preshape=0.5, **common)
    return pool


def kmatrix_adapters(names):
    env = "myogloves_dev/models/myohand_exoglove_env.xml"
    return {n: KMatrixAdapter(n, env) for n in names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "run", "masssweep", "placesweep"])
    ap.add_argument("--devices", default="")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--tmax", type=float, default=T_MAX_DEFAULT)
    ap.add_argument("--masses", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--friction", default="myoassist",
                    choices=["myoassist", "legacy"],
                    help="object friction: myoassist [1.0,0.005,0.0001] or legacy [2.5,0.02,0.002]")
    ap.add_argument("--strap", type=float, default=None,
                    help="series-elastic transmission stiffness N*m/rad "
                         "(0 = ideal; 0.1 = best fit to Zhao et al.)")
    ap.add_argument("--xs", default="")   # placesweep grid
    ap.add_argument("--ys", default="")
    ap.add_argument("--zs", default="")
    a = ap.parse_args()

    global OBJECT_FRICTION
    OBJECT_FRICTION = list(LEGACY_FRICTION if a.friction == 'legacy' else MYOASSIST_FRICTION)
    print(f'object friction: {OBJECT_FRICTION}  ({a.friction})')
    if a.strap is not None:
        global STRAP_STIFFNESS
        STRAP_STIFFNESS = a.strap
    print(f"transmission: STRAP_STIFFNESS = {STRAP_STIFFNESS} N*m/rad"
          + ("  (ideal)" if STRAP_STIFFNESS == 0.0 else "  (as-worn)"))

    pool = glove_dev_adapters()
    pool["healthy_exoglove_env"] = HealthyExogloveEnvAdapter(
        "bare_msk", "myogloves_dev/models/myohand_exoglove_env.xml",
        label="HEALTHY/exoglove_env")
    pool.update(kmatrix_adapters([
        "bare_msk", "D1_underactuated_distal_calibrated",
        "D2_synergy_cross_finger_calibrated", "D3_uniform_single_dof_calibrated",
        "D4_v2_hybrid_per_finger_calibrated",
    ]))

    names = [n for n in a.devices.split(",") if n] or list(pool)
    trials = 3 if a.mode == "pilot" else a.trials

    if a.mode == "placesweep":
        # WARNING -- ranking by hold_median alone is a TRAP, and it has already
        # produced a wrong answer once. hold_median is conditioned on having
        # grasped at all (summarise() takes the median over grasped trials
        # only), so a placement that fails most trials but holds well in the
        # few it happens to win scores HIGHER than one that grasps reliably.
        #
        # Concretely: a 3x3x3 screen over all five devices unanimously ranked
        # (0.09, 0.10, 0.22) first. Re-tested at 15 trials it was clearly worse
        # -- grasp rate fell from 93-100% to 40-80%, and the healthy
        # feasibility control collapsed from 2.06s to 0.21s, i.e. a placement
        # where an unimpaired hand can barely hold the object at all. The
        # authored (0.07, 0.10, 0.22) survived and is still the one in use.
        #
        # Read grasp_rate first, and always re-check a winner against the
        # healthy control before adopting it.
        import itertools
        ad = pool[names[0]]
        xs = [float(v) for v in a.xs.split(",")]
        ys = [float(v) for v in a.ys.split(",")]
        zs = [float(v) for v in a.zs.split(",")]
        ranked = []
        for z, x, y in itertools.product(zs, xs, ys):
            ad.pos_override = (x, y, z)
            res = [run_trial(ad, seed=2000 + i, t_max=a.tmax) for i in range(trials)]
            sm = summarise(res)
            ranked.append((sm["hold_median"], sm["grasp_rate"], x, y, z))
            print(f"  ({x},{y},{z}) grasp={sm['grasp_rate']*100:5.1f}% "
                  f"hold_med={sm['hold_median']:5.2f}s", flush=True)
        ranked.sort(reverse=True)
        print("=" * 60)
        print(f"BEST PLACEMENTS for {ad.name}:")
        for h, g, x, y, z in ranked[:8]:
            print(f"  ({x},{y},{z}) grasp={g*100:5.1f}% hold_med={h:5.2f}s")
        return

    all_out = {}
    for name in names:
        ad = pool[name]
        if a.mode == "masssweep":
            masses = [float(v) for v in a.masses.split(",") if v]
            for mass in masses:
                res = [run_trial(ad, seed=1000 + i, t_max=a.tmax, mass_override=mass)
                       for i in range(trials)]
                s = summarise(res)
                all_out[f"{name}@{mass}kg"] = s
                print(fmt_summary(f"{ad.name} @{mass}kg", s), flush=True)
        else:
            res = [run_trial(ad, seed=1000 + i, t_max=a.tmax) for i in range(trials)]
            s = summarise(res)
            s["preshape"] = ad.preshape
            s["drive"] = ad.drive
            all_out[name] = s
            print(fmt_summary(ad.name, s), flush=True)

    if a.out:
        with open(a.out, "w") as f:
            json.dump(all_out, f, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
