"""Applies exoskeleton coupling-matrix torques (tau_exo = K @ u) to MyoHand.

Joint order (index + middle + thumb + ring + little, 16 DoF):
    index MCP, index PIP, index DIP, middle MCP, middle PIP, middle DIP,
    thumb CMC abduction, thumb CMC flexion, thumb MP flexion, thumb IP flexion,
    ring MCP, ring PIP, ring DIP, little MCP, little PIP, little DIP
Positive torque = flexion (or abduction, for cmc_abduction) assistance.

Which devices drive the thumb, per their primary sources (an earlier version of
this file asserted "D1/D2/D3 don't drive the thumb at all -- their papers only
ever published index+middle data"; that was wrong for D1 and D2, and each error
silently deleted the digit its own paper calls essential):
  D1 (Zhao et al. 2025)     -- YES. Five finger exoskeletons, five pushrod
                               motors, thumb mounted on the side of the
                               backplate with its own dedicated motor (Sec. 2.4).
                               It is in fact the only FIVE-DIGIT device here.
  D2 (Alicea et al. 2021)   -- YES, flexion only. "Only three digits are
                               actively supported, these being the thumb,
                               index, and middle fingers."
  D3 (Thimabut et al. 2022) -- NO. Two-fingered device; the thumb is held
                               immobile by a C-bar splint, not actuated.
  D4 (Gerez et al. 2020)    -- YES, and it is the only device with a separate,
                               independently-motored thumb OPPOSITION tendon
                               (Sec. III / Fig. 4) in addition to thumb flexion.

cmc_abduction was added after discovering (via grasp testing) that flexion
alone cannot produce genuine thumb OPPOSITION: driving cmc_flexion/mp_flexion/
ip_flexion only curls the thumb in roughly the same arc as the fingers (their
fingertip velocity directions were nearly parallel, not opposing), because
that's what flexion does anatomically -- true opposition is what the thumb's
saddle joint's ABDUCTION is for. Without it, no amount of "grip force" can
create a real pincer grasp, which is exactly why the earlier D4 grasp test
failed: squeezing along non-opposing directions just ejects the object faster.

tau_max (N*m) is mostly still a placeholder tuned for visible grasp behavior in
the demo/tests. The "_calibrated" device variants below are the exception:
their K and tau_max are derived from real MyoHand moment arms (MOMENT_ARMS_MM),
not tuned by hand -- see the comment block above them for how and why.
"""

import numpy as np

JOINT_NAMES = [
    "mcp2_flexion",    # index MCP
    "pm2_flexion",     # index PIP
    "md2_flexion",     # index DIP
    "mcp3_flexion",    # middle MCP
    "pm3_flexion",     # middle PIP
    "md3_flexion",     # middle DIP
    "cmc_abduction",   # thumb CMC abduction -- the DoF that actually creates opposition
    "cmc_flexion",     # thumb CMC flexion
    "mp_flexion",      # thumb MP (metacarpophalangeal)
    "ip_flexion",      # thumb IP (interphalangeal)
    "mcp4_flexion",    # ring MCP
    "pm4_flexion",     # ring PIP
    "md4_flexion",     # ring DIP
    "mcp5_flexion",    # little MCP
    "pm5_flexion",     # little PIP
    "md5_flexion",     # little DIP
    "extra_thumb",     # Gerez et al. 2020's actual "telescopic extra thumb" -- confirmed via the primary
                       # source PDF to be a SEPARATE, pneumatically-inflated 6th-digit-like appendage
                       # mounted near the palm (Table 1: 80mm long, 10mm thick, 18g; its own dedicated
                       # air pump), NOT an extension of the anatomical thumb -- an earlier attempt here
                       # wrongly modeled it as a thumb-tip telescoping joint before the actual paper was
                       # available. See myohand_body.xml's extra_thumb joint (mounted on capitate, a
                       # stable central palm bone) for the corrected physical structure. Zero for every
                       # device below except the ones that explicitly add this channel (see
                       # D4_v2_extra_thumb_calibrated) -- it is not part of any existing device's thumb
                       # column; the source paper drives it from its own independent motor/pump.
    "extra_thumb_v2",  # Second mounting/axis for the same real device feature -- v1 (mounted on
                       # capitate, aimed at the closed-fist fingertip-convergence point) looked visually
                       # disconnected from the glove (shoots out sideways from the wrist), nothing like
                       # the paper's own Fig. 3 photo. v2 is mounted on trapezium (the thumb's own carpal
                       # bone, closer to where the photo shows it attaching) and aimed at the OPEN-hand
                       # fingertip position instead, so it extends forward roughly alongside the fingers
                       # like the photo shows, not into the eventual fist-closure point. Kept as a
                       # SEPARATE device (D4_v2_extra_thumb_v2_calibrated) rather than replacing v1, so
                       # both mounting choices stay comparable.
]

PASSIVE_COUPLED_JOINTS = [
    "mcp4_flexion",
    "pm4_flexion",
    "md4_flexion",
    "mcp5_flexion",
    "pm5_flexion",
    "md5_flexion",
]

# Passive ring/little coupling, not extra actuators: these torques are a soft
# byproduct of driving index/middle, standing in for tendon-network and soft-
# tissue coupling in the hand. They are intentionally weaker than the active
# driven digits and only apply flexion, so we do not claim independent hardware
# channels the source devices never published.
PASSIVE_COUPLING_GAIN = np.array([0.22, 0.18, 0.12, 0.12, 0.10, 0.07], dtype=float)
PASSIVE_SOURCE_ROWS = [
    JOINT_NAMES.index("mcp3_flexion"),
    JOINT_NAMES.index("pm3_flexion"),
    JOINT_NAMES.index("md3_flexion"),
    JOINT_NAMES.index("mcp3_flexion"),
    JOINT_NAMES.index("pm3_flexion"),
    JOINT_NAMES.index("md3_flexion"),
]

# Real per-joint moment arms (mm), from MyoHand's own tendons via
# compute_moment_arms.py: FDP2-FDP5 for the four fingers, FPL_tendon for the thumb
# (the analogous single flexor tendon crossing all four thumb joints -- FPL
# also has real leverage over cmc_abduction, not just the flexion DoFs, so the
# same single-tendon assumption used everywhere else in this file still
# applies here: one motor, one tension, real moment arm at each joint it
# crosses). Index/middle moment arm roughly halves from MCP to DIP; the
# thumb's flexion chain does the OPPOSITE -- it roughly SIXES from CMC to IP
# (1.49 -> 8.81 mm) since FPL's leverage at the CMC is small. Either way, it
# is NOT constant across a finger's joints -- the "moment-arm-invariant"
# assumption behind the literature K matrices does not hold for this anatomy,
# in either direction.
MOMENT_ARMS_MM = {
    "mcp2_flexion": 9.583,
    "pm2_flexion": 6.862,
    "md2_flexion": 4.333,
    "mcp3_flexion": 8.457,
    "pm3_flexion": 7.546,
    "md3_flexion": 2.664,
    "cmc_abduction": 3.856,
    "cmc_flexion": 1.492,
    "mp_flexion": 7.120,
    "ip_flexion": 8.813,
    # Ring/little, from FDP4/FDP5. Added when D1 turned out to be a five-finger
    # device; they also retire the proxy profile D4_v2's shared ulnar channel
    # previously had to borrow from the middle finger. Note the ring finger's
    # profile is NOT a scaled copy of the middle's -- its PIP arm (7.142) is
    # LARGER than its MCP arm (6.581), so the peak sits at PIP, unlike every
    # other finger here. A borrowed profile could never have shown that.
    "mcp4_flexion": 6.581,
    "pm4_flexion": 7.142,
    "md4_flexion": 2.305,
    "mcp5_flexion": 5.874,
    "pm5_flexion": 4.836,
    "md5_flexion": 2.654,
}


class ExoDevice:
    def __init__(self, name, K, tau_max, n_inputs, tau_max_is_calibrated=False,
                 passive_coupling=True, controller_overrides=None):
        self.name = name
        self.n_inputs = n_inputs
        self.K = np.asarray(K, dtype=float).reshape(len(JOINT_NAMES), n_inputs)
        self.tau_max = np.asarray(tau_max, dtype=float).reshape(n_inputs)
        # True only when tau_max is derived from real force/moment-arm data (see D1's
        # calibrated variant) rather than tuned by hand for visible demo behavior.
        self.tau_max_is_calibrated = tau_max_is_calibrated
        self.passive_coupling = passive_coupling
        self.controller_overrides = dict(controller_overrides or {})

    def torque(self, u):
        """u: array-like of length n_inputs, each in [0, 1]. Returns per-joint torques."""
        u = np.clip(np.asarray(u, dtype=float).reshape(self.n_inputs), 0.0, 1.0)
        return self.K @ (u * self.tau_max)


DEVICES = {
    # Zhao, Guo, Yang, Li, Zhao, Qu, Li, Liu, Wang & Bu, "A novel underactuated
    # exoskeleton rehabilitation glove for hand flexion and extension training,"
    # Biomimetic Intelligence and Robotics 5, 100248, 2025 -- Section 3.4 / Fig. 15
    # reports measured per-joint loads on the index finger (FSR sensor, n=1
    # subject): MCP 4.1 N, PIP 5.2 N, DIP 5.6 N. K here is those forces
    # normalized by the max (DIP): [0.73, 0.93, 1.00] -- i.e. raw force ratio,
    # NOT torque ratio (that correction is what the "_calibrated" variant below
    # does).
    #
    # CORRECTED from the primary source: this is a FIVE-FINGER device with FIVE
    # independent motors, not an index+middle one. "The complete hand
    # exoskeleton consists of five finger exoskeletons attached to a backplate.
    # The thumb is mounted on the side of the backplate... A pushrod motor is
    # placed on the side of the backplate to drive the thumb, while four
    # additional pushrod motors are positioned above to drive the remaining
    # fingers" (Sec. 2.4). The previous definition zeroed the thumb, ring and
    # little rows AND collapsed the five motors into one shared input -- so
    # three of the device's five actuated digits simply did not exist.
    # n_inputs=5 now matches the motor count; the per-digit force split is the
    # paper's own "uniform drive structure" claim (Sec. 2.2: fingers "vary in
    # size, [but] their skeletal structures are similar, allowing for a uniform
    # drive structure"), since only the index was instrumented.
    #
    # Thumb: the paper models it as MP + IP only (Sec. 2.1), so cmc_flexion
    # stays 0. cmc_abduction also stays 0 -- this device is flexion/extension
    # only by design ("only the flexion/extension function of the MCP joint is
    # retained", Sec. 2.2; the title itself is "for hand flexion and extension
    # training"). MP gets the proximal force (4.1 N) and IP the distal (5.6 N),
    # following the paper's own explanation that the underactuated transmission
    # "prioritizes the force output to the distal structure".
    "D1_underactuated_distal": ExoDevice(
        name="D1_underactuated_distal",
        K=[
            [0.73, 0.00, 0.00, 0.00, 0.00],
            [0.93, 0.00, 0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.73, 0.00, 0.00, 0.00],
            [0.00, 0.93, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],  # cmc_abduction: flexion/extension device only
            [0.00, 0.00, 0.00, 0.00, 0.00],  # cmc_flexion: paper's thumb model has no CMC DoF
            [0.00, 0.00, 0.73, 0.00, 0.00],  # thumb MP (proximal, 4.1 N)
            [0.00, 0.00, 1.00, 0.00, 0.00],  # thumb IP (distal, 5.6 N)
            [0.00, 0.00, 0.00, 0.73, 0.00],
            [0.00, 0.00, 0.00, 0.93, 0.00],
            [0.00, 0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.73],
            [0.00, 0.00, 0.00, 0.00, 0.93],
            [0.00, 0.00, 0.00, 0.00, 1.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],  # extra_thumb: not this device
            [0.00, 0.00, 0.00, 0.00, 0.00],  # extra_thumb_v2: not this device
        ],
        tau_max=[1.0] * 5,  # placeholder; see D1_underactuated_distal_calibrated for the real-torque version
        n_inputs=5,
        passive_coupling=False,  # ring/little are ACTIVELY driven here, not passively coupled
    ),
    # Cross-finger ratio (index vs. middle) is now literature-derived, not guessed:
    # Alicea, Xiloyannis, Chiaradia, Barsotti, Frisoli & Masia, "A soft, synergy-based
    # robotic glove for grasping assistance," Wearable Technologies 2(e4), 2021,
    # Figure 3b. All three tendons (thumb, index, middle) share one motor shaft via a
    # multichannel pulley with channel diameters 2.2 / 0.95 / 1.93 cm, read off the
    # figure in the order given by its own caption ("...channels for the thumb,
    # middle, and index fingers"). Same shaft -> same angular velocity -> tendon
    # velocity (and hence stroke/torque delivered) scales with channel diameter.
    # We only model index+middle, so the thumb channel is dropped and the ratio is
    # renormalized: middle/index = 0.95/1.93 = 0.492. The within-finger MCP/PIP/DIP
    # shape [0.85, 0.95, 1.00] is still a design-intent placeholder -- the paper
    # gives no per-joint split, only the fingertip-anchored-tendon description that
    # the "_calibrated" variant below corrects for.
    "D2_synergy_cross_finger": ExoDevice(
        name="D2_synergy_cross_finger",
        K=[0.85, 0.95, 1.00, 0.4184, 0.4677, 0.4922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # no thumb telescope or v2 either
        tau_max=[1.0],  # placeholder; see fingertip-force ambiguity note below
        n_inputs=1,
    ),
    # Thimabut, Terachinda & Kitisomprayoonkul, "Effectiveness of a Soft Robotic
    # Glove to Assist Hand Function in Stroke Patients: A Cross-Sectional Pilot
    # Study," Rehabilitation Research and Practice 2022, Art. 3738219. The paper
    # reports no per-joint FORCE split (unlike D1/D4), but it does give real,
    # distinguishing per-joint maximal flexion angles for this exact device:
    # "52 deg at the MCP joint, 80 deg at the PIP joint, and 75 deg at the DIP
    # joint" -- a single hoist-and-cable motor drives both index and middle
    # fingers together (matches this device's n_inputs=1, both-fingers-coupled
    # structure). K below is those angles normalized by the max (PIP):
    # 52/80=0.65, 80/80=1.00, 75/80=0.9375. This replaces the old "fully
    # uniform, no data" placeholder -- NOTE it is an ROM ratio (an achieved
    # outcome under the device's own actuation + finger dynamics), not a
    # measured force/torque ratio like D1's, so it's a proxy for relative
    # coupling strength, not a first-principles one.
    "D3_uniform_single_dof": ExoDevice(
        name="D3_uniform_single_dof",
        K=[0.65, 1.00, 0.9375, 0.65, 1.00, 0.9375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # no thumb telescope or v2 either
        tau_max=[1.0],  # placeholder; no published force data to calibrate against (see comment above)
        n_inputs=1,
    ),
    # Gerez, Gao, Dwivedi & Liarokapis, "A Hybrid, Wearable Exoskeleton Glove
    # Equipped With Variable Stiffness Joints, Abduction Capabilities, and a
    # Telescopic Thumb," IEEE Access 8, 173345-173358, 2020 (New Dexterity
    # Group, U. Auckland; open-source: github.com/newdexterity/Hybrid-Exoskeleton-Glove).
    # NOTE: this is a DIFFERENT paper from "Gerez et al. 2019" (the simpler
    # tendon-only exo-glove cited secondhand in Alicea et al.'s Table 3 as the
    # "SEM glove," 55g/15N) -- the "500 N tendon rating" already used below is
    # this 2020 paper's own number ("tendons... can withstand forces up to
    # 500 N"), and "hybrid" in this device's name matches this paper's title,
    # not the 2019 one. Index, middle, AND THUMB each get "dedicated pulleys of
    # individual motors" (Sec. III) -- this is the only device in this set
    # with a real, independently-actuated thumb, so it's the only one where a
    # third input channel is grounded in the source paper rather than faked.
    #
    # CORRECTED from the primary source: thumb FLEXION and thumb OPPOSITION are
    # driven by two DIFFERENT motors, not one. The device has six tendons on
    # five motors: "a tendon connected to each of the tendon termination
    # structures [five fingertips] and an extra tendon that is connected to the
    # thumb's interphalangeal joint region, facilitating the execution of the
    # opposition motion" (Sec. III). Fig. 4's block diagram lists "Fingers
    # Flexion" (4 motors) and "Thumb Opposition" (1 motor) as separate
    # actuation paths, Fig. 3 labels a distinct "Thumb Abduction Internal
    # Cable", and Fig. 5's control app exposes "Thumb Abduction" as its own
    # button. The previous definition bundled cmc_abduction into the thumb
    # flexion column, which forced opposition and curl to be one commanded
    # value -- exactly the coupling this device was designed to avoid ("A
    # tendon-driven solution for the thumb abduction/opposition was chosen over
    # a soft actuator based solution, in order to avoid the obstruction of the
    # [purlicue] region"). cmc_abduction is now its own channel.
    #
    # Uncalibrated K shape (0.80/0.90/1.00) remains a design-intent placeholder;
    # this paper reports no per-joint force/ROM split for any digit to ground
    # it in.
    "D4_hybrid_per_finger": ExoDevice(
        name="D4_hybrid_per_finger",
        K=[
            [0.80, 0.00, 0.00, 0.00],
            [0.90, 0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00],
            [0.00, 0.80, 0.00, 0.00],
            [0.00, 0.90, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1.00],  # cmc_abduction: its own opposition motor
            [0.00, 0.00, 0.80, 0.00],
            [0.00, 0.00, 0.90, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00],  # extra_thumb: this device has no extra thumb channel
            [0.00, 0.00, 0.00, 0.00],  # extra_thumb_v2: this device has no extra thumb channel
        ],
        tau_max=[1.0, 1.0, 1.0, 1.0],  # placeholder; real device has a 500 N tendon rating (safety/material limit,
        n_inputs=4,                     # not a typical operating torque) -- see the _calibrated variant for the real value
    ),
    # Paper-faithful v2: keep index/middle/thumb as dedicated channels, but add a
    # fourth shared tendon channel for ring+pinky (Sec. III: "ring and pinky...
    # coupled together and connected to a single pulley and motor"). This leaves
    # the current 3-channel D4 untouched while providing a separate version that
    # matches the motor count more closely.
    "D4_v2_hybrid_per_finger": ExoDevice(
        name="D4_v2_hybrid_per_finger",
        K=[
            [0.80, 0.00, 0.00, 0.00, 0.00],
            [0.90, 0.00, 0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.80, 0.00, 0.00, 0.00],
            [0.00, 0.90, 0.00, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],  # cmc_abduction: its own opposition motor
            [0.00, 0.00, 0.80, 0.00, 0.00],
            [0.00, 0.00, 0.90, 0.00, 0.00],
            [0.00, 0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.70, 0.00],
            [0.00, 0.00, 0.00, 0.80, 0.00],
            [0.00, 0.00, 0.00, 0.90, 0.00],
            [0.00, 0.00, 0.00, 0.46, 0.00],
            [0.00, 0.00, 0.00, 0.56, 0.00],
            [0.00, 0.00, 0.00, 0.66, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],  # extra_thumb: this device has no extra thumb channel
            [0.00, 0.00, 0.00, 0.00, 0.00],  # extra_thumb_v2: this device has no extra thumb channel
        ],
        tau_max=[1.0, 1.0, 1.0, 1.0, 1.0],
        n_inputs=5,
        passive_coupling=False,
    ),
    # --- MyoHand-calibrated variants ---
    # For D1/D4 (published per-joint/per-tendon force data), these replace the
    # literature force ratios with real joint-torque ratios (tau_j = F_j * MOMENT_ARMS_MM[j]).
    # Report both the original and calibrated K in the paper: the discrepancy
    # (D1 flips from distal-biased to proximal-biased) is itself a finding.
    # tau_j = F_j (Zhao et al. 2025: MCP 4.1N, PIP 5.2N, DIP 5.6N, index-finger
    # measurement reused for the other three fingers per the paper's "uniform
    # drive structure" claim) * MOMENT_ARMS_MM[j]. Five independent motors ->
    # each column is normalized by ITS OWN max, matching the same
    # independent-actuator-per-digit convention D4 uses (a shared normalizer
    # would only make sense for a shared tendon):
    #   index:  39.29 / 35.68 / 24.26 N*mm -> 1.000 / 0.908 / 0.618  (max 39.29)
    #   middle: 34.67 / 39.24 / 14.92      -> 0.884 / 1.000 / 0.380  (max 39.24)
    #   thumb:  MP 4.1*7.120=29.19, IP 5.6*8.813=49.35 -> 0.591 / 1.000 (max 49.35)
    #   ring:   26.98 / 37.14 / 12.91      -> 0.727 / 1.000 / 0.348  (max 37.14)
    #   little: 24.08 / 25.15 / 14.86      -> 0.958 / 1.000 / 0.591  (max 25.15)
    # Note the index/middle absolute torques are UNCHANGED from the previous
    # (thumb-less, single-channel) version -- only the digits that were missing
    # are added, so this correction adds capability without rescaling what was
    # already there.
    #
    # Worth reporting: the paper's headline finding is that its underactuated
    # transmission is DISTAL-biased (MCP 4.1 N < PIP 5.2 N < DIP 5.6 N, "the
    # exoskeleton's force transmission path tends to favor the mid-distal end").
    # Once real moment arms are applied, that reverses for the index finger
    # (1.000 / 0.908 / 0.618, proximal-biased) and lands on the PIP for the
    # middle and ring fingers -- the distal bias in FORCE does not survive
    # conversion to joint TORQUE, because the moment arm shrinks faster than
    # the force grows. The thumb is the exception: FPL's arm grows distally, so
    # there the paper's distal bias is reinforced rather than cancelled.
    "D1_underactuated_distal_calibrated": ExoDevice(
        name="D1_underactuated_distal_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000, 0.000],
            [0.908, 0.000, 0.000, 0.000, 0.000],
            [0.618, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.884, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000, 0.000],
            [0.000, 0.380, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],  # cmc_abduction: flexion/extension device only
            [0.000, 0.000, 0.000, 0.000, 0.000],  # cmc_flexion: paper's thumb model has no CMC DoF
            [0.000, 0.000, 0.591, 0.000, 0.000],
            [0.000, 0.000, 1.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.727, 0.000],
            [0.000, 0.000, 0.000, 1.000, 0.000],
            [0.000, 0.000, 0.000, 0.348, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.958],
            [0.000, 0.000, 0.000, 0.000, 1.000],
            [0.000, 0.000, 0.000, 0.000, 0.591],
            [0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb: not this device
            [0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb_v2: not this device
        ],
        # N*m, per-channel column maxima above -- none of these are placeholders
        tau_max=[0.03929, 0.03924, 0.04935, 0.03714, 0.02515],
        n_inputs=5,
        tau_max_is_calibrated=True,
        passive_coupling=False,
    ),
    # Single fingertip tendon under one tension per finger -> K shape here IS
    # the moment-arm profile itself (tau_j = F_max * r_j, same F_max per finger).
    # tau_max is now real too: Table 1 of Gerez et al. 2020 reports "Maximum
    # fingertip force 13.8 N" (unjammed, Sec. V-B, dummy-hand + load-cell test).
    # The paper doesn't say which finger was tested, so -- same assumption as
    # D1's index-to-middle reuse -- this applies that single measured max to
    # index, middle, AND thumb independently (each has its own dedicated motor,
    # per Sec. III, so treating them as structurally identical is reasonable
    # but not something the paper states directly). Each column is normalized
    # by ITS OWN max joint, matching the independent-tendon-per-digit design:
    #   index:      13.8 N * 9.583 mm = 132.25 N*mm = 0.13225 N*m  (index MCP, K's own max)
    #   middle:     13.8 N * 8.457 mm = 116.71 N*mm = 0.11671 N*m  (middle MCP, K's own max)
    #   thumb flex: 13.8*1.492=20.59 (CMC flexion) | 13.8*7.120=98.26 (MP) | 13.8*8.813=121.62 (IP)
    #               -> normalized by 121.62: [0.169, 0.808, 1.000]; tau_max = 0.12162 N*m
    #   thumb oppos: 13.8 N * 3.856 mm = 53.21 N*mm = 0.05321 N*m, its OWN channel (see the
    #               correction note on D4_hybrid_per_finger above). The paper gives no force for
    #               this tendon specifically; reusing the 13.8 N fingertip figure is defensible
    #               because it is the same motor model (all five are XM430-W350-T, 1.4 N*m max)
    #               pulling the same UHMWPE tendon. NOT the 15.8 N "maximum abduction force" from
    #               Table 1 -- that number belongs to the PNEUMATIC inter-finger chambers, and the
    #               paper is explicit that the thumb went tendon-driven instead of pneumatic.
    #           NOTE the flexion-only rows (cmc_flexion/mp/ip) are proximal-WEAK / distal-strong --
    #           the opposite of index/middle's calibrated profile -- because the thumb's real
    #           flexion moment arm grows from CMC to IP instead of shrinking (see MOMENT_ARMS_MM
    #           comment). cmc_abduction is the row that matters most for actually grasping anything:
    #           grasp testing found that flexion alone only curls the thumb in the same arc as the
    #           fingers (their fingertip velocities were nearly parallel, dot~+0.999, not opposing) --
    #           adding real-moment-arm-weighted abduction is what first produced genuine opposition
    #           (dot flipped negative). This is not a tuned parameter: it's the same F*r derivation
    #           used for every other row here, just for the DoF that was previously left at zero.
    "D4_hybrid_per_finger_calibrated": ExoDevice(
        name="D4_hybrid_per_finger_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000],
            [0.716, 0.000, 0.000, 0.000],
            [0.452, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000],
            [0.000, 0.892, 0.000, 0.000],
            [0.000, 0.315, 0.000, 0.000],
            [0.000, 0.000, 0.000, 1.000],  # cmc_abduction: its own opposition motor
            [0.000, 0.000, 0.169, 0.000],
            [0.000, 0.000, 0.808, 0.000],
            [0.000, 0.000, 1.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000],  # extra_thumb: this device has no extra thumb channel (see D4_v2_extra_thumb_calibrated)
            [0.000, 0.000, 0.000, 0.000],  # extra_thumb_v2: this device has no extra thumb channel
        ],
        tau_max=[0.13225, 0.11671, 0.12162, 0.05321],  # N*m; not a placeholder -- see derivation above
        n_inputs=4,
        tau_max_is_calibrated=True,
    ),
    # v2 calibrated: index/middle/thumb-flexion/thumb-opposition channels keep the
    # real F*r calibration from D4 above. The shared ring+pinky channel is now real
    # too -- FDP4/FDP5 moment arms were extracted (see MOMENT_ARMS_MM), replacing the
    # proxy that borrowed the middle finger's profile and scaled pinky down by eye:
    #   ring:   13.8 * 6.581/7.142/2.305 = 90.82 / 98.56 / 31.81 N*mm
    #   little: 13.8 * 5.874/4.836/2.654 = 81.06 / 66.74 / 36.63 N*mm
    #   normalized by the channel max (ring PIP, 98.56): ring 0.921/1.000/0.323,
    #   little 0.822/0.677/0.372; tau_max = 0.09856 N*m
    # The proxy had ring peaking at MCP (0.883/0.787/0.278) because it copied the
    # middle finger; the real ring finger peaks at PIP, and the real pinky is
    # substantially stronger relative to ring than the hand-set 0.65 scale assumed.
    # Same 13.8 N tendon force as the other flexion channels: one motor, one shared
    # pulley for both digits, per Sec. III.
    "D4_v2_hybrid_per_finger_calibrated": ExoDevice(
        name="D4_v2_hybrid_per_finger_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000, 0.000],
            [0.716, 0.000, 0.000, 0.000, 0.000],
            [0.452, 0.000, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000, 0.000],
            [0.000, 0.892, 0.000, 0.000, 0.000],
            [0.000, 0.315, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 1.000],  # cmc_abduction: its own opposition motor
            [0.000, 0.000, 0.169, 0.000, 0.000],
            [0.000, 0.000, 0.808, 0.000, 0.000],
            [0.000, 0.000, 1.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.921, 0.000],
            [0.000, 0.000, 0.000, 1.000, 0.000],
            [0.000, 0.000, 0.000, 0.323, 0.000],
            [0.000, 0.000, 0.000, 0.822, 0.000],
            [0.000, 0.000, 0.000, 0.677, 0.000],
            [0.000, 0.000, 0.000, 0.372, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb: this device has no extra thumb channel
            [0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb_v2: this device has no extra thumb channel
        ],
        tau_max=[0.13225, 0.11671, 0.12162, 0.09856, 0.05321],
        n_inputs=5,
        tau_max_is_calibrated=True,
        passive_coupling=False,
        controller_overrides={
            "mcp_scale": 0.78,
            "pip_lead": 0.05,
            "pip_full": 0.28,
            "dip_lead": 0.18,
            "dip_full": 0.50,
            "abduct_lead": 0.24,
            "abduct_full": 0.68,
            # Default ramp_seconds=1.5 reached the object at u~0.83 (83% of
            # D4_v2's real calibrated torque) -- measured peak object speed
            # right after first contact was 0.0225 m/s, a real jolt, not a
            # gentle touch. Slowing the ramp to 4.0s drops contact-time u to
            # ~0.40 and post-contact object speed to 0.0053 m/s (4x gentler)
            # while barely changing when contact happens (contact timing is
            # dominated by joint kinematics reaching the object, not ramp
            # rate) and, if anything, slightly IMPROVES sustained touch
            # duration (can_cylinder: 2.0s -> 2.5s) -- verified across all
            # three OBJECT_VARIANTS in run_grasp_baseline.py before locking in.
            "ramp_seconds": 4.0,
        },
    ),
    # D4_v2 + the paper's actual extra thumb, as its own independent 5th channel
    # (own dedicated pneumatic pump in the source device, per Sec. III: "another
    # soft actuator was designed to act as a telescopic extra thumb" -- separate
    # from the "index, middle, and thumb tendons... connected to dedicated
    # pulleys of individual motors" driving channels 0-2). Channels 0-3 (index/
    # middle/thumb/ring+pinky) are byte-for-byte the same as D4_v2 calibrated
    # above; channel 4 is new. tau_max=15.8N reuses the paper's OWN measured
    # "Maximum abduction force" (Sec. V-A) as the closest available real force
    # number for a pneumatic actuator in this same device family -- the paper
    # doesn't report a blocked/max force specifically for the extra thumb
    # actuator itself, so this is an explicit proxy, not a direct measurement.
    "D4_v2_extra_thumb_calibrated": ExoDevice(
        name="D4_v2_extra_thumb_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.716, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.452, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.892, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.315, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 1.000, 0.000],  # cmc_abduction: its own opposition motor
            [0.000, 0.000, 0.169, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.808, 0.000, 0.000, 0.000],
            [0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.921, 0.000, 0.000],
            [0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.323, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.822, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.677, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.372, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # extra_thumb: its own channel, own actuator
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb_v2: not this device (see D4_v2_extra_thumb_v2_calibrated)
        ],
        tau_max=[0.13225, 0.11671, 0.12162, 0.09856, 0.05321, 15.8],
        n_inputs=6,
        tau_max_is_calibrated=True,
        passive_coupling=False,
        controller_overrides={
            "mcp_scale": 0.78,
            "pip_lead": 0.05,
            "pip_full": 0.28,
            "dip_lead": 0.18,
            "dip_full": 0.50,
            "abduct_lead": 0.24,
            "abduct_full": 0.68,
            "ramp_seconds": 4.0,
            # TRIED AND REVERTED: marking extra_thumb (now channel 5, was 4 before the
            # thumb-opposition channel was split out) pneumatic -- matching the
            # source device's actual control law (held at a fixed commanded pressure, 20kPa,
            # Table 1, rather than easing off on contact like the tendon digits) -- is more
            # paper-faithful, but measured WORSE (rotation onset 20.9-37.9deg at 0.2s vs
            # 13.7deg with plain ease-off). Sustained full force through this still-marginal
            # contact amplifies the torque imbalance rather than resisting it; see
            # grasp_controller.py's pneumatic_channels mechanism, which still exists and is
            # correctly implemented, just not beneficial for THIS contact's current quality.
        },
    ),
    # Same device as D4_v2_extra_thumb_calibrated, but driving extra_thumb_v2 (the
    # trapezium-mounted, forward-pointing redesign) instead of extra_thumb (the
    # capitate-mounted, sideways-looking original) -- kept as a genuinely separate
    # device so both mounting choices stay directly comparable rather than
    # overwriting one another. Everything else (index/middle/thumb/ring+pinky
    # channels, force numbers) is identical.
    "D4_v2_extra_thumb_v2_calibrated": ExoDevice(
        name="D4_v2_extra_thumb_v2_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.716, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.452, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.892, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.315, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 1.000, 0.000],  # cmc_abduction: its own opposition motor
            [0.000, 0.000, 0.169, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.808, 0.000, 0.000, 0.000],
            [0.000, 0.000, 1.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.921, 0.000, 0.000],
            [0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.323, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.822, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.677, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.372, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb: not this device
            [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],  # extra_thumb_v2: its own channel, own actuator
        ],
        tau_max=[0.13225, 0.11671, 0.12162, 0.09856, 0.05321, 15.8],
        n_inputs=6,
        tau_max_is_calibrated=True,
        passive_coupling=False,
        controller_overrides={
            "mcp_scale": 0.78,
            "pip_lead": 0.05,
            "pip_full": 0.28,
            "dip_lead": 0.18,
            "dip_full": 0.50,
            "abduct_lead": 0.24,
            "abduct_full": 0.68,
            "ramp_seconds": 4.0,
        },
    ),
    # Paper-like thumb variant for D4_v2, kept separate from the working v2 model:
    # the paper emphasizes thumb opposition rather than a hitchhiker-like,
    # hyper-abducted thumb posture. In this MyoHand model, the thumb's inward
    # grasping direction is reached by NEGATIVE cmc_abduction / mp / ip motion,
    # while cmc_flexion stays positive. This branch therefore uses a somewhat
    # wider abduction-first approach, then folds inward with moderated MP/IP
    # curl so the thumb comes in from the side rather than hitchhiking upward.
    # This is still a modeling assumption, not a published per-joint force table
    # from Gerez et al. 2020, so it lives under a distinct name.
    "D4_v2_paper_like_thumb_calibrated": ExoDevice(
        name="D4_v2_paper_like_thumb_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000, 0.000],
            [0.716, 0.000, 0.000, 0.000, 0.000],
            [0.452, 0.000, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000, 0.000],
            [0.000, 0.892, 0.000, 0.000, 0.000],
            [0.000, 0.315, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, -0.660],  # cmc_abduction: own opposition motor, driven negative
            [0.000, 0.000, 0.360, 0.000, 0.000],
            [0.000, 0.000, -0.500, 0.000, 0.000],
            [0.000, 0.000, -0.240, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.921, 0.000],
            [0.000, 0.000, 0.000, 1.000, 0.000],
            [0.000, 0.000, 0.000, 0.323, 0.000],
            [0.000, 0.000, 0.000, 0.822, 0.000],
            [0.000, 0.000, 0.000, 0.677, 0.000],
            [0.000, 0.000, 0.000, 0.372, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb: this device has no extra thumb channel
            [0.000, 0.000, 0.000, 0.000, 0.000],  # extra_thumb_v2: this device has no extra thumb channel
        ],
        tau_max=[0.13225, 0.11671, 0.12162, 0.09856, 0.05321],
        n_inputs=5,
        tau_max_is_calibrated=True,
        passive_coupling=False,
        controller_overrides={
            "abduct_lead": 0.14,
            "abduct_full": 0.42,
            "hold_fraction": 0.16,
            "slip_gain": 0.010,
        },
    ),
    # D2 and D3 are now distinguished from their PRIMARY SOURCES. An earlier
    # version of this block asserted that "neither reports per-joint forces,
    # [so] D2 and D3 collapse to an IDENTICAL calibrated K". That was wrong on
    # the facts -- both papers publish enough to separate them, and the
    # collapse silently made two different devices numerically identical (same
    # K, same tau_max, same torques), which is how they came to produce
    # bit-identical benchmark results.
    #
    # D2 -- Alicea, Xiloyannis, Chiaradia, Barsotti, Frisoli & Masia, "A soft,
    # synergy-based robotic glove for grasping assistance," Wearable
    # Technologies 2(e4), 2021.
    #   * THREE digits are actively driven, not two: "Only three digits are
    #     actively supported, these being the thumb, index, and middle
    #     fingers" (Glove Design). The previous definition zeroed every thumb
    #     row, which removed the digit the paper calls essential -- "The
    #     support of the thumb is required, given its unique anatomy and
    #     fundamental role in all grasps that require opposition."
    #   * Cross-finger ratio comes from the synergy pulley's channel
    #     diameters, 2.2 / 0.95 / 1.93 cm for thumb / middle / index
    #     (Figure 3b and its caption). One shaft, so tendon stroke -- and
    #     hence delivered drive -- scales with channel diameter:
    #     thumb 1.000, index 0.877, middle 0.432. This IS the postural synergy
    #     the device is built around, and it is what makes D2 "cross-finger".
    #   * Fingertip force 7.5 N ("measured forces delivered by the device
    #     reached up to 7:5N at finger level and up to 15N at palm level").
    #     Table 3 lists 15 N for "This design"; the body text shows that is the
    #     palm-level figure, so the finger-level 7.5 N is used here.
    #   * cmc_abduction stays 0: the thumb is driven by a single tendon
    #     anchored at its dorsal tip and routed along the palmar side, which
    #     crosses the FLEXION joints. No abduction actuator is described.
    #     So this device gets thumb flexion but no independent opposition.
    #
    # D3 -- Thimabut, Terachinda & Kitisomprayoonkul, Rehabilitation Research
    # and Practice 2022, Art. 3738219.
    #   * TWO digits only: "The glove is a two-fingered design covering the
    #     index and middle fingers." The thumb is not actuated at all -- it is
    #     immobilised by a C-bar splint at 50 deg MCP flexion, plus a latex
    #     glove for friction. Thumb rows are correctly zero.
    #   * Genuinely UNIFORM across fingers: one hoist-and-cable motor, 1 DOF,
    #     driving index and middle identically. That is the actual content of
    #     this device's "uniform_single_dof" name, and it contrasts with D2's
    #     deliberately unequal synergy ratios.
    #   * Grip force 12-28 N at the fingertips; the 20 N midpoint is used.
    #
    # Within-finger split for both is the MOMENT_ARMS_MM profile (one
    # continuous tendon under uniform tension across a finger's joints, so
    # torque ratio is the moment-arm ratio) -- same derivation as D1/D4,
    # tau_j = F_j * r_j.
    #
    # MODELLING GAP worth stating: D3's real-world opposition comes from that
    # rigid splinted thumb acting as a post. Nothing in this model splints the
    # thumb, so D3 as simulated has no opposing surface at all and cannot form
    # a pinch on its own. Any D3 result here should be read with that in mind.
    "D2_synergy_cross_finger_calibrated": ExoDevice(
        name="D2_synergy_cross_finger_calibrated",
        # tau_j = 7.5 N * r_j * pulley_scale; normalised by ip_flexion (66.10 N*mm).
        #   thumb  (x1.000): cmc_flex 11.19 -> 0.169 | mp 53.40 -> 0.808 | ip 66.10 -> 1.000
        #   index  (x0.877): mcp2 63.05 -> 0.954 | pm2 45.15 -> 0.683 | md2 28.51 -> 0.431
        #   middle (x0.432): mcp3 27.39 -> 0.414 | pm3 24.44 -> 0.370 | md3  8.63 -> 0.131
        K=[0.954, 0.683, 0.431, 0.414, 0.370, 0.131,
           0.000, 0.169, 0.808, 1.000,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tau_max=[0.06610],  # N*m; = 66.10 N*mm, the thumb-IP torque above
        n_inputs=1,
        tau_max_is_calibrated=True,
    ),
    "D3_uniform_single_dof_calibrated": ExoDevice(
        name="D3_uniform_single_dof_calibrated",
        # tau_j = 20 N * r_j (uniform across both fingers); normalised by
        # index-MCP (191.66 N*mm). Thumb rows zero -- splinted, not actuated.
        K=[1.000, 0.716, 0.452, 0.883, 0.787, 0.278,
           0.0, 0.0, 0.0, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tau_max=[0.19166],  # N*m; = 191.66 N*mm, the index-MCP torque above
        n_inputs=1,
        tau_max_is_calibrated=True,
    ),
}


class ExoApplicator:
    """Injects a device's joint torques into MjData via qfrc_applied each step."""

    def __init__(self, model):
        self.dof_adr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
        self.passive_dof_adr = [model.joint(name).dofadr[0] for name in PASSIVE_COUPLED_JOINTS]

    def apply(self, data, device, u, row_gate=None):
        """row_gate: optional per-joint-row multiplier applied AFTER
        K/tau_max (see GraspController's thumb sequencing) -- a controller-
        level recruitment-order reflex, not part of the device's own
        literature-derived torque profile, so it's opt-in and defaults to
        no-op for every other caller (view_exo_device.py, compare_devices.py,
        grasp_test.py's open-loop devices, etc.)."""
        tau = device.torque(u)
        if row_gate is not None:
            tau = tau * row_gate
        for dof, t in zip(self.dof_adr, tau):
            data.qfrc_applied[dof] = t
        # Weak passive flexion of ring/little fingers driven by the active
        # middle-finger flexion profile. This keeps the extra digits involved
        # in the grasp without claiming separate exoskeleton actuation.
        if device.passive_coupling:
            passive_tau = PASSIVE_COUPLING_GAIN * np.maximum(tau[PASSIVE_SOURCE_ROWS], 0.0)
            for dof, t in zip(self.passive_dof_adr, passive_tau):
                data.qfrc_applied[dof] = t

    def clear(self, data):
        for dof in self.dof_adr:
            data.qfrc_applied[dof] = 0.0
        for dof in self.passive_dof_adr:
            data.qfrc_applied[dof] = 0.0
