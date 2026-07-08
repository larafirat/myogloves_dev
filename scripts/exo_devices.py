"""Applies exoskeleton coupling-matrix torques (tau_exo = K @ u) to MyoHand.

Joint order (index + middle + thumb, 10 DoF):
    index MCP, index PIP, index DIP, middle MCP, middle PIP, middle DIP,
    thumb CMC abduction, thumb CMC flexion, thumb MP flexion, thumb IP flexion
Positive torque = flexion (or abduction, for cmc_abduction) assistance.
D1/D2/D3 don't drive the thumb at all (their papers only ever published
index+middle data) -- their K rows for the four thumb DoFs are all zero.
Only D4 (Gerez et al. 2020) has a genuine, independently-actuated thumb in the
source device, so it's the only one with non-zero thumb columns.

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
# compute_moment_arms.py: FDP2/FDP3 for index/middle, FPL_tendon for the thumb
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
    # No ring/little moment-arm extraction has been added yet. D4_v2's shared
    # ulnar channel therefore uses a proxy coupling profile rather than a true
    # MyoHand-calibrated F*r derivation for those six joints.
}


class ExoDevice:
    def __init__(self, name, K, tau_max, n_inputs, tau_max_is_calibrated=False,
                 passive_coupling=True):
        self.name = name
        self.n_inputs = n_inputs
        self.K = np.asarray(K, dtype=float).reshape(len(JOINT_NAMES), n_inputs)
        self.tau_max = np.asarray(tau_max, dtype=float).reshape(n_inputs)
        # True only when tau_max is derived from real force/moment-arm data (see D1's
        # calibrated variant) rather than tuned by hand for visible demo behavior.
        self.tau_max_is_calibrated = tau_max_is_calibrated
        self.passive_coupling = passive_coupling

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
    # does). Applied identically to the middle finger per the paper's own
    # "uniform drive structure" claim across fingers (Sec. 2.2) -- middle
    # finger forces were not independently measured. Note also: the real device
    # gives each finger its own independent pushrod motor (Sec. 2.4), not one
    # motor shared across index+middle as this n_inputs=1 model assumes --
    # a simplification, not something the paper itself reports.
    "D1_underactuated_distal": ExoDevice(
        name="D1_underactuated_distal",
        K=[0.73, 0.93, 1.00, 0.73, 0.93, 1.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tau_max=[1.0],  # placeholder; see D1_underactuated_distal_calibrated for the real-torque version
        n_inputs=1,
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
        K=[0.85, 0.95, 1.00, 0.4184, 0.4677, 0.4922, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
        K=[0.65, 1.00, 0.9375, 0.65, 1.00, 0.9375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    # Uncalibrated K shape (0.80/0.90/1.00, applied to all three channels
    # including thumb CMC-abduction/CMC-flexion/MP/IP) remains a design-intent
    # placeholder; this paper reports no per-joint force/ROM split for any
    # digit to ground it in. Thumb column is now 4 entries (cmc_abduction
    # added, see module docstring for why); kept the same 0.70/0.80/0.90/1.00
    # placeholder ramp style as the rest.
    "D4_hybrid_per_finger": ExoDevice(
        name="D4_hybrid_per_finger",
        K=[
            [0.80, 0.00, 0.00],
            [0.90, 0.00, 0.00],
            [1.00, 0.00, 0.00],
            [0.00, 0.80, 0.00],
            [0.00, 0.90, 0.00],
            [0.00, 1.00, 0.00],
            [0.00, 0.00, 0.70],
            [0.00, 0.00, 0.80],
            [0.00, 0.00, 0.90],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00],
        ],
        tau_max=[1.0, 1.0, 1.0],  # placeholder; real device has a 500 N tendon rating (safety/material limit,
        n_inputs=3,                # not a typical operating torque) -- see the _calibrated variant for the real value
    ),
    # Paper-faithful v2: keep index/middle/thumb as dedicated channels, but add a
    # fourth shared tendon channel for ring+pinky (Sec. III: "ring and pinky...
    # coupled together and connected to a single pulley and motor"). This leaves
    # the current 3-channel D4 untouched while providing a separate version that
    # matches the motor count more closely.
    "D4_v2_hybrid_per_finger": ExoDevice(
        name="D4_v2_hybrid_per_finger",
        K=[
            [0.80, 0.00, 0.00, 0.00],
            [0.90, 0.00, 0.00, 0.00],
            [1.00, 0.00, 0.00, 0.00],
            [0.00, 0.80, 0.00, 0.00],
            [0.00, 0.90, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.00, 0.70, 0.00],
            [0.00, 0.00, 0.80, 0.00],
            [0.00, 0.00, 0.90, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.70],
            [0.00, 0.00, 0.00, 0.80],
            [0.00, 0.00, 0.00, 0.90],
            [0.00, 0.00, 0.00, 0.46],
            [0.00, 0.00, 0.00, 0.56],
            [0.00, 0.00, 0.00, 0.66],
        ],
        tau_max=[1.0, 1.0, 1.0, 1.0],
        n_inputs=4,
        passive_coupling=False,
    ),
    # --- MyoHand-calibrated variants ---
    # For D1/D4 (published per-joint/per-tendon force data), these replace the
    # literature force ratios with real joint-torque ratios (tau_j = F_j * MOMENT_ARMS_MM[j]).
    # Report both the original and calibrated K in the paper: the discrepancy
    # (D1 flips from distal-biased to proximal-biased) is itself a finding.
    # tau_j = F_j (Zhao et al. 2025: MCP 4.1N, PIP 5.2N, DIP 5.6N, index-finger
    # measurement reused for middle per the paper's "uniform drive structure"
    # claim) * MOMENT_ARMS_MM[j], normalized by the max (index MCP, 39.29 N*mm):
    #   index:  4.1*9.583=39.29 -> 1.000 | 5.2*6.862=35.68 -> 0.908 | 5.6*4.333=24.26 -> 0.618
    #   middle: 4.1*8.457=34.67 -> 0.883 | 5.2*7.546=39.24 -> 0.999 | 5.6*2.664=14.92 -> 0.380
    "D1_underactuated_distal_calibrated": ExoDevice(
        name="D1_underactuated_distal_calibrated",
        K=[1.000, 0.908, 0.618, 0.883, 0.999, 0.380, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tau_max=[0.03929],  # N*m; = 39.29 N*mm, the index-MCP torque above -- not a placeholder
        n_inputs=1,
        tau_max_is_calibrated=True,
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
    #   index:  13.8 N * 9.583 mm = 132.25 N*mm = 0.13225 N*m  (index MCP, K's own max)
    #   middle: 13.8 N * 8.457 mm = 116.71 N*mm = 0.11671 N*m  (middle MCP, K's own max)
    #   thumb:  13.8 N * 3.856 mm =  53.21 N*mm (CMC abduction) | 13.8*1.492=20.59 (CMC flexion)
    #           | 13.8*7.120=98.26 (MP) | 13.8*8.813=121.62 (IP, K's own max)
    #           -> normalized by 121.62: [0.438, 0.169, 0.808, 1.000]; tau_max = 121.62 N*mm = 0.12162 N*m
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
            [1.000, 0.000, 0.000],
            [0.716, 0.000, 0.000],
            [0.452, 0.000, 0.000],
            [0.000, 1.000, 0.000],
            [0.000, 0.892, 0.000],
            [0.000, 0.315, 0.000],
            [0.000, 0.000, 0.438],
            [0.000, 0.000, 0.169],
            [0.000, 0.000, 0.808],
            [0.000, 0.000, 1.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
        ],
        tau_max=[0.13225, 0.11671, 0.12162],  # N*m; not a placeholder -- see derivation above
        n_inputs=3,
        tau_max_is_calibrated=True,
    ),
    # v2 calibrated: index/middle/thumb channels keep the real F*r calibration from
    # D4 above. The new shared ring+pinky channel is only a proxy, because this repo
    # has not yet derived ring/little moment arms from MyoHand or found a published
    # force split for that coupled channel. We therefore borrow the middle-finger
    # calibrated profile for ring and scale pinky down within the same shared motor,
    # while reusing the middle-finger tau_max as the closest available tendon-force
    # proxy. This is explicitly a v2 modeling assumption, not a direct paper number.
    "D4_v2_hybrid_per_finger_calibrated": ExoDevice(
        name="D4_v2_hybrid_per_finger_calibrated",
        K=[
            [1.000, 0.000, 0.000, 0.000],
            [0.716, 0.000, 0.000, 0.000],
            [0.452, 0.000, 0.000, 0.000],
            [0.000, 1.000, 0.000, 0.000],
            [0.000, 0.892, 0.000, 0.000],
            [0.000, 0.315, 0.000, 0.000],
            [0.000, 0.000, 0.438, 0.000],
            [0.000, 0.000, 0.169, 0.000],
            [0.000, 0.000, 0.808, 0.000],
            [0.000, 0.000, 1.000, 0.000],
            [0.000, 0.000, 0.000, 0.883],
            [0.000, 0.000, 0.000, 0.787],
            [0.000, 0.000, 0.000, 0.278],
            [0.000, 0.000, 0.000, 0.575],
            [0.000, 0.000, 0.000, 0.512],
            [0.000, 0.000, 0.000, 0.181],
        ],
        tau_max=[0.13225, 0.11671, 0.12162, 0.11671],
        n_inputs=4,
        tau_max_is_calibrated=True,
        passive_coupling=False,
    ),
    # D2/D3 have no published force data at all (design-intent only), so there is
    # nothing to correct via F*r like D1/D4. What we CAN check: both are described
    # as a single continuous tendon crossing all three joints under one motor/tension
    # (D2: fingertip-anchored synergy; D3: single-DOF cable) -- the same physical
    # setup as D4's per-finger tendon. Under uniform tension, real torque ratio IS
    # the moment-arm ratio, so this is the same MOMENT_ARMS_MM profile as D4's,
    # just combined into ONE shared input (both fingers, single u) instead of two.
    # Because neither reports per-joint forces, D2 and D3 collapse to an IDENTICAL
    # calibrated K -- itself worth flagging: absent that data, this pipeline cannot
    # physically distinguish D2's "flat" claim from D3's "uniform" claim.
    "D2_synergy_cross_finger_calibrated": ExoDevice(
        name="D2_synergy_cross_finger_calibrated",
        K=[1.000, 0.716, 0.452, 0.883, 0.787, 0.278, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tau_max=[1.0],  # placeholder; no force data exists to derive a real scale for D2
        n_inputs=1,
    ),
    "D3_uniform_single_dof_calibrated": ExoDevice(
        name="D3_uniform_single_dof_calibrated",
        K=[1.000, 0.716, 0.452, 0.883, 0.787, 0.278, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tau_max=[1.0],  # placeholder; no force data exists to derive a real scale for D3
        n_inputs=1,
    ),
}


class ExoApplicator:
    """Injects a device's joint torques into MjData via qfrc_applied each step."""

    def __init__(self, model):
        self.dof_adr = [model.joint(name).dofadr[0] for name in JOINT_NAMES]
        self.passive_dof_adr = [model.joint(name).dofadr[0] for name in PASSIVE_COUPLED_JOINTS]

    def apply(self, data, device, u, row_gate=None):
        """row_gate: optional (10,) per-joint-row multiplier applied AFTER
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
