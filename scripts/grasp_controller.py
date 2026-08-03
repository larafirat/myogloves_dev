"""A minimal closed-loop grip controller: ramps each device input up over time,
same as the open-loop demos, but FREEZES a channel the instant the bodies it
drives touch the object -- instead of continuing to max flexion regardless.

This is the simplest version of what a real hand (or a real EMG/FSR-triggered
assistive glove) does and our open-loop tests didn't: stop closing once you
feel something, rather than sweeping through it. grasp_test.py found that
open-loop "ramp to max and hold" lets every device's digits sweep past the
object at different times, so no device -- not even D4 with a real thumb --
holds an object indefinitely. This controller is the fix for that gap, not a
new geometry tweak.

Which bodies belong to which input channel is derived directly from each
device's own K matrix (nonzero rows = joints, hence bodies, that channel
drives) rather than hardcoded per device -- so a device with one shared input
across two fingers (D1-D3) naturally gets "either finger touching freezes the
shared channel," and a device with independent per-digit inputs (D4) naturally
gets independent per-digit freezing, with no special-casing required.
"""

import mujoco
import numpy as np

from exo_devices import JOINT_NAMES

ABDUCTION_ROW = JOINT_NAMES.index("cmc_abduction")


def _quat_from_z_to(direction):
    """Returns the (w,x,y,z) quaternion rotating the canonical +Z axis (MuJoCo
    capsules' symmetry axis) onto the given unit vector -- used to orient the
    extra_thumb_v2 dynamic shaft capsule along whatever direction currently
    connects its fixed base site to its moving tip site."""
    z = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    dot = float(np.dot(z, d))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x, since z and d are anti-parallel
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def thumb_gate_channels(device):
    """Channels that drive cmc_abduction AND at least one other (flexion) row
    from the same shared input -- currently only D4's thumb channel. For
    those channels, returns (channel, [other_row_indices]) so the controller
    can hold the flexion rows back while abduction leads (see GraspController
    docstring): a single tendon can't sequence sub-joints by itself (one u
    scales every row in its column by the same factor), so recruiting
    abduction before flexion has to happen in the controller, not the K
    matrix -- exactly like the ramp/ease-off/slip-reflex logic already here
    is a controller-level stand-in for what a real hand's motor program does."""
    pairs = []
    for col in range(device.n_inputs):
        if device.K[ABDUCTION_ROW, col] == 0.0:
            continue
        other_rows = [row for row in range(len(JOINT_NAMES))
                      if row != ABDUCTION_ROW and device.K[row, col] != 0.0]
        if other_rows:
            pairs.append((col, other_rows))
    return pairs


def finger_gate_channels(device):
    """Per-finger channels whose 3 active rows are MCP/PIP/DIP only.

    For these channels (currently D4's index and middle), return
    (channel, mcp_row, pip_row, dip_row) so the controller can let the
    larger proximal joints lead before the distal tip curls shut. This helps
    the finger wrap around the object instead of pinching empty space first.
    """
    triples = []
    for col in range(device.n_inputs):
        active_rows = [row for row in range(len(JOINT_NAMES)) if device.K[row, col] != 0.0]
        if len(active_rows) != 3:
            continue
        if active_rows[0] == ABDUCTION_ROW:
            continue
        if active_rows[1] == active_rows[0] + 1 and active_rows[2] == active_rows[1] + 1:
            triples.append((col, active_rows[0], active_rows[1], active_rows[2]))
    return triples


def channel_body_ids(model, device):
    """Returns, for each input channel, the set of body ids whose joint(s)
    that channel actually drives (i.e. device.K[row, channel] != 0)."""
    channels = []
    for col in range(device.n_inputs):
        bodies = set()
        for row, joint_name in enumerate(JOINT_NAMES):
            if device.K[row, col] != 0.0:
                bodies.add(model.joint(joint_name).bodyid[0])
        channels.append(bodies)
    return channels


class GraspController:
    """hold_fraction is the key parameter: once a channel's bodies touch the
    object, its u drops to this fraction rather than freezing at whatever the
    ramp had reached. Freezing alone isn't enough -- if contact happens after
    the ramp has already reached 1.0 (real calibrated torques can take longer
    to converge on the object than the 1.5s ramp takes to complete), "freeze
    at current value" freezes at max, i.e. behaves exactly like the old
    open-loop controller and still overpowers the grip. A real hand (or a
    force-feedback assistive glove) eases off once it feels contact rather
    than continuing to drive at whatever force it happened to reach; this is
    the simplest model of that reflex.

    A fixed hold_fraction still isn't the whole story: even a gentle constant
    hold has no way to correct for slow net drift (three independently-timed
    contact points rarely cancel to exactly zero net force), so the object can
    still creep for many seconds and eventually slip free once it crosses some
    threshold. Real hands don't hold at a fixed force either -- cutaneous
    mechanoreceptors sense slip and reflexively increase grip force in
    response (Johansson & Westling's classic finding on human grip control).

    slip_gain/slip_threshold implement that reflex, but against CUMULATIVE
    drift since the grip was first established, not a per-step velocity: the
    failure mode this is catching is slow creep (microns per step, invisible
    step to step) that only becomes a problem after accumulating over
    thousands of steps, not a sudden per-step jump. A per-step threshold would
    need to be smaller than simulation noise to catch it at all.

    OPPOSITION CAPTURE (opposition_threshold/grip_hold_strong): tracing contact
    normals through a full closing motion found something the fixed
    hold_fraction couldn't exploit -- the thumb and a finger's contact normals
    drift through a wide range of relative angles as everything slides into
    its final resting contact, and at ONE point along that slide they
    actually hit genuine opposition (dot ~ -1.0, anti-parallel). But by then
    hold_fraction has already locked grip at a low, constant value, so
    nothing capitalizes on that moment -- the slide just continues past it,
    and a retract-support test confirmed the object falls anyway (drift
    balloons within ~1s even when retraction is delayed until well after the
    good moment passes). Real grasping does the opposite: mechanoreceptors
    detect a secure, load-bearing contact configuration and increase force
    to seize it (the same Johansson & Westling force-scaling behavior
    slip_gain is modeled on, just triggered by CONFIGURATION here instead of
    drift). So each step, for every pair of currently-touched channels, this
    averages each channel's contact normals and tracks the most negative
    pairwise dot product; the instant it crosses opposition_threshold, grip
    on every locked channel jumps to grip_hold_strong and stays there
    (a one-way ratchet, not a continuous retrigger -- modeling a captured
    grasp, not a flickering one).

    THUMB SEQUENCING (abduct_lead/abduct_full): grasp testing found the
    thumb's contact normal stayed ~90 deg off the fingers' (pressing down
    onto the object instead of opposing them) even after cmc_abduction was
    added, because D4's thumb channel drives cmc_abduction and
    cmc_flexion/mp/ip from ONE shared u -- ramping them together means
    flexion (K=1.000/0.808) always dominates abduction (K=0.438) in effective
    torque, so flexion wins the race even though abduction is what's supposed
    to lead. Real thumb opposition recruits the abductor before the flexors
    (Napier's prehension work); row_gate reproduces that by holding the
    flexion/mp/ip rows at zero torque until the channel's OWN u (i.e. how far
    through its ramp it is) passes abduct_lead, then ramping them in linearly
    up to abduct_full -- abduction alone gets the early torque, flexion joins
    once abduction has had a head start. This only ever suppresses rows,
    never adds torque beyond what K/tau_max already allow, so it's a
    controller-level recruitment-order reflex, not a change to the
    literature-derived K matrices themselves.

    FINGER SEQUENCING (mcp_scale/pip_lead/dip_lead/dip_scale): the live large-pillar setup showed
    another ordering issue: the fingertips curled shut before the finger had
    really wrapped around the object, so contact happened late and off the
    distal tips instead of along a broader finger surface. For per-finger
    3-row channels (MCP/PIP/DIP), row_gate now lets MCP lead immediately,
    ramps PIP in next, and delays DIP the longest. That produces a more
    natural "wrap first, pinch later" closing motion without changing the
    device's K matrix or peak torques. mcp_scale optionally softens the MCP
    row if the knuckles are dominating the grasp too early, and dip_scale can
    intentionally cap DIP recruitment when a taller object benefits more from
    flatter finger-pad contact than from full fingertip hook.

    TORQUE BALANCE (torque_balance_gain): retract-testing found the object
    doesn't slip out in a straight line -- it spins up (0.2 rad/s growing to
    several rad/s over ~1s) while still gripped, then loses contact once the
    spin carries it far enough. A first attempt at fixing this (boost
    whichever channel's contact NORMAL direction opposes the current spin)
    failed: it saturated every channel to max within milliseconds regardless
    (normal direction alone doesn't discriminate between channels -- they
    all "opposed" the spin simultaneously by that measure), which is just a
    slower version of "squeeze harder everywhere," already ruled out by a
    grip_hold_strong sweep that produced an identical failure at every
    strength from 0.42 to 1.0.

    This is the corrected version: instead of contact NORMAL direction, it
    uses the actual per-contact FORCE vector (mj_contactForce, transformed
    to world frame) and each channel's real lever arm (contact position
    relative to the object's CoM) to estimate that channel's current torque
    contribution tau_ch = sum(r_i x F_i) over its contacts. Net torque
    tau_net = sum(tau_ch) is what's actually spinning the object up. Each
    channel then gets a gradient step -- du_ch = -gain * dot(tau_net, tau_ch)
    -- which is signed, not a boost-only heuristic: a channel whose own
    torque points the SAME way as tau_net (feeding the spin) gets reduced
    (down to hold_fraction, never below -- this never drops a channel's grip
    entirely), while a channel opposing tau_net gets increased. That's what
    was missing before: real per-channel DISCRIMINATION instead of every
    channel reacting identically to the same global spin signal. Off by
    default (gain=0.0); enable via controller_overrides per device.

    Measured result: this genuinely discriminates (u spreads out across
    channels instead of saturating together) and delays the spin-out
    (drift 300 steps post-retract: 1.33m at gain=0 -> 0.92m at gain=150),
    but doesn't prevent it -- by ~600 steps post-retract every gain tested
    converges to the same ~1.34m failure. With only n_inputs (4 for D4_v2)
    scalar channels, there may not be enough independent directions to fully
    cancel a 3D torque vector, since every row within a channel is still
    forced to move together by that channel's fixed K-ratio.

    ROW-LEVEL TORQUE BALANCE (row_torque_gain): the natural next step given
    the above -- instead of one scalar per channel (4 knobs for D4_v2), do
    the same gradient correction per INDIVIDUAL ROW/joint (up to 16 knobs),
    via a persistent per-row multiplier (row_torque_mult) applied on top of
    row_gate. Only rows whose own body is in direct contact this step get a
    torque reading and an update; others keep their last multiplier (contact
    is intermittent frame to frame, so this has to persist, unlike the
    channel version which recomputes every step from whatever's touching).
    Clipped to [0.2, 3.0] so it can meaningfully suppress OR boost a specific
    joint's torque relative to its K-matrix ratio -- this is the one place
    in this controller that goes beyond "suppress toward 0", because a
    scalar-per-channel space provably couldn't null the torque (see above),
    so recovering real degrees of freedom requires letting individual
    joints diverge from their nominal literature K-ratio, not just gating
    them. Off by default (gain=0.0).

    EXTRA THUMB: measuring the anatomical thumb's reachable Y-range against
    the YCB box's 72mm span found it ~57mm short even at full abduction+
    flexion -- it doesn't even reach as far as where the fingers converge,
    let alone oppose past them. First attempt "fixed" this by bolting a
    telescoping slide joint onto the distal thumb segment, sharing its
    channel -- wrong on inspection of the actual source paper (Gerez et al.
    2020): the real device's "telescopic extra thumb" is a SEPARATE,
    independently-actuated (own dedicated air pump), palm-mounted appendage,
    not an extension of the anatomical thumb at all. Corrected model lives in
    myohand_body.xml (extra_thumb joint, mounted on capitate) and
    exo_devices.py (its own channel, e.g. D4_v2_extra_thumb_calibrated) --
    it needs no special gating here, since it's a normal independent channel
    like any other digit: the standard ramp/lock/hold logic below already
    handles it.

    PNEUMATIC CHANNELS (pneumatic_channels): the ramp/ease-off-on-contact
    behavior (hold_fraction) is a reflex WE modeled for the tendon-driven
    digits -- the paper never claims the tendons back off force on contact,
    but it's a defensible stand-in for cutaneous slip response. The extra
    thumb is different in the paper's own terms: it's driven by "another
    soft actuator" at a fixed operating pressure (20 kPa, Table 1), commanded
    open-loop from the phone app as a single inflate/deflate action -- there
    is no described mechanism for it to sense contact and relax. Modeling it
    with the SAME ease-off reflex as the tendons was an unexamined default,
    not a deliberate choice, and it works against the one thing measured to
    actually stabilize the grasp (the fixed-opposition diagnostic): channels
    listed in pneumatic_channels skip the ease-off and hold at full ramp
    value once locked, matching "inflated to set pressure and held there"
    rather than "touched something, back off."
    """

    def __init__(self, model, device, obj_geom_id, ramp_seconds=1.5, dt=0.002,
                 hold_fraction=0.15, slip_gain=0.01, slip_threshold=0.003,
                 abduct_lead=0.3, abduct_full=0.6,
                 pip_lead=0.15, pip_full=0.45, dip_lead=0.35, dip_full=0.7,
                 mcp_scale=1.0, dip_scale=1.0, torque_balance_gain=0.0,
                 row_torque_gain=0.0, pneumatic_channels=()):
        self.device = device
        self.obj_geom_id = obj_geom_id
        self.obj_body_id = model.geom_bodyid[obj_geom_id]
        self.channel_bodies = channel_body_ids(model, device)
        # Tolerate JOINT_NAMES entries that don't exist in this particular
        # model. extra_thumb/extra_thumb_v2 are bodies added only to the
        # exoglove env's copy of myohand_body.xml, so a model without them
        # (e.g. the glove_dev env, used to run these devices on a rig with a
        # stabilised arm) would otherwise fail to construct the controller at
        # all. Devices that actually drive a missing joint are rejected
        # upstream, so skipping here cannot silently weaken one.
        self.body_to_row = {}
        for row, name in enumerate(JOINT_NAMES):
            try:
                self.body_to_row[model.joint(name).bodyid[0]] = row
            except KeyError:
                continue
        self.thumb_gate_channels = thumb_gate_channels(device)
        self.finger_gate_channels = finger_gate_channels(device)
        self.locked = [False] * device.n_inputs
        self.u = np.zeros(device.n_inputs)
        self.row_gate = np.ones(len(JOINT_NAMES))
        self.ramp_step = dt / ramp_seconds
        self.hold_fraction = hold_fraction
        self.slip_gain = slip_gain
        self.slip_threshold = slip_threshold  # cumulative drift (m) since grip established, not per-step
        self.abduct_lead = abduct_lead
        self.abduct_full = abduct_full
        self.pip_lead = pip_lead
        self.pip_full = pip_full
        self.dip_lead = dip_lead
        self.dip_full = dip_full
        self.mcp_scale = mcp_scale
        self.dip_scale = dip_scale
        self.torque_balance_gain = torque_balance_gain
        self.row_torque_gain = row_torque_gain
        self.pneumatic_channels = set(pneumatic_channels)
        self.row_torque_mult = np.ones(len(JOINT_NAMES))
        self._grip_anchor_pos = None

        # extra_thumb_v2's dynamic shaft: a spatial tendon (visual only, no collision)
        # renders straight through solid objects that sit between the mount and the
        # tip, and a capsule rigidly attached to the translating joint body can't
        # change length -- so instead a dedicated mocap body (extra_thumb_v2_shaft_body,
        # myohand_exoglove_env.xml) is resized/repositioned every step to be a REAL,
        # always-colliding capsule spanning extra_thumb_v2_base_site to
        # extra_thumb_v2_tip_site. Only wired up if this device actually drives that
        # channel AND the model has the shaft body (myohand_exoglove.xml, used by the
        # plain device-torque regression tests, deliberately doesn't -- no grasp object
        # there to contact in the first place).
        self._shaft_channel = None
        if "extra_thumb_v2" in JOINT_NAMES:
            v2_row = JOINT_NAMES.index("extra_thumb_v2")
            nonzero_cols = np.nonzero(device.K[v2_row, :])[0]
            if len(nonzero_cols) > 0:
                try:
                    self._shaft_base_site = model.site("extra_thumb_v2_base_site").id
                    self._shaft_tip_site = model.site("extra_thumb_v2_tip_site").id
                    self._shaft_geom = model.geom("extra_thumb_v2_shaft_geom").id
                    shaft_body = model.body("extra_thumb_v2_shaft_body")
                    self._shaft_mocap = shaft_body.mocapid[0]
                    self._shaft_channel = int(nonzero_cols[0])
                    # The shaft capsule spans the whole mount-to-tip line and will
                    # generally register contact with the object BEFORE the small tip
                    # sphere on the joint body does -- channel_bodies must include this
                    # body too, or "touched"/locked would wait for the tip sphere
                    # specifically even after the capsule already made contact.
                    self.channel_bodies[self._shaft_channel].add(shaft_body.id)
                except KeyError:
                    self._shaft_channel = None

    def _update_extra_thumb_shaft(self, model, data):
        base = data.site_xpos[self._shaft_base_site].copy()
        tip = data.site_xpos[self._shaft_tip_site].copy()
        direction = tip - base
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])
            length = 1e-6
        data.mocap_pos[self._shaft_mocap] = (base + tip) / 2.0
        data.mocap_quat[self._shaft_mocap] = _quat_from_z_to(direction)
        model.geom_size[self._shaft_geom] = [0.012, length / 2.0, 0.0]

    def step(self, model, data):
        """Call once per mj_step (before or after -- reads current data.contact,
        so call it, then apply() the returned u, matching this step's contacts
        from the *previous* step's geometry). Returns the (n_inputs,) u array
        to hand to ExoApplicator.apply."""
        touched = [False] * self.device.n_inputs
        for c in data.contact[:data.ncon]:
            if self.obj_geom_id not in (c.geom1, c.geom2):
                continue
            other_geom = c.geom2 if c.geom1 == self.obj_geom_id else c.geom1
            other_body = model.geom_bodyid[other_geom]
            for ch, bodies in enumerate(self.channel_bodies):
                if other_body in bodies:
                    touched[ch] = True

        obj_pos = data.xpos[self.obj_body_id].copy()
        was_unlocked = not any(self.locked)

        for ch in range(self.device.n_inputs):
            if touched[ch] and not self.locked[ch]:
                self.locked[ch] = True
                if ch not in self.pneumatic_channels:
                    self.u[ch] = self.hold_fraction  # ease off to a gentle hold, not whatever the ramp reached
                # pneumatic channels: no ease-off on contact -- keep ramping to full commanded
                # pressure regardless (see PNEUMATIC CHANNELS docstring above), matching "inflate
                # to a set pressure" rather than a contact reflex the paper doesn't describe.
            if not self.locked[ch] or ch in self.pneumatic_channels:
                self.u[ch] = min(self.u[ch] + self.ramp_step, 1.0)

        if was_unlocked and any(self.locked):
            self._grip_anchor_pos = obj_pos.copy()  # first moment ANY channel grips -- drift is measured from here

        if self._grip_anchor_pos is not None:
            drift = float(np.linalg.norm(obj_pos - self._grip_anchor_pos))
            if drift > self.slip_threshold:
                for ch in range(self.device.n_inputs):
                    if self.locked[ch]:
                        self.u[ch] = min(self.u[ch] + self.slip_gain, 1.0)  # slip detected -> grip harder
                self._grip_anchor_pos = obj_pos.copy()  # re-anchor so the gain keeps responding to further drift

        if self.torque_balance_gain > 0.0 and any(self.locked):
            channel_torque = {ch: np.zeros(3) for ch in range(self.device.n_inputs) if self.locked[ch]}
            forcetorque = np.zeros(6)
            for i in range(data.ncon):
                c = data.contact[i]
                if self.obj_geom_id not in (c.geom1, c.geom2):
                    continue
                other_geom = c.geom2 if c.geom1 == self.obj_geom_id else c.geom1
                other_body = model.geom_bodyid[other_geom]
                ch_hit = next((ch for ch, bodies in enumerate(self.channel_bodies)
                               if self.locked[ch] and other_body in bodies), None)
                if ch_hit is None:
                    continue
                mujoco.mj_contactForce(model, data, i, forcetorque)
                frame = np.array(c.frame, dtype=float).reshape(3, 3)
                force_world = forcetorque[0] * frame[0] + forcetorque[1] * frame[1] + forcetorque[2] * frame[2]
                if c.geom2 == self.obj_geom_id:
                    force_world = -force_world  # consistent "force applied ON the object" convention
                lever = np.array(c.pos, dtype=float) - obj_pos
                channel_torque[ch_hit] += np.cross(lever, force_world)

            tau_net = sum(channel_torque.values(), np.zeros(3))
            for ch, tau_ch in channel_torque.items():
                du = -self.torque_balance_gain * np.dot(tau_net, tau_ch)
                self.u[ch] = float(np.clip(self.u[ch] + du, self.hold_fraction, 1.0))

        if self.row_torque_gain > 0.0 and any(self.locked):
            row_torque = {}
            forcetorque = np.zeros(6)
            for i in range(data.ncon):
                c = data.contact[i]
                if self.obj_geom_id not in (c.geom1, c.geom2):
                    continue
                other_geom = c.geom2 if c.geom1 == self.obj_geom_id else c.geom1
                other_body = model.geom_bodyid[other_geom]
                row_hit = self.body_to_row.get(other_body)
                if row_hit is None:
                    continue
                ch_hit = next((ch for ch, bodies in enumerate(self.channel_bodies)
                               if self.locked[ch] and other_body in bodies), None)
                if ch_hit is None:
                    continue
                mujoco.mj_contactForce(model, data, i, forcetorque)
                frame = np.array(c.frame, dtype=float).reshape(3, 3)
                force_world = forcetorque[0] * frame[0] + forcetorque[1] * frame[1] + forcetorque[2] * frame[2]
                if c.geom2 == self.obj_geom_id:
                    force_world = -force_world
                lever = np.array(c.pos, dtype=float) - obj_pos
                row_torque[row_hit] = row_torque.get(row_hit, np.zeros(3)) + np.cross(lever, force_world)

            if row_torque:
                tau_net_rows = sum(row_torque.values(), np.zeros(3))
                for row, tau_row in row_torque.items():
                    dmult = -self.row_torque_gain * np.dot(tau_net_rows, tau_row)
                    self.row_torque_mult[row] = float(np.clip(self.row_torque_mult[row] + dmult, 0.2, 3.0))

        self.row_gate[:] = 1.0
        for ch, mcp_row, pip_row, dip_row in self.finger_gate_channels:
            pip_span = self.pip_full - self.pip_lead
            dip_span = self.dip_full - self.dip_lead
            self.row_gate[mcp_row] = self.mcp_scale
            self.row_gate[pip_row] = np.clip((self.u[ch] - self.pip_lead) / pip_span, 0.0, 1.0)
            self.row_gate[dip_row] = self.dip_scale * np.clip((self.u[ch] - self.dip_lead) / dip_span, 0.0, 1.0)
        for ch, other_rows in self.thumb_gate_channels:
            span = self.abduct_full - self.abduct_lead
            gate = np.clip((self.u[ch] - self.abduct_lead) / span, 0.0, 1.0)
            for row in other_rows:
                self.row_gate[row] = gate

        if self.row_torque_gain > 0.0:
            self.row_gate *= self.row_torque_mult

        if self._shaft_channel is not None:
            self._update_extra_thumb_shaft(model, data)

        return self.u.copy()
