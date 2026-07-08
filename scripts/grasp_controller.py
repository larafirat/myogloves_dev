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

import numpy as np

from exo_devices import JOINT_NAMES

ABDUCTION_ROW = JOINT_NAMES.index("cmc_abduction")


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

    FINGER SEQUENCING (pip_lead/dip_lead): the live large-pillar setup showed
    another ordering issue: the fingertips curled shut before the finger had
    really wrapped around the object, so contact happened late and off the
    distal tips instead of along a broader finger surface. For per-finger
    3-row channels (MCP/PIP/DIP), row_gate now lets MCP lead immediately,
    ramps PIP in next, and delays DIP the longest. That produces a more
    natural "wrap first, pinch later" closing motion without changing the
    device's K matrix or peak torques.
    """

    def __init__(self, model, device, obj_geom_id, ramp_seconds=1.5, dt=0.002,
                 hold_fraction=0.15, slip_gain=0.01, slip_threshold=0.003,
                 abduct_lead=0.3, abduct_full=0.6,
                 pip_lead=0.15, pip_full=0.45, dip_lead=0.35, dip_full=0.7):
        self.device = device
        self.obj_geom_id = obj_geom_id
        self.obj_body_id = model.geom_bodyid[obj_geom_id]
        self.channel_bodies = channel_body_ids(model, device)
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
        self._grip_anchor_pos = None

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
                self.u[ch] = self.hold_fraction  # ease off to a gentle hold, not whatever the ramp reached
            if not self.locked[ch]:
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

        self.row_gate[:] = 1.0
        for ch, mcp_row, pip_row, dip_row in self.finger_gate_channels:
            pip_span = self.pip_full - self.pip_lead
            dip_span = self.dip_full - self.dip_lead
            self.row_gate[mcp_row] = 1.0
            self.row_gate[pip_row] = np.clip((self.u[ch] - self.pip_lead) / pip_span, 0.0, 1.0)
            self.row_gate[dip_row] = np.clip((self.u[ch] - self.dip_lead) / dip_span, 0.0, 1.0)
        for ch, other_rows in self.thumb_gate_channels:
            span = self.abduct_full - self.abduct_lead
            gate = np.clip((self.u[ch] - self.abduct_lead) / span, 0.0, 1.0)
            for row in other_rows:
                self.row_gate[row] = gate
        return self.u.copy()
