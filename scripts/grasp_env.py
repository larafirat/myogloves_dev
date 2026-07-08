"""GraspEnv: a structured grasp environment modeled on the MyoChallenge'24
myoMPL "Prosthesis Co-Manipulation" task's observation/reward/randomization
conventions (Table I obs space, Eq. 2 reward, object-property randomization
ranges) -- adapted to our single, fixed-arm exoskeleton context.

Kept as a SEPARATE environment from myohand_exoglove_grasp.xml/grasp_test.py
on purpose, so the two can be compared: the old one is a single ad hoc
pass/fail check, this one is a structured obs_dict/reward_dict environment
with per-episode randomization, closer to what an RL-style pipeline expects.

SCOPE NOTE (see myohand_exoglove_env.xml for the model-level version of this):
the source paper's task is bimanual (myoArm reaches with palm/arm muscles,
wraps fingers, hands off to an MPL prosthetic, which then carries the object
to a goal pillar). We are explicitly NOT doing bimanual manipulation, so
there is no goal pillar and no transport phase here -- just a single hand/exo
device reaching for and wrapping around an object on a start pillar. This
project also never drives shoulder/elbow/wrist muscles -- only the
exoskeleton devices in exo_devices.py, which act on finger/thumb joints only
-- so reach_err measures fingertip-to-object distance (the closing motion we
DO have), and the reward only scores reach + wrap.
"""

import numpy as np
import mujoco

from exo_devices import DEVICES, ExoApplicator, JOINT_NAMES
from grasp_controller import GraspController

MODEL_PATH = "myogloves_dev/models/myohand_exoglove_env.xml"

# Adapted from the paper's 5-category touching_body (myoArm/MPL/start/goal/other):
# we only have one hand and no goal pillar (not doing bimanual manipulation).
TOUCH_CATEGORIES = ["hand", "start_pillar", "other"]


class GraspEnv:
    def __init__(self, device_name, seed=None):
        self.device_name = device_name
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.applicator = ExoApplicator(self.model)
        self.rng = np.random.default_rng(seed)

        self.obj_body_id = self.model.body("grasp_object").id
        self.obj_geom_id = self.model.geom("grasp_object_geom").id
        self.start_pillar_id = self.model.body("start_pillar").id
        self.start_pillar_geom_id = self.model.geom("start_pillar_geom").id

        self.hand_qadr = [self.model.joint(n).qposadr[0] for n in JOINT_NAMES]
        self.hand_dofadr = [self.model.joint(n).dofadr[0] for n in JOINT_NAMES]

        obj_jntid = self.model.body_jntadr[self.obj_body_id]
        self.obj_qposadr = self.model.jnt_qposadr[obj_jntid]
        self.obj_dofadr = self.model.jnt_dofadr[obj_jntid]

        # Nominal object geometry/mass/friction -- the randomization baseline,
        # matching the paper's "sampled relative to nominal" convention.
        self._nominal_size = self.model.geom_size[self.obj_geom_id].copy()
        self._nominal_mass = float(self.model.body_mass[self.obj_body_id])
        self._nominal_friction = self.model.geom_friction[self.obj_geom_id].copy()

        self.device = None
        self.controller = None
        self.t = 0.0
        self.max_contact_force = 0.0
        self.touch_steps = 0

    def reset(self, randomize=True):
        mujoco.mj_resetData(self.model, self.data)
        self.device = DEVICES[self.device_name]
        if not self.device.tau_max_is_calibrated:
            self.device.tau_max[:] = 0.03  # same below-saturation convention used everywhere else

        if randomize:
            # Per-axis ranges straight from the paper's own text: "+-0-5% for
            # width, +-0-10% for depth, and +-0-5% for height" applied to the
            # 72x88x28mm box. Axis mapping follows the object geom's own
            # half-extents (X,Y,Z) = (28mm,72mm,88mm) -- X=28mm("height"
            # label, +-5%), Y=72mm("width" label, +-5%, unchanged from the
            # original flat orientation), Z=88mm("depth" label, +-10%, now
            # the vertical axis per user correction -- see myohand_exoglove_env.xml).
            scale = 1.0 + self.rng.uniform([-0.05, -0.05, -0.10], [0.05, 0.05, 0.10])
            self.model.geom_size[self.obj_geom_id] = self._nominal_size * scale
            # Object mass +-50g (matches the paper's X=50 for myoChallengeBimanual-v0).
            self.model.body_mass[self.obj_body_id] = max(0.01, self._nominal_mass + self.rng.uniform(-0.05, 0.05))
            # Friction +-[0.1, 0.001, 0.00002] from nominal (the paper's own ranges,
            # applied around OUR nominal friction rather than literally reusing
            # theirs, since our object's baseline friction differs from theirs).
            friction_delta = self.rng.uniform([-0.1, -0.001, -0.00002], [0.1, 0.001, 0.00002])
            self.model.geom_friction[self.obj_geom_id] = np.clip(self._nominal_friction + friction_delta, 1e-6, None)
        else:
            self.model.geom_size[self.obj_geom_id] = self._nominal_size
            self.model.body_mass[self.obj_body_id] = self._nominal_mass
            self.model.geom_friction[self.obj_geom_id] = self._nominal_friction

        mujoco.mj_forward(self.model, self.data)
        self.controller = GraspController(self.model, self.device, self.obj_geom_id)
        self.t = 0.0
        self.max_contact_force = 0.0
        self.touch_steps = 0
        return self.get_obs_dict()

    def step(self):
        u = self.controller.step(self.model, self.data)
        self.applicator.apply(self.data, self.device, u=u, row_gate=self.controller.row_gate)
        mujoco.mj_step(self.model, self.data)
        self.t += self.model.opt.timestep

        touch = self._touching_body()
        if touch[0] > 0:
            self.touch_steps += 1
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.obj_geom_id in (c.geom1, c.geom2):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                self.max_contact_force = max(self.max_contact_force, abs(force[0]))

        obs = self.get_obs_dict()
        reward_dict = self.get_reward_dict(obs)
        return obs, reward_dict

    def _fingertip_body_ids(self):
        """The most distal body each input channel drives, using JOINT_NAMES'
        established proximal->distal ordering (see exo_devices.py) -- e.g. for
        a channel driving mcp2/pm2/md2, this returns md2_flexion's body."""
        ids = []
        for col in range(self.device.n_inputs):
            active_rows = [row for row in range(len(JOINT_NAMES)) if self.device.K[row, col] != 0.0]
            if not active_rows:
                continue
            last_row = max(active_rows)
            ids.append(self.model.joint(JOINT_NAMES[last_row]).bodyid[0])
        return ids if ids else [self.obj_body_id]

    def _touching_body(self):
        onehot = np.zeros(len(TOUCH_CATEGORIES))
        hand_body_ids = set(self._fingertip_body_ids())
        # also count any driven-chain body, not just the fingertip, as "hand"
        for col in range(self.device.n_inputs):
            for row in range(len(JOINT_NAMES)):
                if self.device.K[row, col] != 0.0:
                    hand_body_ids.add(self.model.joint(JOINT_NAMES[row]).bodyid[0])
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.obj_geom_id not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.obj_geom_id else c.geom1
            other_body = self.model.geom_bodyid[other]
            if other_body in hand_body_ids:
                onehot[0] = 1
            elif other == self.start_pillar_geom_id:
                onehot[1] = 1
            else:
                onehot[2] = 1
        return onehot

    def get_obs_dict(self):
        obj_pos = self.data.xpos[self.obj_body_id].copy()
        obj_quat = self.data.xquat[self.obj_body_id].copy()

        fingertip_ids = self._fingertip_body_ids()
        centroid = np.mean([self.data.xpos[b] for b in fingertip_ids], axis=0)
        reach_err = obj_pos - centroid

        return {
            "time": np.array([self.t]),
            "hand_qpos": self.data.qpos[self.hand_qadr].copy(),
            "hand_qvel": self.data.qvel[self.hand_dofadr].copy(),
            "object_qpos": np.concatenate([obj_pos, obj_quat]),
            "object_qvel": self.data.qvel[self.obj_dofadr:self.obj_dofadr + 6].copy(),
            "start_pos": self.data.xpos[self.start_pillar_id].copy(),
            "touching_body": self._touching_body(),
            "u": self.controller.u.copy(),
            "reach_err": reach_err,
        }

    def get_reward_dict(self, obs):
        # Phase 1 (reach): same exp(-sigma * ||.||^2) form as the paper's Eq. 2,
        # applied to fingertip-centroid-to-object distance (our stand-in for
        # palm-to-object distance -- see module SCOPE NOTE).
        reach_dist2 = float(np.sum(obs["reach_err"] ** 2))
        reach_reward = float(np.exp(-8.0 * reach_dist2))

        # Phase 2 (wrap/grasp): contact-based. Mirrors the paper's curriculum
        # shifting weight from palm-distance to finger-distance/contact.
        wrap_reward = float(obs["touching_body"][0])

        total = 0.6 * reach_reward + 0.4 * wrap_reward
        return {"reach": reach_reward, "wrap": wrap_reward, "transport": None, "total": total}

    def success(self, min_touch_steps=500, max_force_n=5.0):
        """Adapted from the paper's success condition (sustained touch + force
        cap); their goal-proximity clause is dropped since there's no goal
        pillar here (not doing bimanual manipulation -- see module SCOPE
        NOTE). max_force_n is NOT their 1500N (sized for a full prosthetic
        arm) -- picked as a small but real cap appropriate to our glove-scale
        forces."""
        return self.touch_steps >= min_touch_steps and self.max_contact_force <= max_force_n
