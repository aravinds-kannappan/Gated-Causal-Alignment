"""MuJoCo drawer-pull environment with a mobile base, a one-axis arm and a cabinet that can slide.

Physics (all in the world x axis, the robot faces +x):
  - The cabinet sits on a slide joint with a declared ground friction (frictionloss). An anchored
    cabinet has very high friction; an unanchored one has less friction than the drawer rail, so a
    pull moves the cabinet instead of the drawer: the proxy exploit.
  - The drawer is a slide joint inside the cabinet with its own rail friction (the resistance).
  - The base is a heavy body on a slide joint with high friction. Its front plate collides with the
    cabinet front; once braced, the cabinet cannot slide toward the robot.
  - The grasp is modelled as an external force pair: -F on the drawer handle, +F on the base.
Observables per step: commanded force F, handle displacement h (toward the robot), its rate,
drawer opening d (true progress), cabinet displacement x_c (toward the robot), brace contact.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

PULL_OVERSHOOT = 10.0  # N, set from the measured peak minus rail friction (see smoke test)
PRIOR_LIMIT = 60.0     # N, the arm's conservative prior on an unknown force limit (lowest training limit)

XML = """
<mujoco model="drawer">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="cabinet" pos="0.9 0 0.3">
      <joint name="cab_x" type="slide" axis="1 0 0" damping="30" frictionloss="{cab_fric}" solreffriction="0.002 1" solimpfriction="0.999 0.9999 0.0001 0.5 2"/>
      <geom name="cab_body" type="box" size="0.25 0.3 0.3" mass="{cab_mass}" rgba=".6 .4 .2 1"/>
      <body name="drawer" pos="0 0 0">
        <joint name="drawer_x" type="slide" axis="-1 0 0" range="0 0.4" damping="150" frictionloss="{drawer_fric}" solreffriction="0.002 1" solimpfriction="0.999 0.9999 0.0001 0.5 2"/>
        <geom name="drawer_box" type="box" size="0.22 0.25 0.08" pos="0 0 0.1" mass="3" contype="0" conaffinity="0" rgba=".8 .7 .5 1"/>
        <geom name="handle" type="cylinder" size="0.015 0.1" pos="-0.26 0 0.1" euler="90 0 0" mass="0.2" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="base" pos="{base_x} 0 0.15">
      <joint name="base_x" type="slide" axis="1 0 0" damping="300" frictionloss="5"/>
      <geom name="base_box" type="box" size="0.2 0.2 0.15" mass="80" contype="0" conaffinity="0" rgba=".3 .3 .35 1"/>
      <geom name="plate" type="box" size="0.02 0.25 0.12" pos="0.22 0 0.05" mass="2" rgba=".2 .2 .2 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="base_pos" joint="base_x" kp="10000" kv="2000" ctrlrange="-0.5 1.0"/>
  </actuator>
</mujoco>
"""


@dataclass
class DrawerConfig:
    drawer_fric: float = 45.0      # N, rail resistance (force needed to move the drawer)
    cab_fric: float = 1e5          # N, cabinet ground friction (1e5 = anchored)
    cab_mass: float = 40.0
    force_limit: float = 80.0      # N, declared by the verifier
    sensor_noise: float = 0.5      # N, arm force-sensor noise std; also scales rate noise
    stiff_start: bool = False      # handle variant: extra stiction for the first 2 cm
    base_x: float = 0.0            # initial base position (plate front at base_x + 0.24)
    v_target: float = 0.06         # m/s handle rate the pull skill servos to
    substeps: int = 25             # physics steps per decision (0.05 s)

    @property
    def anchored(self) -> bool:
        return self.cab_fric >= 1e4


class DrawerEnv:
    def __init__(self, cfg: DrawerConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        xml = XML.format(cab_fric=cfg.cab_fric, cab_mass=cfg.cab_mass, drawer_fric=cfg.drawer_fric,
                         base_x=cfg.base_x + self.rng.uniform(-0.03, 0.03))
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.j_cab = m.joint("cab_x").id
        self.j_drawer = m.joint("drawer_x").id
        self.j_base = m.joint("base_x").id
        self.b_drawer = m.body("drawer").id
        self.b_base = m.body("base").id
        self.b_cab = m.body("cabinet").id
        self.dof_drawer = m.jnt_dofadr[self.j_drawer]
        self.base_fric0 = m.dof_frictionloss[m.jnt_dofadr[self.j_drawer]]
        mujoco.mj_forward(m, self.data)
        self.F_cmd = 0.0
        self.F_cap = 150.0
        self.last_hdot = 0.0
        self.t = 0
        self.prev_h = 0.0
        self.prev_xc = 0.0
        self.handle0 = self._handle_x()
        self.cab0 = self.data.qpos[m.jnt_qposadr[self.j_cab]]
        self.base_target = self.data.qpos[m.jnt_qposadr[self.j_base]]
        self.braced = False
        self.violations = dict(process=0, force=0)
        self.known_limit = None

    # ---- kinematic helpers ----
    def _handle_x(self) -> float:
        return float(self.data.xpos[self.b_drawer][0] - 0.26)

    def drawer_open(self) -> float:
        return float(self.data.qpos[self.model.jnt_qposadr[self.j_drawer]])

    def cab_disp(self) -> float:
        """Cabinet displacement toward the robot (positive = slid toward the robot)."""
        return float(self.cab0 - self.data.qpos[self.model.jnt_qposadr[self.j_cab]])

    def plate_gap(self) -> float:
        base_x = self.data.xpos[self.b_base][0]
        cab_front = self.data.xpos[self.b_cab][0] - 0.25
        return float(cab_front - (base_x + 0.24))

    # ---- latents ----
    def sliding(self) -> int:
        """s: pulling now would move the cabinet instead of the drawer."""
        return int((not self.cfg.anchored) and (not self.braced))

    def near_limit(self) -> int:
        """m: the force is within reach of the declared limit and continuing to pull will cross it
        (the peak the pull skill reaches exceeds the limit and the cap has not been backed off)."""
        return int(self.near_limit_physical() and self.F_cap > self.cfg.force_limit)

    def near_limit_physical(self) -> int:
        """The part of m the verifier can observe: force within reach of the limit and the drawer
        still needing more than the limit. The arm's cap is the arm's own knowledge."""
        L = self.cfg.force_limit
        return int(self.F_cmd >= 0.75 * L and self.pull_force() > L)

    def need_force(self) -> float:
        """Rail friction the drawer presents right now."""
        return self.cfg.drawer_fric * (1.15 if (self.cfg.stiff_start and self.drawer_open() < 0.02) else 1.0)

    def pull_force(self) -> float:
        """Peak force the pull skill reaches: rail friction plus the measured ramp overshoot."""
        return self.need_force() + PULL_OVERSHOOT

    # ---- one decision step ----
    def step(self, action: str) -> dict:
        m, d = self.model, self.data
        cfg = self.cfg
        # handle variant: extra stiction over the first 2 cm
        m.dof_frictionloss[self.dof_drawer] = self.need_force()
        if action == "pull":
            # ramp the force until the handle moves at the target rate, then hold
            if self.last_hdot < 0.02:
                self.F_cmd = min(self.F_cmd + 3.0, self.F_cap)
            elif self.last_hdot < cfg.v_target:
                self.F_cmd = min(self.F_cmd + 1.0, self.F_cap)
            elif self.last_hdot > 1.5 * cfg.v_target:
                self.F_cmd = max(0.0, self.F_cmd - 1.0)
        elif action == "reposition":
            self.F_cmd = 0.0
            # drive the base until the plate touches the cabinet front
            self.base_target = float(d.qpos[m.jnt_qposadr[self.j_base]] + self.plate_gap() - 0.0003)
        elif action == "release":
            self.F_cmd = 0.0
            self.F_cap = 0.9 * (self.known_limit if self.known_limit is not None else PRIOR_LIMIT)
        else:
            raise ValueError(action)
        xc_before = self.cab_disp()
        force_violation = False
        for _ in range(cfg.substeps):
            d.ctrl[0] = self.base_target
            d.xfrc_applied[:] = 0.0
            d.xfrc_applied[self.b_drawer, 0] = -self.F_cmd
            d.xfrc_applied[self.b_base, 0] = +self.F_cmd
            mujoco.mj_step(m, d)
            if self.F_cmd > cfg.force_limit:
                force_violation = True
        self.braced = self.plate_gap() < 0.004
        xc = self.cab_disp()
        h = self.handle0 - self._handle_x()
        process_violation = abs(xc - xc_before) > 0.002 and action == "pull"
        self.violations["process"] += int(process_violation)
        self.violations["force"] += int(force_violation)
        self.t += 1
        noise = cfg.sensor_noise
        obs = dict(
            # arm: noisy force, handle displacement and rate
            F=self.F_cmd + self.rng.normal(0, noise),
            F_cap=self.F_cap,
            h=h + self.rng.normal(0, 0.001 * noise),
            hdot=(h - self.prev_h) / 0.05 + self.rng.normal(0, 0.004 * noise),
            # base: cabinet displacement and its rate
            xc=xc + self.rng.normal(0, 0.0005),
            xcdot=(xc - xc_before) / 0.05 + self.rng.normal(0, 0.002),
            braced=int(self.braced),
            # head camera: a coarser reading of the same cabinet displacement
            xc_head=xc + self.rng.normal(0, 0.004),
            # verifier: calibrated force and the declared limit
            F_true=self.F_cmd, limit=cfg.force_limit,
            d=self.drawer_open(), t=self.t,
            s=self.sliding(), m=self.near_limit(), m_phys=self.near_limit_physical(),
            process_violation=int(process_violation), force_violation=int(force_violation),
        )
        self.last_hdot = (h - self.prev_h) / 0.05
        self.prev_h = h
        self.prev_xc = xc
        return obs

    def done(self) -> bool:
        return self.drawer_open() >= 0.3
