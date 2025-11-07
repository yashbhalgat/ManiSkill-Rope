from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PickRope-v1", max_episode_steps=75)
class PickRopeEnv(BaseEnv):
    """
    Pick a rope made of small cube links connected by revolute joints.
    Success: grasp any link and lift part of the rope above a height threshold while robot is static.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch", "xarm6_robotiq", "so100", "widowxai"]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]

    def __init__(
        self,
        *args,
        robot_uids: Union[str, tuple] = "panda",
        robot_init_qpos_noise: float = 0.02,
        num_links: int = 20,
        link_half_size: float = 0.01,
        link_gap: float = 0.001,
        joint_limit_deg: float = 90.0,
        joint_damping: float = 2.0,
        joint_friction: float = 0.02,
        lift_height_thresh: float = 0.05,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        # Rope params
        self.num_links = int(num_links)
        self.link_half_size = float(link_half_size)
        self.link_gap = float(link_gap)
        self.joint_limit = float(np.deg2rad(joint_limit_deg))
        self.joint_damping = float(joint_damping)
        self.joint_friction = float(joint_friction)
        self.lift_height_thresh = float(lift_height_thresh)

        # Camera defaults similar to PickCube
        self.sensor_cam_eye_pos = [0.3, 0, 0.6]
        self.sensor_cam_target_pos = [-0.1, 0, 0.1]
        self.human_cam_eye_pos = [0.6, 0.7, 0.6]
        self.human_cam_target_pos = [0.0, 0.0, 0.35]

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        # place robot similar to PickCube
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _build_rope_articulation(self):
        # Build an articulation with N box links connected by revolute joints
        builder = self.scene.create_articulation_builder()
        # Enable self-collisions for physical correctness
        builder.disable_self_collisions = False

        s = self.link_half_size
        # Rotate joint frames so hinge axis becomes Z (90 deg about Y), per robel example
        joint_frame_q = [0.70710678, 0.0, 0.70710678, 0.0]

        # Prepare a rope-like brown material
        rope_mat = sapien.render.RenderMaterial(
            base_color=[0.36, 0.24, 0.15, 1.0],  # brown
            roughness=0.8,
            specular=0.2,
            metallic=0.0,
        )

        # Root link
        parent = builder.create_link_builder(parent=None)
        parent.set_name("link_0")
        parent.add_box_collision(half_size=[s, s, s])
        parent.add_box_visual(half_size=[s, s, s], material=rope_mat)

        # Chain
        for i in range(1, self.num_links):
            child = builder.create_link_builder(parent=parent)
            child.set_name(f"link_{i}")
            child.add_box_collision(half_size=[s, s, s])
            child.add_box_visual(half_size=[s, s, s], material=rope_mat)
            child.set_joint_name(f"joint_{i}")

            # Connect near face centers along +X / -X but with a small gap to avoid initial intersection
            gap = self.link_gap
            pose_in_parent = sapien.Pose(p=[s + gap, 0.0, 0.0], q=joint_frame_q)
            pose_in_child = sapien.Pose(p=[-s - gap, 0.0, 0.0], q=joint_frame_q)
            child.set_joint_properties(
                type="revolute",
                limits=[[-self.joint_limit, self.joint_limit]],
                pose_in_parent=pose_in_parent,
                pose_in_child=pose_in_child,
                friction=self.joint_friction,
                damping=self.joint_damping,
            )
            parent = child

        # Initial pose above table; randomized in reset
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, s + 0.05])
        # Free-floating rope (not fixed to world)
        self.rope = builder.build(name="rope", fix_root_link=False)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._build_rope_articulation()
        # Track whether each env has ever succeeded in the current episode (for success_once metric)
        self._success_once = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize rope root pose on table
            p = torch.zeros((b, 3))
            p[..., 0:2] = torch.rand((b, 2)) * 0.20 - 0.10
            # Give some clearance above the table to avoid initial contact explosions
            p[..., 2] = self.link_half_size + 0.03
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.rope.set_pose(Pose.create_from_pq(p=p, q=q))

            # Initialize joints into a randomized rope-like curve per environment
            # Number of active hinge joints equals number of links - 1
            num_active = len(self.rope.get_active_joints())
            if num_active > 0:
                link_len = 2 * self.link_half_size + 2 * self.link_gap
                # Precompute indices mask to ignore adjacent links when checking distances
                if num_active + 1 >= 3:
                    idxs = torch.arange(self.num_links, device=self.device)
                # Generate per-env shapes with rejection sampling to avoid self-collisions
                accepted_angles = torch.zeros((b, num_active), device=self.device)
                for bi in range(b):
                    for _ in range(8):  # up to 8 attempts
                        # Conservative parameters to reduce overlap probability
                        amp = (0.1 + 0.2 * torch.rand(1, device=self.device)) * self.joint_limit
                        cycles = 0.5 + 1.5 * torch.rand(1, device=self.device)
                        phase = 2 * np.pi * torch.rand(1, device=self.device)
                        j_axis = torch.linspace(0.0, 1.0, num_active, device=self.device)
                        base = amp * torch.sin(phase + 2 * np.pi * cycles * j_axis)
                        noise = torch.randn((num_active,), device=self.device) * (0.04 * self.joint_limit)
                        smooth = torch.cumsum(noise, dim=0) / 4.0
                        ang = torch.clamp(base + 0.2 * smooth, -self.joint_limit, self.joint_limit)

                        # Approximate forward kinematics in plane to test self-overlap
                        # link 0 center at (0,0); each subsequent link translates by rotated [link_len, 0]
                        thetas = torch.zeros((self.num_links,), device=self.device)
                        thetas[1:] = torch.cumsum(ang, dim=0)
                        # cumulative rotation for each segment
                        cos_t = torch.cos(thetas)
                        sin_t = torch.sin(thetas)
                        dx = link_len * cos_t
                        dy = link_len * sin_t
                        # centers as cumulative sum; link 0 at (0,0)
                        xs = torch.cumsum(dx, dim=0)
                        ys = torch.cumsum(dy, dim=0)
                        # compute pairwise distances between non-adjacent links
                        pts = torch.stack([xs, ys], dim=1)
                        dmat = torch.cdist(pts, pts)
                        # mask out i==j and adjacent pairs
                        mask = torch.ones_like(dmat, dtype=torch.bool)
                        mask.fill_(True)
                        mask.fill_diagonal_(False)
                        for k in range(self.num_links - 1):
                            mask[k, k + 1] = False
                            mask[k + 1, k] = False
                        min_non_adj = dmat[mask].min() if mask.any() else torch.tensor(float('inf'), device=self.device)
                        if min_non_adj > (2 * self.link_half_size + self.link_gap):
                            accepted_angles[bi] = ang
                            break
                    else:
                        # fallback: tiny curvature
                        accepted_angles[bi] = torch.zeros_like(accepted_angles[bi])

                qpos = torch.zeros((b, int(self.rope.max_dof)), device=self.device)
                qpos[:, :num_active] = accepted_angles
                self.rope.set_qpos(qpos)

            # reset success-once flag for these envs
            self._success_once[env_idx] = False

    def _rope_link_positions(self) -> torch.Tensor:
        ps = torch.stack([link.pose.p for link in self.rope.links], dim=0)
        return ps.permute(1, 0, 2)

    def _nearest_link_and_dist(self) -> tuple[torch.Tensor, torch.Tensor]:
        ps = self._rope_link_positions()
        tcp = self.agent.tcp_pose.p
        dists = torch.linalg.norm(ps - tcp[:, None, :], dim=2)
        idx = torch.argmin(dists, dim=1)
        mind = dists[torch.arange(len(dists), device=self.device), idx]
        return idx, mind

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped_any"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            idx, _ = self._nearest_link_and_dist()
            all_poses = torch.stack([l.pose.raw_pose for l in self.rope.links], dim=1)
            obj_pose = all_poses[torch.arange(len(idx), device=self.device), idx]
            obs.update(obj_pose=obj_pose)
        return obs

    def evaluate(self):
        grasp_flags = [self.agent.is_grasping(link) for link in self.rope.links]
        is_grasped_any = torch.stack(grasp_flags, dim=0).any(dim=0)

        ps = self._rope_link_positions()
        max_z = ps[..., 2].max(dim=1).values
        lifted = max_z > (0.0 + self.lift_height_thresh)

        is_robot_static = self.agent.is_static(0.2)
        success = is_grasped_any & lifted & is_robot_static
        # accumulate success across the episode for compatibility with ppo_fast logging
        self._success_once = torch.logical_or(self._success_once, success)
        return {
            "success": success,
            "success_once": self._success_once,
            "is_grasped_any": is_grasped_any,
            "is_lifted": lifted,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        _, mind = self._nearest_link_and_dist()
        reaching = 1 - torch.tanh(5 * mind)

        grasp_bonus = info["is_grasped_any"].to(mind.dtype)

        ps = self._rope_link_positions()
        max_z = ps[..., 2].max(dim=1).values
        lift_bonus = torch.clamp((max_z - self.lift_height_thresh) * 10.0, min=0.0, max=1.0)

        reward = reaching + grasp_bonus + lift_bonus
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5.0
