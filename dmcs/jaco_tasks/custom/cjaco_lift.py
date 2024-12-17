# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import collections
import itertools

from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation import props as m_props  # custom 
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.utils import rewards
import numpy as np


_LiftWorkspace = collections.namedtuple(
    "_LiftWorkspace", ["prop_bbox", "tcp_bbox", "arm_offset"]
)

# ensure that the props are not touching the table before setting.
_PROP_Z_OFFSET = 0.001

# duplo workspace
_DUPLO_WORKSPACE = _LiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET),
        upper=(0.1, 0.1, _PROP_Z_OFFSET),
    ),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2),
        upper=(0.1, 0.1, 0.4),
    ),
    arm_offset=robots.ARM_OFFSET,
)

# box parameters
_BOX_SIZE = 0.025
_BOX_MASS = 0.1

# box workspace
_BOX_WORKSPACE = _LiftWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET),
        upper=(0.1, 0.1, _PROP_Z_OFFSET),
    ),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2),
        upper=(0.1, 0.1, 0.4),
    ),
    arm_offset=robots.ARM_OFFSET,
)

# common parameters
_TARGET_Z = 0.18
_TARGET_Z_MARGIN = 0.01
_TARGET_POSITION = [0.0, 0.0, _TARGET_Z]
_TARGET_SIZE = 0.02
_TARGET_MARGIN = 0.05
_SUCCESS_MARGIN = 0.01
_TIME_LIMIT = 10.0

# reward parameters
_TOUCH_FORCE_MARGIN = 5.0
_VERTICAL_MARGIN = 0.99
_FLOAT_MARGIN = 0.035
_OBJ_Z_MARGIN = 0.03


class _VertexSitesMixin:
    """Mixin class that adds sites corresponding to the vertices of a box."""

    def _add_vertex_sites(self, box_geom_or_site):
        """Add sites corresponding to the vertices of a box geom or site."""
        offsets = ((-half_length, half_length) for half_length in box_geom_or_site.size)
        site_positions = np.vstack(itertools.product(*offsets))
        if box_geom_or_site.pos is not None:
            site_positions += box_geom_or_site.pos
        self._vertices = []
        for i, pos in enumerate(site_positions):
            site = box_geom_or_site.parent.add(
                "site",
                name="vertex_" + str(i),
                pos=pos,
                type="sphere",
                size=[0.002],
                rgba=constants.RED,
                group=constants.TASK_SITE_GROUP,
            )
            self._vertices.append(site)

    @property
    def vertices(self):
        return self._vertices


class _BoxWithVertexSites(m_props.Primitive, _VertexSitesMixin):
    """Subclass of `Box` with sites marking the vertices of the box geom."""

    def _build(self, *args, **kwargs):
        super()._build(*args, geom_type="box", **kwargs)
        self._add_vertex_sites(self.geom)


class Lift(composer.Task):
    """A task where the goal is to elevate a prop."""

    def __init__(
        self, arena, arm, hand, prop, obs_settings, workspace, control_timestep
    ):
        """Initializes a new `Lift` task.

        Args:
          arena: `composer.Entity` instance.
          arm: `robot_base.RobotArm` instance.
          hand: `robot_base.RobotHand` instance.
          prop: `composer.Entity` instance.
          obs_settings: `observations.ObservationSettings` instance.
          workspace: `_LiftWorkspace` specifying the placement of the prop and TCP.
          control_timestep: Float specifying the control timestep in seconds.
        """
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._total_reward_num = 5.0

        # Add custom camera observable.
        self._task_observables = cameras.add_camera_observables(
            arena, obs_settings, cameras.FRONT_CLOSE
        )

        # hand tcp
        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand,
            self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=workspaces.DOWN_QUATERNION,
        )

        # prop
        self._prop = prop
        self._prop_frame = self._arena.add_free_entity(prop)
        self._prop_placer = initializers.PropPlacer(
            props=[prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=True,
            settle_physics=True,
        )

        # target
        self._target = workspaces.add_target_site(
            body=self.root_entity.mjcf_model.worldbody,
            radius=_TARGET_SIZE,
            visible=False,  # True
            rgba=constants.GREEN,
            name="target",
        )

        # Add sites for visualizing bounding boxes and target height.
        self._target_height_site = workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=(-1, -1, 0),
            upper=(1, 1, 0),
            visible=False,
            rgba=constants.RED,
            name="target_height",
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.tcp_bbox.lower,
            upper=workspace.tcp_bbox.upper,
            visible=False,
            rgba=constants.GREEN,
            name="tcp_spawn_area",
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower,
            upper=workspace.prop_bbox.upper,
            visible=False,
            rgba=constants.BLUE,
            name="prop_spawn_area",
        )

    @property
    def root_entity(self):
        return self._arena

    @property
    def arm(self):
        return self._arm

    @property
    def hand(self):
        return self._hand

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode(self, physics, random_state):
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        self._prop_placer(physics, random_state)
        physics.bind(self._target).pos = _TARGET_POSITION

    def _cos_vec(self, s, e1, e2):
        vec1 = np.array(e1) - s
        vec2 = np.array(e2) - s
        v1u = vec1 / np.linalg.norm(vec1)
        v2u = vec2 / np.linalg.norm(vec2)
        return np.clip(np.dot(v1u, v2u), -1.0, 1.0)

    def _cos_dist(self, a, b):
        return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

    def get_reward(self, physics):
        # observations
        obj = physics.bind(self._prop_frame).xpos.copy()
        obj_z = obj[-1]
        tcp = physics.bind(self._hand.tool_center_point).xpos.copy()
        target = physics.bind(self._target).xpos.copy()
        hand = physics.bind(self._hand.hand_geom).xpos.copy()

        # reach reward
        tcp_to_obj = np.linalg.norm(tcp - obj)
        reach_reward = rewards.tolerance(
            tcp_to_obj,
            bounds=(0.0, _TARGET_MARGIN),
            margin=_TARGET_MARGIN,
        )
        _completed_reach = True if tcp_to_obj < _TARGET_MARGIN else False

        # angle reward
        tcp_to_hand = tcp - hand[0]
        angle = self._cos_dist(tcp_to_hand, np.array([0, 0, -1]))
        angle_reward = rewards.tolerance(
            angle,
            bounds=(_VERTICAL_MARGIN, 1.0),
            margin=_VERTICAL_MARGIN,
        )

        # float reward
        obj_z_reward = (obj_z - _OBJ_Z_MARGIN) / (
            _TARGET_Z - _TARGET_Z_MARGIN - _OBJ_Z_MARGIN
        )
        float_reward = max(0.0, min(2.0 * obj_z_reward, 1.0))

        # lift reward
        obj_to_target = np.linalg.norm(obj - target)
        lift_reward = rewards.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_MARGIN),
            margin=_TARGET_MARGIN,
            sigmoid="long_tail",
        )

        # success reward
        success_reward = rewards.tolerance(
            obj_to_target,
            bounds=(0, _SUCCESS_MARGIN),
            margin=_SUCCESS_MARGIN,
            sigmoid="long_tail",
        )

        _obj_floated = True if obj_z > _FLOAT_MARGIN and _completed_reach else False
        float_reward = float_reward if _obj_floated else 0.0
        lift_reward = lift_reward if _obj_floated else 0.0
        success_reward = success_reward if _obj_floated else 0.0

        return (
            1.0 * reach_reward / self._total_reward_num
            + 1.0 * angle_reward / self._total_reward_num
            + 1.0 * float_reward / self._total_reward_num
            + 1.0 * lift_reward / self._total_reward_num
            + 1.0 * success_reward / self._total_reward_num
        )


def _lift(obs_settings, prop_name):
    """Configure and instantiate a Lift task.

    Args:
      obs_settings: `observations.ObservationSettings` instance.
      prop_name: The name of the prop to be lifted. Must be either 'duplo' or
        'box'.

    Returns:
      An instance of `lift.Lift`.

    Raises:
      ValueError: If `prop_name` is neither 'duplo' nor 'box'.
    """
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=obs_settings)
    hand = robots.make_hand(obs_settings=obs_settings)

    if prop_name == "duplo":
        workspace = _DUPLO_WORKSPACE
        prop = props.Duplo(
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            ),
        )
    elif prop_name == "box":
        workspace = _BOX_WORKSPACE
        prop = _BoxWithVertexSites(
            size=[_BOX_SIZE] * 3,
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            ),
        )
        prop.geom.mass = _BOX_MASS
        prop.geom.rgba = [1, 0, 0, 1.0]
    else:
        raise ValueError("`prop_name` must be either 'duplo' or 'box'.")

    task = Lift(
        arena=arena,
        arm=arm,
        hand=hand,
        prop=prop,
        workspace=workspace,
        obs_settings=obs_settings,
        control_timestep=constants.CONTROL_TIMESTEP,
    )

    return task


def make(prop, seed):
    obs_settings = observations.VISION
    task = _lift(obs_settings=obs_settings, prop_name=prop)
    return composer.Environment(task, time_limit=_TIME_LIMIT, random_state=seed)
