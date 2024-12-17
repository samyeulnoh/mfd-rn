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
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer import initializers
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation import props as m_props
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


_StackWorkspace = collections.namedtuple(
    "_StackWorkspace", ["prop_bbox", "target_bbox", "tcp_bbox", "arm_offset"]
)

# Ensures that the prop does not collide with the table during initialization.
_PROP_Z_OFFSET = 1e-6

# Box parameters
_BOX_SIZE = 0.025
_BOX_MASS = 0.1

# reward parameters
_VERTICAL_MARGIN = 0.30
_REACH_MARGIN = 0.05
_SUCCEED_MARGIN = 0.01
_PICK_MARGIN = 0.035
_TARGET_Z_MARGIN = 0.01
_OBJ_Z_MARGIN = 0.03
_TIME_LIMIT = 10.0

# pedestal(obstacle) parameters
_TARGET_RADIUS = 0.05
_PEDESTAL_RADIUS = 0.045
_PEDESTAL_HEIGHT  = 0.015
_PEDESTAL_START_Z = 0.185


_WORKSPACE = _StackWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET), upper=(0.1, 0.1, _PROP_Z_OFFSET)
    ),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PEDESTAL_START_Z + 0.1), upper=(0.1, 0.1, 0.4)
    ),
    target_bbox=workspaces.BoundingBox(
        # lower=(0.0, 0.230, _PEDESTAL_START_Z), upper=(0.0, 0.230, _PEDESTAL_START_Z),
        lower=(0.0, 0.0, _PEDESTAL_START_Z), upper=(0.0, 0.0, _PEDESTAL_START_Z),
    ),
    arm_offset=robots.ARM_OFFSET,
)


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


class SphereCradle(composer.Entity):
    """A concave shape for easy placement."""

    _SPHERE_COUNT = 3

    def _build(self):
        self._mjcf_root = mjcf.element.RootElement(model="cradle")
        sphere_radius = _PEDESTAL_RADIUS * 0.7
        for ang in np.linspace(0, 2 * np.pi, num=self._SPHERE_COUNT, endpoint=False):
            pos = 0.7 * sphere_radius * np.array([np.sin(ang), np.cos(ang), -1])
            self._mjcf_root.worldbody.add(
                "geom", type="sphere", size=[sphere_radius], condim=6, pos=pos
            )

    @property
    def mjcf_model(self):
        return self._mjcf_root


class Pedestal(composer.Entity):
    """A narrow pillar to elevate the target."""

    _HEIGHT = _PEDESTAL_HEIGHT  # 0.075

    def _build(self, cradle, target_radius):
        self._mjcf_root = mjcf.element.RootElement(model="pedestal")

        self._mjcf_root.worldbody.add(
            "geom",
            # type="capsule",
            type="cylinder",
            size=[_PEDESTAL_RADIUS],
            # fromto=[0, 0, -_PEDESTAL_RADIUS, 0, 0, -(self._HEIGHT + _PEDESTAL_RADIUS)],
            fromto=[0, 0, 0, 0, 0, (self._HEIGHT)],
        )
        attachment_site = self._mjcf_root.worldbody.add(
            "site", type="sphere", size=(0.003,), group=constants.TASK_SITE_GROUP
        )
        # self.attach(cradle, attachment_site)
        self._target_site = workspaces.add_target_site(
            body=self.mjcf_model.worldbody, radius=target_radius, rgba=constants.RED
        )

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def target_site(self):
        return self._target_site

    def _build_observables(self):
        return PedestalObservables(self)


class PedestalObservables(composer.Observables):
    """Observables for the `Pedestal` prop."""

    @define.observable
    def position(self):
        return observable.MJCFFeature("xpos", self._entity.target_site)


class Stack(composer.Task):
    """Pick and place the prop on top of the fixed obstacle composed by a pedestal."""

    def __init__(
        self, arena, arm, hand, prop, obs_settings, workspace, control_timestep, cradle
    ):
        """Initializes a new `Stack` task.
        Args:
          arena: `composer.Entity` instance.
          arm: `robot_base.RobotArm` instance.
          hand: `robot_base.RobotHand` instance.
          prop: `composer.Entity` instance.
          obs_settings: `observations.ObservationSettings` instance.
          workspace: A `_StackWorkspace` instance.
          control_timestep: Float specifying the control timestep in seconds.
          cradle: `composer.Entity` onto which the `prop` must be placed.
        """
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._total_reward_num = 6.0

        # Add custom camera observable.
        self._task_observables = cameras.add_camera_observables(
            arena, obs_settings, cameras.FRONT_CLOSE
        )

        self._tcp_initializer = initializers.ToolCenterPointInitializer(
            self._hand,
            self._arm,
            position=distributions.Uniform(*workspace.tcp_bbox),
            quaternion=workspaces.DOWN_QUATERNION,
        )

        self._prop = prop
        self._prop_frame = self._arena.add_free_entity(prop)
        self._pedestal = Pedestal(cradle=cradle, target_radius=_TARGET_RADIUS)
        self._arena.attach(self._pedestal)

        for obs in self._pedestal.observables.as_dict().values():
            obs.configure(**obs_settings.prop_pose._asdict())

        self._prop_placer = initializers.PropPlacer(
            props=[prop],
            position=distributions.Uniform(*workspace.prop_bbox),
            quaternion=workspaces.uniform_z_rotation,
            ignore_collisions=True,
            settle_physics=True,
            max_attempts_per_prop=50,
        )

        self._pedestal_placer = initializers.PropPlacer(
            props=[self._pedestal],
            position=distributions.Uniform(*workspace.target_bbox),
            settle_physics=False,
        )

        # Add sites for visual debugging.
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.tcp_bbox.lower,
            upper=workspace.tcp_bbox.upper,
            rgba=constants.GREEN,
            name="tcp_spawn_area",
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.prop_bbox.lower,
            upper=workspace.prop_bbox.upper,
            rgba=constants.BLUE,
            name="prop_spawn_area",
        )
        workspaces.add_bbox_site(
            body=self.root_entity.mjcf_model.worldbody,
            lower=workspace.target_bbox.lower,
            upper=workspace.target_bbox.upper,
            rgba=constants.CYAN,
            name="pedestal_spawn_area",
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
        self._pedestal_placer(physics, random_state, ignore_contacts_with_entities=[self._prop])
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        self._prop_placer(physics, random_state)

    # ================================================

    def _cos_vec(self, s, e1, e2):
        vec1 = np.array(e1) - s
        vec2 = np.array(e2) - s
        v1u = vec1 / np.linalg.norm(vec1)
        v2u = vec2 / np.linalg.norm(vec2)
        return np.clip(np.dot(v1u, v2u), -1.0, 1.0)

    def _cos_dist(self, a, b):
        return np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)

    # ================================================
    
    def get_reward(self, physics):
        # observations
        target = physics.bind(self._pedestal.target_site).xpos.copy()
        target[2] = _PEDESTAL_START_Z + _PEDESTAL_HEIGHT + _BOX_SIZE # 0.11
        obj = physics.bind(self._prop_frame).xpos.copy()
        tcp = physics.bind(self._hand.tool_center_point).xpos.copy()
        hand = physics.bind(self._hand.hand_geom).xpos.copy()

        # vertical_reward
        tcp_to_hand = tcp - hand[0]
        angle = self._cos_dist(tcp_to_hand, np.array([0, 0, -1]))
        vertical_reward = rewards.tolerance(
            angle,
            bounds=(_VERTICAL_MARGIN, 1.0),
            margin=_VERTICAL_MARGIN,
        )

        # reach reward
        tcp_to_obj = np.linalg.norm(tcp - obj)
        reach_reward = rewards.tolerance(
            tcp_to_obj,
            bounds=(0.0, _REACH_MARGIN),
            margin=_REACH_MARGIN,
        )

        # pick reward
        _TARGET_Z = target[2]
        obj_z_score = (obj[-1] - _OBJ_Z_MARGIN) / (
            _TARGET_Z - _TARGET_Z_MARGIN - _OBJ_Z_MARGIN
        )
        pick_reward = max(0.0, min(2.0 * obj_z_score, 1.0))

        # hold reward
        hold_target = target.copy()
        hold_target[2] = target[2] + _BOX_SIZE
        tcp_to_target = np.linalg.norm(obj - hold_target)
        hold_reward = rewards.tolerance(
            tcp_to_target,
            bounds=(0, _SUCCEED_MARGIN),
            margin=_SUCCEED_MARGIN,
            sigmoid="long_tail",
        )

        # place reward
        obj_to_target = np.linalg.norm(obj - target)
        place_reward = rewards.tolerance(
            obj_to_target,
            bounds=(0, _REACH_MARGIN),
            margin=_REACH_MARGIN,
            sigmoid="long_tail",
        )

        # success reward
        success_reward = rewards.tolerance(
            obj_to_target,
            bounds=(0, _SUCCEED_MARGIN),
            margin=_SUCCEED_MARGIN,
            sigmoid="long_tail",
        )

        # hand_away_reward
        hand_target = target.copy()
        # hand_target[0] = 0.0
        # hand_target[1] = 0.20
        hand_target[2] = target[2] + 0.10 
        tcp_to_origin = np.linalg.norm(tcp - hand_target)
        handaway_reward = rewards.tolerance(
            tcp_to_origin,
            bounds=(0, _SUCCEED_MARGIN),
            margin=_SUCCEED_MARGIN,
            sigmoid="long_tail",
        )

        _completed_reach = True if tcp_to_obj < _REACH_MARGIN else False
        _obj_floated = True if obj[-1] > _PICK_MARGIN else False
        _obj_picked = True if obj[-1] > _PICK_MARGIN and _completed_reach else False
        _obj_placed = True if obj_to_target < _SUCCEED_MARGIN and _obj_floated else False

        # 1. vertical reward
        # 2. reach object
        # 3. pick object
        pick_reward = pick_reward if _obj_picked else 0.0
        # 4. hold object
        hold_reward = hold_reward if _obj_picked else 0.0
        # 5. place object
        place_reward = place_reward if _obj_picked else 0.0
        # 5. place object
        success_reward = success_reward if _obj_picked else 0.0
        ### completed
        reach_reward = 1.0 if _obj_placed else reach_reward
        pick_reward = 1.0 if _obj_placed else pick_reward
        hold_reward = 1.0 if _obj_placed else hold_reward
        place_reward = 1.0 if _obj_placed else place_reward
        success_reward = 1.0 if _obj_placed else success_reward
        # 6. handaway object
        handaway_reward = handaway_reward if _obj_placed else 0.0

        return (
            1.0 * reach_reward / self._total_reward_num
            + 1.0 * vertical_reward / self._total_reward_num
            + 1.0 * pick_reward / self._total_reward_num
            # + 1.0 * hold_reward / self._total_reward_num
            + 1.0 * place_reward / self._total_reward_num
            + 1.0 * success_reward / self._total_reward_num
            + 1.0 * handaway_reward / self._total_reward_num
        )


def _stack(obs_settings, cradle_prop_name):
    """Configure and instantiate a Stack task.

    Args:
      obs_settings: `observations.ObservationSettings` instance.
      cradle_prop_name: The name of the prop onto which the Duplo brick must be
        placed. Must be either 'duplo' or 'cradle'.

    Returns:
      An instance of `Stack`.

    Raises:
      ValueError: If `prop_name` is neither 'duplo' nor 'cradle'.
    """
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=obs_settings)
    hand = robots.make_hand(obs_settings=obs_settings)

    if cradle_prop_name == "duplo":
        # workspace = _DUPLO_WORKSPACE
        prop = props.Duplo(
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            ),
        )
    elif cradle_prop_name == "box":
        # workspace = _BOX_WORKSPACE
        # NB: The box is intentionally too large to pick up with a pinch grip.
        prop = _BoxWithVertexSites(
            size=[_BOX_SIZE] * 3,
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            ),
        )
        prop.geom.mass = _BOX_MASS
        prop.geom.rgba = [1, 0, 0, 1.0]  # TODO for manipulation.props (m_props)
    else:
        raise ValueError("'prop_name' must be either 'duplo' or 'box'.")

    if cradle_prop_name == "duplo":
        cradle = props.Duplo()
    elif cradle_prop_name == "box":
        cradle = m_props.Sphere()
    elif cradle_prop_name == "cradle":
        cradle = SphereCradle()
    else:
        raise ValueError("`cradle_prop_name` must be either 'duplo' or 'cradle'.")

    task = Stack(
        arena=arena,
        arm=arm,
        hand=hand,
        prop=prop,
        obs_settings=obs_settings,
        workspace=_WORKSPACE,
        control_timestep=constants.CONTROL_TIMESTEP,
        cradle=cradle,
    )
    return task


def make(prop, seed):
    obs_settings = observations.VISION
    task = _stack(obs_settings=obs_settings, cradle_prop_name=prop)
    return composer.Environment(task, time_limit=_TIME_LIMIT, random_state=seed)
