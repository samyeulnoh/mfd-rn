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
from dm_control.manipulation import props as m_props  # TODO
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


_PushWorkspace = collections.namedtuple(
    "_PushWorkspace", ["prop_bbox", "tcp_bbox", "target_bbox", "obs_bbox", "arm_offset"]
)

# ensure that the props are not touching the table before setting.
_PROP_Z_OFFSET = 0.001

# box parameters
_BOX_SIZE = 0.025
_BOX_MASS = 0.1

# pedestal parameters
_TARGET_RADIUS = 0.05
_PEDESTAL_RADIUS = 0.085 # 0.10 # 0.05 # 0.05
_PEDESTAL_HEIGHT = 0.015 # 0.02 # 0.03 # 0.05
_PEDESTAL_START_Z = 0.135 # 0.15 # 0.12 # 0.15

# common parameters
_TARGET_SIZE = 0.025
_TARGET_MARGIN = 0.05
_SUCCESS_MARGIN = 0.01
_TIME_LIMIT = 10.0


# box workspace
_BOX_WORKSPACE = _PushWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET), upper=(0.1, 0.1, _PROP_Z_OFFSET),
    ),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.2), upper=(0.1, 0.1, 0.4),
    ),
    target_bbox=workspaces.BoundingBox(
        # lower=(-0.2, -0.2, _PROP_Z_OFFSET), upper=(0.2, 0.2, _PROP_Z_OFFSET),
        lower=(-0.0, -0.2, _BOX_SIZE / 2.0), upper=(0.0, -0.2, _BOX_SIZE / 2.0),
    ),
    obs_bbox=workspaces.BoundingBox(
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

    _HEIGHT = _PEDESTAL_HEIGHT 

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


class Push(composer.Task):
    """A task where the goal is to push a prop under an environment with an obstacle."""

    def __init__(
        self, arena, arm, hand, prop, obs_settings, workspace, control_timestep, cradle
    ):
        """Initializes a new `Push` task.

        Args:
          arena: `composer.Entity` instance.
          arm: `robot_base.RobotArm` instance.
          hand: `robot_base.RobotHand` instance.
          prop: `composer.Entity` instance.
          obs_settings: `observations.ObservationSettings` instance.
          workspace: `_PushWorkspace` specifying the placement of the prop and TCP.
          control_timestep: Float specifying the control timestep in seconds.
          cradle: 'composer.Entity' instance (the obstacle constraint).
        """
        self._arena = arena
        self._arm = arm
        self._hand = hand
        self._arm.attach(self._hand)
        self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
        self.control_timestep = control_timestep
        self._total_reward_number = 3.0

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
            visible=True,
            rgba=constants.GREEN,
            name="target",
        )
        self._target_placer = distributions.Uniform(*workspace.target_bbox)

        # pedestal(obstacle)
        self._pedestal = Pedestal(cradle=cradle, target_radius=_TARGET_RADIUS)
        self._arena.attach(self._pedestal)

        for obs in self._pedestal.observables.as_dict().values():
            obs.configure(**obs_settings.prop_pose._asdict())

        self._pedestal_placer = initializers.PropPlacer(
            props=[self._pedestal],
            position=distributions.Uniform(*workspace.obs_bbox),
            settle_physics=False,
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
        self._pedestal_placer(physics, random_state, ignore_contacts_with_entities=[self._prop])
        self._hand.set_grasp(physics, close_factors=random_state.uniform())
        self._tcp_initializer(physics, random_state)
        self._prop_placer(physics, random_state)
        # physics.bind(self._target).pos = _TARGET_POSITION
        physics.bind(self._target).pos = self._target_placer(random_state=random_state) 

    def get_reward(self, physics):
        # observations
        obj = physics.bind(self._prop_frame).xpos.copy()
        tcp = physics.bind(self._hand.tool_center_point).xpos.copy()
        target = physics.bind(self._target).xpos.copy()

        # reach reward
        tcp_to_obj = np.linalg.norm(tcp - obj)
        reach_reward = rewards.tolerance(
            tcp_to_obj,
            bounds=(0, _TARGET_MARGIN),
            margin=_TARGET_MARGIN,
        )
        _completed_reach = True if tcp_to_obj < _TARGET_MARGIN else False

        # push reward
        obj_to_target = np.linalg.norm(obj - target)
        push_reward = rewards.tolerance(
            obj_to_target,
            bounds=(0, _TARGET_MARGIN),
            margin=_TARGET_MARGIN,
            sigmoid="long_tail",
        )

        # success_reward
        success_reward = rewards.tolerance(
            obj_to_target,
            bounds=(0, _SUCCESS_MARGIN),
            margin=_SUCCESS_MARGIN,
            sigmoid="long_tail",
        )

        push_reward = push_reward if _completed_reach else 0.0
        success_reward = success_reward if _completed_reach else 0.0

        return (
            1.0 * reach_reward / self._total_reward_number
            + 1.0 * push_reward / self._total_reward_number
            + 1.0 * success_reward / self._total_reward_number
        )


def _push(obs_settings, prop_name):
    """Configure and instantiate a Push task.

    Args:
      obs_settings: `observations.ObservationSettings` instance.
      prop_name: The name of the prop to be pushed. Must be either 'duplo' or
        'box'.

    Returns:
      An instance of `push.Push`.

    Raises:
      ValueError: If `prop_name` is neither 'duplo' nor 'box'.
    """
    arena = arenas.Standard()
    arm = robots.make_arm(obs_settings=obs_settings)
    hand = robots.make_hand(obs_settings=obs_settings)

    if prop_name == "duplo":
        prop = props.Duplo(
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            )
        )
        cradle = props.Duplo()
    elif prop_name == "box":
        prop = _BoxWithVertexSites(
            size=[_BOX_SIZE] * 3,
            observable_options=observations.make_options(
                obs_settings, observations.FREEPROP_OBSERVABLES
            ),
        )
        prop.geom.mass = _BOX_MASS
        prop.geom.rgba = [1, 0, 0, 1.0]
        cradle = m_props.Sphere()
    else:
        raise ValueError("`prop_name` must be either 'duplo' or 'box'.")

    task = Push(
        arena=arena,
        arm=arm,
        hand=hand,
        prop=prop,
        workspace=_BOX_WORKSPACE,
        obs_settings=obs_settings,
        control_timestep=constants.CONTROL_TIMESTEP,
        cradle=cradle,
    )

    return task


def make(prop, seed):
    obs_settings = observations.VISION
    task = _push(obs_settings=obs_settings, prop_name=prop)
    return composer.Environment(task, time_limit=_TIME_LIMIT, random_state=seed)
