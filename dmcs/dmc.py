from collections import OrderedDict, deque
from typing import Any, NamedTuple
import numpy as np

from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
import dm_env
from dm_env import StepType, specs

import dmcs.jaco_tasks as cjaco


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env, state_key):
        self._env = env
        self._state_key = state_key
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if "front_close" in wrapped_obs_spec:
            spec = wrapped_obs_spec["front_close"]
            # drop batch dim
            self._obs_spec["pixels"] = specs.BoundedArray(
                shape=spec.shape[1:],
                dtype=spec.dtype,
                minimum=spec.minimum,
                maximum=spec.maximum,
                name="pixels",
            )
            wrapped_obs_spec.pop("front_close")

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array

        dim = 0
        for key, value in wrapped_obs_spec.items():
            if key in state_key:
                dim += np.sum(np.fromiter([np.int(np.prod(value.shape))], np.int32))

        self._obs_spec["observations"] = specs.Array(
            shape=(dim,), dtype=np.float32, name="observations"
        )

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if "front_close" in time_step.observation:
            pixels = time_step.observation["front_close"]
            time_step.observation.pop("front_close")
            pixels = np.squeeze(pixels)
            obs["pixels"] = pixels

        features = []
        for key, feature in time_step.observation.items():
            if key in self._state_key:
                features.append(feature.ravel())
        obs["observations"] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    state: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels", states_key="states"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key
        self._states_key = states_key
        self._obs_spec = OrderedDict()

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        pixels_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

        states_shape = wrapped_obs_spec["observations"].shape
        states_spec = specs.Array(
            states_shape,
            dtype=np.float32,
            name="state",
        )

        self._obs_spec[pixels_key] = pixels_spec
        self._obs_spec[states_key] = states_spec

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = OrderedDict()
        obs[self._pixels_key] = np.concatenate(list(self._frames), axis=0)
        obs[self._states_key] = time_step.observation["observations"]
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype, pixels_key="pixels", states_key="states"):
        self._env = env
        self._dtype = dtype
        self._pixels_key = pixels_key
        self._states_key = states_key
        wrapped_states_spec = env.observation_spec()[states_key]
        states_spec = specs.Array(wrapped_states_spec.shape, dtype, "state")
        self._obs_spec = OrderedDict()
        self._obs_spec[pixels_key] = env.observation_spec()[pixels_key]
        self._obs_spec[states_key] = states_spec

    def _transform_observation(self, time_step):
        obs = OrderedDict()
        obs[self._pixels_key] = time_step.observation[self._pixels_key]
        obs[self._states_key] = time_step.observation[self._states_key].astype(
            self._dtype
        )
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation["pixels"],
            state=time_step.observation["states"],
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


def _make_manipulation(state_key, domain, task, action_repeat, seed):
    env = cjaco.make_jaco(domain, task, seed)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env, state_key)

    return env


def _make_suite(state_key, domain, task, action_repeat, seed):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            task_kwargs=dict(random=seed),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    else:
        raise ValueError("(domain, task) should be in DeepMind Control Suite.")

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    # zoom in camera for quadruped
    camera_id = dict(quadruped=2).get(domain, 0)
    render_kwargs = dict(height=84, width=84, camera_id=camera_id)
    env = pixels.Wrapper(env, pixels_only=False, render_kwargs=render_kwargs)

    return env


def make(name, state_key, frame_stack, action_repeat, seed):
    domain, task = name.split("_", 1)
    domain = dict(cup="ball_in_cup").get(domain, domain)

    # suite: tasks from the DMC suite
    # manipulation: tasks built on top of the DMC manipulation
    make_fn = _make_manipulation if domain in ("jaco", "dmc") else _make_suite
    env = make_fn(state_key, domain, task, action_repeat, seed)
    env = FrameStackWrapper(env, frame_stack)
    env = ObservationDTypeWrapper(env, np.float32)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)

    return env
