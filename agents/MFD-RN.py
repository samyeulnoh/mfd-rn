import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common.utils as utils


class ImageEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_rgb_dim = 32 * 35 * 35

        conv_layers = []
        conv_layers += [
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
        ]
        for _ in range(3):
            conv_layers += [
                nn.Conv2d(32, 32, 3, stride=1),
                nn.ReLU(),
            ]
        self.convnet = nn.Sequential(*conv_layers)

        self.apply(utils.weight_init)

    def forward(self, obs_rgb):
        norm_obs_rgb = obs_rgb / 255.0 - 0.5
        repr_rgb = self.convnet(norm_obs_rgb)
        repr_rgb = repr_rgb.view(repr_rgb.shape[0], -1)
        return repr_rgb


class PropEncoder(nn.Module):
    def __init__(self, obs_prop_dim):
        super().__init__()

        self.repr_prop_dim = 128
        hidden_dim = 128

        self.propnet = nn.Sequential(
            nn.Linear(obs_prop_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.repr_prop_dim),
            nn.LayerNorm(self.repr_prop_dim),
            nn.ReLU(inplace=True),
        )

        self.apply(utils.weight_init)

    def forward(self, obs_prop):
        repr_prop = self.propnet(obs_prop)
        return repr_prop


class IdentityEncoder(nn.Module):
    def __init__(self, obs_prop_dim):
        super().__init__()
        self.repr_prop_dim = obs_prop_dim

    def forward(self, obs_prop):
        return obs_prop


class MultimodalFusion(nn.Module):
    def __init__(
        self,
        repr_rgb_dim,
        repr_prop_dim,
        latent_rgb_dim,
        latent_prop_dim,
    ):
        super().__init__()

        self.trunk_rgb = nn.Sequential(
            nn.Linear(repr_rgb_dim, latent_rgb_dim),
            nn.LayerNorm(latent_rgb_dim),
            # nn.Tanh(),
            utils.SimplexNorm(),
        )

        self.trunk_prop = nn.Sequential(
            nn.Linear(repr_prop_dim, latent_prop_dim),
            nn.LayerNorm(latent_prop_dim),
            # nn.Tanh(),
            utils.SimplexNorm(),
        )

        self.apply(utils.weight_init)

    def forward(self, repr_rgb, repr_prop):
        latent_rgb = self.trunk_rgb(repr_rgb)
        latent_prop = self.trunk_prop(repr_prop)
        repr_multimodal = torch.cat([latent_rgb, latent_prop], dim=-1)

        return repr_multimodal


class Actor(nn.Module):
    def __init__(
        self,
        repr_multimodal_dim,
        hidden_dim,
        action_dim,
    ):
        super().__init__()

        policy_layers = []
        policy_layers += [
            nn.Linear(repr_multimodal_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(4):
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        policy_layers += [
            nn.Linear(hidden_dim, action_dim),
        ]
        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, repr_multimodal, std):
        mu = self.policy(repr_multimodal)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)

        return dist


class Critic(nn.Module):
    def __init__(
        self,
        repr_multimodal_dim,
        hidden_dim,
        action_dim,
    ):
        super().__init__()

        def make_Q():
            q_layers = []
            q_layers += [
                nn.Linear(repr_multimodal_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            for _ in range(4):
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            q_layers += [
                nn.Linear(hidden_dim, 1),
            ]
            return nn.Sequential(*q_layers)

        self.Q1 = make_Q()
        self.Q2 = make_Q()

        self.apply(utils.weight_init)

    def forward(self, repr_multimodal, action):
        repr_multimodal = torch.cat([repr_multimodal, action], dim=-1)
        q1 = self.Q1(repr_multimodal)
        q2 = self.Q2(repr_multimodal)

        return q1, q2


class MFDRNAgent:
    def __init__(
        self,
        obs_shape,
        state_shape,
        action_shape,
        device,
        lr,
        prop_encoder_type,
        latent_rgb_dim,
        latent_prop_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
    ):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.prop_encoder_type = prop_encoder_type

        # models
        obs_prop_dim = state_shape[0]
        self.aug = utils.RandomShiftsAug(pad=4)
        self.image_encoder = ImageEncoder(obs_shape).to(device)
        assert prop_encoder_type in ["MLP", "identity"]
        if prop_encoder_type == "MLP":
            self.prop_encoder = PropEncoder(obs_prop_dim).to(device)
        else: # "identity"
            self.prop_encoder = IdentityEncoder(obs_prop_dim).to(device)
        repr_rgb_dim = self.image_encoder.repr_rgb_dim
        repr_prop_dim = self.prop_encoder.repr_prop_dim

        self.multimodal_fusion_actor = MultimodalFusion(
            repr_rgb_dim,
            repr_prop_dim,
            latent_rgb_dim,
            latent_prop_dim,
        ).to(device)
        self.multimodal_fusion_critic = MultimodalFusion(
            repr_rgb_dim,
            repr_prop_dim,
            latent_rgb_dim,
            latent_prop_dim,
        ).to(device)
        repr_multimodal_dim = latent_rgb_dim + latent_prop_dim

        self.actor = Actor(
            repr_multimodal_dim,
            hidden_dim,
            action_shape[0],
        ).to(device)

        self.critic = Critic(
            repr_multimodal_dim,
            hidden_dim,
            action_shape[0],
        ).to(device)
        self.critic_target = Critic(
            repr_multimodal_dim,
            hidden_dim,
            action_shape[0],
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.image_encoder_opt = torch.optim.Adam(self.image_encoder.parameters(), lr=lr)
        if prop_encoder_type == "MLP":
            self.prop_encoder_opt = torch.optim.Adam(self.prop_encoder.parameters(), lr=lr)
        self.multimodal_fusion_actor_opt = torch.optim.Adam(self.multimodal_fusion_actor.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.multimodal_fusion_critic_opt = torch.optim.Adam(self.multimodal_fusion_critic.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.image_encoder.train(training)
        self.prop_encoder.train(training)
        self.multimodal_fusion_actor.train(training)
        self.actor.train(training)
        self.multimodal_fusion_critic.train(training)
        self.critic.train(training)

    def act(self, obs_rgb, obs_prop, step, eval_mode):
        obs_rgb = torch.as_tensor(obs_rgb, device=self.device).unsqueeze(0)
        repr_rgb = self.image_encoder(obs_rgb)
        obs_prop = torch.as_tensor(
            obs_prop, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        repr_prop = self.prop_encoder(obs_prop)
        repr_multimodal_actor = self.multimodal_fusion_actor(repr_rgb, repr_prop)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(repr_multimodal_actor, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self,
        repr_rgb,
        repr_prop,
        action,
        reward,
        discount,
        next_repr_rgb,
        next_repr_prop,
        step,
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            next_repr_multimodal_actor = self.multimodal_fusion_actor(next_repr_rgb, next_repr_prop)
            dist = self.actor(next_repr_multimodal_actor, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            next_repr_multimodal_critic = self.multimodal_fusion_critic(next_repr_rgb, next_repr_prop)
            target_Q1, target_Q2 = self.critic_target(next_repr_multimodal_critic, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        repr_multimodal_critic = self.multimodal_fusion_critic(repr_rgb, repr_prop)
        Q1, Q2 = self.critic(repr_multimodal_critic, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        self.image_encoder_opt.zero_grad(set_to_none=True)
        if self.prop_encoder_type == "MLP":
            self.prop_encoder_opt.zero_grad(set_to_none=True)
        self.multimodal_fusion_critic_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.multimodal_fusion_critic_opt.step()
        if self.prop_encoder_type == "MLP":
            self.prop_encoder_opt.step()
        self.image_encoder_opt.step()

        return metrics

    def update_actor(
        self,
        repr_rgb,
        repr_prop,
        step,
    ):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        repr_multimodal_actor = self.multimodal_fusion_actor(repr_rgb, repr_prop)
        dist = self.actor(repr_multimodal_actor, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        repr_multimodal_critic = self.multimodal_fusion_critic(repr_rgb, repr_prop)
        Q1, Q2 = self.critic(repr_multimodal_critic, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.multimodal_fusion_actor_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.multimodal_fusion_actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs_rgb):
        aug_obs_rgb = self.aug(obs_rgb)
        return self.image_encoder(aug_obs_rgb)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        (
            obs_rgb,
            obs_prop,
            action,
            reward,
            discount,
            next_obs_rgb,
            next_obs_prop,
        ) = utils.to_torch(batch, self.device)

        # squeeze
        obs_prop = torch.as_tensor(obs_prop, device=self.device, dtype=torch.float32).squeeze()
        next_obs_prop = torch.as_tensor(next_obs_prop, device=self.device, dtype=torch.float32).squeeze()

        # augment and encode
        repr_rgb = self.aug_and_encode(obs_rgb)
        repr_prop = self.prop_encoder(obs_prop)

        with torch.no_grad():
            next_repr_rgb = self.aug_and_encode(next_obs_rgb)
            next_repr_prop = self.prop_encoder(next_obs_prop)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                repr_rgb,
                repr_prop,
                action,
                reward,
                discount,
                next_repr_rgb,
                next_repr_prop,
                step,
            )
        )

        # update actor
        metrics.update(self.update_actor(repr_rgb.detach(), repr_prop.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
