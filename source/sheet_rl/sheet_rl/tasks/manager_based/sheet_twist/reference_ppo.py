"""PPO fine-tuning with stochastic critic preparation and a frozen policy reference.

Collection and GAE use RSL-RL's PPO implementation in both phases. Only optimization
changes: warm-up fits the critic, and fine-tuning adds KL(reference || actor).
This implementation deliberately supports feedforward, single-GPU PPO only.
"""

from copy import deepcopy

import torch
from rsl_rl.algorithms import PPO


def explained_variance(prediction, target):
    target_var = target.float().var(unbiased=False)
    return 1.0 - (target - prediction).float().var(unbiased=False) / target_var.clamp_min(1e-8)


class ReferencePPO(PPO):
    def __init__(self, *args, critic_only=False, critic_learning_rate=3e-4,
                 reference_coef=1.0, max_update_kl=0.01, std_learning_rate=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.actor.is_recurrent or self.critic.is_recurrent or self.is_multi_gpu or self.rnd or self.symmetry:
            raise ValueError("ReferencePPO supports feedforward, single-GPU PPO without RND/symmetry.")
        if self.schedule != "fixed":
            raise ValueError("ReferencePPO uses fixed, separate actor and critic learning rates.")
        if self.actor.distribution is None:
            raise ValueError("ReferencePPO needs a stochastic actor.")
        if self.actor.obs_normalization:
            raise ValueError("ReferencePPO expects the student's fixed, unnormalized observations.")
        self.critic_only = critic_only
        self.critic_learning_rate = critic_learning_rate
        self.reference_coef = reference_coef
        self.max_update_kl = max_update_kl
        self.actor.requires_grad_(not critic_only)
        # Identical exploration in critic preparation and RL. Loading weights keeps
        # these requires_grad flags, unlike changing init_std in a resumed config.
        self.actor.distribution.requires_grad_(False)
        self.reference_actor = deepcopy(self.actor).eval().requires_grad_(False)
        # A learned state-independent std (FastSACActorModel ``init_std``) gets its own rate: at the actor's ~1e-5
        # Adam moves a log-std by at most ~1e-5 per step, i.e. it would stay put.
        std = getattr(self.actor, "learned_log_std", None)
        self.std_learning_rate = std_learning_rate if std_learning_rate is not None else self.learning_rate
        groups = [
            {"params": [p for p in self.actor.parameters() if p is not std], "lr": self.learning_rate, "name": "actor"},
            {"params": self.critic.parameters(), "lr": critic_learning_rate, "name": "critic"},
        ]
        if std is not None:
            groups.append({"params": [std], "lr": self.std_learning_rate, "name": "actor_std"})
        self.optimizer = torch.optim.Adam(groups)
        self.updates = 0
        self.last_metrics = {}

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        # Align the potential's discount even when gamma is overridden on the CLI.
        reward = env.unwrapped.reward_manager.get_term_cfg("grasp_progress")
        reward.params["gamma"] = cfg["algorithm"]["gamma"]
        env.unwrapped.cfg.rewards.grasp_progress.params["gamma"] = cfg["algorithm"]["gamma"]
        return PPO.construct_algorithm(obs, env, cfg, device)

    def _warmup_update(self):
        st = self.storage
        # Hold out complete environment trajectories, not correlated adjacent rows.
        order = torch.randperm(st.num_envs, device=self.device)
        n_valid = max(1, st.num_envs // 5)
        if n_valid == st.num_envs:
            raise ValueError("Critic warm-up needs at least two environments.")
        valid, train = order[:n_valid], order[n_valid:]
        train_obs = st.observations[:, train].flatten(0, 1)
        target = st.returns[:, train].flatten(0, 1)
        valid_obs = st.observations[:, valid].flatten(0, 1)
        valid_target = st.returns[:, valid].flatten(0, 1)
        batch_size = max(1, len(target) // self.num_mini_batches)
        losses = []
        for _ in range(self.num_learning_epochs):
            for rows in torch.randperm(len(target), device=self.device).split(batch_size):
                prediction = self.critic(train_obs[rows])
                loss = (prediction - target[rows]).square().mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(loss.detach())
        with torch.no_grad():
            prediction = self.critic(valid_obs)
            metrics = {
                "value": torch.stack(losses).mean().item(),
                "validation_value": (prediction - valid_target).square().mean().item(),
                "validation_explained_variance": explained_variance(prediction, valid_target).item(),
                "explained_variance_before": explained_variance(st.values, st.returns).item(),
                "critic_learning_rate": self.critic_learning_rate,
            }
        return metrics

    def _ppo_update(self):
        totals = {key: 0.0 for key in ("value", "surrogate", "entropy", "reference_kl", "update_kl", "clip_fraction")}
        updates = 0
        stopped = False
        before = explained_variance(self.storage.values, self.storage.returns).item()
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self.actor(batch.observations, stochastic_output=True)
            params = self.actor.output_distribution_params
            log_prob = self.actor.get_output_log_prob(batch.actions)
            entropy = self.actor.output_entropy.mean()
            with torch.no_grad():
                old_kl = self.actor.get_kl_divergence(batch.old_distribution_params, params).mean()
                self.reference_actor.distribution.update(self.reference_actor(batch.observations))
                reference_params = self.reference_actor.output_distribution_params
            # Stop updating when this rollout's policy has already moved too far.
            if old_kl.item() > self.max_update_kl:
                stopped = True
                break
            reference_kl = self.actor.get_kl_divergence(reference_params, params).mean()
            advantages = batch.advantages.squeeze(-1)
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = (log_prob - batch.old_actions_log_prob.squeeze(-1)).exp()
            surrogate = torch.maximum(-advantages * ratio,
                -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param)).mean()
            values = self.critic(batch.observations)
            value_loss = (values - batch.returns).square().mean()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum((values - batch.returns).square(),
                                           (clipped - batch.returns).square()).mean()
            loss = (surrogate + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy + self.reference_coef * reference_kl)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            for key, value in zip(totals, (value_loss, surrogate, entropy, reference_kl, old_kl,
                    ((ratio - 1).abs() > self.clip_param).float().mean())):
                totals[key] += value.detach().item()
            updates += 1
        metrics = {key: value / max(1, updates) for key, value in totals.items()}
        metrics.update(explained_variance_before=before, kl_stopped=float(stopped),
                       optimizer_steps=updates, critic_learning_rate=self.critic_learning_rate)
        return metrics

    def update(self):
        metrics = self._warmup_update() if self.critic_only else self._ppo_update()
        self.storage.clear()
        self.updates += 1
        self.last_metrics = metrics
        return metrics

    def save(self):
        saved = super().save()
        saved["reference_actor_state_dict"] = self.reference_actor.state_dict()
        saved["sheet_finetune"] = dict(critic_only=self.critic_only, gamma=self.gamma, lam=self.lam,
                                      updates=self.updates, metrics=self.last_metrics)
        return saved

    def load(self, loaded_dict, load_cfg=None, strict=True):
        requested = dict(load_cfg) if load_cfg is not None else dict(actor=True, critic=True, optimizer=True, iteration=True)
        previous = loaded_dict.get("sheet_finetune", {})
        same_phase = previous.get("critic_only") == self.critic_only
        if previous and not self.critic_only and requested.get("critic"):
            if previous["gamma"] != self.gamma or previous["lam"] != self.lam:
                raise ValueError("Discount/GAE changed after critic preparation; warm the critic again.")
        if not same_phase:
            # DAgger/old PPO import, or critic preparation -> RL: fresh optimizer and iteration counter.
            requested.update(optimizer=False, iteration=False)
        result = super().load(loaded_dict, requested, strict)
        # Configured rates win even when resuming an optimizer from this algorithm.
        for group, rate in zip(self.optimizer.param_groups,
                               (self.learning_rate, self.critic_learning_rate, self.std_learning_rate)):
            group["lr"] = rate
        if requested.get("actor"):
            reference = loaded_dict.get("reference_actor_state_dict")
            if reference is None or (self.critic_only and not same_phase):
                reference = loaded_dict["actor_state_dict"]
            self.reference_actor.load_state_dict(reference, strict=strict)
        self.updates = previous.get("updates", 0) if result else 0
        print(f"[ReferencePPO] critic_only={self.critic_only}, actor_lr={self.learning_rate:g}, "
              f"critic_lr={self.critic_learning_rate:g}, fixed_std="
              f"{self.actor.distribution.std_param.detach().cpu().tolist()}")
        return result
