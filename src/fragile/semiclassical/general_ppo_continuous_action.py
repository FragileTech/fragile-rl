# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""
    actor_learning_rate: float | None = None
    """actor optimizer learning rate; defaults to learning_rate when unset"""
    critic_learning_rate: float | None = None
    """critic optimizer learning rate; defaults to learning_rate when unset"""
    actor_update_epochs: int | None = None
    """actor update epochs; defaults to update_epochs when unset"""
    critic_update_epochs: int | None = None
    """critic update epochs; defaults to update_epochs when unset"""
    min_explained_variance_for_actor: float | None = None
    """skip actor updates when the rollout explained variance falls below this threshold"""
    max_clipfrac_for_actor: float | None = None
    """skip actor updates when minibatch clipfrac exceeds this threshold"""
    min_advantage_snr_for_actor: float | None = None
    """skip actor updates when minibatch advantage SNR falls below this threshold"""
    max_state_sensitivity_score_for_actor: float | None = None
    """skip actor updates when minibatch mean state-sensitivity score exceeds this threshold"""
    state_sensitivity_coef: float = 0.0
    """scales how strongly state sensitivity down-weights actor updates; 0 disables it"""
    state_sensitivity_eps: float = 1e-8
    """numerical stabilizer for state-sensitivity RMS calculations"""
    state_sensitivity_max_scale: float = 1.0
    """upper bound applied to the combined state-sensitivity score"""
    state_sensitivity_policy_weight: float = 1.0
    """weight for the policy observation-Fisher proxy in the sensitivity score"""
    state_sensitivity_value_weight: float = 1.0
    """weight for the critic observation-gradient proxy in the sensitivity score"""
    hidden_dim: int = 64
    """hidden width used by both the actor and critic MLPs"""
    num_layers: int = 2
    """number of hidden layers used by both the actor and critic MLPs"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=gym.spaces.Box(
                low=-10.0,
                high=10.0,
                shape=env.observation_space.shape,
                dtype=env.observation_space.dtype,
            ),
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def log_vector_episode_stats(writer: SummaryWriter, global_step: int, infos: dict) -> None:
    """Log episodic returns for both old and new Gymnasium vector-info formats."""
    if "episode" in infos:
        episode = infos["episode"]
        returns = np.atleast_1d(np.asarray(episode["r"]))
        lengths = np.atleast_1d(np.asarray(episode["l"]))
        finished = infos.get("_episode")
        if finished is None:
            finished = infos.get("_r")
        if finished is None:
            finished = np.ones_like(returns, dtype=bool)
        finished = np.atleast_1d(np.asarray(finished, dtype=bool))

        for episodic_return, episodic_length, is_finished in zip(returns, lengths, finished):
            if not is_finished:
                continue
            episodic_return = float(episodic_return)
            episodic_length = float(episodic_length)
            print(f"global_step={global_step}, episodic_return={episodic_return}")
            writer.add_scalar("charts/episodic_return", episodic_return, global_step)
            writer.add_scalar("charts/episodic_length", episodic_length, global_step)
        return

    if "final_info" in infos:
        for info in infos["final_info"]:
            if info and "episode" in info:
                episodic_return = float(info["episode"]["r"])
                episodic_length = float(info["episode"]["l"])
                print(f"global_step={global_step}, episodic_return={episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, *, output_std: float) -> nn.Sequential:
    """Build a tanh MLP with configurable width/depth."""
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(layer_init(nn.Linear(in_dim, hidden_dim)))
        layers.append(nn.Tanh())
        in_dim = hidden_dim
    layers.append(layer_init(nn.Linear(in_dim, output_dim), std=output_std))
    return nn.Sequential(*layers)


def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def compute_state_sensitivity_proxy(
    obs: torch.Tensor,
    newlogprob: torch.Tensor,
    newvalue: torch.Tensor,
    args: Args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate cheap per-sample observation-space sensitivity proxies.

    The proxy is detached before being reused as a trust signal so the main PPO
    backward pass remains first-order in the model parameters.
    """
    policy_obs_grad = torch.autograd.grad(
        newlogprob.sum(),
        obs,
        retain_graph=True,
        create_graph=False,
    )[0]
    value_obs_grad = torch.autograd.grad(
        newvalue.sum(),
        obs,
        retain_graph=True,
        create_graph=False,
    )[0]

    policy_proxy = policy_obs_grad.flatten(start_dim=1).square().mean(dim=1)
    value_proxy = torch.sqrt(value_obs_grad.flatten(start_dim=1).square().mean(dim=1) + args.state_sensitivity_eps)

    sensitivity_score = (
        args.state_sensitivity_policy_weight * policy_proxy
        + args.state_sensitivity_value_weight * value_proxy
    )
    sensitivity_score = sensitivity_score.clamp(max=args.state_sensitivity_max_scale)
    caution_weight = 1.0 / (1.0 + args.state_sensitivity_coef * sensitivity_score)
    return policy_proxy, value_proxy, sensitivity_score, caution_weight


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance for value predictions."""
    var_y = np.var(y_true)
    return float(np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)


def compute_advantage_snr(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Estimate a simple minibatch signal-to-noise ratio for raw advantages."""
    return advantages.mean().abs() / (advantages.std() + eps)


def mean_or_nan(values: list[float]) -> float:
    """Return the mean of a list, or NaN when it is empty."""
    return float(np.mean(values)) if values else float("nan")


def use_two_timescale_tweaks(args: Args) -> bool:
    """Enable split actor/critic optimization only when explicitly requested."""
    return any(
        value is not None
        for value in (
            args.actor_learning_rate,
            args.critic_learning_rate,
            args.actor_update_epochs,
            args.critic_update_epochs,
            args.min_explained_variance_for_actor,
            args.max_clipfrac_for_actor,
            args.min_advantage_snr_for_actor,
            args.max_state_sensitivity_score_for_actor,
        )
    )


class Agent(nn.Module):
    def __init__(self, envs, args: Args | None = None):
        super().__init__()
        self.args = args or Args()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = build_mlp(
            obs_dim,
            self.args.hidden_dim,
            self.args.num_layers,
            1,
            output_std=1.0,
        )
        self.actor_mean = build_mlp(
            obs_dim,
            self.args.hidden_dim,
            self.args.num_layers,
            act_dim,
            output_std=0.01,
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def print_model_summary(agent: Agent, envs: gym.vector.VectorEnv) -> None:
    """Print a compact startup summary with parameter counts."""
    actor_params = count_parameters(agent.actor_mean) + agent.actor_logstd.numel()
    critic_params = count_parameters(agent.critic)
    total_params = actor_params + critic_params

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    print("Model summary:")
    print(f"  Env:        {envs.spec.id if envs.spec is not None else 'unknown'}")
    print(f"  Obs dim:    {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Actor:      {actor_params:>10,} params")
    print(f"  Critic:     {critic_params:>10,} params")
    print(f"  TOTAL:      {total_params:>10,} params")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    use_two_timescale_updates = use_two_timescale_tweaks(args)
    effective_actor_learning_rate = args.learning_rate if args.actor_learning_rate is None else args.actor_learning_rate
    effective_critic_learning_rate = args.learning_rate if args.critic_learning_rate is None else args.critic_learning_rate
    effective_actor_update_epochs = args.update_epochs if args.actor_update_epochs is None else args.actor_update_epochs
    effective_critic_update_epochs = args.update_epochs if args.critic_update_epochs is None else args.critic_update_epochs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args).to(device)
    print_model_summary(agent, envs)
    all_params = list(agent.parameters())
    actor_params = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
    critic_params = list(agent.critic.parameters())
    optimizer = None
    actor_optimizer = None
    critic_optimizer = None
    if use_two_timescale_updates:
        actor_optimizer = optim.Adam(actor_params, lr=effective_actor_learning_rate, eps=1e-5)
        critic_optimizer = optim.Adam(critic_params, lr=effective_critic_learning_rate, eps=1e-5)
    else:
        optimizer = optim.Adam(all_params, lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            if use_two_timescale_updates:
                actor_lrnow = frac * effective_actor_learning_rate
                critic_lrnow = frac * effective_critic_learning_rate
                actor_optimizer.param_groups[0]["lr"] = actor_lrnow
                critic_optimizer.param_groups[0]["lr"] = critic_lrnow
            else:
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            log_vector_episode_stats(writer, global_step, infos)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        rollout_explained_var = compute_explained_variance(y_pred, y_true)

        critic_value_losses = []
        old_approx_kls = []
        approx_kls = []
        clipfracs = []
        advantage_snrs = []
        actor_policy_losses = []
        actor_entropies = []
        state_sensitivity_policy_means = []
        state_sensitivity_value_means = []
        state_sensitivity_score_means = []
        state_sensitivity_caution_means = []
        state_sensitivity_enabled = args.state_sensitivity_coef > 0.0
        actor_minibatches_total = 0
        actor_steps_taken = 0
        actor_gate_kl = 0
        actor_gate_explained_variance = 0
        actor_gate_clipfrac = 0
        actor_gate_advantage_snr = 0
        actor_gate_state_sensitivity = 0
        if use_two_timescale_updates:
            critic_b_inds = np.arange(args.batch_size)
            for epoch in range(effective_critic_update_epochs):
                np.random.shuffle(critic_b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = critic_b_inds[start:end]

                    newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    critic_optimizer.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(critic_params, args.max_grad_norm)
                    critic_optimizer.step()
                    critic_value_losses.append(v_loss.item())

            actor_b_inds = np.arange(args.batch_size)
            stop_actor_updates = False
            for epoch in range(effective_actor_update_epochs):
                np.random.shuffle(actor_b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = actor_b_inds[start:end]

                    mb_obs = b_obs[mb_inds]
                    if state_sensitivity_enabled:
                        mb_obs = mb_obs.detach().requires_grad_(True)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    actor_minibatches_total += 1

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        old_approx_kls.append(old_approx_kl.item())
                        approx_kls.append(approx_kl.item())
                        clipfracs.append(clipfrac)

                    raw_mb_advantages = b_advantages[mb_inds]
                    advantage_snr = compute_advantage_snr(raw_mb_advantages)
                    advantage_snr_value = float(advantage_snr.detach())
                    advantage_snrs.append(advantage_snr_value)
                    mb_advantages = raw_mb_advantages
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    state_sensitivity_score_value = 0.0
                    if state_sensitivity_enabled:
                        policy_proxy, value_proxy, sensitivity_score, caution_weight = compute_state_sensitivity_proxy(
                            mb_obs,
                            newlogprob,
                            newvalue,
                            args,
                        )
                        caution_weight = caution_weight.detach()
                        state_sensitivity_policy_means.append(float(policy_proxy.mean().detach()))
                        state_sensitivity_value_means.append(float(value_proxy.mean().detach()))
                        state_sensitivity_score_value = float(sensitivity_score.mean().detach())
                        state_sensitivity_score_means.append(state_sensitivity_score_value)
                        state_sensitivity_caution_means.append(float(caution_weight.mean()))
                    else:
                        caution_weight = torch.ones_like(mb_advantages)
                        state_sensitivity_policy_means.append(0.0)
                        state_sensitivity_value_means.append(0.0)
                        state_sensitivity_score_means.append(0.0)
                        state_sensitivity_caution_means.append(1.0)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = (caution_weight * torch.max(pg_loss1, pg_loss2)).mean()
                    actor_policy_losses.append(pg_loss.item())
                    entropy_loss = entropy.mean()
                    actor_entropies.append(entropy_loss.item())

                    actor_step_allowed = True
                    if args.target_kl is not None and approx_kl.item() > args.target_kl:
                        actor_step_allowed = False
                        actor_gate_kl += 1
                        stop_actor_updates = True
                    if (
                        args.min_explained_variance_for_actor is not None
                        and rollout_explained_var < args.min_explained_variance_for_actor
                    ):
                        actor_step_allowed = False
                        actor_gate_explained_variance += 1
                    if args.max_clipfrac_for_actor is not None and clipfrac > args.max_clipfrac_for_actor:
                        actor_step_allowed = False
                        actor_gate_clipfrac += 1
                    if (
                        args.min_advantage_snr_for_actor is not None
                        and advantage_snr_value < args.min_advantage_snr_for_actor
                    ):
                        actor_step_allowed = False
                        actor_gate_advantage_snr += 1
                    if (
                        args.max_state_sensitivity_score_for_actor is not None
                        and state_sensitivity_score_value > args.max_state_sensitivity_score_for_actor
                    ):
                        actor_step_allowed = False
                        actor_gate_state_sensitivity += 1

                    if actor_step_allowed:
                        actor_loss = pg_loss - args.ent_coef * entropy_loss
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor_params, args.max_grad_norm)
                        actor_optimizer.step()
                        actor_steps_taken += 1

                    if stop_actor_updates:
                        break

                if stop_actor_updates:
                    break
        else:
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    mb_obs = b_obs[mb_inds]
                    if state_sensitivity_enabled:
                        mb_obs = mb_obs.detach().requires_grad_(True)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    actor_minibatches_total += 1

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        old_approx_kls.append(old_approx_kl.item())
                        approx_kls.append(approx_kl.item())
                        clipfracs.append(clipfrac)

                    raw_mb_advantages = b_advantages[mb_inds]
                    advantage_snr_value = float(compute_advantage_snr(raw_mb_advantages).detach())
                    advantage_snrs.append(advantage_snr_value)
                    mb_advantages = raw_mb_advantages
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    if state_sensitivity_enabled:
                        policy_proxy, value_proxy, sensitivity_score, caution_weight = compute_state_sensitivity_proxy(
                            mb_obs,
                            newlogprob,
                            newvalue,
                            args,
                        )
                        caution_weight = caution_weight.detach()
                        state_sensitivity_policy_means.append(float(policy_proxy.mean().detach()))
                        state_sensitivity_value_means.append(float(value_proxy.mean().detach()))
                        state_sensitivity_score_means.append(float(sensitivity_score.mean().detach()))
                        state_sensitivity_caution_means.append(float(caution_weight.mean()))
                    else:
                        caution_weight = torch.ones_like(mb_advantages)
                        state_sensitivity_policy_means.append(0.0)
                        state_sensitivity_value_means.append(0.0)
                        state_sensitivity_score_means.append(0.0)
                        state_sensitivity_caution_means.append(1.0)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = (caution_weight * torch.max(pg_loss1, pg_loss2)).mean()
                    actor_policy_losses.append(pg_loss.item())

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    critic_value_losses.append(v_loss.item())

                    entropy_loss = entropy.mean()
                    actor_entropies.append(entropy_loss.item())
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(all_params, args.max_grad_norm)
                    optimizer.step()
                    actor_steps_taken += 1

                if args.target_kl is not None and approx_kl.item() > args.target_kl:
                    actor_gate_kl += 1
                    break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if use_two_timescale_updates:
            learning_rate = actor_optimizer.param_groups[0]["lr"]
            actor_learning_rate = actor_optimizer.param_groups[0]["lr"]
            critic_learning_rate = critic_optimizer.param_groups[0]["lr"]
        else:
            learning_rate = optimizer.param_groups[0]["lr"]
            actor_learning_rate = learning_rate
            critic_learning_rate = learning_rate
        writer.add_scalar("charts/learning_rate", learning_rate, global_step)
        writer.add_scalar("charts/actor_learning_rate", actor_learning_rate, global_step)
        writer.add_scalar("charts/critic_learning_rate", critic_learning_rate, global_step)
        writer.add_scalar("losses/value_loss", mean_or_nan(critic_value_losses), global_step)
        writer.add_scalar("losses/policy_loss", mean_or_nan(actor_policy_losses), global_step)
        writer.add_scalar("losses/entropy", mean_or_nan(actor_entropies), global_step)
        writer.add_scalar("losses/old_approx_kl", mean_or_nan(old_approx_kls), global_step)
        writer.add_scalar("losses/approx_kl", mean_or_nan(approx_kls), global_step)
        writer.add_scalar("losses/clipfrac", mean_or_nan(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", rollout_explained_var, global_step)
        writer.add_scalar("losses/explained_variance_pre_update", rollout_explained_var, global_step)
        writer.add_scalar("losses/advantage_snr", mean_or_nan(advantage_snrs), global_step)
        writer.add_scalar("losses/state_sensitivity_policy", mean_or_nan(state_sensitivity_policy_means), global_step)
        writer.add_scalar("losses/state_sensitivity_value", mean_or_nan(state_sensitivity_value_means), global_step)
        writer.add_scalar("losses/state_sensitivity_score", mean_or_nan(state_sensitivity_score_means), global_step)
        writer.add_scalar("losses/state_sensitivity_caution", mean_or_nan(state_sensitivity_caution_means), global_step)
        actor_gate_denominator = max(actor_minibatches_total, 1)
        writer.add_scalar("losses/actor_step_fraction", actor_steps_taken / actor_gate_denominator, global_step)
        writer.add_scalar("losses/actor_gate_kl", actor_gate_kl / actor_gate_denominator, global_step)
        writer.add_scalar(
            "losses/actor_gate_explained_variance",
            actor_gate_explained_variance / actor_gate_denominator,
            global_step,
        )
        writer.add_scalar("losses/actor_gate_clipfrac", actor_gate_clipfrac / actor_gate_denominator, global_step)
        writer.add_scalar(
            "losses/actor_gate_advantage_snr",
            actor_gate_advantage_snr / actor_gate_denominator,
            global_step,
        )
        writer.add_scalar(
            "losses/actor_gate_state_sensitivity",
            actor_gate_state_sensitivity / actor_gate_denominator,
            global_step,
        )
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=lambda eval_envs: Agent(eval_envs, args),
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
