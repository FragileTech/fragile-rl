# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from dataclasses import dataclass
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tyro

from fragile.losses.macro import AbsoluteEnclosureProbe, compose_absolute_structured_state


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
    hidden_dim: int = 255
    """hidden width used by both the actor and critic MLPs"""
    num_layers: int = 2
    """number of hidden layers used by both the actor and critic MLPs"""

    # Enclosure-only auxiliary loss stack
    enclosure_encoder_hidden_dim: int = 13
    """hidden width of the lightweight enclosure symbolizer MLPs"""
    enclosure_latent_dim: int = 3
    """latent dimension used by the enclosure-only structured/texture encoders"""
    enclosure_codes_per_chart: int = 6
    """number of enclosure symbols per chart; a single chart is always used here"""
    enclosure_encoder_weight: float = 0.05
    """weight applied to the enclosure loss on the main optimizer"""
    enclosure_probe_weight: float = 0.05
    """weight applied to the detached enclosure-probe training loss"""
    enclosure_probe_hidden_dim: int = 8
    """hidden width of the detached enclosure probe MLPs"""
    enclosure_dropout: float = 0.0
    """dropout used inside the enclosure probe"""
    enclosure_alpha: float = 1.0
    """gradient-reversal strength applied to texture inputs in the enclosure probe"""
    enclosure_probe_lr: float = 1e-3
    """learning rate for the detached enclosure probe optimizer"""
    log_enclosure_every_update: bool = False
    """if toggled, enclosure diagnostics are emitted to TensorBoard every PPO update"""

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
        obs_space = env.observation_space
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            observation_space=gym.spaces.Box(
                low=np.full(obs_space.shape, -10.0, dtype=obs_space.dtype),
                high=np.full(obs_space.shape, 10.0, dtype=obs_space.dtype),
                dtype=obs_space.dtype,
            ),
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        return gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))

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


def compute_explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute explained variance for value predictions."""
    var_y = np.var(y_true)
    return float(np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)


def _average_metric_dicts(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of flat metric dictionaries."""
    if not metric_dicts:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / float(counts[key]) for key in totals}


def set_module_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Toggle gradient tracking for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad_(requires_grad)


class EnclosureAuxEncoder(nn.Module):
    """Tiny symbolizer used only to feed the enclosure loss."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, codes_per_chart: int):
        super().__init__()
        self.codes_per_chart = int(codes_per_chart)
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.z_n_head = layer_init(nn.Linear(hidden_dim, latent_dim), std=0.01)
        self.z_tex_head = layer_init(nn.Linear(hidden_dim, latent_dim), std=0.01)
        self.code_logits_head = layer_init(nn.Linear(hidden_dim, codes_per_chart), std=0.01)
        self.chart_centers = nn.Parameter(torch.empty(1, latent_dim))
        self.codebook = nn.Parameter(torch.empty(1, codes_per_chart, latent_dim))
        nn.init.normal_(self.chart_centers, std=0.01)
        nn.init.normal_(self.codebook, std=0.01)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        code_idx = self.code_logits_head(features).argmax(dim=-1)
        chart_idx = torch.zeros_like(code_idx)
        return {
            "chart_idx": chart_idx,
            "code_idx": code_idx,
            "z_n_tan": torch.tanh(self.z_n_head(features)),
            "z_tex": torch.tanh(self.z_tex_head(features)),
        }


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

        self.obs_total_codes = int(self.args.enclosure_codes_per_chart)
        self.act_total_codes = int(self.args.enclosure_codes_per_chart)
        self.obs_enclosure_encoder = EnclosureAuxEncoder(
            obs_dim,
            self.args.enclosure_encoder_hidden_dim,
            self.args.enclosure_latent_dim,
            self.obs_total_codes,
        )
        self.act_enclosure_encoder = EnclosureAuxEncoder(
            act_dim,
            self.args.enclosure_encoder_hidden_dim,
            self.args.enclosure_latent_dim,
            self.act_total_codes,
        )
        self.enclosure_probe = AbsoluteEnclosureProbe(
            obs_struct_dim=self.args.enclosure_latent_dim,
            act_struct_dim=self.args.enclosure_latent_dim,
            obs_tex_dim=self.args.enclosure_latent_dim,
            act_tex_dim=self.args.enclosure_latent_dim,
            num_obs_charts=1,
            obs_codes_per_chart=self.obs_total_codes,
            hidden_dim=self.args.enclosure_probe_hidden_dim,
            alpha=self.args.enclosure_alpha,
            dropout=self.args.enclosure_dropout,
        )

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

    def compute_enclosure_regularization(
        self,
        obs_batch: torch.Tensor,
        next_obs_batch: torch.Tensor,
        next_done_batch: torch.Tensor,
        action_batch: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        """Compute only the enclosure loss on valid transitions."""
        zero = torch.zeros((), device=obs_batch.device)
        valid_mask = ~next_done_batch.bool()
        if not valid_mask.any():
            return {
                "loss": zero,
                "probe_loss": zero,
                "scalars": {
                    "enclosure/encoder_total": 0.0,
                    "enclosure/probe_total": 0.0,
                },
            }

        curr_obs = self.obs_enclosure_encoder(obs_batch[valid_mask])
        curr_act = self.act_enclosure_encoder(action_batch[valid_mask])
        with torch.no_grad():
            next_obs = self.obs_enclosure_encoder(next_obs_batch[valid_mask])

        target = next_obs["chart_idx"].long() * self.obs_total_codes + next_obs["code_idx"].long()
        u_obs = compose_absolute_structured_state(
            self.obs_enclosure_encoder.chart_centers,
            self.obs_enclosure_encoder.codebook,
            curr_obs["chart_idx"],
            curr_obs["code_idx"],
            curr_obs["z_n_tan"],
        )
        u_act = compose_absolute_structured_state(
            self.act_enclosure_encoder.chart_centers,
            self.act_enclosure_encoder.codebook,
            curr_act["chart_idx"],
            curr_act["code_idx"],
            curr_act["z_n_tan"],
        )

        # Freeze probe weights on the live encoder-loss pass so autograd only
        # tracks the paths needed for the enclosure encoders.
        set_module_requires_grad(self.enclosure_probe, False)
        logits_live = self.enclosure_probe(u_obs, u_act, curr_obs["z_tex"], curr_act["z_tex"])
        set_module_requires_grad(self.enclosure_probe, True)

        ce_obs = F.cross_entropy(logits_live["obs"], target)
        ce_act = F.cross_entropy(logits_live["act"], target)
        ce_both = F.cross_entropy(logits_live["both"], target)
        enclosure_encoder_loss = (ce_obs + ce_act + ce_both) / 3.0

        enclosure_probe_loss = zero
        enclosure_diag = {
            "loss_encoder": float(enclosure_encoder_loss.detach()),
            "loss_probe": 0.0,
        }
        should_log_full_diag = bool(self.args.log_enclosure_every_update)
        should_train_probe = float(self.args.enclosure_probe_weight) > 0.0
        if should_train_probe or should_log_full_diag:
            logits_det = self.enclosure_probe(
                u_obs.detach(),
                u_act.detach(),
                curr_obs["z_tex"].detach(),
                curr_act["z_tex"].detach(),
            )
            ce_base_det = F.cross_entropy(logits_det["baseline"], target)
            ce_obs_det = F.cross_entropy(logits_det["obs"], target)
            ce_act_det = F.cross_entropy(logits_det["act"], target)
            ce_both_det = F.cross_entropy(logits_det["both"], target)
            enclosure_probe_loss = (ce_base_det + ce_obs_det + ce_act_det + ce_both_det) / 4.0
            enclosure_diag["loss_probe"] = float(enclosure_probe_loss.detach())

            if should_log_full_diag:
                with torch.no_grad():
                    acc_base = (logits_det["baseline"].argmax(dim=-1) == target).float().mean().item()
                    acc_obs = (logits_det["obs"].argmax(dim=-1) == target).float().mean().item()
                    acc_act = (logits_det["act"].argmax(dim=-1) == target).float().mean().item()
                    acc_both = (logits_det["both"].argmax(dim=-1) == target).float().mean().item()

                enclosure_diag.update(
                    {
                        "acc_base": acc_base,
                        "acc_obs": acc_obs,
                        "acc_act": acc_act,
                        "acc_both": acc_both,
                        "defect_acc_obs": acc_obs - acc_base,
                        "defect_acc_act": acc_act - acc_base,
                        "defect_acc_both": acc_both - acc_base,
                        "ce_base": float(ce_base_det.detach()),
                        "ce_obs": float(ce_obs_det.detach()),
                        "ce_act": float(ce_act_det.detach()),
                        "ce_both": float(ce_both_det.detach()),
                        "defect_ce_obs": float((ce_base_det - ce_obs_det).detach()),
                        "defect_ce_act": float((ce_base_det - ce_act_det).detach()),
                        "defect_ce_both": float((ce_base_det - ce_both_det).detach()),
                    }
                )

        return {
            "loss": float(self.args.enclosure_encoder_weight) * enclosure_encoder_loss,
            "probe_loss": float(self.args.enclosure_probe_weight) * enclosure_probe_loss,
            "scalars": {
                "enclosure/encoder_total": enclosure_diag["loss_encoder"],
                "enclosure/probe_total": enclosure_diag["loss_probe"],
                **{f"enclosure/{key}": float(value) for key, value in enclosure_diag.items()},
            },
        }


def print_model_summary(agent: Agent, envs: gym.vector.VectorEnv) -> None:
    """Print a startup summary with counted params excluding the detached probe."""
    actor_params = count_parameters(agent.actor_mean) + agent.actor_logstd.numel()
    critic_params = count_parameters(agent.critic)
    obs_aux_params = count_parameters(agent.obs_enclosure_encoder)
    act_aux_params = count_parameters(agent.act_enclosure_encoder)
    counted_total = actor_params + critic_params + obs_aux_params + act_aux_params
    probe_params = count_parameters(agent.enclosure_probe)
    all_trainable = counted_total + probe_params

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    print("Model summary:")
    print(f"  Env:                 {envs.spec.id if envs.spec is not None else 'unknown'}")
    print(f"  Obs dim:             {obs_dim}")
    print(f"  Action dim:          {act_dim}")
    print(f"  PPO Actor:           {actor_params:>10,} params")
    print(f"  PPO Critic:          {critic_params:>10,} params")
    print(f"  Enclosure Obs Aux:   {obs_aux_params:>10,} params")
    print(f"  Enclosure Act Aux:   {act_aux_params:>10,} params")
    print(f"  COUNTED TOTAL:       {counted_total:>10,} params")
    print(f"  Probe excluded:      {probe_params:>10,} params")
    print(f"  ALL TRAINABLE:       {all_trainable:>10,} params")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    hyperparameter_table = "\n".join(f"|{key}|{value}|" for key, value in vars(args).items())
    writer.add_text(
        "hyperparameters",
        f"|param|value|\n|-|-|\n{hyperparameter_table}",
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
    main_params = [
        param
        for name, param in agent.named_parameters()
        if param.requires_grad and not name.startswith("enclosure_probe.")
    ]
    optimizer = optim.Adam(main_params, lr=args.learning_rate, eps=1e-5)
    probe_optimizer = optim.Adam(
        agent.enclosure_probe.parameters(),
        lr=args.enclosure_probe_lr,
        eps=1e-5,
    )

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, *envs.single_observation_space.shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, *envs.single_action_space.shape)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_obs_buf = torch.zeros((args.num_steps, args.num_envs, *envs.single_observation_space.shape)).to(
        device
    )
    next_dones_buf = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    should_compute_enclosure = bool(
        args.log_enclosure_every_update
        or args.enclosure_encoder_weight > 0
        or args.enclosure_probe_weight > 0
    )

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.eval()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.inference_mode():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            next_obs_buf[step] = next_obs
            next_dones_buf[step] = next_done

            log_vector_episode_stats(writer, global_step, infos)

        # bootstrap value if not done
        with torch.inference_mode():
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
        b_obs = obs.reshape((-1, *envs.single_observation_space.shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *envs.single_action_space.shape))
        b_next_obs = next_obs_buf.reshape((-1, *envs.single_observation_space.shape))
        b_next_dones = next_dones_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        enclosure_metric_dicts: list[dict[str, float]] = []
        probe_loss_value = 0.0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                probe_loss = None
                if should_compute_enclosure:
                    enclosure_aux = agent.compute_enclosure_regularization(
                        b_obs[mb_inds],
                        b_next_obs[mb_inds],
                        b_next_dones[mb_inds],
                        b_actions[mb_inds],
                    )
                    loss = loss + enclosure_aux["loss"]
                    probe_loss = enclosure_aux["probe_loss"]
                    probe_loss_value = float(probe_loss.detach())
                    enclosure_metric_dicts.append(
                        {key: float(value) for key, value in enclosure_aux["scalars"].items()}
                    )

                optimizer.zero_grad()
                probe_optimizer.zero_grad()
                retain_graph = probe_loss is not None and probe_loss.requires_grad and probe_loss_value > 0.0
                loss.backward(retain_graph=retain_graph)
                nn.utils.clip_grad_norm_(main_params, args.max_grad_norm)
                optimizer.step()
                if retain_graph:
                    probe_optimizer.zero_grad()
                    probe_loss.backward()
                    nn.utils.clip_grad_norm_(list(agent.enclosure_probe.parameters()), args.max_grad_norm)
                    probe_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        explained_var = compute_explained_variance(y_pred, y_true)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if should_compute_enclosure and enclosure_metric_dicts:
            averaged_enclosure_metrics = _average_metric_dicts(enclosure_metric_dicts)
            writer.add_scalar(
                "losses/enclosure_encoder",
                averaged_enclosure_metrics.get("enclosure/encoder_total", 0.0),
                global_step,
            )
            writer.add_scalar(
                "losses/enclosure_probe",
                averaged_enclosure_metrics.get("enclosure/probe_total", 0.0),
                global_step,
            )
            if args.log_enclosure_every_update:
                for key, value in averaged_enclosure_metrics.items():
                    writer.add_scalar(key, value, global_step)
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
