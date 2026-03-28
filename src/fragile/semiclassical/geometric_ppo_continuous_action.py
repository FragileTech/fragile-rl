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

from fragile.layers import BeliefGeometryEncoder, SpectralLinear, TopoEncoder, TopologicalDecoder
from fragile.layers.gauge import poincare_weighted_mean
from fragile.losses.markov_model import (
    compose_absolute_macro_dictionary,
    soft_macro_state_distribution,
)


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
    wandb_entity: str = None
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
    target_kl: float = None
    """the target KL divergence threshold"""

    # Geometry architecture
    obs_hidden_dim: int = 256
    """hidden width of the observation topoencoder stack"""
    obs_latent_dim: int = 16
    """latent poincare-ball dimension of the observation topoencoder"""
    obs_num_charts: int = 8
    """number of observation atlas charts"""
    obs_codes_per_chart: int = 32
    """number of observation codes per chart"""
    act_hidden_dim: int = 256
    """hidden width of the action topoencoder stack"""
    act_latent_dim: int = 16
    """latent poincare-ball dimension of the action topoencoder"""
    act_num_charts: int = 8
    """number of action atlas charts"""
    act_codes_per_chart: int = 32
    """number of action codes per chart"""
    geometry_hidden_dim: int = 128
    """hidden width used by the geometry-aware policy/value heads"""
    routing_tau: float = 1.0
    """routing temperature used by the observation encoder and action decoders"""
    actor_std_min: float = 1e-3
    """minimum standard deviation added after the std decoder"""

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


def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _state_probs_to_chart_probs(
    state_probs: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> torch.Tensor:
    """Marginalize flattened chart/code probabilities down to chart probabilities."""
    return state_probs.reshape(*state_probs.shape[:-1], num_charts, codes_per_chart).sum(dim=-1)


def _topoencoder_kwargs(
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_charts: int,
    codes_per_chart: int,
    input_affine_enabled: bool,
) -> dict[str, int | float | bool]:
    """Mirror the geometry command's topoencoder architecture defaults."""
    return {
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "num_charts": int(num_charts),
        "codes_per_chart": int(codes_per_chart),
        "covariant_attn_tau_min": 1e-2,
        "covariant_attn_denom_min": 1e-3,
        "covariant_attn_transport_eps": 1e-3,
        "soft_equiv_metric": True,
        "soft_equiv_bundle_size": None,
        "soft_equiv_hidden_dim": 64,
        "soft_equiv_use_spectral_norm": True,
        "soft_equiv_zero_self_mixing": False,
        "soft_equiv_soft_assign": True,
        "soft_equiv_temperature": 1.0,
        "film_conditioning": True,
        "commitment_beta": 0.25,
        "codebook_loss_weight": 1.0,
        "input_affine_enabled": bool(input_affine_enabled),
        "input_affine_learnable": False,
        "input_affine_min_scale": 1e-3,
    }


class Agent(nn.Module):
    def __init__(self, envs, args: Args | None = None):
        super().__init__()
        self.args = args or Args()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        geo_hidden_dim = int(self.args.geometry_hidden_dim)

        self.obs_encoder = TopoEncoder(
            **_topoencoder_kwargs(
                input_dim=obs_dim,
                hidden_dim=self.args.obs_hidden_dim,
                latent_dim=self.args.obs_latent_dim,
                num_charts=self.args.obs_num_charts,
                codes_per_chart=self.args.obs_codes_per_chart,
                input_affine_enabled=False,
            ),
        )
        self.act_encoder = TopoEncoder(
            **_topoencoder_kwargs(
                input_dim=act_dim,
                hidden_dim=self.args.act_hidden_dim,
                latent_dim=self.args.act_latent_dim,
                num_charts=self.args.act_num_charts,
                codes_per_chart=self.args.act_codes_per_chart,
                input_affine_enabled=True,
            ),
        )

        self.obs_belief_encoder = BeliefGeometryEncoder(self.args.obs_latent_dim, geo_hidden_dim)
        self.obs_expected_tangent_proj = SpectralLinear(self.args.obs_latent_dim, geo_hidden_dim)
        self.obs_context = nn.Sequential(
            SpectralLinear(2 * geo_hidden_dim, geo_hidden_dim),
            nn.GELU(),
            SpectralLinear(geo_hidden_dim, geo_hidden_dim),
            nn.GELU(),
        )
        self.action_key_proj = SpectralLinear(self.args.act_latent_dim, geo_hidden_dim, bias=False)
        self.action_logit_bias = nn.Parameter(
            torch.zeros(self.args.act_num_charts * self.args.act_codes_per_chart),
        )
        self.value_head = nn.Sequential(
            SpectralLinear(geo_hidden_dim, geo_hidden_dim),
            nn.GELU(),
            SpectralLinear(geo_hidden_dim, 1),
        )
        self.actor_std_decoder = TopologicalDecoder(
            latent_dim=self.args.act_latent_dim,
            hidden_dim=self.args.act_hidden_dim,
            num_charts=self.args.act_num_charts,
            output_dim=act_dim,
            covariant_attn_tau_min=1e-2,
            covariant_attn_denom_min=1e-3,
            covariant_attn_transport_eps=1e-3,
            film_conditioning=True,
        )

    def _obs_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode observations with the observation topoencoder only."""
        x_norm = self.obs_encoder.normalize_input(x)
        (
            _chart_idx,
            _code_idx,
            _z_n,
            _z_tex,
            _router_weights,
            z_geo,
            _vq_loss,
            _indices_stack,
            _z_n_all,
            _c_bar,
            _v_local,
            _z_q_blended,
        ) = self.obs_encoder.encoder(
            x_norm,
            routing_tau=float(self.args.routing_tau),
        )
        obs_state = soft_macro_state_distribution(
            z_geo,
            self.obs_encoder.encoder.chart_centers,
            self.obs_encoder.encoder.codebook,
            chart_tau=float(self.args.routing_tau),
            code_tau=float(self.args.routing_tau),
        )
        belief = self.obs_belief_encoder(
            obs_state["state_probs"],
            obs_state["state_tangent_points"],
        )
        obs_context = self.obs_context(
            torch.cat(
                [
                    belief["summary"],
                    self.obs_expected_tangent_proj(belief["expected_tangent"]),
                ],
                dim=-1,
            ),
        )
        return {
            "z_geo": z_geo,
            "state": obs_state,
            "belief": belief,
            "context": obs_context,
        }

    def _action_symbol_geometry(self) -> dict[str, torch.Tensor]:
        """Return the live action symbol dictionary induced by the action atlas."""
        return compose_absolute_macro_dictionary(
            self.act_encoder.encoder.chart_centers,
            self.act_encoder.encoder.codebook,
        )

    def _policy_distribution(self, x: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Build the geometry-backed Gaussian policy from the current observations."""
        obs_features = self._obs_features(x)
        action_geometry = self._action_symbol_geometry()
        action_keys = self.action_key_proj(
            action_geometry["state_tangent_points"].to(
                device=x.device,
                dtype=x.dtype,
            ),
        )
        action_logits = (
            torch.einsum("bh,sh->bs", obs_features["context"], action_keys)
            + self.action_logit_bias
        )
        action_probs = torch.softmax(action_logits, dim=-1)
        action_chart_probs = _state_probs_to_chart_probs(
            action_probs,
            self.args.act_num_charts,
            self.args.act_codes_per_chart,
        )
        action_latent = poincare_weighted_mean(
            action_geometry["state_points"].to(device=x.device, dtype=x.dtype),
            action_probs,
        )
        action_mean, _, _ = self.act_encoder.decode(
            action_latent,
            router_weights=action_chart_probs,
            routing_tau=float(self.args.routing_tau),
        )
        action_std_model, _, _ = self.actor_std_decoder(
            action_latent,
            router_weights=action_chart_probs,
            routing_tau=float(self.args.routing_tau),
        )
        action_scale = self.act_encoder.io_affine.scale().to(
            device=action_std_model.device,
            dtype=action_std_model.dtype,
        )
        action_std = torch.nn.functional.softplus(action_std_model) * action_scale
        action_std = action_std + float(self.args.actor_std_min)
        return Normal(action_mean, action_std), obs_features["context"]

    def get_value(self, x):
        obs_features = self._obs_features(x)
        return self.value_head(obs_features["context"])

    def get_action_and_value(self, x, action=None):
        probs, obs_context = self._policy_distribution(x)
        if action is None:
            action = probs.sample()
        value = self.value_head(obs_context)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value


def print_model_summary(agent: Agent, envs: gym.vector.VectorEnv) -> None:
    """Print a compact startup summary with parameter counts."""
    obs_stack = count_parameters(agent.obs_encoder)
    act_stack = count_parameters(agent.act_encoder)
    actor_heads = (
        count_parameters(agent.obs_belief_encoder)
        + count_parameters(agent.obs_expected_tangent_proj)
        + count_parameters(agent.obs_context)
        + count_parameters(agent.action_key_proj)
        + agent.action_logit_bias.numel()
        + count_parameters(agent.actor_std_decoder)
    )
    critic_head = count_parameters(agent.value_head)
    total_params = obs_stack + act_stack + actor_heads + critic_head

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))

    print("Model summary:")
    print(f"  Env:        {envs.spec.id if envs.spec is not None else 'unknown'}")
    print(f"  Obs dim:    {obs_dim}")
    print(f"  Action dim: {act_dim}")
    print(f"  Obs stack:  {obs_stack:>10,} params")
    print(f"  Act stack:  {act_stack:>10,} params")
    print(f"  Actor:      {actor_heads:>10,} params")
    print(f"  Critic:     {critic_head:>10,} params")
    print(f"  TOTAL:      {total_params:>10,} params")


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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        agent.eval()
        for step in range(0, args.num_steps):
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
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
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
