"""Smoke tests for the standalone macro RL runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf

from fragile.__main__ import run
from fragile.checkpoints import load_checkpoint


train_macro_rl_module = importlib.import_module("fragile.commands.train_macro_rl")
CONFIG_PATH = train_macro_rl_module.CONFIG_PATH


class _FakeActionSpec:
    def __init__(self) -> None:
        self.shape = (2,)
        self.minimum = np.array([-1.0, -1.0], dtype=np.float32)
        self.maximum = np.array([1.0, 1.0], dtype=np.float32)


class _FakeTimeStep:
    def __init__(self, obs: np.ndarray, *, reward: float = 0.0, done: bool = False) -> None:
        self.observation = {"state": obs.astype(np.float32, copy=False)}
        self.reward = float(reward)
        self._done = bool(done)

    def last(self) -> bool:
        return self._done


class _FakeDMEnv:
    def __init__(self, *, max_steps: int = 4) -> None:
        self._max_steps = int(max_steps)
        self._spec = _FakeActionSpec()
        self._t = 0
        self._state = np.zeros(3, dtype=np.float32)

    def action_spec(self) -> _FakeActionSpec:
        return self._spec

    def reset(self) -> _FakeTimeStep:
        self._t = 0
        self._state = np.zeros(3, dtype=np.float32)
        return _FakeTimeStep(self._state, reward=0.0, done=False)

    def step(self, action: np.ndarray) -> _FakeTimeStep:
        action = np.asarray(action, dtype=np.float32)
        self._t += 1
        self._state[:2] = np.clip(self._state[:2] + 0.25 * action, -2.0, 2.0)
        self._state[2] = float(self._t) / float(self._max_steps)
        reward = 1.0 - float(np.linalg.norm(self._state[:2] - 0.5))
        done = self._t >= self._max_steps
        return _FakeTimeStep(self._state.copy(), reward=reward, done=done)


def _make_runner(
    tmp_path: Path,
    **overrides,
) -> train_macro_rl_module.MacroRLRunner:
    cfg = OmegaConf.load(CONFIG_PATH)
    test_overrides = OmegaConf.create({
        "output_dir": str(tmp_path / "macro-rl"),
        "epochs": 1,
        "seed_episodes": 2,
        "collect_episodes_per_epoch": 1,
        "eval_episodes": 1,
        "num_collect_envs": 2,
        "num_eval_envs": 2,
        "updates_per_epoch": 1,
        "batch_size": 2,
        "replay_capacity": 64,
        "max_episode_steps": 4,
        "device": "cpu",
        "save_every": 1,
        "log_every": 1,
        "eval_every": 1,
        "sigma_motor": 0.0,
        "agent": {
            "enclosure_hidden_dim": 16,
            "markov_hidden_dim": 16,
            "obs_encoder": {
                "hidden_dim": 24,
                "latent_dim": 4,
                "num_charts": 2,
                "codes_per_chart": 2,
                "chart_ot_iters": 4,
            },
            "act_encoder": {
                "hidden_dim": 24,
                "latent_dim": 4,
                "num_charts": 2,
                "codes_per_chart": 2,
                "chart_ot_iters": 4,
            },
        },
        "trainer": {
            "weight_enclosure_encoder": 0.0,
            "weight_enclosure_probe": 0.0,
            "weight_markov_shape": 0.0,
        },
    })
    merged = OmegaConf.merge(cfg, test_overrides, OmegaConf.create(overrides))
    return instantiate(merged)


def test_macro_rl_config_loads_and_instantiates() -> None:
    cfg = OmegaConf.load(CONFIG_PATH)
    runner = instantiate(cfg)
    assert isinstance(runner, train_macro_rl_module.MacroRLRunner)
    assert runner.domain == "cartpole"
    assert runner.task == "balance"
    assert runner.output_dir == "outputs/rl/macro-cartpole-balance"
    assert runner.agent.obs_encoder.num_charts == 8
    assert runner.agent.act_encoder.input_affine_enabled is True
    assert runner.gamma == 0.99
    assert runner.trainer.weight_enclosure_encoder == 1.0
    assert runner.trainer.weight_enclosure_probe == 1.0
    assert runner.trainer.weight_markov_transition == 1.0
    assert runner.trainer.weight_markov_shape == 1.0


def test_macro_rl_runner_smoke_writes_checkpoints_and_logs(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        train_macro_rl_module,
        "_make_env",
        lambda _domain, _task: _FakeDMEnv(max_steps=4),
    )

    runner = _make_runner(tmp_path)
    runner.run()

    output = capsys.readouterr().out
    output_dir = Path(runner.output_dir)
    assert (output_dir / "macro_rl_epoch_00000.pt").exists()
    assert (output_dir / "macro_rl_final.pt").exists()
    assert (output_dir / "run_metadata.json").exists()
    assert (output_dir / ".hydra" / "config.yaml").exists()
    assert (output_dir / ".hydra" / "hydra.yaml").exists()
    assert (output_dir / ".hydra" / "overrides.yaml").exists()
    assert "Train metrics:" in output
    assert "Eval metrics:" in output
    assert "collect:" in output
    assert "q:" in output
    assert "proto:" in output
    assert "train obs symbol dist/chart:" in output
    assert "train act symbol dist/chart:" in output
    assert "eval obs symbol dist/chart:" in output
    assert "eval act symbol dist/chart:" in output
    ckpt = load_checkpoint(str(output_dir / "macro_rl_final.pt"))
    assert "q_state" in ckpt
    assert "replay_buffer" in ckpt
    assert int(ckpt["epoch"]) == 0
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["dimensions"] == {"obs_dim": 3, "act_dim": 2}
    assert metadata["resolved_config"]["epochs"] == 1
    saved_cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    assert int(saved_cfg.epochs) == 1


def test_macro_rl_cli_registers_command() -> None:
    assert "macro-rl" in run.commands
