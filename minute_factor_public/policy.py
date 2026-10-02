"""Valid reverse Polish expression sampling and an optional masked PPO policy.

The random sampler needs only NumPy. ``MaskedPPO`` requires PyTorch when it is
constructed; importing this module does not require PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from .expression import Expr


_UNARY = ("abs", "log1p_abs")
_BINARY = ("add", "sub", "mul", "div")
_ROLLING = ("mean", "std", "delta")
_WINDOWS = (3, 5, 10)


@dataclass(frozen=True)
class _Action:
    op: str
    name: str | None = None
    window: int | None = None


def _validate_inputs(feature_names: tuple[str, ...], max_tokens: int) -> None:
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if not feature_names or any(not isinstance(name, str) or not name for name in feature_names):
        raise ValueError("feature_names must contain nonempty strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must be unique")


def _actions(feature_names: tuple[str, ...]) -> tuple[_Action, ...]:
    return (
        *(_Action("feature", name=name) for name in feature_names),
        *(_Action(op) for op in _UNARY),
        *(_Action(op) for op in _BINARY),
        *(_Action(op, window=window) for op, window in product(_ROLLING, _WINDOWS)),
        *(_Action("corr", window=window) for window in _WINDOWS),
        _Action("stop"),
    )


def _legal_mask(actions: tuple[_Action, ...], depth: int, used: int, max_tokens: int) -> np.ndarray:
    """Keep only actions that can still end in one complete expression."""
    remaining_after = max_tokens - used - 1
    mask = np.zeros(len(actions), dtype=np.bool_)
    for index, action in enumerate(actions):
        if action.op == "stop":
            mask[index] = depth == 1
            continue
        if action.op == "feature":
            next_depth = depth + 1
        elif action.op in _UNARY or action.op in _ROLLING:
            if depth < 1:
                continue
            next_depth = depth
        else:
            if depth < 2:
                continue
            next_depth = depth - 1
        # Each binary action can remove at most one stack item.
        mask[index] = remaining_after >= next_depth - 1
    return mask


def _apply(stack: list[Expr], action: _Action) -> None:
    if action.op == "feature":
        stack.append(Expr("feature", name=action.name))
    elif action.op in _UNARY or action.op in _ROLLING:
        child = stack.pop()
        stack.append(Expr(action.op, (child,), window=action.window))
    else:
        right = stack.pop()
        left = stack.pop()
        stack.append(Expr(action.op, (left, right), window=action.window))


def sample_expression(
    feature_names: tuple[str, ...], rng: np.random.Generator, max_tokens: int = 10
) -> Expr:
    """Draw a syntactically valid expression with no more than ``max_tokens`` nodes."""
    _validate_inputs(feature_names, max_tokens)
    actions = _actions(feature_names)
    stack: list[Expr] = []
    for used in range(max_tokens):
        legal = np.flatnonzero(_legal_mask(actions, len(stack), used, max_tokens))
        choice = actions[int(rng.choice(legal))]
        if choice.op == "stop":
            break
        _apply(stack, choice)
    assert len(stack) == 1
    return stack[0]


@dataclass(frozen=True)
class _Trajectory:
    actions: tuple[int, ...]
    masks: tuple[tuple[bool, ...], ...]
    old_log_probs: tuple[float, ...]


class MaskedPPO:
    """A small LSTM actor-critic trained from terminal expression rewards.

    Each ``sample`` call returns an expression and a trajectory. Pass trajectories
    back with their scalar rewards to ``update``. The policy runs on CPU so it can
    be used for small demonstrations without a GPU.
    """

    def __init__(self, feature_names: tuple[str, ...], seed: int = 0, max_tokens: int = 10):
        _validate_inputs(feature_names, max_tokens)
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("MaskedPPO requires PyTorch; sample_expression needs only NumPy") from exc

        self.feature_names = feature_names
        self.max_tokens = max_tokens
        self._actions = _actions(feature_names)
        self._torch = torch
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        action_count = len(self._actions)
        begin_token = action_count
        self._begin_token = begin_token

        class ActorCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(action_count + 1, 64)
                self.memory = nn.LSTM(64, 64, batch_first=True)
                self.actor = nn.Linear(64, action_count)
                self.critic = nn.Linear(64, 1)

            def forward(self, tokens: Any) -> tuple[Any, Any]:
                hidden, _ = self.memory(self.embedding(tokens))
                return self.actor(hidden), self.critic(hidden).squeeze(-1)

        # Restore the caller's global RNG state after deterministic initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._model = ActorCritic()
        self._learning_rate = 3e-4

    def sample(self) -> tuple[Expr, object]:
        """Sample a valid expression and return its PPO trajectory."""
        torch = self._torch
        stack: list[Expr] = []
        selected: list[int] = []
        masks: list[tuple[bool, ...]] = []
        old_log_probs: list[float] = []
        used = 0
        self._model.eval()
        while used < self.max_tokens:
            prefix = torch.tensor([[self._begin_token, *selected]], dtype=torch.long)
            mask = _legal_mask(self._actions, len(stack), used, self.max_tokens)
            with torch.no_grad():
                logits, _ = self._model(prefix)
                masked_logits = logits[0, -1].masked_fill(
                    ~torch.as_tensor(mask), torch.finfo(logits.dtype).min
                )
                log_probs = torch.log_softmax(masked_logits, dim=0)
                choice = int(torch.multinomial(log_probs.exp(), 1, generator=self._generator).item())
            selected.append(choice)
            masks.append(tuple(bool(value) for value in mask))
            old_log_probs.append(float(log_probs[choice].item()))
            action = self._actions[choice]
            if action.op == "stop":
                break
            _apply(stack, action)
            used += 1
        assert len(stack) == 1
        return stack[0], _Trajectory(tuple(selected), tuple(masks), tuple(old_log_probs))

    def update(self, experiences: list[tuple[object, float]]) -> None:
        """Apply clipped PPO updates using one terminal reward per trajectory."""
        if not experiences:
            return
        torch = self._torch
        checked: list[tuple[_Trajectory, float]] = []
        for trajectory, reward in experiences:
            if not isinstance(trajectory, _Trajectory):
                raise TypeError("each experience needs a trajectory returned by sample()")
            if not np.isfinite(reward):
                raise ValueError("rewards must be finite")
            checked.append((trajectory, float(reward)))

        self._model.train()
        for _ in range(4):
            policy_terms = []
            value_terms = []
            entropy_terms = []
            for trajectory, reward in checked:
                prefix = torch.tensor(
                    [[self._begin_token, *trajectory.actions[:-1]]], dtype=torch.long
                )
                logits, values = self._model(prefix)
                legal = torch.tensor(trajectory.masks, dtype=torch.bool)
                masked_logits = logits[0].masked_fill(~legal, torch.finfo(logits.dtype).min)
                distribution = torch.distributions.Categorical(logits=masked_logits)
                chosen = torch.tensor(trajectory.actions, dtype=torch.long)
                current_log_probs = distribution.log_prob(chosen)
                old_log_probs = torch.tensor(trajectory.old_log_probs, dtype=torch.float32)
                returns = torch.full_like(values[0], reward)
                advantages = returns - values[0].detach()
                ratio = (current_log_probs - old_log_probs).exp()
                clipped_ratio = ratio.clamp(0.8, 1.2)
                policy_terms.append(-torch.minimum(ratio * advantages, clipped_ratio * advantages).mean())
                value_terms.append((values[0] - returns).square().mean())
                entropy_terms.append(distribution.entropy().mean())
            loss = (
                torch.stack(policy_terms).mean()
                + 0.5 * torch.stack(value_terms).mean()
                - 0.01 * torch.stack(entropy_terms).mean()
            )
            self._model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            with torch.no_grad():
                for parameter in self._model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-self._learning_rate)
