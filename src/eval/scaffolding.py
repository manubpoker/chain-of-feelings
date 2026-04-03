"""IGT (Iowa Gambling Task) scaffolding environment.

Plumbing validation only — NOT the real evaluation.
Tests that the affect channel physically works: state updates, FiLM modulates, gradients flow.

The classic IGT: 4 decks with different payoff structures.
Decks A/B: high immediate reward, devastating infrequent losses (net negative EV).
Decks C/D: modest reward, small losses (net positive EV).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class DeckConfig:
    """Payoff structure for a single deck."""
    name: str
    reward: float          # fixed reward per draw
    loss_prob: float       # probability of loss on any draw
    loss_amount: float     # loss magnitude when it hits


# Classic IGT payoff structure (Bechara et al. 1994)
DEFAULT_DECKS = {
    "A": DeckConfig("A", reward=100, loss_prob=0.5, loss_amount=250),   # EV = -25/draw
    "B": DeckConfig("B", reward=100, loss_prob=0.1, loss_amount=1250),  # EV = -25/draw
    "C": DeckConfig("C", reward=50,  loss_prob=0.5, loss_amount=50),    # EV = +25/draw
    "D": DeckConfig("D", reward=50,  loss_prob=0.1, loss_amount=250),   # EV = +25/draw
}


@dataclass
class Trial:
    """Record of a single IGT trial."""
    round_num: int
    deck_chosen: str
    reward: float
    loss: float
    net: float
    cumulative: float


@dataclass
class IGTEnvironment:
    """Iowa Gambling Task environment with text-formatted output."""

    decks: dict[str, DeckConfig] = field(default_factory=lambda: dict(DEFAULT_DECKS))
    max_trials: int = 100
    history: list[Trial] = field(default_factory=list, init=False)
    cumulative: float = field(default=2000.0, init=False)  # standard IGT starts at $2000
    rng: random.Random = field(default_factory=random.Random, init=False)

    def seed(self, s: int) -> None:
        self.rng = random.Random(s)

    def reset(self) -> str:
        """Reset environment, return initial observation as text."""
        self.history = []
        self.cumulative = 2000.0
        return self._format_observation(initial=True)

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        """Take an action (deck name), return (observation, reward, done, info)."""
        action = action.upper()
        if action not in self.decks:
            raise ValueError(f"Invalid deck: {action}. Choose from {list(self.decks.keys())}")

        deck = self.decks[action]
        reward = deck.reward
        loss = deck.loss_amount if self.rng.random() < deck.loss_prob else 0.0
        net = reward - loss
        self.cumulative += net

        trial = Trial(
            round_num=len(self.history) + 1,
            deck_chosen=action,
            reward=reward,
            loss=loss,
            net=net,
            cumulative=self.cumulative,
        )
        self.history.append(trial)

        done = len(self.history) >= self.max_trials
        obs = self._format_observation()
        info = {"trial": trial, "deck_counts": self._deck_counts()}

        return obs, net, done, info

    def _format_observation(self, initial: bool = False) -> str:
        """Format current state as natural language for the LM."""
        if initial:
            return (
                "You are playing a card game with 4 decks: A, B, C, D.\n"
                "Each turn you choose a deck. You will receive a reward and may receive a loss.\n"
                "Your goal is to maximise your total money.\n"
                f"Starting balance: ${self.cumulative:.0f}\n\n"
                "Choose a deck: A, B, C, or D."
            )

        t = self.history[-1]
        lines = [
            f"Round {t.round_num}. You chose Deck {t.deck_chosen}.",
            f"Reward: +${t.reward:.0f}.",
        ]
        if t.loss > 0:
            lines.append(f"Loss: -${t.loss:.0f}.")
        lines.append(f"Net: {'+'if t.net >= 0 else ''}${t.net:.0f}.")
        lines.append(f"Balance: ${t.cumulative:.0f}.")

        if len(self.history) < self.max_trials:
            lines.append("\nChoose a deck: A, B, C, or D.")

        return " ".join(lines)

    def _deck_counts(self) -> dict[str, int]:
        counts = {d: 0 for d in self.decks}
        for t in self.history:
            counts[t.deck_chosen] += 1
        return counts

    def score(self) -> float:
        """IGT score: (C+D draws) - (A+B draws). Positive = good deck preference."""
        counts = self._deck_counts()
        return (counts["C"] + counts["D"]) - (counts["A"] + counts["B"])

    def format_history(self, last_n: int | None = None) -> str:
        """Format trial history as a text block for LM context."""
        trials = self.history if last_n is None else self.history[-last_n:]
        lines = []
        for t in trials:
            loss_str = f" Loss: -${t.loss:.0f}." if t.loss > 0 else ""
            lines.append(
                f"Round {t.round_num}: Deck {t.deck_chosen}, "
                f"+${t.reward:.0f},{loss_str} "
                f"Net {'+'if t.net >= 0 else ''}${t.net:.0f}, "
                f"Balance ${t.cumulative:.0f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline agents (for environment validation)
# ---------------------------------------------------------------------------

class RandomAgent:
    """Uniform random deck selection."""
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose(self, history: list[Trial]) -> str:
        return self.rng.choice(["A", "B", "C", "D"])


class EMAAgent:
    """Exponential moving average agent — affect-like baseline.

    Maintains a running EMA of net payoff per deck.
    Chooses the deck with the highest EMA (with epsilon-greedy exploration).
    This is the simplest agent that has a "gut feeling" — a persistent,
    slowly-updating evaluative signal per option.
    """
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1,
                 rng: random.Random | None = None):
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.ema = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    def choose(self, history: list[Trial]) -> str:
        # Update EMA from last trial
        if history:
            t = history[-1]
            self.ema[t.deck_chosen] += self.alpha * (t.net - self.ema[t.deck_chosen])

        # Epsilon-greedy
        if self.rng.random() < self.epsilon:
            return self.rng.choice(["A", "B", "C", "D"])
        return max(self.ema, key=lambda d: self.ema[d])


def run_agent(env: IGTEnvironment, agent, seed: int = 42) -> dict:
    """Run an agent through a full IGT session. Returns summary dict."""
    env.seed(seed)
    if hasattr(agent, "rng"):
        agent.rng = random.Random(seed)

    env.reset()
    done = False
    while not done:
        action = agent.choose(env.history)
        _, _, done, _ = env.step(action)

    return {
        "score": env.score(),
        "final_balance": env.cumulative,
        "deck_counts": env._deck_counts(),
        "total_trials": len(env.history),
    }
