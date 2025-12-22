from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimClock:
    """Simulation clock based on fixed step length."""
    step_length_s: float
    step: int = 0

    @property
    def t(self) -> float:
        """Current simulation time in seconds."""
        return self.step * self.step_length_s

    def tick(self) -> None:
        """Advance one step."""
        self.step += 1
