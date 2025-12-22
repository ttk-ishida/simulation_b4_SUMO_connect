from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class WorldState:
    """Holds the simulation state.

    This is intentionally lightweight at the beginning. As you implement more
    components (Vehicle/RSU/MEC), extend this class or replace it with richer
    dataclasses in src/vecsim/models.
    """
    vehicles: Dict[str, dict] = field(default_factory=dict)  # placeholder
    rsu: Optional[dict] = None                               # placeholder
    mec: Optional[dict] = None                               # placeholder
    prediction: Optional[dict] = None                        # placeholder
