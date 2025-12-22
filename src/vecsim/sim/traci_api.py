from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VehicleSnapshot:
    veh_id: str
    x_m: float
    y_m: float
    speed_mps: float


class TraciAPI:
    """Thin wrapper around TraCI.

    Keep all direct `traci.*` calls inside this module so the rest of the code
    stays testable.
    """

    def __init__(self) -> None:
        try:
            import traci  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TraCI (traci) is not available. Make sure SUMO and traci are installed "
                "and that you are running inside an environment where SUMO_HOME is set."
            ) from e
        self._traci = traci

    def start(self, sumo_cmd: List[str]) -> None:
        log.info("Starting SUMO with cmd: %s", " ".join(sumo_cmd))
        self._traci.start(sumo_cmd)

    def close(self) -> None:
        log.info("Closing TraCI.")
        self._traci.close()

    def simulation_step(self) -> None:
        self._traci.simulationStep()

    def get_departed_vehicles(self) -> List[str]:
        return list(self._traci.simulation.getDepartedIDList())

    def get_arrived_vehicles(self) -> List[str]:
        return list(self._traci.simulation.getArrivedIDList())

    def get_vehicle_snapshot(self, veh_id: str) -> VehicleSnapshot:
        x, y = self._traci.vehicle.getPosition(veh_id)
        v = self._traci.vehicle.getSpeed(veh_id)
        return VehicleSnapshot(veh_id=veh_id, x_m=float(x), y_m=float(y), speed_mps=float(v))

    def get_all_vehicle_ids(self) -> List[str]:
        return list(self._traci.vehicle.getIDList())
