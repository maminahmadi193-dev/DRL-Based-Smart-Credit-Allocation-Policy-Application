from dataclasses import dataclass
from .customer_state import CustomerState


@dataclass
class SimulationOutcome:
    """
    Output of CustomerSimulator.
    """
    next_state: CustomerState

    probability_of_default: float

    default_event: bool