from dataclasses import dataclass
from enum import Enum, auto


class OriginDataset(Enum):
    WILDCHAT = auto()
    ALPACA = auto()
    MATH = auto()


@dataclass
class Task:
    prompt: str
    origin: OriginDataset
    id: str
    metadata: dict
