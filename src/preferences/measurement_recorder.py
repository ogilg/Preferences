"""Record measurements to YAML for human review."""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MeasurementRecord:
    """A single measurement record."""

    model: str
    measurement_type: str  # PreferenceType value (e.g., "PRE_TASK_STATED")
    tasks: list[dict[str, str]]  # [{"id": ..., "prompt": ...}]
    response_format: str
    template: str
    temperature: float
    sample_index: int
    prompt: str  # Full prompt text
    response: str  # Raw model response
    result: dict[str, Any]  # {"choice": "a"} or {"score": 7.0}


class MeasurementRecorder:
    """Records measurements to a YAML file."""

    def __init__(self, output_path: Path | str):
        self.output_path = Path(output_path)
        self.records: list[MeasurementRecord] = []

    def record(self, record: MeasurementRecord) -> None:
        """Add a measurement record."""
        self.records.append(record)

    def save(self) -> None:
        """Write all records to YAML file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(r) for r in self.records]

        with open(self.output_path, "w") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

    def __enter__(self) -> "MeasurementRecorder":
        return self

    def __exit__(self, *args: Any) -> None:
        self.save()
