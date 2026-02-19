from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class CategoryCondition(BaseModel):
    condition_id: str
    system_prompt: str
    category: str
    direction: Literal["pos", "neg"]
    prompt_type: Literal["persona", "experiential", "value_laden", "instruction", "identity", "casual"]


class CompetingCondition(BaseModel):
    condition_id: str
    system_prompt: str
    pair_id: str
    subject: str
    task_type: str
    direction: Literal["love_subject", "love_task_type"]


class RolePlayingCondition(BaseModel):
    condition_id: str
    system_prompt: str


_EXPERIMENT_TO_CONDITION = {
    "category_preference": CategoryCondition,
    "targeted_preference": CategoryCondition,
    "competing_preference": CompetingCondition,
    "role_playing": RolePlayingCondition,
    "narrow_preference": RolePlayingCondition,
}


class OODPromptSet(BaseModel):
    experiment: str
    baseline_prompt: str
    conditions: list[CategoryCondition] | list[CompetingCondition] | list[RolePlayingCondition]

    @classmethod
    def load(cls, path: Path) -> OODPromptSet:
        with open(path) as f:
            data = json.load(f)

        experiment = data["experiment"]
        condition_cls = _EXPERIMENT_TO_CONDITION[experiment]
        conditions = [condition_cls(**c) for c in data["conditions"]]

        return cls(
            experiment=experiment,
            baseline_prompt=data["baseline_prompt"],
            conditions=conditions,
        )
