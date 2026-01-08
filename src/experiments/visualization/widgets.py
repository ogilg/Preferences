from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from src.preferences.ranking import PairwiseData, ThurstonianResult, fit_thurstonian
from src.preferences.storage import MEASUREMENTS_DIR, load_yaml
from src.preferences.templates.template import load_templates_from_yaml
from src.task_data import OriginDataset, Task


@dataclass
class RunConfig:
    template_name: str
    template_tags: dict
    model_short: str
    run_dir: Path


def load_thurstonian_from_measurements(run_dir: Path) -> ThurstonianResult:
    """Fit thurstonian model from measurements in run directory."""
    measurements_path = run_dir / "measurements.yaml"
    raw = load_yaml(measurements_path)

    task_ids = sorted({m["task_a"] for m in raw} | {m["task_b"] for m in raw})
    tasks = [Task(prompt="", origin=OriginDataset.WILDCHAT, id=tid, metadata={}) for tid in task_ids]
    id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

    n = len(tasks)
    wins = np.zeros((n, n), dtype=np.int32)
    for m in raw:
        i, j = id_to_idx[m["task_a"]], id_to_idx[m["task_b"]]
        if m["choice"] == "a":
            wins[i, j] += 1
        else:
            wins[j, i] += 1

    data = PairwiseData(tasks=tasks, wins=wins)
    return fit_thurstonian(data)


class RunBrowser:
    """Usage: RunBrowser("results/measurements/").display()"""

    def __init__(self, results_dir: str | Path = MEASUREMENTS_DIR):
        self.results_dir = Path(results_dir)
        self._current_config: RunConfig | None = None
        self._current_result: ThurstonianResult | None = None
        self._setup_widgets()

    def _discover_runs(self) -> list[str]:
        runs = []
        if not self.results_dir.exists():
            return runs

        for run_dir in sorted(self.results_dir.iterdir()):
            if (run_dir / "config.yaml").exists():
                runs.append(run_dir.name)
        return runs

    def _setup_widgets(self) -> None:
        run_options = self._discover_runs()
        if not run_options:
            run_options = ["(no runs found)"]

        self.run_dropdown = widgets.Dropdown(
            options=run_options,
            description="Run:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="300px"),
        )

        self.template_accordion = widgets.Accordion(
            children=[widgets.Textarea(
                value="",
                layout=widgets.Layout(width="100%", height="150px"),
                disabled=True,
            )],
            titles=["Template"],
            selected_index=None,
        )

        self.plot_output = widgets.Output()

        self.task_dropdown = widgets.Dropdown(
            options=[],
            description="Task:",
            style={"description_width": "40px"},
            layout=widgets.Layout(width="300px"),
        )

        self.run_dropdown.observe(self._on_run_change, names="value")

    def _on_run_change(self, change: dict) -> None:
        self._render()

    def _render(self) -> None:
        self.plot_output.clear_output(wait=True)

        run_name = self.run_dropdown.value
        if run_name == "(no runs found)":
            with self.plot_output:
                print("No measurement runs found in directory.")
            return

        run_dir = self.results_dir / run_name

        with self.plot_output:
            try:
                config_data = load_yaml(run_dir / "config.yaml")
                self._current_config = RunConfig(
                    template_name=config_data["template_name"],
                    template_tags=config_data["template_tags"],
                    model_short=config_data["model_short"],
                    run_dir=run_dir,
                )
                self._current_result = load_thurstonian_from_measurements(run_dir)

                # Try to load template for display
                try:
                    template_file = Path("src/preferences/templates/generated_templates.yaml")
                    templates = load_templates_from_yaml(template_file)
                    template = next(
                        (t for t in templates if t.name == self._current_config.template_name),
                        None,
                    )
                    if template:
                        self.template_accordion.children[0].value = template.template
                    else:
                        self.template_accordion.children[0].value = "(template not found)"
                except Exception:
                    self.template_accordion.children[0].value = "(template file not found)"

                # Update task dropdown (sorted by utility)
                order = np.argsort(-self._current_result.mu)
                sorted_ids = [self._current_result.tasks[i].id for i in order]
                self.task_dropdown.options = sorted_ids

                # Create plot
                self._plot_utility_ranking()
                plt.show()

            except Exception as e:
                print(f"Error loading run: {e}")

    def _plot_utility_ranking(self) -> None:
        if self._current_result is None or self._current_config is None:
            return

        result = self._current_result
        order = np.argsort(-result.mu)
        sorted_ids = [result.tasks[i].id for i in order]
        sorted_mu = result.mu[order]
        sorted_sigma = result.sigma[order]

        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(sorted_ids))
        ax.bar(x, sorted_mu, yerr=sorted_sigma, capsize=3, color="steelblue", alpha=0.8)

        ax.set_xlabel("Task")
        ax.set_ylabel("Utility (mu)")
        ax.set_xticks(x)
        ax.set_xticklabels(sorted_ids, rotation=45, ha="right")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

        title = f"Utility Ranking: {self._current_config.template_name} / {self._current_config.model_short}"
        if not result.converged:
            title += " (not converged)"
        ax.set_title(title)

        fig.tight_layout()

    def display(self) -> None:
        container = widgets.VBox([
            self.run_dropdown,
            self.template_accordion,
            self.plot_output,
            self.task_dropdown,
        ])
        display(container)
        self._render()
