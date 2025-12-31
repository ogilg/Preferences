from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from src.preferences.storage import (
    load_thurstonian_data,
    load_run,
    MeasurementRun,
    ThurstonianData,
    RESULTS_DIR,
)
from .plots import plot_utility_ranking


class RunBrowser:
    """Usage: RunBrowser("results/").display()"""

    def __init__(self, results_dir: str | Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self._current_run: MeasurementRun | None = None
        self._current_data: ThurstonianData | None = None
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
        self.task_prompt = widgets.Textarea(
            value="Select a task to see its prompt",
            layout=widgets.Layout(width="100%", height="100px"),
            disabled=True,
        )

        self.run_dropdown.observe(self._on_run_change, names="value")
        self.task_dropdown.observe(self._on_task_change, names="value")

    def _on_run_change(self, change: dict) -> None:
        self._render()

    def _on_task_change(self, change: dict) -> None:
        if self._current_run is None:
            return

        task_id = self.task_dropdown.value
        if not task_id:
            return

        prompt = self._current_run.config.task_prompts.get(task_id, "(prompt not stored)")
        self.task_prompt.value = prompt

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
                self._current_data = load_thurstonian_data(run_dir)
                self._current_run = load_run(run_dir)

                # Update template display
                try:
                    template = self._current_run.config.load_template()
                    self.template_accordion.children[0].value = template.template
                except Exception:
                    self.template_accordion.children[0].value = "(template not found)"

                # Update task dropdown (sorted by utility)
                order = self._current_data.ranking_order()
                sorted_ids = [self._current_data.task_ids[i] for i in order]
                self.task_dropdown.options = sorted_ids
                if sorted_ids:
                    self.task_dropdown.value = sorted_ids[0]
                    self._on_task_change(None)

                # Create plot
                fig, bars = plot_utility_ranking(
                    self._current_data,
                    config=self._current_run.config,
                )
                plt.show()

            except Exception as e:
                print(f"Error loading run: {e}")

    def display(self) -> None:
        task_box = widgets.VBox([self.task_dropdown, self.task_prompt])

        container = widgets.VBox([
            self.run_dropdown,
            self.template_accordion,
            self.plot_output,
            task_box,
        ])
        display(container)
        self._render()
