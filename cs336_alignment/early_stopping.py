from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EarlyStopping:
    metric_name: str = "reward"
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "max"  # "max" or "min"
    min_steps: int = 0
    smoothing_window: int = 1
    save_best: bool = True
    output_dir: Optional[str] = None

    best_metric: float = field(init=False)
    patience_counter: int = field(default=0, init=False)
    step: int = field(default=0, init=False)
    metric_window: deque = field(init=False)

    def __post_init__(self):
        assert self.mode in ["max", "min"]
        self.best_metric = -float("inf") if self.mode == "max" else float("inf")
        self.metric_window = deque(maxlen=self.smoothing_window)

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_metric + self.min_delta
        else:
            return value < self.best_metric - self.min_delta

    def update(self, metrics: dict, model=None) -> tuple[bool, dict]:
        """
        Args:
            metrics: dict of eval metrics (e.g., avg_scores)
            model: optional HF model to save best checkpoint

        Returns:
            stop: bool → whether to stop training
            info: dict → logging info
        """
        self.step += 1

        if self.metric_name not in metrics:
            raise ValueError(f"{self.metric_name} not found in metrics: {list(metrics.keys())}")

        raw_value = metrics[self.metric_name]
        self.metric_window.append(raw_value)
        smoothed_value = sum(self.metric_window) / len(self.metric_window)

        info = {
            "raw_metric": raw_value,
            "smoothed_metric": smoothed_value,
            "best_metric": self.best_metric,
            "patience_counter": self.patience_counter,
        }

        # Warmup period
        if self.step < self.min_steps:
            return False, info

        if self._is_improvement(smoothed_value):
            self.best_metric = smoothed_value
            self.patience_counter = 0
            info["improved"] = True

            if self.save_best and model is not None and self.output_dir is not None:
                save_path = f"{self.output_dir}/best-checkpoint"
                model.save_pretrained(save_path)
                info["saved"] = save_path

        else:
            self.patience_counter += 1
            info["improved"] = False

        stop = self.patience_counter >= self.patience
        return stop, info