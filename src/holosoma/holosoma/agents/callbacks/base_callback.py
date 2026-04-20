from __future__ import annotations

from typing import Any

from torch.nn import Module


class RLEvalCallback(Module):
    def __init__(self, config, training_loop):
        super().__init__()
        self.config = config
        self.training_loop = training_loop
        self.device = self.training_loop.device

    def _get_env(self):
        """Get the unwrapped BaseTask environment."""
        return self.training_loop._unwrap_env()

    def _require_recording_cb(self) -> Any:
        """Find and return the EvalRecordingCallback, or raise."""
        from holosoma.agents.callbacks.recording import EvalRecordingCallback

        for cb in self.training_loop.eval_callbacks:
            if isinstance(cb, EvalRecordingCallback):
                return cb
        raise RuntimeError(f"{type(self).__name__} requires EvalRecordingCallback. Set --recording.config.enabled=True")

    def on_pre_evaluate_policy(self):
        pass

    def on_pre_eval_env_step(self, actor_state):
        return actor_state

    def on_post_eval_env_step(self, actor_state):
        return actor_state

    def on_post_evaluate_policy(self):
        pass
