from nsgp.interpretability.AutomaticInterpretabilityEstimateUpdater import AutomaticInterpretabilityEstimateUpdater
from threads.StoppableThread import StoppableThread


class FeedbackThread(StoppableThread):
    def __init__(self, interpretability_estimate_updater: AutomaticInterpretabilityEstimateUpdater, delay: float = None, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.interpretability_estimate_updater = interpretability_estimate_updater
        self.delay = delay

    def run(self) -> None:
        while not self.stopped():
            if self.delay is not None and self.delay > 0:
                self.interpretability_estimate_updater.delayed_update(self.delay)
            else:
                self.interpretability_estimate_updater.immediate_update()
