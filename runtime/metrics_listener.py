from runtime.event import WorkflowEvent
from runtime.event_types import WorkflowEventType


class MetricsListener:

    def __init__(self):

        self.workflows_started = 0
        self.workflows_completed = 0

        self.steps_started = 0
        self.steps_completed = 0
        self.steps_failed = 0
        self.steps_skipped = 0

    def __call__(self, event: WorkflowEvent):

        match event.type:

            case WorkflowEventType.WORKFLOW_STARTED:
                self.workflows_started += 1

            case WorkflowEventType.WORKFLOW_COMPLETED:
                self.workflows_completed += 1

            case WorkflowEventType.STEP_STARTED:
                self.steps_started += 1

            case WorkflowEventType.STEP_COMPLETED:
                self.steps_completed += 1

            case WorkflowEventType.STEP_FAILED:
                self.steps_failed += 1

            case WorkflowEventType.STEP_SKIPPED:
                self.steps_skipped += 1

    def snapshot(self):

        return {
            "workflows_started": self.workflows_started,
            "workflows_completed": self.workflows_completed,
            "steps_started": self.steps_started,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "steps_skipped": self.steps_skipped,
        }