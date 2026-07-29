import logging

from runtime.event import WorkflowEvent
from runtime.event_types import WorkflowEventType


logger = logging.getLogger(__name__)


class LoggingEventListener:

    def __call__(self, event: WorkflowEvent):

        match event.type:

            case WorkflowEventType.WORKFLOW_STARTED:
                logger.info("Workflow started.")

            case WorkflowEventType.WORKFLOW_COMPLETED:
                logger.info("Workflow completed.")

            case WorkflowEventType.STEP_STARTED:
                logger.info(
                    "Step %s started (%s).",
                    event.step_id,
                    event.tool,
                )

            case WorkflowEventType.STEP_COMPLETED:
                logger.info(
                    "Step %s completed (%s).",
                    event.step_id,
                    event.tool,
                )

            case WorkflowEventType.STEP_FAILED:
                logger.error(
                    "Step %s failed (%s).",
                    event.step_id,
                    event.tool,
                )

            case WorkflowEventType.STEP_SKIPPED:
                logger.warning(
                    "Step %s skipped (%s).",
                    event.step_id,
                    event.tool,
                )