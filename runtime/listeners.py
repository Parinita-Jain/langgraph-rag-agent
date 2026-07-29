from runtime.event import WorkflowEvent


class ConsoleEventListener:

    def __call__(self, event: WorkflowEvent):

        print(
            f"[EVENT] "
            f"{event.type} "
            f"step={event.step_id} "
            f"tool={event.tool}"
        )