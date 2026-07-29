from runtime.event import WorkflowEvent


class AuditListener:

    def __init__(self):

        self._events: list[WorkflowEvent] = []

    def __call__(self, event: WorkflowEvent):

        self._events.append(event)

    def history(self) -> list[WorkflowEvent]:

        return self._events.copy()

    def clear(self):

        self._events.clear()