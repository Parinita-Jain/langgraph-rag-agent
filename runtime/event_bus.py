import time

from runtime.event import WorkflowEvent


class EventBus:

    def __init__(self):

        self._listeners = []

    def subscribe(self, listener):

        self._listeners.append(listener)

    def emit(self, event):

        for listener in self._listeners:

            listener(event)