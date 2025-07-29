class BaseAgent:
    """
    Base class for all agents. Defines the standard interface.
    """
    def execute(self, *args, **kwargs):
        raise NotImplementedError("The execute method must be implemented by subclasses.")
