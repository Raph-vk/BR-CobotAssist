import abc

class TeachbotInterface(metaclass=abc.ABCMeta):
    """
    Abstract base class for a 'Teachbot' — 
    a manipulator used for teaching paths/trajectories.
    """

    @abc.abstractmethod
    def connect(self):
        """Establish connection to the teachbot hardware or simulator."""
        pass

