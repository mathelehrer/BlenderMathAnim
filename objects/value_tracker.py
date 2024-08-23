import numpy as np

from objects.bobject import BObject


class ValueTracker(BObject):

    def __init__(self, value=0, **kwargs):
        super().__init__(**kwargs)
        self.points = np.zeros((1, 3))
        self.set_value(value)

    def get_value(self) -> float:
        """Get the current value of this ValueTracker."""
        return self.points[0, 0]

    def set_value(self, value: float):
        """Sets a new scalar value to the ValueTracker"""
        self.points[0, 0] = value
        return self

    def increment_value(self, d_value: float):
        """Increments (adds) a scalar value  to the ValueTracker"""
        self.set_value(self.get_value() + d_value)
        return self

    #########################
    # Overloaded operations #
    #########################

    def __bool__(self):
        """Return whether the value of this value tracker evaluates as true."""
        return bool(self.get_value())

    def __iadd__(self, d_value: float):
        """adds ``+=`` syntax to increment the value of the ValueTracker"""
        self.increment_value(d_value)
        return self

    def __ifloordiv__(self, d_value: float):
        """Set the value of this value tracker to the floor division of the current value by ``d_value``."""
        self.set_value(self.get_value() // d_value)
        return self

    def __imod__(self, d_value: float):
        """Set the value of this value tracker to the current value modulo ``d_value``."""
        self.set_value(self.get_value() % d_value)
        return self

    def __imul__(self, d_value: float):
        """Set the value of this value tracker to the product of the current value and ``d_value``."""
        self.set_value(self.get_value() * d_value)
        return self

    def __ipow__(self, d_value: float):
        """Set the value of this value tracker to the current value raised to the power of ``d_value``."""
        self.set_value(self.get_value() ** d_value)
        return self

    def __isub__(self, d_value: float):
        """adds ``-=`` syntax to decrement the value of the ValueTracker"""
        self.increment_value(-d_value)
        return self

    def __itruediv__(self, d_value: float):
        """Sets the value of this value tracker to the current value divided by ``d_value``."""
        self.set_value(self.get_value() / d_value)
        return self
