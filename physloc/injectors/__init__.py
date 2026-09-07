"""Injector registry. Importing this module registers every injector."""
from .base import (Injector, InterventionPlan, available, get,  # noqa: F401
                   register)
from . import (appearance, contact, dynamics,  # noqa: F401
               equilibrium, identity, kinematics, optical)
