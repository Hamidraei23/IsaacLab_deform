"""FastSAC (holosoma's soft actor-critic for massively parallel simulation), ported for Isaac Lab.

See ``fast_sac.py`` for what was kept and changed, and ``scripts/train_sac.py`` for the entry point.
"""

from .buffers import EmpiricalNormalization, IdentityNormalization, SimpleReplayBuffer
from .config import FastSACCfg
from .fast_sac import FastSAC
from .networks import Actor, Critic, DistributionalQNetwork
from .runner import BehaviorPolicy, FastSACRunner, IsaacLabSacEnv
from .teacher import FrozenTeacher
from .validation import CriticGate, CriticValidator

__all__ = [
    "Actor",
    "BehaviorPolicy",
    "Critic",
    "CriticGate",
    "CriticValidator",
    "DistributionalQNetwork",
    "EmpiricalNormalization",
    "FastSAC",
    "FastSACCfg",
    "FastSACRunner",
    "FrozenTeacher",
    "IdentityNormalization",
    "IsaacLabSacEnv",
    "SimpleReplayBuffer",
]
