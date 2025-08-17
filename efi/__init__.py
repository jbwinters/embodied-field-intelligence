"""
Embodied Field Intelligence (EFI)
A CA-based framework for embodied artificial intelligence using cellular automata,
chemotaxis fields, and schema learning.
"""

__version__ = "0.1.0"
__author__ = "EFI Team"

from .configs import EnvConfig, AgentConfig, SchemaConfig, Ablations
from .agents import ChemotaxisAgentCA, SchemaField
from .envs import ForageWorld

__all__ = [
    "EnvConfig",
    "AgentConfig", 
    "SchemaConfig",
    "Ablations",
    "ChemotaxisAgentCA",
    "SchemaField",
    "ForageWorld",
]