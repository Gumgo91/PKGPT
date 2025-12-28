"""
Phase-specific prompt templates for NONMEM optimization
Each phase has a focused, concise prompt for targeted improvements
"""

from .phase1_establish import Phase1Establish
from .phase2_diagnose import Phase2Diagnose
from .phase3_reduce import Phase3Reduce
from .phase4_optimize import Phase4Optimize
from .phase5_covariates import Phase5Covariates

__all__ = [
    'Phase1Establish',
    'Phase2Diagnose',
    'Phase3Reduce',
    'Phase4Optimize',
    'Phase5Covariates',
]
