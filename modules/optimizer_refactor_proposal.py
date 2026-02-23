"""
PROPOSAL: Clearer Phase Transition Logic
This is a proposed refactoring to make phase transitions more explicit and understandable.
"""

from enum import Enum
from typing import Optional, Dict


class ModelPhase(Enum):
    ESTABLISH_BASE = 1
    DIAGNOSE_STRUCTURE = 2
    REDUCE_OVERFITTING = 3
    OPTIMIZE_IIV = 4
    COVARIATE_ANALYSIS = 5


class PhaseTransitionManager:
    """Manages phase transitions with clear, explicit logic"""

    MAX_ITERATIONS_PER_PHASE = 5

    def __init__(self, data_loader, current_phase, iterations_in_phase, improvement_history):
        self.data_loader = data_loader
        self.current_phase = current_phase
        self.iterations_in_phase = iterations_in_phase
        self.improvement_history = improvement_history

    def determine_next_phase(self, parsed_results: Dict) -> ModelPhase:
        """
        Determine next phase using clear state machine logic

        Returns:
            Next phase (always >= current_phase, never backward!)
        """
        # Step 1: Check if current phase is COMPLETE
        if self._is_phase_complete(parsed_results):
            # Move to next logical phase
            next_phase = self._get_next_phase(parsed_results)
            return next_phase

        # Step 2: Check if STUCK in current phase (max iterations)
        if self.iterations_in_phase >= self.MAX_ITERATIONS_PER_PHASE:
            # Force transition to avoid infinite loop
            return self._force_next_phase(parsed_results)

        # Step 3: Stay in current phase (not complete, not stuck)
        return self.current_phase

    def _is_phase_complete(self, parsed_results: Dict) -> bool:
        """
        Check if current phase has completed its objectives

        EXPLICIT completion criteria for each phase
        """
        minimization_ok = parsed_results.get('minimization_successful', False)
        ofv = parsed_results.get('objective_function')
        shrinkage = parsed_results.get('eta_shrinkage', [])
        avg_shrink = sum([s['shrinkage'] for s in shrinkage]) / len(shrinkage) if shrinkage else None

        metadata = self.data_loader.get_metadata()
        n_subjects = metadata.get('n_subjects', 100)

        # Phase 1: ESTABLISH_BASE
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            # Complete when: Minimization successful
            return minimization_ok

        # Phase 2: DIAGNOSE_STRUCTURE
        if self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            # Complete when:
            # - Spent at least 2 iterations checking structure
            # - OR max iterations reached
            return self.iterations_in_phase >= 2

        # Phase 3: REDUCE_OVERFITTING
        if self.current_phase == ModelPhase.REDUCE_OVERFITTING:
            # Complete when:
            # - Shrinkage reduced to <90%
            # - OR OFV no longer extremely negative
            # - OR spent enough iterations trying
            if avg_shrink and avg_shrink < 90:
                return True
            if ofv and ofv > -50:
                return True
            return self.iterations_in_phase >= 3

        # Phase 4: OPTIMIZE_IIV
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            # Complete when:
            # - Shrinkage is good (<50%) → Ready for covariates
            # - OR spent enough iterations optimizing
            if avg_shrink and avg_shrink < 50:
                # Check if covariates available
                covariates_available = len(metadata.get('covariates', [])) > 0
                if covariates_available and minimization_ok:
                    # Check OFV stability
                    if self._is_ofv_stable():
                        return True
            return self.iterations_in_phase >= 4

        # Phase 5: COVARIATE_ANALYSIS
        if self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            # Final phase - complete when max iterations or converged
            return self.iterations_in_phase >= 4

        return False

    def _get_next_phase(self, parsed_results: Dict) -> ModelPhase:
        """
        Determine the NEXT phase to transition to

        EXPLICIT phase progression rules
        """
        minimization_ok = parsed_results.get('minimization_successful', False)
        ofv = parsed_results.get('objective_function')
        shrinkage = parsed_results.get('eta_shrinkage', [])
        avg_shrink = sum([s['shrinkage'] for s in shrinkage]) / len(shrinkage) if shrinkage else None

        metadata = self.data_loader.get_metadata()
        n_subjects = metadata.get('n_subjects', 100)

        # From Phase 1: ESTABLISH_BASE
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            # Check for overfitting first
            if self._is_true_overfitting(ofv, avg_shrink):
                print("[TRANSITION] Phase 1 → Phase 3 (overfitting detected)")
                return ModelPhase.REDUCE_OVERFITTING

            # Check dataset size
            if n_subjects < 20:
                print("[TRANSITION] Phase 1 → Phase 4 (N<20, skipping structure diagnosis)")
                return ModelPhase.OPTIMIZE_IIV

            # Normal progression
            print("[TRANSITION] Phase 1 → Phase 2 (minimization successful, diagnosing structure)")
            return ModelPhase.DIAGNOSE_STRUCTURE

        # From Phase 2: DIAGNOSE_STRUCTURE
        if self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            # Check for overfitting
            if self._is_true_overfitting(ofv, avg_shrink):
                print("[TRANSITION] Phase 2 → Phase 3 (overfitting detected)")
                return ModelPhase.REDUCE_OVERFITTING

            print("[TRANSITION] Phase 2 → Phase 4 (structure diagnosis complete)")
            return ModelPhase.OPTIMIZE_IIV

        # From Phase 3: REDUCE_OVERFITTING
        if self.current_phase == ModelPhase.REDUCE_OVERFITTING:
            print("[TRANSITION] Phase 3 → Phase 4 (overfitting reduced)")
            return ModelPhase.OPTIMIZE_IIV

        # From Phase 4: OPTIMIZE_IIV
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            # Check if ready for covariate analysis
            covariates_available = len(metadata.get('covariates', [])) > 0
            if (covariates_available and
                minimization_ok and
                avg_shrink and avg_shrink < 50 and
                self._is_ofv_stable()):
                print("[TRANSITION] Phase 4 → Phase 5 (ready for covariate analysis)")
                return ModelPhase.COVARIATE_ANALYSIS

            print("[TRANSITION] Phase 4 complete (no covariates or not ready)")
            return ModelPhase.OPTIMIZE_IIV  # Stay in Phase 4

        # From Phase 5: COVARIATE_ANALYSIS (final phase)
        print("[TRANSITION] Phase 5 complete (final phase)")
        return ModelPhase.COVARIATE_ANALYSIS

    def _force_next_phase(self, parsed_results: Dict) -> ModelPhase:
        """
        Force transition when stuck (max iterations reached)

        Always moves forward, never backward
        """
        print(f"\n{'!'*70}")
        print(f"[FORCE TRANSITION] Max iterations ({self.MAX_ITERATIONS_PER_PHASE}) reached in {self.current_phase}")
        print(f"{'!'*70}")

        minimization_ok = parsed_results.get('minimization_successful', False)
        metadata = self.data_loader.get_metadata()
        covariates_available = len(metadata.get('covariates', [])) > 0

        # All forced transitions go to Phase 4 (safest) or Phase 5
        if self.current_phase.value < ModelPhase.OPTIMIZE_IIV.value:
            print(f"[FORCE] {self.current_phase} → Phase 4 (safest fallback)")
            return ModelPhase.OPTIMIZE_IIV

        # From Phase 4: Try Phase 5 if possible
        if self.current_phase == ModelPhase.OPTIMIZE_IIV:
            if covariates_available and minimization_ok:
                print(f"[FORCE] Phase 4 → Phase 5 (attempting covariates)")
                return ModelPhase.COVARIATE_ANALYSIS
            else:
                print(f"[FORCE] Phase 4 complete (no covariates, staying)")
                return ModelPhase.OPTIMIZE_IIV

        # Phase 5: Stay (final phase)
        print(f"[FORCE] Phase 5 complete (final phase)")
        return ModelPhase.COVARIATE_ANALYSIS

    def _is_ofv_stable(self) -> bool:
        """Check if OFV has stabilized"""
        if len(self.improvement_history) < 2:
            return False

        recent_ofvs = [h.get('ofv') for h in self.improvement_history[-2:]]
        if not all(o is not None for o in recent_ofvs):
            return False

        ofv_change = abs(recent_ofvs[-1] - recent_ofvs[-2])
        return ofv_change < 5

    def _is_true_overfitting(self, ofv, avg_shrink) -> bool:
        """Check if TRUE overfitting (vs underparameterization)"""
        # Only detect overfitting if not already past Phase 3
        if self.current_phase.value >= ModelPhase.OPTIMIZE_IIV.value:
            return False

        # Criteria for TRUE overfitting
        if ofv is not None and ofv < -50:
            return True

        if avg_shrink is not None and avg_shrink > 95:
            return True

        return False


"""
USAGE EXAMPLE:

class NONMEMOptimizer:
    def __init__(self, ...):
        self.phase_manager = None

    def _determine_current_phase(self, parser):
        if self.phase_manager is None:
            self.phase_manager = PhaseTransitionManager(
                self.data_loader,
                self.current_phase,
                self.iterations_in_phase,
                self.improvement_history
            )

        # Update state
        self.phase_manager.current_phase = self.current_phase
        self.phase_manager.iterations_in_phase = self.iterations_in_phase

        # Get parsed results
        parsed_results = parser.get_parsed_data() if parser else {}

        # Determine next phase using clear logic
        next_phase = self.phase_manager.determine_next_phase(parsed_results)

        return next_phase
"""
