"""
Recursive NONMEM Optimization Engine
Iteratively generates and improves NONMEM control stream files

Phase-aware optimization following systematic pharmacometric model development
"""

import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time

from .gemini_client import MultiModelGeminiClient
from .data_loader import PKDataLoader
from .prompt_templates import PromptTemplates
from .nonmem_parser import NONMEMParser
from .phase_transition_manager import PhaseTransitionManager


class ModelPhase(Enum):
    """
    Systematic model development phases

    Following best practices in pharmacometric model building
    """
    ESTABLISH_BASE = 1       # Phase 1: Fix syntax/execution, get minimization working
    DIAGNOSE_STRUCTURE = 2   # Phase 2: Check structural model adequacy (compartments)
    REDUCE_OVERFITTING = 3   # Phase 3: Simplify model if overfitting detected
    OPTIMIZE_IIV = 4         # Phase 4: Fine-tune random effects structure
    COVARIATE_ANALYSIS = 5   # Phase 5: Add covariates to explain variability

    def __str__(self):
        phase_names = {
            ModelPhase.ESTABLISH_BASE: "Phase 1: Establish Base Model",
            ModelPhase.DIAGNOSE_STRUCTURE: "Phase 2: Diagnose Structure",
            ModelPhase.REDUCE_OVERFITTING: "Phase 3: Reduce Overfitting",
            ModelPhase.OPTIMIZE_IIV: "Phase 4: Optimize IIV",
            ModelPhase.COVARIATE_ANALYSIS: "Phase 5: Covariate Analysis"
        }
        return phase_names.get(self, "Unknown Phase")


class NONMEMOptimizer:
    """Recursive optimizer for NONMEM models"""

    def __init__(
        self,
        data_file: str,
        output_base: str,
        api_key: Optional[str] = None,
        min_iterations: int = 3,
        max_iterations: int = 40,
        nmfe_command: str = 'nmfe75',
        model: str = 'flash'
    ):
        """
        Initialize NONMEM optimizer

        Args:
            data_file: Path to input dataset
            output_base: Base name for output files (without extension)
            api_key: Google Gemini API key
            min_iterations: Minimum number of optimization iterations
            max_iterations: Maximum number of optimization iterations
            nmfe_command: NONMEM execution command
            model: Gemini model to use ('flash', 'flash-lite', 'pro')
        """
        self.data_file = data_file
        self.output_base = output_base
        self.min_iterations = min_iterations
        self.max_iterations = max_iterations
        self.nmfe_command = nmfe_command
        self.model = model

        # Initialize components
        print("=" * 70)
        print("NONMEM RECURSIVE OPTIMIZER")
        print("=" * 70)

        self.data_loader = PKDataLoader(data_file)
        self.gemini_client = MultiModelGeminiClient(api_key)

        # Set the selected model
        self.gemini_client.current_model_type = model
        current_model = self.gemini_client.clients[model].get_current_model()
        print(f"Using model: {current_model}")

        # Optimization state
        self.iteration = 0
        self.current_code = None
        self.improvement_history = []
        self.best_ofv = None
        self.best_iteration = 0
        self.best_composite_score = float('inf')  # Lower is better
        self.best_code = None  # Store code from best iteration for revert
        self.code_history = []  # Store control stream code from each iteration

        # Phase-based optimization tracking
        self.current_phase = ModelPhase.ESTABLISH_BASE
        self.phase_history = []  # Track phase transitions
        self.iterations_in_phase = 0

        # Phase transition manager (cleaner state machine logic)
        self.phase_manager = None  # Will be initialized on first use

        # Failed strategy tracking to prevent infinite loops
        self.failed_strategies = []  # Track what we've tried that failed

        # Parameter stabilization history (THETA/OMEGA/SIGMA across iterations)
        # Used ONLY to suggest narrower initial values and bounds; does not change other logic
        self.parameter_history: list[dict] = []

        # Print dataset summary
        print("\n" + "=" * 70)
        print("DATASET SUMMARY")
        print("=" * 70)
        print(self.data_loader.get_column_summary())
        print("\n" + self.data_loader.get_data_summary())
        print("=" * 70 + "\n")

    def run(self) -> Dict:
        """
        Run the recursive optimization process

        Returns:
            Dictionary with optimization results
        """
        print("\n" + "=" * 70)
        print("STARTING OPTIMIZATION")
        print("=" * 70)
        print(f"Min iterations: {self.min_iterations}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Output base: {self.output_base}")
        print("=" * 70 + "\n")

        # Step 1: Generate initial NONMEM code
        self._generate_initial_code()

        # Step 2: Recursive improvement loop with phase-based control
        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {self.iteration}/{self.max_iterations}")
            print(f"CURRENT PHASE: {self.current_phase}")
            print(f"Iterations in phase: {self.iterations_in_phase}")
            print("=" * 70)

            # Run NONMEM
            success = self._run_nonmem()

            if success:
                # Parse results
                parsed_results = self._parse_results()

                # Determine and update current phase
                new_phase = self._determine_current_phase(parsed_results)
                self._update_phase(new_phase)

                # CRITICAL: Phase 5 max iterations check
                # Phase 5 (COVARIATE_ANALYSIS) limited to 20 iterations
                if (self.current_phase == ModelPhase.COVARIATE_ANALYSIS and
                    self.iterations_in_phase >= 20):
                    print("\n" + "="*70)
                    print("[PHASE 5 COMPLETE] Covariate analysis finished (20 iterations)")
                    print("[OK] Optimization complete - maximum phase iterations reached")
                    print("="*70)
                    break

                # Check for improvement
                should_continue = self._evaluate_improvement(parsed_results)

                # Decide whether to continue
                if self.iteration >= self.min_iterations and not should_continue:
                    print("\n[OK] Optimization converged successfully!")
                    break

                # Generate improved code (phase-aware)
                if self.iteration < self.max_iterations:
                    self._generate_improved_code(parsed_results)
            else:
                # NONMEM failed - stay in ESTABLISH_BASE phase
                self.current_phase = ModelPhase.ESTABLISH_BASE
                if self.iteration < self.max_iterations:
                    print("[WARNING] NONMEM execution failed - attempting to fix...")
                    self._generate_improved_code(None)
                else:
                    print("[ERROR] Maximum iterations reached with errors")
                    break

        # Final summary
        return self._generate_final_summary()

    def _generate_initial_code(self):
        """Generate initial NONMEM control stream"""
        print("Generating initial NONMEM control stream...")

        metadata = self.data_loader.get_metadata()

        prompt = PromptTemplates.initial_generation_prompt(
            dataset_info=self.data_loader.get_column_summary(),
            data_summary=self.data_loader.get_data_summary(),
            columns=metadata['columns'],
            nonmem_columns=metadata.get('nonmem_columns', {}),
            covariates=metadata.get('covariates', [])
        )

        response = self.gemini_client.generate(prompt, model_type=self.model)

        # Extract NONMEM code from response
        self.current_code = self._extract_nonmem_code(response)

        # Save to file
        iteration_file = f"{self.output_base}_iter0.txt"
        with open(iteration_file, 'w', encoding='utf-8') as f:
            f.write(self.current_code)

        # Store in code history
        self.code_history.append({
            'iteration': 0,
            'code': self.current_code,
            'description': 'Initial generation'
        })

        print(f"[OK] Initial code generated and saved to: {iteration_file}")
        print(f"  Lines of code: {len(self.current_code.splitlines())}")

    def _extract_nonmem_code(self, response: str) -> str:
        """Extract NONMEM code from Gemini response"""
        # Try to find code block
        code_block_pattern = r'```(?:nonmem|txt)?\s*\n(.*?)\n```'
        match = re.search(code_block_pattern, response, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # If no code block, look for $PROBLEM (start of NONMEM code)
        problem_match = re.search(r'\$PROBLEM', response, re.IGNORECASE)
        if problem_match:
            # Extract from $PROBLEM to end or until analysis text
            code = response[problem_match.start():]

            # Try to find where the code ends
            end_patterns = [
                r'\n\n[A-Z]{2,}:',  # Section headers like "ANALYSIS:"
                r'\n\n\*\*',  # Markdown headers
                r'\n\nNote:',  # Explanation notes
            ]

            for pattern in end_patterns:
                end_match = re.search(pattern, code)
                if end_match:
                    code = code[:end_match.start()]
                    break

            return code.strip()

        # Last resort - return full response
        return response.strip()

    def _validate_advan_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate ADVAN-specific syntax requirements before execution

        Args:
            code: NONMEM control stream code

        Returns:
            (is_valid, error_message) tuple
        """
        # Extract ADVAN type
        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        if not advan_match:
            return True, None  # No ADVAN specified, let NONMEM handle it

        advan_num = int(advan_match.group(1))

        # ADVAN4/5/6 (2-compartment) validation
        if advan_num in [4, 5, 6]:
            # Check for V3 error: ADVAN4 uses compartments 1,2,3 but we define V1,V2 (not V3)
            # The correct approach is to use K10, K12, K21 (not V3)

            # Common error: Defining S2 and S3 when using V1 and V2
            # ADVAN4 compartments: 1=Depot, 2=Central, 3=Peripheral
            # We should define S2 = V1 (central) but NOT S3 unless using V3

            pk_block = re.search(r'\$PK(.*?)(?=\$|$)', code, re.DOTALL | re.IGNORECASE)
            if pk_block:
                pk_content = pk_block.group(1)

                # Check if S3 is defined
                has_s3 = re.search(r'S3\s*=', pk_content, re.IGNORECASE)
                # Check if V3 is defined
                has_v3 = re.search(r'\bV3\s*=', pk_content, re.IGNORECASE)

                # Error: S3 defined but V3 not defined (common mistake)
                if has_s3 and not has_v3:
                    error_msg = (
                        f"ADVAN{advan_num} syntax error: S3 is defined but V3 is not defined. "
                        f"For 2-compartment models with ADVAN4, use:\n"
                        f"  - V1 (central volume) and V2 (peripheral volume)\n"
                        f"  - S2 = V1 (scaling for central compartment)\n"
                        f"  - Do NOT define S3 unless you explicitly define V3\n"
                        f"OR use micro-rate constants (K10, K12, K21) without S3."
                    )
                    return False, error_msg

                # Check if using V naming (V, not V1) which can confuse NONMEM
                has_v_not_v1 = re.search(r'\bV\s*=\s*TV', pk_content, re.IGNORECASE)
                has_v1 = re.search(r'\bV1\s*=\s*TV', pk_content, re.IGNORECASE)

                if advan_num == 4 and has_v_not_v1 and not has_v1:
                    error_msg = (
                        f"ADVAN4 requires explicit V1 for central volume, not just 'V'. "
                        f"Use 'V1 = TVV1 * EXP(ETA(2))' instead of 'V = TVV * EXP(ETA(2))'"
                    )
                    return False, error_msg

        return True, None

    def _run_nonmem(self) -> bool:
        """
        Execute NONMEM

        Returns:
            True if execution completed (even with errors), False if command failed
        """
        input_file = f"{self.output_base}_iter{self.iteration}.txt"
        output_file = f"{self.output_base}_iter{self.iteration}.lst"

        # Validate ADVAN syntax before execution
        is_valid, validation_error = self._validate_advan_syntax(self.current_code)
        if not is_valid:
            print(f"\n{'!'*70}")
            print("ADVAN SYNTAX VALIDATION FAILED")
            print(f"{'!'*70}")
            print(f"{validation_error}")
            print(f"{'!'*70}")
            print("\n[ERROR] Skipping NONMEM execution due to syntax error")
            print("[INFO] Will attempt to fix in next iteration")

            # Create a mock error output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"SYNTAX VALIDATION ERROR\n\n{validation_error}\n")
            return False

        # Write current code to input file
        with open(input_file, 'w', encoding='utf-8') as f:
            # Update $DATA line to point to actual data file
            code = self.current_code
            # Use absolute path to ensure NONMEM can find the data file
            # Convert to forward slashes for cross-platform compatibility
            absolute_data_path = os.path.abspath(self.data_file).replace('\\', '/')
            # Find $DATA line and replace filename with absolute path
            code = re.sub(
                r'\$DATA\s+\S+',
                f'$DATA {absolute_data_path}',
                code,
                flags=re.IGNORECASE
            )
            f.write(code)

        print(f"\nExecuting NONMEM: {self.nmfe_command} {input_file} {output_file}")
        print("  [INFO] This may take a few minutes...")

        try:
            # Execute NONMEM
            # Simple execution, rely on lst file for results
            result = subprocess.run(
                [self.nmfe_command, input_file, output_file],
                timeout=600,
                cwd=os.path.dirname(os.path.abspath(input_file)) or '.'
            )

            print(f"  [INFO] NONMEM process finished (exit code: {result.returncode})")

            # Wait for output file to be fully written
            # NONMEM might still be writing even after process returns
            if os.path.exists(output_file):
                print("  [INFO] Waiting for output file to be complete...")
                max_wait = 30  # seconds
                wait_interval = 1  # second
                waited = 0
                prev_size = 0
                stable_count = 0

                while waited < max_wait:
                    time.sleep(wait_interval)
                    waited += wait_interval

                    # Check if file size is stable
                    try:
                        current_size = os.path.getsize(output_file)
                        if current_size == prev_size:
                            stable_count += 1
                            if stable_count >= 3:  # Stable for 3 seconds
                                # Also check if "Stop Time" appears in file
                                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read()
                                    if 'Stop Time' in content or 'Stop time' in content.lower():
                                        print("  [OK] Output file complete")
                                        return True
                        else:
                            stable_count = 0
                            prev_size = current_size
                    except:
                        pass

                print(f"  [OK] NONMEM execution completed")
                return True
            else:
                print("[WARNING] Output file not created")
                return False

        except FileNotFoundError:
            print(f"[WARNING] NONMEM command '{self.nmfe_command}' not found")
            print("  This is expected if NONMEM is not installed on this machine")
            print("  Creating mock output for testing...")

            # Create a mock output file for testing
            self._create_mock_output(output_file)
            return True

        except subprocess.TimeoutExpired:
            print("[ERROR] NONMEM execution timed out (>10 minutes)")
            return False

        except Exception as e:
            print(f"[ERROR] Error executing NONMEM: {e}")
            return False

    def _create_mock_output(self, output_file: str):
        """Create mock NONMEM output for testing when NONMEM is not available"""
        mock_output = f"""Mock NONMEM output for testing
ITERATION: {self.iteration}

This is a placeholder because NONMEM is not installed on this system.
When running on a system with NONMEM, this will contain actual output.

MINIMIZATION TERMINATED
OBJECTIVE FUNCTION VALUE: {1000 + self.iteration * 10}

This mock allows the optimizer to continue and demonstrate the recursive improvement process.
"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(mock_output)

    def _parse_results(self) -> Optional[NONMEMParser]:
        """Parse NONMEM output file"""
        output_file = f"{self.output_base}_iter{self.iteration}.lst"

        try:
            # Pass gemini_client to parser for AI-based parsing
            # Get the underlying client for the selected model
            selected_client = self.gemini_client.clients.get(self.model)
            parser = NONMEMParser(
                output_file,
                gemini_client=selected_client,
                use_ai_parsing=True
            )
            print("\n" + parser.get_summary())
            return parser

        except Exception as e:
            print(f"[WARNING] Error parsing NONMEM output: {e}")
            return None

    def _calculate_composite_score(self, ofv: Optional[float], shrinkage: Optional[float],
                                   cov_success: bool, minimization_ok: bool,
                                   omega_values: List[float]) -> float:
        """
        Calculate composite quality score (lower is better)
        Penalizes overfitting, extreme shrinkage, and numerical instability

        Args:
            ofv: Objective function value
            shrinkage: Average ETA shrinkage percentage
            cov_success: Whether covariance step succeeded
            minimization_ok: Whether minimization succeeded
            omega_values: List of OMEGA diagonal values

        Returns:
            Composite score (lower = better model)
        """
        score = 0

        # OFV component
        if ofv is not None:
            if ofv < -50:
                # CRITICAL: Negative OFV in SAEM often indicates overfitting
                # Give huge penalty - likely fitting noise
                score += 50000
                print(f"  [WARNING] Negative OFV penalty: +50000")
            elif ofv < 0:
                # Small negative OFV might be acceptable in some methods
                score += 10000
                print(f"  [WARNING] Small negative OFV penalty: +10000")
            else:
                # Normal positive OFV - use as-is
                score += ofv
        else:
            # No OFV available - large penalty
            score += 100000

        # Shrinkage penalty (CRITICAL: extreme shrinkage = overfitting)
        if shrinkage is not None:
            if shrinkage > 95:
                # Catastrophic shrinkage - model has lost individual variability
                score += 20000
                print(f"  [CRITICAL] Extreme shrinkage (>95%) penalty: +20000")
            elif shrinkage > 90:
                # Severe shrinkage - likely overparameterized
                score += 10000
                print(f"  [CRITICAL] Severe shrinkage (>90%) penalty: +10000")
            elif shrinkage > 70:
                # High shrinkage - concerning
                score += 2000
                print(f"  [WARNING] High shrinkage (>70%) penalty: +2000")
            elif shrinkage > 50:
                # Moderate shrinkage - some concern
                score += 500

        # OMEGA near-zero detection (informational; mild penalty)
        if omega_values:
            collapsed_omegas = [o for o in omega_values if o < 0.0001]
            if collapsed_omegas:
                penalty = len(collapsed_omegas) * 500
                score += penalty
                print(f"  [WARNING] {len(collapsed_omegas)} OMEGA(s) very small (<0.0001): +{penalty}")

        # Covariance failure penalty (context-dependent)
        if not cov_success:
            if shrinkage is not None and shrinkage < 40:
                # If shrinkage is good (<40%), covariance failure is less critical
                # Model may still be useful for simulation/prediction
                score += 100
                print(f"  [INFO] Covariance failed but shrinkage good (<40%): +100")
            elif shrinkage is not None and shrinkage < 60:
                # Moderate shrinkage - covariance failure is concerning
                score += 200
                print(f"  [WARNING] Covariance failed with moderate shrinkage: +200")
            else:
                # Poor shrinkage + covariance failure = serious problem
                score += 300
                print(f"  [WARNING] Covariance failed with poor shrinkage: +300")
        else:
            print(f"  [OK] Covariance step successful: +0")

        # Minimization failure penalty
        if not minimization_ok:
            score += 2000

        return score

    def _is_true_overfitting(self, ofv: Optional[float], avg_shrink: Optional[float]) -> bool:
        """
        Distinguish true overfitting from underparameterization

        True overfitting: Model too complex, shrinkage high AND model stable/improving
        Underparameterization: Shrinkage high BUT model got worse (OFV increased significantly)

        Args:
            ofv: Current objective function value
            avg_shrink: Average ETA shrinkage

        Returns:
            True if this is true overfitting (should simplify)
        """
        if avg_shrink is None or avg_shrink <= 90:
            # Shrinkage not critical
            return False

        # Check recent OFV trend
        if len(self.improvement_history) >= 2:
            recent_ofvs = [h.get('ofv') for h in self.improvement_history[-2:]]
            if all(o is not None for o in recent_ofvs):
                prev_ofv, current_ofv = recent_ofvs

                # If OFV got much worse (>30%), this is likely underparameterization
                # (we removed too much and model can't fit data anymore)
                if current_ofv > prev_ofv * 1.3:
                    shrink_val = avg_shrink
                    prev_val = prev_ofv
                    curr_val = current_ofv
                    if shrink_val is not None:
                        print(f"  [ANALYSIS] High shrinkage ({shrink_val:.1f}%) BUT OFV worsened")
                    else:
                        print(f"  [ANALYSIS] High shrinkage (N/A%) BUT OFV worsened")
                    print(f"  [ANALYSIS] OFV changed from {prev_val:.1f} to {curr_val:.1f}")
                    print(f"  [ANALYSIS] This suggests UNDERPARAMETERIZATION, not overfitting")
                    return False

        # If OFV very negative, definitely overfitting
        if ofv is not None and ofv < -50:
            return True

        # High shrinkage without OFV worsening -> likely overfitting
        return True

    def _determine_current_phase(self, parser: Optional[NONMEMParser]) -> ModelPhase:
        """
        Determine current optimization phase based on model state

        Uses PhaseTransitionManager for cleaner, explicit state machine logic

        Phase progression:
        1. ESTABLISH_BASE: Until minimization successful
        2. DIAGNOSE_STRUCTURE: Check if compartment model is adequate
        3. REDUCE_OVERFITTING: If TRUE overfitting detected (not underparameterization)
        4. OPTIMIZE_IIV: Fine-tune random effects
        5. COVARIATE_ANALYSIS: If base model stable and covariates available

        CRITICAL: Phases can ONLY move forward, never backward!
        - Prevents infinite loops (Phase 4 → Phase 1 → Phase 4 ...)
        - Once a phase is complete, it stays complete
        - If issues occur in later phases, fix within current phase

        CRITICAL: For very small datasets (N<20), avoid Phase 2 (DIAGNOSE_STRUCTURE)
        after successful minimization, as structural changes often break small-sample models.

        Args:
            parser: NONMEM output parser

        Returns:
            Current model phase (always >= self.current_phase)
        """
        if parser is None:
            # Can't determine phase without output - stay in current phase
            return self.current_phase

        # Initialize phase manager on first use
        if self.phase_manager is None:
            self.phase_manager = PhaseTransitionManager(
                self.data_loader,
                self.current_phase,
                self.iterations_in_phase,
                self.improvement_history
            )

        # Update phase manager state
        self.phase_manager.current_phase = self.current_phase
        self.phase_manager.iterations_in_phase = self.iterations_in_phase

        # Get parsed results
        parsed_results = parser.get_parsed_data() if parser else {}

        # Determine next phase using clean state machine logic
        next_phase = self.phase_manager.determine_next_phase(parsed_results)

        return next_phase

    def _update_phase(self, new_phase: ModelPhase):
        """
        Update current phase and track transition

        CRITICAL: Only allows FORWARD phase transitions!
        - Prevents backward movement (e.g., Phase 4 → Phase 1)
        - Prevents infinite loops
        - Ensures systematic progression
        """
        # SAFETY CHECK: Prevent backward phase transitions
        if new_phase.value < self.current_phase.value:
            print(f"\n{'⚠'*70}")
            print(f"[WARNING] Attempted backward phase transition blocked!")
            print(f"[WARNING] Current: {self.current_phase}, Requested: {new_phase}")
            print(f"[WARNING] Phases can only move FORWARD. Staying in {self.current_phase}")
            print(f"{'⚠'*70}\n")
            # Stay in current phase
            self.iterations_in_phase += 1
            return

        if new_phase != self.current_phase:
            print(f"\n{'='*70}")
            print(f"PHASE TRANSITION: {self.current_phase} -> {new_phase}")
            print(f"{'='*70}")
            self.phase_history.append({
                'iteration': self.iteration,
                'from_phase': self.current_phase,
                'to_phase': new_phase,
                'iterations_in_previous_phase': self.iterations_in_phase
            })
            self.current_phase = new_phase
            self.iterations_in_phase = 0
        else:
            self.iterations_in_phase += 1

    def _should_revert_to_best(self, current_composite: float, current_ofv: Optional[float]) -> bool:
        """
        Check if we should revert to best iteration

        Revert if:
        1. Current model is 2x worse than best (by composite score)
        2. OFV increased by >2x
        3. We've been getting worse for 2+ consecutive iterations

        Args:
            current_composite: Current composite quality score
            current_ofv: Current objective function value

        Returns:
            True if should revert to best model
        """
        # Need at least best model and 2 iterations to compare
        if not self.best_code or len(self.improvement_history) < 2:
            return False

        # Don't revert too early
        if self.iteration - self.best_iteration < 2:
            return False

        # Check 1: Composite score 2x worse
        if self.best_composite_score != float('inf'):
            if current_composite > self.best_composite_score * 2:
                print(f"  [REVERT TRIGGER] Composite score 2x worse than best")
                return True

        # Check 2: OFV doubled
        if self.best_ofv is not None and current_ofv is not None:
            if current_ofv > self.best_ofv * 2:
                print(f"  [REVERT TRIGGER] OFV doubled from best")
                return True

        # Check 3: Consistent deterioration for 2+ iterations
        if len(self.improvement_history) >= 3:
            recent_scores = [h.get('composite_score', float('inf'))
                           for h in self.improvement_history[-3:]]
            # All getting worse
            if (recent_scores[0] < recent_scores[1] < recent_scores[2] and
                recent_scores[2] > self.best_composite_score * 1.5):
                print(f"  [REVERT TRIGGER] Consistent deterioration for 3 iterations")
                return True

        return False

    def _should_stop_early(self) -> tuple[bool, str]:
        """
        Check if optimization should stop early due to CATASTROPHIC persistent issues

        Philosophy: Let the model keep trying to improve. Only stop if truly hopeless.

        Returns:
            (should_stop, reason) tuple
        """
        if len(self.improvement_history) < 5:
            # Don't stop before 5 iterations - give it a real chance
            return False, ""

        # Check last N iterations for persistent problems
        recent_n = 6
        recent_history = self.improvement_history[-recent_n:]

        # 1. CATASTROPHIC SHRINKAGE: >95% for 4+ consecutive iterations
        #    (This means model is truly hopeless, not just suboptimal)
        catastrophic_shrink = [h for h in recent_history[-4:]
                              if h.get('avg_eta_shrinkage') is not None
                              and h.get('avg_eta_shrinkage') > 95]
        if len(catastrophic_shrink) >= 4:
            reason = (f"ETA shrinkage >95% (catastrophic) for {len(catastrophic_shrink)} consecutive iterations. "
                     "The model has completely failed to estimate individual variability despite multiple attempts. "
                     "Dataset may be too small or model fundamentally misspecified.")
            return True, reason

        # 2. PERSISTENT NEGATIVE OFV: Negative for 4+ consecutive iterations
        #    (Clear sign of severe overfitting that won't resolve)
        negative_ofv = [h for h in recent_history[-4:]
                       if h.get('ofv') is not None and h.get('ofv') < -50]
        if len(negative_ofv) >= 4:
            reason = (f"Negative OFV (<-50) for {len(negative_ofv)} consecutive iterations. "
                     "Severe overfitting persists despite attempts to fix. Model is fitting noise.")
            return True, reason

        # 3. TOTAL COLLAPSE: All OMEGAs collapsed for 3+ iterations
        #    (Not just some, but ALL OMEGAs collapsed = hopeless)
        recent_with_omega = [h for h in recent_history[-3:]
                           if h.get('omega_values') is not None and len(h.get('omega_values', [])) > 0]
        if len(recent_with_omega) >= 3:
            all_collapsed_count = 0
            for entry in recent_with_omega:
                omega_values = entry.get('omega_values', [])
                collapsed = [o for o in omega_values if o < 0.001]
                if len(collapsed) == len(omega_values) and len(omega_values) > 0:
                    all_collapsed_count += 1

            if all_collapsed_count >= 3:
                reason = (f"ALL OMEGA parameters collapsed (<0.001) for {all_collapsed_count} consecutive iterations. "
                         f"Individual variability structure is completely lost and cannot be recovered.")
                return True, reason

        # 4. NO PROGRESS AT ALL: Score getting worse for 5+ iterations
        #    (Model is actively degrading, not improving)
        if len(self.improvement_history) >= 6:
            recent_scores = [h.get('composite_score', float('inf'))
                           for h in self.improvement_history[-6:]]
            if all(score != float('inf') for score in recent_scores):
                # Check if scores are monotonically increasing (getting worse)
                getting_worse = all(recent_scores[i] <= recent_scores[i+1]
                                   for i in range(len(recent_scores)-1))
                if getting_worse and recent_scores[-1] > recent_scores[0] * 1.5:
                    reason = (f"Composite score has been degrading for 6 consecutive iterations "
                             f"({recent_scores[0]:.1f} -> {recent_scores[-1]:.1f}). "
                             f"Model quality is getting worse, not better.")
                    return True, reason

        # OTHERWISE: Keep trying! Don't give up too easily.
        return False, ""

    def _evaluate_improvement(self, parser: Optional[NONMEMParser]) -> bool:
        """
        Evaluate if improvement occurred and should continue

        Args:
            parser: NONMEM output parser

        Returns:
            True if should continue optimizing
        """
        if parser is None:
            self.improvement_history.append({
                'iteration': self.iteration,
                'status': 'failed',
                'ofv': None,
                'issues': ['Failed to parse output'],
                'changes': 'N/A',
                'composite_score': float('inf')
            })
            return True  # Try to fix

        parsed_data = parser.get_parsed_data()
        # Update parameter stabilization history based on latest successful run
        self._update_parameter_history(parsed_data)
        current_ofv = parsed_data.get('objective_function')
        minimization_ok = parsed_data.get('minimization_successful', False)
        issues = parser.get_issues()

        # Extract RSE and Shrinkage metrics
        rse_data = parsed_data.get('rse_percent', {})
        eta_shrinkage = parsed_data.get('eta_shrinkage', [])
        cov_step = parsed_data.get('covariance_step', {})
        cov_success = cov_step.get('successful', False)

        max_rse = rse_data.get('max_rse')
        high_rse_count = rse_data.get('high_rse_count', 0)
        avg_eta_shrinkage = sum([s['shrinkage'] for s in eta_shrinkage]) / len(eta_shrinkage) if eta_shrinkage else None

        # Extract OMEGA values for collapse detection
        params = parsed_data.get('parameter_estimates', {})
        omega_values = [float(o.get('estimate', 1.0)) for o in params.get('omega', []) if 'estimate' in o]

        # Calculate composite quality score
        print(f"\n{'='*70}")
        print("COMPOSITE QUALITY SCORE CALCULATION")
        print(f"{'='*70}")
        current_composite = self._calculate_composite_score(
            current_ofv, avg_eta_shrinkage, cov_success, minimization_ok, omega_values
        )
        print(f"Total Composite Score: {current_composite:.2f} (lower is better)")
        print(f"{'='*70}")

        # Get dataset size
        metadata = self.data_loader.get_metadata()
        n_subjects = metadata.get('n_subjects', 0)

        # Record history
        history_entry = {
            'iteration': self.iteration,
            'status': 'success' if minimization_ok else 'failed',
            'ofv': current_ofv,
            'max_rse': max_rse,
            'high_rse_count': high_rse_count,
            'avg_eta_shrinkage': avg_eta_shrinkage,
            'issues': issues,
            'minimization_successful': minimization_ok,
            'covariance_successful': cov_success,
            'composite_score': current_composite,
            'omega_values': omega_values,
            'n_subjects': n_subjects
        }

        self.improvement_history.append(history_entry)

        # Evaluate model quality using composite score
        if current_composite < self.best_composite_score:
            self.best_composite_score = current_composite
            self.best_ofv = current_ofv  # Keep for display
            self.best_iteration = self.iteration
            self.best_code = self.current_code  # Store best code for potential revert

            quality_parts = []
            comp_val = current_composite
            quality_parts.append(f"Composite={comp_val:.2f}")
            if current_ofv is not None:
                ofv_val = current_ofv
                quality_parts.append(f"OFV={ofv_val:.2f}")
            if avg_eta_shrinkage is not None:
                shrink_val = avg_eta_shrinkage
                if shrink_val is not None:
                    quality_parts.append(f"Shrink={shrink_val:.1f}%")
            quality_str = ', '.join(quality_parts)
            print(f"\n[OK] NEW BEST MODEL: {quality_str}")

        # Check if we should revert to best model
        elif self._should_revert_to_best(current_composite, current_ofv):
            print(f"\n{'='*70}")
            print("MODEL DETERIORATION DETECTED - REVERTING TO BEST")
            print(f"{'='*70}")
            current_ofv_str = f"{current_ofv:.2f}" if current_ofv is not None else "N/A"
            best_ofv_str = f"{self.best_ofv:.2f}" if self.best_ofv is not None else "N/A"
            print(f"Current: OFV={current_ofv_str}, Composite={current_composite:.2f}")
            print(f"Best (Iter {self.best_iteration}): OFV={best_ofv_str}, Composite={self.best_composite_score:.2f}")
            print(f"Action: Reverting to iteration {self.best_iteration} code")
            print(f"{'='*70}")

            # Revert to best code
            if self.best_code:
                self.current_code = self.best_code
                # Record revert in history
                history_entry['reverted_to_best'] = True
                history_entry['reverted_from_iteration'] = self.iteration

        # Check for early stopping conditions FIRST
        should_stop, stop_reason = self._should_stop_early()
        if should_stop:
            print(f"\n{'='*70}")
            print("EARLY STOPPING TRIGGERED")
            print(f"{'='*70}")
            print(f"Reason: {stop_reason}")
            print(f"{'='*70}")
            return False  # Stop optimization

        # ALWAYS perform AI quality evaluation first (for all cases)
        print("\n[INFO] Performing comprehensive AI quality evaluation...")
        should_continue_ai = self._ai_quality_check(parsed_data)

        # If AI evaluation succeeded, use its decision
        if should_continue_ai is not None:
            # Override: If AI says stop but model isn't actually good, keep going
            if not should_continue_ai:
                # Stricter quality checks before accepting stop decision

                # Check 1: Minimization must be successful
                if not minimization_ok:
                    print("\n[OVERRIDE] AI suggested stopping but minimization failed - continuing")
                    return True

                # Check 2: Shrinkage must be acceptable for dataset size
                metadata = self.data_loader.get_metadata()
                num_subjects = metadata.get('n_subjects', 100)

                shrinkage_threshold = 70 if num_subjects < 20 else 60 if num_subjects < 50 else 50

                if avg_eta_shrinkage is not None and avg_eta_shrinkage > shrinkage_threshold:
                    shrink_val = avg_eta_shrinkage
                    thresh_val = shrinkage_threshold
                    n_val = num_subjects
                    print(f"\n[OVERRIDE] AI suggested stopping but shrinkage too high")
                    if shrink_val is not None:
                        print(f"[OVERRIDE] Shrinkage {shrink_val:.1f}% > threshold {thresh_val}% for N={n_val}")
                    else:
                        print(f"[OVERRIDE] Shrinkage N/A% > threshold {thresh_val}% for N={n_val}")
                    print(f"[OVERRIDE] Continuing optimization")
                    return True

                # Check 3: Model must have reasonable quality score
                # Only stop if quality score is at least 60/100
                if self.improvement_history and 'ai_evaluation' in self.improvement_history[-1]:
                    last_eval = self.improvement_history[-1]['ai_evaluation']
                    quality_score = last_eval.get('quality_score', 0)
                    if quality_score < 60:
                        score_val = quality_score
                        print(f"\n[OVERRIDE] AI suggested stopping but quality score too low")
                        print(f"[OVERRIDE] Quality score {score_val}/100 is less than 60")
                        print(f"[OVERRIDE] Continuing optimization")
                        return True

                # Check 4: Must have tried enough iterations
                if self.iteration < 5:
                    iter_val = self.iteration
                    print(f"\n[OVERRIDE] AI suggested stopping but not enough iterations")
                    print(f"[OVERRIDE] Only {iter_val} iterations completed, need at least 5")
                    print(f"[OVERRIDE] Continuing optimization")
                    return True

                # All checks passed - model is genuinely good enough
                print("\n[ACCEPTED] AI suggests stopping and model quality is sufficient:")
                min_status = 'OK' if minimization_ok else 'FAILED'
                print(f"  - Minimization: {min_status}")
                shrink_val = avg_eta_shrinkage
                thresh_val = shrinkage_threshold
                if shrink_val is not None:
                    print(f"  - Shrinkage: {shrink_val:.1f}% (threshold: <{thresh_val}%)")
                else:
                    print(f"  - Shrinkage: N/A (threshold: <{thresh_val}%)")
                if self.improvement_history and 'ai_evaluation' in self.improvement_history[-1]:
                    last_eval = self.improvement_history[-1]['ai_evaluation']
                    quality_score = last_eval.get('quality_score', 0)
                    print(f"  - Quality Score: {quality_score}/100")
                iter_val = self.iteration
                print(f"  - Iterations: {iter_val}")
                return False
            else:
                # AI says continue - but check if we should override to STOP
                # CRITICAL: If only 1 OMEGA left and shrinkage is acceptable, must STOP
                if len(omega_values) == 1 and avg_eta_shrinkage is not None:
                    # For small datasets (N<20), shrinkage 50-75% is acceptable
                    # Further simplification would destroy the model
                    metadata = self.data_loader.get_metadata()
                    num_subjects = metadata.get('n_subjects', 100)

                    if num_subjects < 20 and avg_eta_shrinkage < 75:
                        n_val = num_subjects
                        shrink_val = avg_eta_shrinkage
                        print(f"\n{'='*70}")
                        print("[CRITICAL OVERRIDE] AI suggested CONTINUE, but STOPPING instead")
                        print(f"{'='*70}")
                        print(f"  Reason:")
                        print(f"    - Only 1 OMEGA parameter remains")
                        print(f"    - Dataset size: {n_val} subjects (small)")
                        if shrink_val is not None:
                            print(f"    - Shrinkage {shrink_val:.1f}% is ACCEPTABLE for N<20")
                        else:
                            print(f"    - Shrinkage N/A% is ACCEPTABLE for N<20")
                        print(f"    - Further simplification would remove last OMEGA")
                        print(f"    - Model would become non-population (NONMEM will fail)")
                        print(f"  Decision: STOP to preserve model viability")
                        print(f"{'='*70}")
                        return False

                # Otherwise trust AI's decision to continue
                return True

        # Fallback: Traditional decision logic if AI evaluation failed
        print("\n[WARNING] AI evaluation unavailable, using fallback logic...")

        # Default: Keep trying to improve
        if not minimization_ok:
            print("\n[WARNING] Minimization not successful - continuing optimization")
            return True

        if issues:
            print(f"\n[INFO] Found {len(issues)} issue(s) - continuing to address them")
            return True

        # Even if things look ok, keep going unless really converged
        if len(self.improvement_history) >= 2:
            prev_entry = self.improvement_history[-2]
            prev_ofv = prev_entry.get('ofv')

            if current_ofv is not None and prev_ofv is not None:
                ofv_change = current_ofv - prev_ofv
                print(f"\nOFV change: {ofv_change:.2f}")

                # Only stop if truly converged (very small change + good quality)
                if abs(ofv_change) < 0.1 and avg_eta_shrinkage is not None and avg_eta_shrinkage < 50:
                    print("  Model appears to have converged with good quality")
                    return False

        # Default: Continue optimization
        print("\n[DEFAULT] Continuing optimization")
        return True

    def _ai_quality_check(self, parsed_data: Dict) -> Optional[bool]:
        """
        Use AI to evaluate model quality comprehensively

        Args:
            parsed_data: Parsed NONMEM results dictionary

        Returns:
            True to continue optimization, False to stop, None if evaluation failed
        """
        import json

        # Pre-check: Detect obvious overfitting before AI call
        ofv = parsed_data.get('objective_function')
        eta_shrinkage = parsed_data.get('eta_shrinkage', [])
        avg_shrink = sum([s['shrinkage'] for s in eta_shrinkage]) / len(eta_shrinkage) if eta_shrinkage else None
        params = parsed_data.get('parameter_estimates', {})
        omega_values = [float(o.get('estimate', 1.0)) for o in params.get('omega', []) if 'estimate' in o]

        # Detect critical overfitting signals
        overfitting_warnings = []
        if ofv is not None and ofv < -50:
            overfitting_warnings.append(f"CRITICAL: Negative OFV ({ofv:.2f}) indicates overfitting")
        if avg_shrink is not None and avg_shrink > 95:
            overfitting_warnings.append(f"CRITICAL: Catastrophic shrinkage ({avg_shrink:.1f}%) - IIV lost")
        collapsed_omegas = [o for o in omega_values if o < 0.001]
        if len(collapsed_omegas) > 0:
            overfitting_warnings.append(f"CRITICAL: {len(collapsed_omegas)} OMEGA(s) collapsed - overparameterization")

        if overfitting_warnings:
            print(f"\n{'!'*70}")
            print("OVERFITTING DETECTED - IMMEDIATE ACTION REQUIRED")
            print(f"{'!'*70}")
            for warning in overfitting_warnings:
                print(f"  {warning}")
            print(f"{'!'*70}")

        try:
            prompt = PromptTemplates.quality_evaluation_prompt(
                iteration=self.iteration,
                parsed_data=parsed_data,
                previous_improvements=self.improvement_history,
                overfitting_warnings=overfitting_warnings  # Pass warnings to AI
            )

            # Use JSON mode for structured output
            response = self.gemini_client.generate(prompt, model_type=self.model)

            # Extract JSON from code block if present
            json_text = response.strip()

            # Remove markdown code block markers if present
            if json_text.startswith('```'):
                # Find the actual JSON content
                lines = json_text.split('\n')
                # Skip first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                json_text = '\n'.join(lines).strip()

            # Parse JSON response
            evaluation = json.loads(json_text)

            quality_score = evaluation.get('quality_score', 0)
            should_continue = evaluation.get('should_continue', True)
            grade = evaluation.get('model_grade', 'F')
            reason = evaluation.get('decision_reason', '')
            score_breakdown = evaluation.get('score_breakdown', {})
            critical_issues = evaluation.get('critical_issues', [])
            recommendations = evaluation.get('recommendations_if_continuing', [])

            # Display evaluation results
            print(f"\n{'=' * 70}")
            print("AI QUALITY EVALUATION")
            print(f"{'=' * 70}")
            print(f"Quality Score: {quality_score}/100")
            print(f"Model Grade: {grade}")
            print(f"\nScore Breakdown:")
            for criterion, score in score_breakdown.items():
                print(f"  - {criterion.capitalize()}: {score}/100")

            print(f"\nDecision: {'CONTINUE OPTIMIZATION' if should_continue else 'STOP - MODEL GOOD ENOUGH'}")
            print(f"Reason: {reason}")

            if critical_issues:
                print(f"\nCritical Issues:")
                for issue in critical_issues:
                    print(f"  - {issue}")

            if should_continue and recommendations:
                print(f"\nRecommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")

            print(f"{'=' * 70}")

            # Store evaluation in history
            if self.improvement_history:
                self.improvement_history[-1]['ai_evaluation'] = {
                    'quality_score': quality_score,
                    'grade': grade,
                    'should_continue': should_continue,
                    'reason': reason
                }

            return should_continue

        except json.JSONDecodeError as e:
            print(f"[WARNING] AI quality evaluation returned invalid JSON: {e}")
            if response:
                snippet = response[:200]
            else:
                snippet = 'N/A'
            print(f"  Response snippet: {snippet}...")
            return None

        except Exception as e:
            print(f"[WARNING] AI quality evaluation failed: {e}")
            return None

    def _detect_simplification_needed(self) -> tuple[bool, str]:
        """
        Detect if mandatory simplification is required based on history

        Returns:
            (is_mandatory, reason) tuple
        """
        if len(self.improvement_history) < 2:
            return False, ""

        # Check recent shrinkage history
        recent_shrinkage = []
        for entry in self.improvement_history[-3:]:
            shrink = entry.get('avg_eta_shrinkage')
            if shrink is not None:
                recent_shrinkage.append(shrink)

        # CRITICAL: 2+ consecutive iterations with >90% shrinkage
        extreme_shrink_count = sum(1 for s in recent_shrinkage if s > 90)
        if extreme_shrink_count >= 2:
            avg_shrink = sum(recent_shrinkage) / len(recent_shrinkage)
            reason = (
                f"MANDATORY SIMPLIFICATION: ETA shrinkage >{90}% for {extreme_shrink_count} consecutive iterations "
                f"(average: {avg_shrink:.1f}%). The model is severely overparameterized. "
                f"You MUST reduce model complexity - DO NOT just adjust boundaries or initial values."
            )
            return True, reason

        # Check for persistent covariance failures with high shrinkage
        cov_failures = [e for e in self.improvement_history[-3:]
                       if not e.get('covariance_successful', False)]
        if len(cov_failures) >= 2 and recent_shrinkage and recent_shrinkage[-1] > 70:
            reason = (
                f"MANDATORY SIMPLIFICATION: Covariance failed {len(cov_failures)} times "
                f"with shrinkage >{70}%. Model complexity exceeds data information content. "
                f"You MUST simplify the random effects structure."
            )
            return True, reason

        # Check for OMEGA collapse (informational only; do not force simplification)
        if self.improvement_history:
            last_entry = self.improvement_history[-1]
            omega_values = last_entry.get('omega_values', [])
            collapsed = [o for o in omega_values if o < 0.0001]
            if len(collapsed) >= len(omega_values) // 2 and len(omega_values) > 0:
                reason = (
                    f"SUGGESTED REVIEW: {len(collapsed)}/{len(omega_values)} OMEGA parameters "
                    f"are very small (<0.0001). This may indicate limited information for some random effects. "
                    f"Consider reviewing the structural and random-effects model; simplification of clearly "
                    f"redundant ETAs can be considered, but avoid blindly fixing OMEGA to zero."
                )
                print(f"[INFO] {reason}")

        return False, ""

    def _get_phase_specific_guidance(self) -> str:
        """
        Generate phase-specific guidance for the improvement prompt

        Returns:
            Focused guidance text for current phase
        """
        if self.current_phase == ModelPhase.ESTABLISH_BASE:
            return """
**CURRENT FOCUS: ESTABLISH BASE MODEL (Phase 1)**

Priority: Get minimization working
Actions ONLY:
1. Fix syntax errors (missing K=CL/V, S2=V, wrong $INPUT order)
2. Fix parameter boundaries (THETA out of bounds -> adjust or FIX)
3. Fix estimation convergence (try METHOD=ZERO if METHOD=1 INTER fails)
4. Ensure NONMEM executes without fatal errors

DO NOT:
- Change structural model (ADVAN)
- Add/remove OMEGA
- Add covariates
- Make complex changes

Goal: Minimization successful, even if quality is poor
"""
        elif self.current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            return """
**CURRENT FOCUS: DIAGNOSE STRUCTURAL MODEL (Phase 2)**

Priority: Check if compartment structure is adequate
Actions:
1. Review NONMEM output for systematic patterns
2. If residuals show bi-phasic decline -> Consider ADVAN2->ADVAN4 (2-compartment)
3. If residuals show U-shape -> Structure inadequate
4. Check error model adequacy (proportional vs combined)

Key indicators:
- Flat random residuals -> Structure OK, proceed to Phase 4
- Systematic bias -> Need structural change
- Funnel-shaped residuals -> Add proportional error component

DO NOT yet:
- Remove OMEGA (unless >90% shrinkage)
- Add covariates
"""
        elif self.current_phase == ModelPhase.REDUCE_OVERFITTING:
            return """
**CURRENT FOCUS: REDUCE OVERFITTING (Phase 3)**

CRITICAL: Model shows signs of overfitting - consider simplification

Priority: Reduce model complexity CAREFULLY
Suggested actions (check OFV after each change):
1. **IF OMEGA count >1**: Consider removing 1 OMEGA (highest shrinkage first)
   - WARNING: Only remove if OFV remains similar or improves
   - If OFV worsens by >30%, model may already be optimal

2. Simplify error model if using combined (additive+proportional)
   - Try proportional-only first

3. Consider fixing problematic THETA to typical value
   - If parameter hitting upper/lower bound repeatedly

4. Try METHOD=ZERO if using METHOD=1 INTER
   - More robust for small datasets

Overfitting indicators:
- Shrinkage >90% AND model stable/improving
- OFV <-50 (extremely negative)
- OMEGA <0.001 (collapsed)

**CRITICAL CHECK**: After simplification, compare OFV to previous iteration:
- OFV similar/better -> Simplification successful ✓
- OFV much worse (>30%) -> Simplification too aggressive, revert strategy ✗

IMPORTANT: Keep at least 1 OMEGA for population model
"""
        elif self.current_phase == ModelPhase.OPTIMIZE_IIV:
            return """
**CURRENT FOCUS: OPTIMIZE RANDOM EFFECTS (Phase 4)**

Priority: Fine-tune IIV structure
Actions:
1. Check shrinkage for each ETA
   - >70%: Consider removing if OMEGA>1
   - <50%: Good, retain
2. Optimize OMEGA structure
   - For N<30: Keep DIAGONAL
   - For N>30: Can try BLOCK if warranted
3. Adjust THETA bounds if parameters near boundaries
4. Fine-tune estimation method if needed

Goal: Shrinkage <50-60% with stable estimates
"""
        elif self.current_phase == ModelPhase.COVARIATE_ANALYSIS:
            return """
**CURRENT FOCUS: COVARIATE ANALYSIS (Phase 5)**

Prerequisites met: Base model stable, shrinkage good

Priority: Add covariates to explain variability
Actions:
1. Add ONE covariate at a time
2. Test physiologically plausible relationships:
   - Weight on CL: TVCL = THETA(1) * (WT/70)**THETA(X)
   - Age on V: TVV = THETA(2) * (1 + THETA(X)*(AGE-40))
   - Sex on CL: IF(SEX.EQ.1) TVCL = TVCL * THETA(X)
3. Check if OFV improves by >3.84 (p<0.05)
4. Verify parameter estimates are reasonable

DO NOT:
- Add multiple covariates at once
- Add covariates without physiological justification
- Overfit with too many covariates
"""
        else:
            return ""



    def _update_parameter_history(self, parsed_data: Dict) -> None:
        """Store THETA/OMEGA/SIGMA estimates for stabilization across iterations.

        This function does NOT change any decision logic. It only records the
        latest parameter estimates so that we can suggest narrower initial
        values/bounds for the next NONMEM run.
        """
        try:
            params = parsed_data.get('parameter_estimates', {}) or {}
        except Exception:
            return

        theta_list = params.get('theta', []) or []
        omega_list = params.get('omega', []) or []
        sigma_list = params.get('sigma', []) or []

        def _extract_vals(items):
            names = []
            vals = []
            for it in items:
                name = it.get('name') or ''
                est = it.get('estimate')
                try:
                    if est is None:
                        continue
                    v = float(est)
                except Exception:
                    continue
                names.append(str(name))
                vals.append(v)
            return names, vals

        th_names, th_vals = _extract_vals(theta_list)
        om_names, om_vals = _extract_vals(omega_list)
        sg_names, sg_vals = _extract_vals(sigma_list)

        if not (th_vals or om_vals or sg_vals):
            return

        entry = {
            'iteration': self.iteration,
            'theta_names': th_names,
            'theta_vals': th_vals,
            'omega_names': om_names,
            'omega_vals': om_vals,
            'sigma_names': sg_names,
            'sigma_vals': sg_vals,
        }
        self.parameter_history.append(entry)

    def _build_parameter_stabilization_guidance(self) -> str:
        """Build textual guidance to narrow THETA/OMEGA/SIGMA around previous estimates.

        This produces human-readable instructions that are appended to the
        NONMEM output given to the LLM so that the next control stream uses
        tighter initial values and boundaries, while keeping all other logic
        unchanged.
        """
        if not getattr(self, 'parameter_history', None):
            return ""

        last = self.parameter_history[-1]
        lines: list[str] = []
        lines.append("Use the following parameter estimates to tighten initial values and bounds")
        lines.append("for the NEXT model iteration (do NOT change structural model unless required).")
        lines.append("For each parameter, set the initial value close to the estimate and")
        lines.append("shrink the bounds to roughly 50-150% of the estimate (or positive-only for variances).")
        lines.append("")

        def _theta_bounds(v: float) -> tuple[float, float]:
            # Symmetric shrinkage around current estimate; keep wide enough to avoid trapping
            if v == 0.0:
                return -1.0, 1.0
            lower = v * 0.5
            upper = v * 1.5
            # Avoid collapsing very small parameters
            if 0 < abs(v) < 1e-3:
                lower = v * 0.1
                upper = v * 10.0
            return lower, upper

        def _var_bounds(v: float) -> tuple[float, float]:
            # Variances (OMEGA/SIGMA) must stay positive but can vary on log-scale
            base = max(v, 1e-6)
            lower = base * 0.3
            upper = base * 3.0
            return lower, upper

        th_names = last.get('theta_names') or []
        th_vals = last.get('theta_vals') or []
        if th_vals:
            lines.append("THETA (fixed effects):")
            for i, v in enumerate(th_vals, start=1):
                name = th_names[i-1] if i-1 < len(th_names) else f"THETA({i})"
                lo, hi = _theta_bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, recommended bounds ≈ [{lo:.6g}, {hi:.6g}]")
            lines.append("")

        om_names = last.get('omega_names') or []
        om_vals = last.get('omega_vals') or []
        if om_vals:
            lines.append("OMEGA (IIV variances):")
            for i, v in enumerate(om_vals, start=1):
                name = om_names[i-1] if i-1 < len(om_names) else f"OMEGA({i})"
                lo, hi = _var_bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, recommended bounds ≈ [{lo:.6g}, {hi:.6g}] (keep >0)")
            lines.append("")

        sg_names = last.get('sigma_names') or []
        sg_vals = last.get('sigma_vals') or []
        if sg_vals:
            lines.append("SIGMA (residual error variances):")
            for i, v in enumerate(sg_vals, start=1):
                name = sg_names[i-1] if i-1 < len(sg_names) else f"SIGMA({i})"
                lo, hi = _var_bounds(v)
                lines.append(f"  - {name}: estimate={v:.6g}, recommended bounds ≈ [{lo:.6g}, {hi:.6g}] (keep >0)")
            lines.append("")

        return "\n".join(lines)

    def _track_failed_strategy(self, code: str, error_type: str):
        """
        Track strategies that failed to prevent repeating them

        Args:
            code: The code that failed
            error_type: Type of error (e.g., "V3_ERROR", "MINIMIZATION_FAILED")
        """
        # Extract key characteristics
        advan_match = re.search(r'ADVAN(\d+)', code, re.IGNORECASE)
        advan = advan_match.group(1) if advan_match else "unknown"

        omega_count = 0
        omega_pattern = r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)'
        omega_match = re.search(omega_pattern, code, re.DOTALL | re.IGNORECASE)
        if omega_match:
            omega_section = omega_match.group(1)
            for line in omega_section.split('\n'):
                if ';' in line:
                    line = line.split(';')[0]
                line = line.strip()
                if line:
                    numbers = re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?', line)
                    omega_count += len(numbers)

        strategy_signature = f"ADVAN{advan}_OMEGA{omega_count}_{error_type}"

        if strategy_signature not in self.failed_strategies:
            self.failed_strategies.append(strategy_signature)
            print(f"  [TRACK] Recording failed strategy: {strategy_signature}")

    def _generate_improved_code(self, parser: Optional[NONMEMParser]):
        """Generate improved NONMEM code based on results and current phase"""
        print(f"\nGenerating improved model for iteration {self.iteration + 1}...")
        print(f"Phase-specific guidance: {self.current_phase}")

        # Get output text
        if parser:
            nonmem_output = parser.get_full_output()
            issues = parser.get_issues()

            # Track failed strategy if minimization failed
            parsed_data = parser.get_parsed_data()
            if not parsed_data.get('minimization_successful', False):
                self._track_failed_strategy(self.current_code, "MINIMIZATION_FAILED")
        else:
            output_file = f"{self.output_base}_iter{self.iteration}.lst"
            try:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    nonmem_output = f.read()

                # Check if this is a V3 error (ADVAN issue)
                if 'V3' in nonmem_output and 'ERROR' in nonmem_output.upper():
                    self._track_failed_strategy(self.current_code, "V3_ERROR")
            except:
                nonmem_output = "Could not read output file"
            issues = ["Output file could not be parsed properly"]

        # Gemini has 1M token context - can handle full NONMEM output
        # Only truncate if extremely large (>500KB = ~250K tokens)
        if len(nonmem_output) > 500000:
            output_len = len(nonmem_output)
            print(f"  [INFO] Large output detected")
            print(f"  [INFO] Output size: {output_len} chars, using smart truncation")
            nonmem_output = self._smart_truncate_output(nonmem_output, max_length=200000)

        # Check if mandatory simplification is needed
        simplification_required, simplification_reason = self._detect_simplification_needed()

        if simplification_required:
            print(f"\n{'!'*70}")
            print("MANDATORY SIMPLIFICATION REQUIRED")
            print(f"{'!'*70}")
            print(f"{simplification_reason}")
            print(f"{'!'*70}\n")

        # Count current OMEGA parameters to prevent destruction
        current_omega_count = 0
        # Match $OMEGA block (handles both "$OMEGA\n0.1" and "$OMEGA 0.1" formats)
        omega_pattern = r'\$OMEGA\s*(.*?)(?=\n\s*\$|\Z)'
        omega_match = re.search(omega_pattern, self.current_code, re.DOTALL | re.IGNORECASE)
        if omega_match:
            omega_section = omega_match.group(1)
            # Count numeric values (each OMEGA parameter)
            # Split by lines and look for numeric values
            for line in omega_section.split('\n'):
                # Remove comments
                if ';' in line:
                    line = line.split(';')[0]
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Count numbers in the line (each number = 1 OMEGA)
                # Match floats like 0.1, 1.0, 0.001, etc.
                numbers = re.findall(r'\d+\.?\d*(?:[eE][+-]?\d+)?', line)
                current_omega_count += len(numbers)

        # Get phase-specific guidance
        phase_guidance = self._get_phase_specific_guidance()

        # Generate failed strategies warning
        failed_strategies_warning = ""
        if self.failed_strategies:
            failed_strategies_warning = "\n\n" + "="*70 + "\n"
            failed_strategies_warning += "FAILED STRATEGIES - DO NOT REPEAT THESE\n"
            failed_strategies_warning += "="*70 + "\n"
            failed_strategies_warning += "The following model configurations have already been tried and FAILED:\n"
            for strategy in self.failed_strategies[-5:]:  # Show last 5 failed attempts
                failed_strategies_warning += f"  ✗ {strategy}\n"
            failed_strategies_warning += "\nIMPORTANT: Do NOT generate code matching these patterns.\n"
            failed_strategies_warning += "If you see ADVAN4_OMEGA3_V3_ERROR, do NOT use ADVAN4 again.\n"
            failed_strategies_warning += "Try a different approach instead.\n"
            failed_strategies_warning += "="*70 + "\n\n"

        # Get last 2 models from code history for context
        previous_models = []
        if len(self.code_history) >= 2:
            previous_models = self.code_history[-2:]
        elif len(self.code_history) == 1:
            previous_models = self.code_history[-1:]

        # Prepare parsed_results for v2 API
        if parser:
            parsed_results = parser.get_parsed_data()
        else:
            parsed_results = {
                'objective_function': None,
                'minimization_successful': False,
                'warnings': [],
                'eta_shrinkage': []
            }

        # Get metadata
        metadata = self.data_loader.get_metadata()

        # Generate improvement prompt with phase-aware routing (V2)
        print(f"  [INFO] Using phase-specific prompt: {self.current_phase}")

        # Build parameter stabilization guidance and append to NONMEM output.
        # This guides the LLM to tighten THETA/OMEGA/SIGMA bounds around the
        # latest estimates without changing the rest of the optimization logic.
        param_guidance = self._build_parameter_stabilization_guidance()
        if param_guidance:
            augmented_output = (
                nonmem_output
                + "\n\n" + "="*70 + "\n"
                + "AUTO-GENERATED PARAMETER STABILIZATION GUIDANCE (FOR THETA/OMEGA/SIGMA)\n"
                + "="*70 + "\n"
                + param_guidance
            )
        else:
            augmented_output = nonmem_output

        prompt = PromptTemplates.improvement_prompt_v2(
            iteration=self.iteration,
            current_code=self.current_code,
            nonmem_output=augmented_output,
            parsed_results=parsed_results,
            current_phase=self.current_phase,  # NEW: Phase routing
            metadata=metadata,
            previous_improvements=self.improvement_history,
            issues_found=issues
        )

        # Prepend failed strategies warning if any
        if failed_strategies_warning:
            prompt = failed_strategies_warning + "\n\n" + prompt

        response = self.gemini_client.generate(prompt)

        # Extract improved code
        self.current_code = self._extract_nonmem_code(response)

        # Extract analysis and changes
        analysis = self._extract_section(response, 'ANALYSIS')
        changes = self._extract_section(response, 'CHANGES MADE')

        if analysis:
            print(f"\nAnalysis: {analysis}")
        if changes:
            print(f"Changes: {changes}")

        # Store code in history
        self.code_history.append({
            'iteration': self.iteration + 1,
            'code': self.current_code,
            'description': changes or 'Not specified'
        })

        # Update history with changes
        if self.improvement_history:
            self.improvement_history[-1]['changes'] = changes or 'Not specified'

        print(f"[OK] Improved code generated for next iteration")

    def _smart_truncate_output(self, output: str, max_length: int = 8000) -> str:
        """
        Intelligently truncate NONMEM output to keep most important parts

        Priority:
        1. Error messages (first 2000 chars)
        2. Final parameter estimates (search for "FINAL")
        3. Objective function value
        4. Middle section if space remains
        """
        if len(output) <= max_length:
            return output

        # Always keep the beginning (errors usually here)
        head_size = min(2000, max_length // 2)
        head = output[:head_size]

        # Try to find and keep final parameter estimates
        final_match = re.search(r'FINAL PARAMETER ESTIMATE.*?(?=\n\s*\n|\Z)', output, re.DOTALL | re.IGNORECASE)

        if final_match:
            final_section = final_match.group(0)
            remaining = max_length - head_size - len(final_section) - 200

            if remaining > 0:
                # Include some middle context
                middle_start = head_size
                middle_end = middle_start + remaining
                middle = output[middle_start:middle_end]

                return f"{head}\n\n... [middle section truncated] ...\n\n{middle}\n\n{final_section}"
            else:
                return f"{head}\n\n... [truncated] ...\n\n{final_section}"
        else:
            # No final estimates found, keep head and tail
            tail_size = min(1000, max_length - head_size)
            tail = output[-tail_size:]
            return f"{head}\n\n... [middle section truncated ({len(output) - head_size - tail_size} chars)] ...\n\n{tail}"

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """Extract a section from response text"""
        pattern = rf'{section_name}:\s*(.+?)(?:\n\n|\n[A-Z]{{2,}}:|\Z)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _generate_final_summary(self) -> Dict:
        """Generate final optimization summary"""
        print("\n" + "=" * 70)
        print("OPTIMIZATION COMPLETE")
        print("=" * 70)

        print(f"\nTotal iterations: {self.iteration}")
        print(f"Best iteration: {self.best_iteration} (selected by composite score)")
        if self.best_composite_score is not None and self.best_composite_score != float('inf'):
            print(f"Best composite score: {self.best_composite_score:.2f} (lower is better)")
        if self.best_ofv is not None:
            print(f"Best OFV: {self.best_ofv:.2f}")

        # Phase progression summary
        if self.phase_history:
            print(f"\n{'='*70}")
            print("PHASE PROGRESSION")
            print(f"{'='*70}")
            for transition in self.phase_history:
                print(
                    f"Iteration {transition['iteration']}: "
                    f"{transition['from_phase']} -> {transition['to_phase']} "
                    f"(spent {transition['iterations_in_previous_phase']} iterations in previous phase)"
                )
            print(f"Final phase: {self.current_phase}")
            print(f"{'='*70}")

        print(f"\n{'='*70}")
        print("ITERATION HISTORY (sorted by quality)")
        print(f"{'='*70}")
        print(f"{'Iter':<6} {'Status':<8} {'OFV':<12} {'Composite':<12} {'Shrink':<10} {'CoV':<6}")
        print(f"{'-'*70}")

        for entry in self.improvement_history:
            # Extract values safely
            status_icon = "OK" if entry.get('minimization_successful') else "FAIL"

            ofv_val = entry.get('ofv')
            if ofv_val is not None:
                ofv_text = f"{ofv_val:.2f}"
            else:
                ofv_text = "N/A"

            comp_val = entry.get('composite_score', float('inf'))
            if comp_val != float('inf'):
                comp_text = f"{comp_val:.1f}"
            else:
                comp_text = "N/A"

            shrink_val = entry.get('avg_eta_shrinkage')
            if shrink_val is not None:
                shrink_text = f"{shrink_val:.1f}%"
            else:
                shrink_text = "N/A"

            cov_text = "Yes" if entry.get('covariance_successful') else "No"

            iter_num = entry.get('iteration')
            if iter_num == self.best_iteration:
                is_best = " ⭐"
            else:
                is_best = ""

            print(f"{iter_num:<6} {status_icon:<8} {ofv_text:<12} {comp_text:<12} {shrink_text:<10} {cov_text:<6}{is_best}")

        print(f"{'-'*70}")

        # Quality assessment of best model
        best_entry = next((e for e in self.improvement_history if e['iteration'] == self.best_iteration), None)
        if best_entry:
            print(f"\n{'='*70}")
            print("BEST MODEL QUALITY ASSESSMENT")
            print(f"{'='*70}")

            shrink = best_entry.get('avg_eta_shrinkage')
            if shrink is not None:
                shrink_val = shrink
                if shrink < 30:
                    if shrink_val is not None:
                        print(f"ETA Shrinkage: {shrink_val:.1f}% - EXCELLENT ✓")
                    else:
                        print(f"ETA Shrinkage: N/A - EXCELLENT ✓")
                elif shrink < 50:
                    if shrink_val is not None:
                        print(f"ETA Shrinkage: {shrink_val:.1f}% - GOOD ✓")
                    else:
                        print(f"ETA Shrinkage: N/A - GOOD ✓")
                elif shrink < 70:
                    if shrink_val is not None:
                        print(f"ETA Shrinkage: {shrink_val:.1f}% - ACCEPTABLE ⚠")
                    else:
                        print(f"ETA Shrinkage: N/A - ACCEPTABLE ⚠")
                elif shrink < 90:
                    if shrink_val is not None:
                        print(f"ETA Shrinkage: {shrink_val:.1f}% - CONCERNING ⚠")
                    else:
                        print(f"ETA Shrinkage: N/A - CONCERNING ⚠")
                else:
                    if shrink_val is not None:
                        print(f"ETA Shrinkage: {shrink_val:.1f}% - CRITICAL ✗ (overfitting)")
                    else:
                        print(f"ETA Shrinkage: N/A - CRITICAL ✗ (overfitting)")

            if best_entry.get('covariance_successful'):
                print("Covariance Step: SUCCESS ✓")
            else:
                print("Covariance Step: FAILED ✗")

            ofv = best_entry.get('ofv')
            if ofv is not None:
                ofv_val = ofv
                if ofv < -50:
                    print(f"OFV: {ofv_val:.2f} - NEGATIVE (overfitting suspected) ✗")
                elif ofv < 0:
                    print(f"OFV: {ofv_val:.2f} - Small negative (check model) ⚠")
                else:
                    print(f"OFV: {ofv_val:.2f} - POSITIVE ✓")

            print(f"{'='*70}")

        # Save final model
        final_file = f"{self.output_base}_final.txt"
        best_file = f"{self.output_base}_iter{self.best_iteration}.txt"

        if os.path.exists(best_file):
            import shutil
            shutil.copy(best_file, final_file)
            print(f"\n[OK] Best model saved to: {final_file}")

        print("=" * 70 + "\n")

        return {
            'total_iterations': self.iteration,
            'best_iteration': self.best_iteration,
            'best_ofv': self.best_ofv,
            'history': self.improvement_history,
            'final_file': final_file if os.path.exists(best_file) else None
        }
