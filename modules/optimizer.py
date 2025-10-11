"""
Recursive NONMEM Optimization Engine
Iteratively generates and improves NONMEM control stream files
"""

import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple
import time

from .gemini_client import MultiModelGeminiClient
from .data_loader import PKDataLoader
from .prompt_templates import PromptTemplates
from .nonmem_parser import NONMEMParser


class NONMEMOptimizer:
    """Recursive optimizer for NONMEM models"""

    def __init__(
        self,
        data_file: str,
        output_base: str,
        api_key: Optional[str] = None,
        min_iterations: int = 3,
        max_iterations: int = 10,
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
        print(f"Using model: {self.gemini_client.clients[model].get_current_model()}")

        # Optimization state
        self.iteration = 0
        self.current_code = None
        self.improvement_history = []
        self.best_ofv = None
        self.best_iteration = 0

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

        # Step 2: Recursive improvement loop
        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {self.iteration}/{self.max_iterations}")
            print("=" * 70)

            # Run NONMEM
            success = self._run_nonmem()

            if success:
                # Parse results
                parsed_results = self._parse_results()

                # Check for improvement
                should_continue = self._evaluate_improvement(parsed_results)

                # Decide whether to continue
                if self.iteration >= self.min_iterations and not should_continue:
                    print("\n[OK] Optimization converged successfully!")
                    break

                # Generate improved code
                if self.iteration < self.max_iterations:
                    self._generate_improved_code(parsed_results)
            else:
                # NONMEM failed - try to fix
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
        with open(iteration_file, 'w') as f:
            f.write(self.current_code)

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

    def _run_nonmem(self) -> bool:
        """
        Execute NONMEM

        Returns:
            True if execution completed (even with errors), False if command failed
        """
        input_file = f"{self.output_base}_iter{self.iteration}.txt"
        output_file = f"{self.output_base}_iter{self.iteration}.lst"

        # Write current code to input file
        with open(input_file, 'w') as f:
            # Update $DATA line to point to actual data file
            code = self.current_code
            # Find $DATA line and replace filename
            code = re.sub(
                r'\$DATA\s+\S+',
                f'$DATA {os.path.basename(self.data_file)}',
                code,
                flags=re.IGNORECASE
            )
            f.write(code)

        print(f"\nExecuting NONMEM: {self.nmfe_command} {input_file} {output_file}")

        try:
            # Check if NONMEM command exists
            result = subprocess.run(
                [self.nmfe_command, input_file, output_file],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=os.path.dirname(os.path.abspath(input_file)) or '.'
            )

            print(f"[OK] NONMEM execution completed (exit code: {result.returncode})")

            # Check if output file was created
            if os.path.exists(output_file):
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
        with open(output_file, 'w') as f:
            f.write(mock_output)

    def _parse_results(self) -> Optional[NONMEMParser]:
        """Parse NONMEM output file"""
        output_file = f"{self.output_base}_iter{self.iteration}.lst"

        try:
            parser = NONMEMParser(output_file)
            print("\n" + parser.get_summary())
            return parser

        except Exception as e:
            print(f"[WARNING] Error parsing NONMEM output: {e}")
            return None

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
                'changes': 'N/A'
            })
            return True  # Try to fix

        parsed_data = parser.get_parsed_data()
        current_ofv = parsed_data.get('objective_function')
        minimization_ok = parsed_data.get('minimization_successful', False)
        issues = parser.get_issues()

        # Record history
        history_entry = {
            'iteration': self.iteration,
            'status': 'success' if minimization_ok else 'failed',
            'ofv': current_ofv,
            'issues': issues,
            'minimization_successful': minimization_ok
        }

        self.improvement_history.append(history_entry)

        # Update best OFV
        if current_ofv is not None and minimization_ok:
            if self.best_ofv is None or current_ofv < self.best_ofv:
                self.best_ofv = current_ofv
                self.best_iteration = self.iteration
                print(f"\n[OK] NEW BEST MODEL: OFV = {self.best_ofv:.2f}")

        # Decision logic
        if not minimization_ok:
            print("\n[WARNING] Minimization not successful - continuing optimization")
            return True

        if issues:
            print(f"\n[WARNING] Found {len(issues)} issue(s) - may continue optimization")
            return True

        # Check OFV improvement
        if len(self.improvement_history) >= 2:
            prev_entry = self.improvement_history[-2]
            prev_ofv = prev_entry.get('ofv')

            if current_ofv is not None and prev_ofv is not None:
                ofv_change = current_ofv - prev_ofv
                print(f"\nOFV change: {ofv_change:.2f}")

                if abs(ofv_change) < 0.1:
                    print("  OFV change < 0.1 - model may have converged")
                    if not issues:
                        return False  # Converged!

        return True  # Continue by default

    def _generate_improved_code(self, parser: Optional[NONMEMParser]):
        """Generate improved NONMEM code based on results"""
        print(f"\nGenerating improved model for iteration {self.iteration + 1}...")

        # Get output text
        if parser:
            nonmem_output = parser.get_full_output()
            issues = parser.get_issues()
        else:
            output_file = f"{self.output_base}_iter{self.iteration}.lst"
            try:
                with open(output_file, 'r', encoding='utf-8', errors='ignore') as f:
                    nonmem_output = f.read()
            except:
                nonmem_output = "Could not read output file"
            issues = ["Output file could not be parsed properly"]

        # Generate improvement prompt
        prompt = PromptTemplates.improvement_prompt(
            iteration=self.iteration,
            dataset_info=self.data_loader.get_column_summary(),
            current_code=self.current_code,
            nonmem_output=nonmem_output[:5000],  # Limit size
            previous_improvements=self.improvement_history,
            issues_found=issues
        )

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

        # Update history with changes
        if self.improvement_history:
            self.improvement_history[-1]['changes'] = changes or 'Not specified'

        print(f"[OK] Improved code generated for next iteration")

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
        print(f"Best iteration: {self.best_iteration}")
        if self.best_ofv is not None:
            print(f"Best OFV: {self.best_ofv:.2f}")

        print("\nIteration history:")
        for entry in self.improvement_history:
            status_icon = "[OK]" if entry.get('minimization_successful') else "[ERROR]"
            ofv_text = f"{entry['ofv']:.2f}" if entry['ofv'] is not None else "N/A"
            print(f"  {status_icon} Iteration {entry['iteration']}: OFV={ofv_text}, Status={entry['status']}")

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
