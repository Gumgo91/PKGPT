"""
NONMEM output parser
Extracts key information from NONMEM .lst output files
"""

import re
from typing import Dict, List, Optional, Tuple


class NONMEMParser:
    """Parse NONMEM output files and extract key information"""

    def __init__(self, output_file: str):
        """
        Initialize parser

        Args:
            output_file: Path to NONMEM .lst output file
        """
        self.output_file = output_file
        self.content = ""
        self.parsed_data = {}

        self._read_file()
        self._parse_output()

    def _read_file(self):
        """Read the output file"""
        try:
            with open(self.output_file, 'r', encoding='utf-8', errors='ignore') as f:
                self.content = f.read()
        except Exception as e:
            raise Exception(f"Failed to read output file: {e}")

    def _parse_output(self):
        """Parse the NONMEM output"""
        self.parsed_data = {
            'minimization_successful': self._check_minimization(),
            'objective_function': self._extract_ofv(),
            'termination_status': self._extract_termination_status(),
            'parameter_estimates': self._extract_parameters(),
            'warnings': self._extract_warnings(),
            'errors': self._extract_errors(),
            'covariance_step': self._check_covariance_step(),
            'condition_number': self._extract_condition_number(),
            'runtime': self._extract_runtime()
        }

    def _check_minimization(self) -> bool:
        """Check if minimization was successful"""
        success_patterns = [
            r'MINIMIZATION SUCCESSFUL',
            r'MINIMUM VALUE OF OBJECTIVE FUNCTION',
        ]

        for pattern in success_patterns:
            if re.search(pattern, self.content, re.IGNORECASE):
                return True

        return False

    def _extract_ofv(self) -> Optional[float]:
        """Extract objective function value"""
        patterns = [
            r'MINIMUM VALUE OF OBJECTIVE FUNCTION\s*:\s*([-+]?\d+\.?\d*)',
            r'OBJECTIVE FUNCTION VALUE\s*:\s*([-+]?\d+\.?\d*)',
            r'#OBJV:\s*([-+]?\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, self.content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass

        return None

    def _extract_termination_status(self) -> str:
        """Extract termination status"""
        if re.search(r'MINIMIZATION SUCCESSFUL', self.content, re.IGNORECASE):
            return "SUCCESSFUL"
        elif re.search(r'MINIMIZATION TERMINATED', self.content, re.IGNORECASE):
            return "TERMINATED"
        elif re.search(r'ERROR', self.content, re.IGNORECASE):
            return "ERROR"
        else:
            return "UNKNOWN"

    def _extract_parameters(self) -> Dict:
        """Extract parameter estimates"""
        params = {
            'theta': [],
            'omega': [],
            'sigma': []
        }

        # Try to find final parameter estimates section
        final_est_match = re.search(
            r'FINAL PARAMETER ESTIMATE.*?(?=\n\s*\n|\Z)',
            self.content,
            re.DOTALL | re.IGNORECASE
        )

        if final_est_match:
            section = final_est_match.group(0)

            # Extract THETA values
            theta_matches = re.finditer(
                r'THETA\((\d+)\)\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
                section,
                re.IGNORECASE
            )
            for match in theta_matches:
                params['theta'].append({
                    'index': int(match.group(1)),
                    'value': float(match.group(2))
                })

            # Extract OMEGA values
            omega_matches = re.finditer(
                r'OMEGA\((\d+),(\d+)\)\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
                section,
                re.IGNORECASE
            )
            for match in omega_matches:
                params['omega'].append({
                    'row': int(match.group(1)),
                    'col': int(match.group(2)),
                    'value': float(match.group(3))
                })

            # Extract SIGMA values
            sigma_matches = re.finditer(
                r'SIGMA\((\d+),(\d+)\)\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
                section,
                re.IGNORECASE
            )
            for match in sigma_matches:
                params['sigma'].append({
                    'row': int(match.group(1)),
                    'col': int(match.group(2)),
                    'value': float(match.group(3))
                })

        return params

    def _extract_warnings(self) -> List[str]:
        """Extract warning messages"""
        warnings = []

        # Common warning patterns
        warning_patterns = [
            r'WARNING[:\s]+(.*?)(?:\n|$)',
            r'PARAMETER.*?NEAR.*?BOUNDARY',
            r'ILL-CONDITIONED',
            r'ROUNDING ERRORS',
        ]

        for pattern in warning_patterns:
            matches = re.finditer(pattern, self.content, re.IGNORECASE)
            for match in matches:
                warning_text = match.group(0).strip()
                if warning_text and warning_text not in warnings:
                    warnings.append(warning_text)

        return warnings

    def _extract_errors(self) -> List[str]:
        """Extract error messages"""
        errors = []

        error_patterns = [
            r'ERROR[:\s]+(.*?)(?:\n|$)',
            r'FATAL ERROR',
            r'EXECUTION.*?FAILED',
        ]

        for pattern in error_patterns:
            matches = re.finditer(pattern, self.content, re.IGNORECASE)
            for match in matches:
                error_text = match.group(0).strip()
                if error_text and error_text not in errors:
                    errors.append(error_text)

        return errors

    def _check_covariance_step(self) -> Dict:
        """Check covariance step status"""
        result = {
            'attempted': False,
            'successful': False,
            'issues': []
        }

        if re.search(r'\$COV', self.content, re.IGNORECASE):
            result['attempted'] = True

            if re.search(r'COVARIANCE STEP.*?SUCCESSFUL', self.content, re.IGNORECASE):
                result['successful'] = True
            elif re.search(r'COVARIANCE STEP.*?ABORT', self.content, re.IGNORECASE):
                result['issues'].append('Covariance step aborted')

        return result

    def _extract_condition_number(self) -> Optional[float]:
        """Extract condition number"""
        match = re.search(
            r'CONDITION NUMBER\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
            self.content,
            re.IGNORECASE
        )

        if match:
            try:
                return float(match.group(1))
            except:
                pass

        return None

    def _extract_runtime(self) -> Optional[float]:
        """Extract runtime in seconds"""
        patterns = [
            r'Elapsed.*?time.*?(\d+\.?\d*)\s*second',
            r'TOTAL.*?TIME.*?(\d+\.?\d*)',
        ]

        for pattern in patterns:
            match = re.search(pattern, self.content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass

        return None

    def get_summary(self) -> str:
        """
        Get human-readable summary of NONMEM output

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("NONMEM RUN SUMMARY")
        lines.append("=" * 60)

        # Status
        status = "[OK] SUCCESS" if self.parsed_data['minimization_successful'] else "[ERROR] FAILED"
        lines.append(f"Status: {status} ({self.parsed_data['termination_status']})")

        # OFV
        if self.parsed_data['objective_function'] is not None:
            lines.append(f"Objective Function Value: {self.parsed_data['objective_function']:.2f}")
        else:
            lines.append("Objective Function Value: Not available")

        # Covariance
        cov = self.parsed_data['covariance_step']
        if cov['attempted']:
            cov_status = "[OK] Successful" if cov['successful'] else "[ERROR] Failed"
            lines.append(f"Covariance Step: {cov_status}")

        # Condition number
        if self.parsed_data['condition_number'] is not None:
            lines.append(f"Condition Number: {self.parsed_data['condition_number']:.2e}")

        # Runtime
        if self.parsed_data['runtime'] is not None:
            lines.append(f"Runtime: {self.parsed_data['runtime']:.1f} seconds")

        # Warnings
        if self.parsed_data['warnings']:
            lines.append(f"\nWarnings ({len(self.parsed_data['warnings'])}):")
            for warning in self.parsed_data['warnings'][:5]:  # Limit to 5
                lines.append(f"  - {warning}")

        # Errors
        if self.parsed_data['errors']:
            lines.append(f"\nErrors ({len(self.parsed_data['errors'])}):")
            for error in self.parsed_data['errors'][:5]:  # Limit to 5
                lines.append(f"  - {error}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def get_issues(self) -> List[str]:
        """
        Get list of issues that need attention

        Returns:
            List of issue descriptions
        """
        issues = []

        if not self.parsed_data['minimization_successful']:
            issues.append("Minimization did not complete successfully")

        if self.parsed_data['errors']:
            issues.append(f"Found {len(self.parsed_data['errors'])} error(s)")

        if self.parsed_data['warnings']:
            issues.append(f"Found {len(self.parsed_data['warnings'])} warning(s)")

        cov = self.parsed_data['covariance_step']
        if cov['attempted'] and not cov['successful']:
            issues.append("Covariance step failed")

        condition_num = self.parsed_data['condition_number']
        if condition_num is not None and condition_num > 1000:
            issues.append(f"High condition number ({condition_num:.2e}) indicates ill-conditioning")

        return issues

    def get_parsed_data(self) -> Dict:
        """Get complete parsed data dictionary"""
        return self.parsed_data

    def get_full_output(self) -> str:
        """Get the full NONMEM output text"""
        return self.content
