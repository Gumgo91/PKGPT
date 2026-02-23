"""
NONMEM output parser
Extracts key information from NONMEM .lst output files
Uses Gemini API for robust parsing with regex fallback
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# Pydantic schemas for structured AI parsing
class ParameterEstimate(BaseModel):
    """Single parameter estimate"""
    index: int
    value: float

class MatrixParameterEstimate(BaseModel):
    """Matrix parameter (OMEGA/SIGMA)"""
    row: int
    col: int
    value: float

class ParameterEstimates(BaseModel):
    """All parameter estimates"""
    theta: List[ParameterEstimate] = Field(default_factory=list)
    omega: List[MatrixParameterEstimate] = Field(default_factory=list)
    sigma: List[MatrixParameterEstimate] = Field(default_factory=list)

class RSEItem(BaseModel):
    """RSE for single parameter"""
    index: int
    rse: float

class RSEMatrixItem(BaseModel):
    """RSE for matrix parameter"""
    row: int
    col: int
    rse: float

class RSEData(BaseModel):
    """RSE% data"""
    theta: List[RSEItem] = Field(default_factory=list)
    omega: List[RSEMatrixItem] = Field(default_factory=list)
    sigma: List[RSEMatrixItem] = Field(default_factory=list)
    max_rse: Optional[float] = None
    high_rse_count: int = 0

class ShrinkageItem(BaseModel):
    """Shrinkage for single ETA/EPS"""
    eta: Optional[int] = None
    eps: Optional[int] = None
    shrinkage: float

class CovarianceStep(BaseModel):
    """Covariance step results"""
    attempted: bool = False
    successful: bool = False
    issues: List[str] = Field(default_factory=list)

class NONMEMParsingResult(BaseModel):
    """Complete NONMEM parsing result from AI"""
    minimization_successful: bool = False
    objective_function: Optional[float] = None
    termination_status: str = "UNKNOWN"
    parameter_estimates: ParameterEstimates = Field(default_factory=ParameterEstimates)
    rse_percent: RSEData = Field(default_factory=RSEData)
    eta_shrinkage: List[ShrinkageItem] = Field(default_factory=list)
    eps_shrinkage: List[ShrinkageItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    covariance_step: CovarianceStep = Field(default_factory=CovarianceStep)
    condition_number: Optional[float] = None
    runtime: Optional[float] = None


class NONMEMParser:
    """Parse NONMEM output files and extract key information"""

    def __init__(self, output_file: str, gemini_client=None, use_ai_parsing: bool = True):
        """
        Initialize parser

        Args:
            output_file: Path to NONMEM .lst output file
            gemini_client: Optional GeminiClient instance for AI-based parsing
            use_ai_parsing: Whether to use AI parsing (default True, falls back to regex if fails)
        """
        self.output_file = output_file
        self.content = ""
        self.parsed_data = {}
        self.gemini_client = gemini_client
        self.use_ai_parsing = use_ai_parsing

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
        """Parse the NONMEM output - try AI first, fallback to regex"""
        # Try AI-based parsing first if enabled and client available
        if self.use_ai_parsing and self.gemini_client:
            try:
                print("  [INFO] Attempting AI-based parsing...")
                ai_result = self._parse_with_ai()
                if ai_result:
                    self.parsed_data = ai_result
                    print("  [OK] AI parsing successful")
                    return
                else:
                    print("  [WARNING] AI parsing returned empty result, falling back to regex")
            except Exception as e:
                print(f"  [WARNING] AI parsing failed ({str(e)}), falling back to regex")

        # Fallback to regex-based parsing
        print("  [INFO] Using regex-based parsing...")
        self.parsed_data = {
            'minimization_successful': self._check_minimization(),
            'objective_function': self._extract_ofv(),
            'termination_status': self._extract_termination_status(),
            'parameter_estimates': self._extract_parameters(),
            'rse_percent': self._extract_rse(),
            'eta_shrinkage': self._extract_eta_shrinkage(),
            'eps_shrinkage': self._extract_eps_shrinkage(),
            'warnings': self._extract_warnings(),
            'errors': self._extract_errors(),
            'covariance_step': self._check_covariance_step(),
            'condition_number': self._extract_condition_number(),
            'runtime': self._extract_runtime()
        }

    def _parse_with_ai(self) -> Optional[Dict]:
        """
        Parse NONMEM output using Gemini AI with structured output

        Returns:
            Parsed data dictionary or None if parsing failed
        """
        import google.generativeai as genai

        # Create parsing prompt
        prompt = f"""You are a NONMEM output parser. Parse the following NONMEM .lst file and extract all relevant information in JSON format.

IMPORTANT INSTRUCTIONS:
1. Extract Objective Function Value (OFV) from "MINIMUM VALUE OF OBJECTIVE FUNCTION" or "OBJECTIVE FUNCTION VALUE"
2. Check if minimization was successful (look for "MINIMIZATION SUCCESSFUL")
3. Extract final parameter estimates (THETA, OMEGA, SIGMA)
4. Extract RSE% (Relative Standard Error) from STANDARD ERROR OF ESTIMATE section
   - Calculate RSE% = (Standard Error / Estimate) * 100
   - Flag high RSE: >30% for THETA, >50% for OMEGA
5. Extract ETA shrinkage % and EPS shrinkage %
6. Extract warnings and errors
7. Check covariance step status
8. Extract condition number if available
9. Extract runtime if available

If a value is not found in the output, set it to null or empty list.

Return ONLY valid JSON in this exact structure:
{{
  "minimization_successful": boolean,
  "objective_function": number or null,
  "termination_status": "SUCCESSFUL" | "TERMINATED" | "ERROR" | "UNKNOWN",
  "parameter_estimates": {{
    "theta": [{{"index": int, "value": float}}, ...],
    "omega": [{{"row": int, "col": int, "value": float}}, ...],
    "sigma": [{{"row": int, "col": int, "value": float}}, ...]
  }},
  "rse_percent": {{
    "theta": [{{"index": int, "rse": float}}, ...],
    "omega": [{{"row": int, "col": int, "rse": float}}, ...],
    "sigma": [{{"row": int, "col": int, "rse": float}}, ...],
    "max_rse": number or null,
    "high_rse_count": int
  }},
  "eta_shrinkage": [{{"eta": int, "shrinkage": float}}, ...],
  "eps_shrinkage": [{{"eps": int, "shrinkage": float}}, ...],
  "warnings": [string, ...],
  "errors": [string, ...],
  "covariance_step": {{
    "attempted": boolean,
    "successful": boolean,
    "issues": [string, ...]
  }},
  "condition_number": number or null,
  "runtime": number or null
}}

NONMEM OUTPUT:
{self.content}
"""

        try:
            # Call Gemini with JSON response mode
            response = self.gemini_client.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type='application/json',
                )
            )

            # Parse JSON response
            result_json = json.loads(response.text)

            # Convert to our internal dictionary format
            parsed_data = self._convert_ai_result_to_dict(result_json)

            return parsed_data

        except Exception as e:
            print(f"  [ERROR] AI parsing exception: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_ai_result_to_dict(self, ai_result: Dict) -> Dict:
        """
        Convert AI parsing result (Pydantic format) to internal dictionary format

        Args:
            ai_result: Dictionary from AI parsing (JSON from Pydantic schema)

        Returns:
            Dictionary in internal format compatible with existing code
        """
        return {
            'minimization_successful': ai_result.get('minimization_successful', False),
            'objective_function': ai_result.get('objective_function'),
            'termination_status': ai_result.get('termination_status', 'UNKNOWN'),
            'parameter_estimates': ai_result.get('parameter_estimates', {'theta': [], 'omega': [], 'sigma': []}),
            'rse_percent': ai_result.get('rse_percent', {'theta': [], 'omega': [], 'sigma': [], 'max_rse': None, 'high_rse_count': 0}),
            'eta_shrinkage': ai_result.get('eta_shrinkage', []),
            'eps_shrinkage': ai_result.get('eps_shrinkage', []),
            'warnings': ai_result.get('warnings', []),
            'errors': ai_result.get('errors', []),
            'covariance_step': ai_result.get('covariance_step', {'attempted': False, 'successful': False, 'issues': []}),
            'condition_number': ai_result.get('condition_number'),
            'runtime': ai_result.get('runtime')
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
            r'#OBJV:\s*\*+\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',  # #OBJV:****  3215.442  ****
            r'MINIMUM VALUE OF OBJECTIVE FUNCTION\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
            r'OBJECTIVE FUNCTION VALUE\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
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

    def _extract_rse(self) -> Dict:
        """
        Extract Relative Standard Error % (RSE%)
        RSE% = (SE / Estimate) * 100
        Good model: RSE < 30% for fixed effects, RSE < 50% for random effects
        """
        rse_data = {
            'theta': [],
            'omega': [],
            'sigma': [],
            'max_rse': None,
            'high_rse_count': 0
        }

        # Look for STANDARD ERROR OF ESTIMATE section
        se_match = re.search(
            r'STANDARD ERROR OF ESTIMATE.*?(?=\n\s*\n|\Z)',
            self.content,
            re.DOTALL | re.IGNORECASE
        )

        if se_match:
            se_section = se_match.group(0)

            # Extract THETA RSE
            theta_matches = re.finditer(
                r'THETA\((\d+)\).*?:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
                se_section,
                re.IGNORECASE
            )
            for match in theta_matches:
                rse_value = float(match.group(2))
                rse_data['theta'].append({
                    'index': int(match.group(1)),
                    'rse': rse_value
                })
                if rse_value > 30:  # High RSE for fixed effects
                    rse_data['high_rse_count'] += 1

            # Extract OMEGA RSE
            omega_matches = re.finditer(
                r'OMEGA\((\d+),(\d+)\).*?:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)',
                se_section,
                re.IGNORECASE
            )
            for match in omega_matches:
                rse_value = float(match.group(3))
                rse_data['omega'].append({
                    'row': int(match.group(1)),
                    'col': int(match.group(2)),
                    'rse': rse_value
                })
                if rse_value > 50:  # High RSE for random effects
                    rse_data['high_rse_count'] += 1

            # Calculate max RSE
            all_rse = [item['rse'] for item in rse_data['theta']] + \
                      [item['rse'] for item in rse_data['omega']]
            if all_rse:
                rse_data['max_rse'] = max(all_rse)

        return rse_data

    def _extract_eta_shrinkage(self) -> List[Dict]:
        """
        Extract ETA shrinkage %
        Low shrinkage (<30%) is desirable for individual parameter estimates
        """
        shrinkage_data = []

        # Look for ETA SHRINKAGE section
        patterns = [
            r'ETAshrink.*?%.*?:\s*([-+]?\d+\.?\d*)',
            r'ETA.*?SHRINKAGE.*?:\s*([-+]?\d+\.?\d*)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, self.content, re.IGNORECASE)
            for i, match in enumerate(matches, 1):
                try:
                    shrinkage_value = float(match.group(1))
                    shrinkage_data.append({
                        'eta': i,
                        'shrinkage': shrinkage_value
                    })
                except:
                    pass

        return shrinkage_data

    def _extract_eps_shrinkage(self) -> List[Dict]:
        """
        Extract EPS shrinkage %
        High EPS shrinkage (>30%) suggests data are informative
        """
        shrinkage_data = []

        patterns = [
            r'EPSshrink.*?%.*?:\s*([-+]?\d+\.?\d*)',
            r'EPS.*?SHRINKAGE.*?:\s*([-+]?\d+\.?\d*)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, self.content, re.IGNORECASE)
            for i, match in enumerate(matches, 1):
                try:
                    shrinkage_value = float(match.group(1))
                    shrinkage_data.append({
                        'eps': i,
                        'shrinkage': shrinkage_value
                    })
                except:
                    pass

        return shrinkage_data

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

        # RSE%
        rse = self.parsed_data['rse_percent']
        if rse['max_rse'] is not None:
            lines.append(f"\nParameter Uncertainty:")
            lines.append(f"  Max RSE%: {rse['max_rse']:.1f}%")
            if rse['high_rse_count'] > 0:
                lines.append(f"  High RSE count: {rse['high_rse_count']} parameters")

        # ETA Shrinkage
        eta_shrink = self.parsed_data['eta_shrinkage']
        if eta_shrink:
            lines.append(f"\nETA Shrinkage:")
            for item in eta_shrink:
                status = "Good" if item['shrinkage'] < 30 else "High"
                lines.append(f"  ETA({item['eta']}): {item['shrinkage']:.1f}% [{status}]")

        # EPS Shrinkage
        eps_shrink = self.parsed_data['eps_shrinkage']
        if eps_shrink:
            lines.append(f"\nEPS Shrinkage:")
            for item in eps_shrink:
                lines.append(f"  EPS({item['eps']}): {item['shrinkage']:.1f}%")

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

    def diagnose_structural_model(self) -> Dict:
        """
        Diagnose structural model adequacy from residual patterns (Chapter 13, basic.pdf)

        Returns:
            Dictionary with structural diagnostics and recommendations
        """
        diagnostics = {
            'has_table_data': False,
            'residual_patterns': [],
            'structural_adequate': True,
            'recommendations': []
        }

        # Check for systematic patterns in residuals (would need table data)
        # Since we're parsing .lst file, we can only provide guidance
        # Real residual analysis would be done with table files (CWRES vs TIME/PRED)

        # Check for warnings about structural misspecification
        warnings = self.parsed_data.get('warnings', [])
        for warning in warnings:
            if any(keyword in warning.upper() for keyword in ['TREND', 'PATTERN', 'SYSTEMATIC']):
                diagnostics['residual_patterns'].append(warning)
                diagnostics['structural_adequate'] = False

        # Provide guidance based on common structural issues
        ofv = self.parsed_data.get('objective_function')
        if ofv is not None and ofv < 0:
            diagnostics['recommendations'].append(
                "Negative OFV suggests potential structural misspecification"
            )

        # Check if multi-phasic decline might indicate need for 2-cmt model
        if not diagnostics['structural_adequate']:
            diagnostics['recommendations'].append(
                "Check CWRES vs TIME plot for U-shaped or systematic patterns"
            )
            diagnostics['recommendations'].append(
                "U-shaped residuals → Consider changing 1-cmt to 2-cmt model (ADVAN2→ADVAN4)"
            )

        return diagnostics
