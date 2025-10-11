"""
Prompt templates for NONMEM code generation and optimization
"""

from typing import Dict, List, Optional


class PromptTemplates:
    """Templates for generating prompts for Gemini API"""

    @staticmethod
    def initial_generation_prompt(
        dataset_info: str,
        data_summary: str,
        columns: List[str],
        nonmem_columns: Dict[str, str],
        covariates: List[str]
    ) -> str:
        """
        Generate initial NONMEM code generation prompt

        Args:
            dataset_info: Dataset column information
            data_summary: Summary statistics
            columns: List of all columns
            nonmem_columns: Mapping of NONMEM standard columns
            covariates: List of covariate columns

        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert pharmacometrician specializing in NONMEM modeling.

TASK: Generate a complete NONMEM control stream file for population pharmacokinetic analysis.

DATASET INFORMATION:
{dataset_info}

DATA SUMMARY:
{data_summary}

AVAILABLE COLUMNS:
{', '.join(columns)}

NONMEM STANDARD COLUMNS DETECTED:
{_format_dict(nonmem_columns)}

COVARIATES (Subject-level variables):
{', '.join(covariates) if covariates else 'None detected'}

INSTRUCTIONS:
1. Create a complete NONMEM control stream with all necessary sections:
   - $PROBLEM: Clear description
   - $INPUT: Map all columns (use DROP for unused columns)
   - $DATA: Reference the input file
   - $SUBROUTINES: Choose appropriate ADVAN/TRANS
   - $PK: Define structural model with relevant covariates
   - $ERROR: Define residual error model (consider proportional, additive, or combined)
   - $THETA: Initial estimates and bounds for structural parameters
   - $OMEGA: Inter-individual variability
   - $SIGMA: Residual variability
   - $ESTIMATION: Use appropriate method (FOCE INTER is common)
   - $COVARIANCE
   - $TABLE: Output relevant parameters

2. Model selection guidance:
   - For most oral/IV drugs: 1-compartment or 2-compartment model
   - Consider absorption lag time if oral
   - Include relevant covariates on clearance and volume
   - Use allometric scaling for weight if appropriate

3. Initial parameter estimates:
   - Provide reasonable starting values based on typical PK parameters
   - Set appropriate bounds (lower, initial, upper)
   - Typical population values with ~20-50% IIV

4. Output ONLY the NONMEM control stream code.
   - Start with $PROBLEM
   - End with $TABLE
   - Use semicolons for comments
   - Ensure proper NONMEM syntax

5. Make this a scientifically sound, executable model that can run successfully.

GENERATE THE NONMEM CONTROL STREAM NOW:
"""
        return prompt

    @staticmethod
    def improvement_prompt(
        iteration: int,
        dataset_info: str,
        current_code: str,
        nonmem_output: str,
        previous_improvements: List[Dict],
        issues_found: List[str]
    ) -> str:
        """
        Generate prompt for iterative improvement

        Args:
            iteration: Current iteration number
            dataset_info: Dataset information
            current_code: Current NONMEM code
            nonmem_output: Output from NONMEM execution
            previous_improvements: History of previous improvements
            issues_found: List of issues detected in current run

        Returns:
            Formatted prompt string
        """
        history_text = _format_improvement_history(previous_improvements)

        issues_text = "\n".join([f"- {issue}" for issue in issues_found]) if issues_found else "None detected"

        prompt = f"""You are an expert pharmacometrician iteratively improving a NONMEM model.

ITERATION: {iteration}

DATASET INFORMATION:
{dataset_info}

CURRENT NONMEM CODE:
```
{current_code}
```

NONMEM OUTPUT:
```
{nonmem_output}
```

DETECTED ISSUES:
{issues_text}

PREVIOUS IMPROVEMENT HISTORY:
{history_text}

TASK: Analyze the NONMEM output and improve the model. Consider:

1. MINIMIZATION STATUS:
   - Did minimization succeed?
   - Are there any boundary issues?
   - Is the gradient acceptable?
   - Are there rounding errors?

2. PARAMETER ESTIMATES:
   - Are estimates reasonable?
   - Any parameters hitting bounds?
   - High RSE% (>50%) indicating poor precision?
   - High correlations between parameters?

3. OBJECTIVE FUNCTION VALUE:
   - What is the current OFV?
   - How does it compare to previous iterations?

4. MODEL DIAGNOSTICS:
   - Any warnings or errors?
   - Condition number issues?
   - Covariance step successful?

5. POSSIBLE IMPROVEMENTS:
   - Adjust initial estimates
   - Simplify model (remove unnecessary IIV)
   - Try different parameterization
   - Fix parameters with poor identifiability
   - Adjust bounds
   - Change estimation method settings
   - Add or remove covariates
   - Try different structural model

INSTRUCTIONS:
1. First, provide a brief analysis (2-3 sentences) of the current model status
2. Then, provide the IMPROVED NONMEM control stream code
3. Make ONE focused improvement per iteration
4. Explain what you changed and why (after the code)

Format your response as:
ANALYSIS: [Your analysis here]

IMPROVED CODE:
```
[NONMEM control stream]
```

CHANGES MADE: [Brief explanation of changes]

EXPECTED IMPROVEMENT: [What you expect this change to achieve]
"""
        return prompt

    @staticmethod
    def convergence_check_prompt(
        iteration: int,
        improvement_history: List[Dict],
        current_ofv: Optional[float],
        previous_ofv: Optional[float]
    ) -> str:
        """
        Generate prompt to assess if further improvements are needed

        Args:
            iteration: Current iteration number
            improvement_history: History of improvements
            current_ofv: Current objective function value
            previous_ofv: Previous objective function value

        Returns:
            Formatted prompt string
        """
        history_text = _format_improvement_history(improvement_history)

        ofv_text = "Not available"
        if current_ofv is not None:
            ofv_text = f"Current: {current_ofv:.2f}"
            if previous_ofv is not None:
                change = current_ofv - previous_ofv
                ofv_text += f" | Previous: {previous_ofv:.2f} | Change: {change:.2f}"

        prompt = f"""Assess whether the NONMEM model has converged satisfactorily.

ITERATION: {iteration}

OBJECTIVE FUNCTION VALUE:
{ofv_text}

IMPROVEMENT HISTORY:
{history_text}

ASSESSMENT CRITERIA:
1. Successful minimization with no major warnings
2. All parameter estimates within reasonable bounds
3. Acceptable RSE% for key parameters (<50%)
4. Successful covariance step
5. OFV change < 0.1 from previous iteration (if converged)
6. No obvious model misspecification

Based on the improvement history, answer these questions:

1. Has the model converged satisfactorily? (YES/NO)
2. Are there any remaining critical issues? (List them)
3. Would additional iterations likely provide meaningful improvement? (YES/NO)
4. Confidence level in current model (LOW/MEDIUM/HIGH)

Respond in this format:
CONVERGED: [YES/NO]
CRITICAL_ISSUES: [List or "None"]
CONTINUE_OPTIMIZATION: [YES/NO]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [Brief explanation]
"""
        return prompt


def _format_dict(d: Dict) -> str:
    """Format dictionary for display"""
    if not d:
        return "None"
    return "\n".join([f"  {k}: {v}" for k, v in d.items()])


def _format_improvement_history(history: List[Dict]) -> str:
    """Format improvement history for display"""
    if not history:
        return "No previous improvements"

    lines = []
    for i, item in enumerate(history, 1):
        lines.append(f"Iteration {i}:")
        lines.append(f"  Status: {item.get('status', 'unknown')}")
        lines.append(f"  Changes: {item.get('changes', 'N/A')}")
        if 'ofv' in item:
            lines.append(f"  OFV: {item['ofv']:.2f}")
        if 'issues' in item:
            lines.append(f"  Issues: {', '.join(item['issues']) if item['issues'] else 'None'}")
        lines.append("")

    return "\n".join(lines)
