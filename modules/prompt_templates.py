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
        prompt = f"""You are an expert pharmacometrician tasked with developing a population pharmacokinetic model using NONMEM.

═══════════════════════════════════════════════════════════════════
DATASET CHARACTERISTICS
═══════════════════════════════════════════════════════════════════
{dataset_info}

{data_summary}

Available columns: {', '.join(columns)}
Identified NONMEM columns: {_format_dict(nonmem_columns)}
Subject-level covariates: {', '.join(covariates) if covariates else 'None'}

═══════════════════════════════════════════════════════════════════
MODEL DEVELOPMENT FRAMEWORK
═══════════════════════════════════════════════════════════════════

PHASE 1: STRUCTURAL MODEL SELECTION
Assess administration route and disposition characteristics:
- Oral administration (AMT>0, EVID=1, CMT=1): Use ADVAN2 TRANS2
  * ADVAN2: 1-compartment with first-order absorption
  * Parameters: CL, V, KA
  * Scaling: S2 = V (depot compartment scales by central volume)

- IV bolus/infusion (AMT>0, RATE>=0): Use ADVAN1 TRANS2 or ADVAN2
  * ADVAN1: 1-compartment IV
  * Parameters: CL, V
  * Scaling: S1 = V

Start with the simplest model that captures the essential PK processes.

PHASE 2: PARAMETER ESTIMATION STRATEGY
Typical population values (adjust based on drug class):
- Clearance (CL): 1-10 L/h for small molecules
  * Renal clearance: ~GFR × fu (~8 L/h)
  * Hepatic clearance: can approach liver blood flow (~90 L/h)

- Volume of distribution (V): 10-100 L
  * Hydrophilic drugs: ~0.2-0.3 L/kg (~15-20 L/70kg)
  * Lipophilic drugs: ~1-10 L/kg (70-700 L/70kg)

- Absorption rate constant (KA): 0.5-3 h⁻¹ for oral drugs
  * Fast absorption: >2 h⁻¹
  * Moderate: 0.5-2 h⁻¹
  * Slow/sustained release: <0.5 h⁻¹

PHASE 3: COVARIATE MODEL SPECIFICATION
Implement physiologically-based covariate relationships:

Weight effects (use allometric scaling):
- On CL: TVCL = THETA(CL) * (WT/70)^0.75
  * Exponent 0.75: metabolic scaling law
- On V: TVV = THETA(V) * (WT/70)^1
  * Exponent 1: proportional to body size

Other covariates (if data supports):
- Age on CL: consider maturation/decline functions
- Renal function on CL: CRCL effects for renally eliminated drugs
- Sex, genotype: include only if mechanistically justified

PHASE 4: RANDOM EFFECTS STRUCTURE
Inter-individual variability (IIV):
- Lognormal distribution: P_i = TV_P * EXP(ETA_i)
- Interpretation: ETA ~ N(0, OMEGA)
- Initial OMEGA values: 0.1-0.3 (corresponds to ~30-55% CV)

Start with diagonal OMEGA matrix (assume independence):
$OMEGA
0.1  ; IIV on CL
0.1  ; IIV on V
0.2  ; IIV on KA (if oral)

Consider BLOCK structure only if:
- Strong physiological basis for correlation
- Sufficient data to support estimation
- Improved model diagnostics

PHASE 5: RESIDUAL ERROR MODEL
Select error structure based on data characteristics:

a) Proportional error (recommended for PK data):
   IPRED = F
   Y = IPRED * (1 + EPS(1))

   When: Error scales with concentration
   SIGMA: 0.04-0.1 (20-30% CV)

b) Combined proportional + additive:
   IPRED = F
   W = IPRED
   Y = IPRED + W*EPS(1) + EPS(2)

   When: Constant error at low concentrations, proportional at high
   SIGMA(1): 0.04-0.1 (proportional)
   SIGMA(2): 0.1-1 (additive, in concentration units)

c) Additive (rarely used alone):
   IPRED = F
   Y = IPRED + EPS(1)

   When: Constant absolute error (uncommon for PK)

PHASE 6: ESTIMATION METHOD CONFIGURATION
Use first-order conditional estimation:
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC

Parameters:
- METHOD=1: FOCE (First-Order Conditional Estimation)
- INTER: Interaction between IIV (ETA) and residual error (EPS)
- MAXEVAL=9999: Maximum function evaluations
- PRINT=5: Print every 5th iteration
- POSTHOC: Compute individual parameter estimates

Add covariance step:
$COVARIANCE PRINT=E
- Estimates parameter uncertainty (SE, RSE%)
- Computes correlations between parameters

═══════════════════════════════════════════════════════════════════
CRITICAL SYNTAX REQUIREMENTS
═══════════════════════════════════════════════════════════════════

1. $INPUT Declaration:
   - List ALL columns from dataset in order
   - Use DROP for unused columns
   - Example: $INPUT ID TIME AMT DV EVID MDV CMT WT SEX DROP DROP

2. $PK Block Requirements:
   - Define all typical values (TVCL, TVV, TVKA)
   - Apply covariate effects to typical values
   - Define individual parameters with EXP(ETA) for lognormal
   - MUST define scaling: S2 = V (for ADVAN2) or S1 = V (for ADVAN1)
   - Assign CL, V, KA to NONMEM-recognized names

3. $ERROR Block Requirements:
   - MUST define IPRED = F first
   - For combined error: define W = IPRED before Y equation
   - Ensure proper operator precedence: use parentheses

4. Parameter Bounds Format:
   Format: (lower, initial) or (lower, initial, upper)
   - Lower bound: Use 0 for positive parameters (CL, V, KA)
   - Initial: Physiologically reasonable value
   - Upper: Optional, use if parameter should be constrained
   - Example: (0, 3) or (0, 3, 100)

5. $TABLE Statement:
   Include diagnostic variables for model evaluation:
   - ID, TIME, DV: observed data
   - IPRED: individual predictions
   - PRED: population predictions (if needed)
   - CWRES: conditional weighted residuals
   - File format: ONEHEADER NOPRINT FILE=outputname

═══════════════════════════════════════════════════════════════════
GENERATE NONMEM CONTROL STREAM
═══════════════════════════════════════════════════════════════════
Based on the dataset provided, generate a complete, syntactically correct, executable NONMEM control stream.

Requirements:
1. Use appropriate ADVAN based on data characteristics
2. Include physiologically plausible initial estimates
3. Implement covariate effects if supported by data
4. Use proven syntax for all blocks
5. Ensure all referenced columns exist in dataset
6. Add brief comments explaining key decisions
7. Output ONLY the NONMEM control stream code

Structure:
$PROBLEM [descriptive title]
$DATA [filename] IGNORE=@
$INPUT [all columns]
$SUBROUTINES [ADVAN TRANS]
$PK [parameter model]
$ERROR [error model]
$THETA [initial estimates with bounds]
$OMEGA [IIV structure]
$SIGMA [residual error]
$ESTIMATION [method settings]
$COVARIANCE [options]
$TABLE [output specification]

Generate the control stream now:
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

        prompt = f"""Diagnose and resolve issues in the NONMEM model.

═══════════════════════════════════════════════════════════════════
ITERATION {iteration} - MODEL REFINEMENT
═══════════════════════════════════════════════════════════════════

CURRENT MODEL:
```
{current_code}
```

NONMEM EXECUTION OUTPUT:
```
{nonmem_output}
```

IDENTIFIED ISSUES:
{issues_text}

OPTIMIZATION HISTORY:
{history_text}

═══════════════════════════════════════════════════════════════════
DIAGNOSTIC FRAMEWORK
═══════════════════════════════════════════════════════════════════

LEVEL 1: SYNTAX AND STRUCTURAL ERRORS
Check for:
□ Column name mismatches ($INPUT vs dataset)
□ Missing scaling parameter (S1 or S2 in $PK)
□ Incorrect $ERROR syntax (especially combined error model)
□ Parameter bound violations (initial outside [lower, upper])
□ Undefined variables referenced in equations
□ FORTRAN syntax errors (operators, parentheses)

Common fixes:
- $ERROR combined model: Must use IPRED=F, W=IPRED, then Y=IPRED+W*EPS(1)+EPS(2)
- Scaling: Add S2=V (ADVAN2) or S1=V (ADVAN1) in $PK
- Columns: Verify all referenced columns exist in dataset

LEVEL 2: NUMERICAL STABILITY ISSUES
Check for:
□ Initial estimates orders of magnitude off
□ OMEGA/SIGMA too large (>1) or too small (<0.001)
□ Parameters hitting bounds during estimation
□ Rounding errors or ill-conditioning warnings

Remediation strategies:
- Adjust THETA initial values to physiologically realistic ranges
- Reduce IIV initial estimates (try 0.1-0.3)
- Simplify OMEGA structure (use diagonal instead of BLOCK)
- Remove poorly estimated random effects

LEVEL 3: MODEL OVERPARAMETERIZATION
Symptoms:
- High RSE% (>50%) on multiple parameters
- Correlation coefficients near ±1
- Failure to compute covariance matrix
- Boundary estimates (OMEGA or SIGMA near zero)

Solutions:
- Fix or remove IIV on poorly estimated parameters
- Simplify covariate relationships
- Use fewer random effects
- Consider simpler structural model

LEVEL 4: CONVERGENCE PROBLEMS
If minimization not successful:
1. Check for programming errors first (Level 1)
2. Try different initial estimates
3. Simplify random effects structure
4. Increase MAXEVAL or try different METHOD
5. Fix problematic parameters temporarily

═══════════════════════════════════════════════════════════════════
IMPROVEMENT STRATEGY
═══════════════════════════════════════════════════════════════════

Priority order:
1. Fix syntax errors (model won't run)
2. Correct structural problems (S2 missing, wrong ADVAN)
3. Adjust initial estimates (numerical issues)
4. Simplify if overparameterized
5. Refine once stable

Make ONE targeted change per iteration:
- Focus on the most critical issue
- Preserve what works
- Avoid multiple simultaneous changes
- Document the rationale

═══════════════════════════════════════════════════════════════════
GENERATE IMPROVED MODEL
═══════════════════════════════════════════════════════════════════

Based on the diagnostic output, provide:

1. ANALYSIS (2-3 sentences):
   - Primary issue identified
   - Root cause (syntax, numerical, structural)
   - Proposed solution approach

2. IMPROVED CODE:
   - Complete corrected NONMEM control stream
   - Implement one focused improvement
   - Maintain overall model structure
   - Ensure syntactic correctness

3. CHANGES MADE:
   - Specific modifications with rationale
   - Expected impact on model performance

Format your response:

ANALYSIS:
[Concise diagnosis of the primary issue]

IMPROVED CODE:
```
[Complete corrected NONMEM control stream]
```

CHANGES MADE:
[List specific modifications and reasons]

EXPECTED OUTCOME:
[What this change should achieve]
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
        if 'ofv' in item and item['ofv'] is not None:
            lines.append(f"  OFV: {item['ofv']:.2f}")
        elif 'ofv' in item:
            lines.append(f"  OFV: Not available")
        if 'issues' in item:
            lines.append(f"  Issues: {', '.join(item['issues']) if item['issues'] else 'None'}")
        lines.append("")

    return "\n".join(lines)
