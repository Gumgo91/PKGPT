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
Choose ADVAN based on data:
- Oral 1-compartment: ADVAN2 TRANS2
  * MUST define: CL, V, KA
  * MUST define: K = CL/V (elimination rate constant)
  * MUST define: S2 = V (scaling factor)
- IV 1-compartment: ADVAN1 TRANS2
  * MUST define: CL, V
  * MUST define: K = CL/V (elimination rate constant)
  * MUST define: S1 = V (scaling factor)
- Oral 2-compartment: ADVAN4 TRANS4 (CL, V2, Q, V3, KA, S2=V2)
- IV 2-compartment: ADVAN3 TRANS4 (CL, V1, Q, V2, S1=V1)

**CRITICAL: K=CL/V MUST be explicitly defined for ADVAN1/ADVAN2 TRANS2.**
Without explicit K definition, numerical instability will occur.

Start simple. Don't overcomplicate unless data demands it.

PHASE 2: INITIAL PARAMETER ESTIMATES
Typical values for small molecules:
- CL: 1-10 L/h
- V: 10-100 L
- KA: 0.5-3 h⁻¹

PHASE 3: COVARIATES
Weight effects (if available):
- CL: TVCL = THETA(1) * (WT/70)^0.75
- V: TVV = THETA(2) * (WT/70)^1

PHASE 4: RANDOM EFFECTS
Use diagonal OMEGA (one value per line):
$OMEGA
0.1  ; IIV on CL
0.1  ; IIV on V
0.2  ; IIV on KA

Never use BLOCK OMEGA or multi-value lines.

PHASE 5: RESIDUAL ERROR
Start with proportional error:
$ERROR
IPRED = F
Y = IPRED * (1 + EPS(1))

$SIGMA
0.04  ; Proportional error

PHASE 6: ESTIMATION
Use METHOD=1 INTER (FOCE with interaction) as default:
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC
$COVARIANCE PRINT=E

**NOTE: Use METHOD=1 (FOCE-I) for most models. Use METHOD=SAEM only for:**
- Complex models that fail with METHOD=1
- Models with categorical outcomes
- After 3+ failed attempts with METHOD=1

Starting with METHOD=SAEM often causes numerical problems for basic PK models.

═══════════════════════════════════════════════════════════════════
CRITICAL SYNTAX REQUIREMENTS
═══════════════════════════════════════════════════════════════════

1. $INPUT Declaration:
   - List ALL columns from dataset in order
   - Use DROP for unused columns
   - Example: $INPUT ID TIME AMT DV EVID MDV CMT WT SEX DROP DROP

2. $PK Block Requirements (IN THIS ORDER):
   a. Define all typical values (TVCL, TVV, TVKA)
   b. Apply covariate effects to typical values
   c. Define individual parameters with EXP(ETA) for lognormal
   d. **MUST define K = CL/V for ADVAN1/ADVAN2 TRANS2**
   e. MUST define scaling: S2 = V (for ADVAN2) or S1 = V (for ADVAN1)

   Example for ADVAN2 TRANS2:
   ```
   TVCL = THETA(1) * (WT/70)**0.75  ; if covariate
   TVV  = THETA(2) * (WT/70)**1.0
   TVKA = THETA(3)

   CL = TVCL * EXP(ETA(1))
   V  = TVV * EXP(ETA(2))
   KA = TVKA * EXP(ETA(3))

   K = CL/V    ; CRITICAL: Must define explicitly
   S2 = V      ; Scaling factor
   ```

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
$INPUT [all columns IN THE EXACT ORDER they appear in the dataset]
$SUBROUTINES [ADVAN TRANS]
$PK [parameter model]
$ERROR [error model]
$THETA [initial estimates with bounds]
$OMEGA [IIV structure]
$SIGMA [residual error]
$ESTIMATION [method settings]
$COVARIANCE [options]
$TABLE [output specification]

**CRITICAL: DO NOT add $END or any other statement after $TABLE. The file ends naturally.**

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

        # Check if we have OFV from history
        has_ofv = any(h.get('ofv') is not None for h in previous_improvements)

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
TWO-PHASE OPTIMIZATION STRATEGY
═══════════════════════════════════════════════════════════════════

{'PHASE 1: MAKE IT RUN (Current Phase - No OFV yet)' if not has_ofv else 'PHASE 2: MAKE IT BETTER (OFV available - Focus on improvement)'}

{'**CRITICAL: The model is not producing OFV yet. Focus ONLY on getting estimation to run successfully.**' if not has_ofv else '**The model runs successfully. Focus on improving OFV, RSE%, and Shrinkage.**'}

{'PHASE 1 PRIORITIES (Make it run):' if not has_ofv else 'PHASE 2 PRIORITIES (Make it better):'}
{'1. SYNTAX ERRORS - Fix immediately' if not has_ofv else '1. OFV IMPROVEMENT'}
{'   - Missing S2=V or S1=V in $PK' if not has_ofv else '   - Try different error models'}
{'   - Wrong $OMEGA format (use diagonal: one value per line)' if not has_ofv else '   - Add covariates if RSE is good'}
{'   - $DATA file path issues' if not has_ofv else '   - Optimize OMEGA values'}
{'   - Column name mismatches' if not has_ofv else ''}
{'2. STRUCTURAL ISSUES' if not has_ofv else '2. RSE% REDUCTION'}
{'   - Data/model mismatch (CMT issues, ADVAN mismatch)' if not has_ofv else '   - Simplify if RSE > 50%'}
{'   - Missing K=CL/V for ADVAN2' if not has_ofv else '   - Add covariates if RSE < 30%'}
{'   - Wrong compartment for observations' if not has_ofv else ''}
{'3. INITIAL VALUES' if not has_ofv else '3. SHRINKAGE REDUCTION'}
{'   - Use safe defaults: CL=3, V=30, KA=1' if not has_ofv else '   - Target < 30% ETA shrinkage'}
{'   - Conservative OMEGA (0.04-0.1)' if not has_ofv else '   - Adjust IIV structure'}
{'4. SIMPLIFICATION' if not has_ofv else '4. MODEL COMPLEXITY'}
{'   - Start with simplest model' if not has_ofv else '   - Add complexity only if justified'}
{'   - Remove INTER if unstable' if not has_ofv else '   - Balance fit vs parsimony'}
{'   - Use proportional error only' if not has_ofv else ''}

═══════════════════════════════════════════════════════════════════
TROUBLESHOOTING CHECKLIST
═══════════════════════════════════════════════════════════════════

1. Check NONMEM output for:
   - "MINIMIZATION SUCCESSFUL" → Model ran ✓
   - "#OBJV:" with a number → OFV available ✓
   - Only "NM-TRAN" + "Stop Time" → Compilation only, no estimation ✗
   - "UNKNOWN CONTROL RECORD" → Syntax error (often $END) ✗

2. Common reasons estimation doesn't run or produces bad results:
   - **$END statement** → Remove it! NONMEM doesn't recognize $END
   - **$INPUT order wrong** → Must match exact column order in CSV
   - **Missing K=CL/V** → MUST define explicitly for ADVAN1/ADVAN2 TRANS2
   - **Using METHOD=SAEM for simple models** → Switch to METHOD=1 INTER
   - **Parameters hitting boundaries** → CL/V at lower/upper bounds = bad model
   - Data file path wrong
   - CMT mismatch (observations in wrong compartment)
   - Missing scaling (S1, S2)
   - Invalid initial values (negative, too extreme)

3. Signs of fundamental model problems (requires structural fix):
   - Parameters hitting boundaries (CL=0.1, V=100)
   - OMEGA collapsing to ~0 (< 1e-10)
   - OFV > 100,000 (unreasonably high)
   - ETA shrinkage > 90%
   - SE larger than estimate
   - "GRADIENT TO THETA IS ZERO" warnings

   → These indicate model misspecification, not just numerical issues

4. Don't change ADVAN randomly:
   - Stick with ADVAN2 for oral until confirmed wrong
   - Only switch after 3+ iterations with same error

5. CRITICAL: Never add $END - the file ends naturally after $TABLE

═══════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════

Make ONE focused change:
- Fix the most critical issue blocking estimation
- Preserve everything else that works
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
