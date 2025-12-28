"""
Phase 2: Diagnose Structural Model
Focus: Determine if compartment structure and error model are adequate
"""

from typing import Dict, List, Optional


class Phase2Diagnose:
    """Phase 2 prompts for diagnosing structural model adequacy"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        nonmem_output: str,
        parsed_results: Dict,
        warnings: List[str],
        n_subjects: int
    ) -> str:
        """
        Generate Phase 2 prompt focused on structural diagnosis

        Goals:
        - Check if compartment structure is adequate
        - Diagnose residual patterns
        - Assess error model appropriateness
        - Decide if structural changes needed
        """

        ofv = parsed_results.get('objective_function', 'N/A')
        minimization_ok = parsed_results.get('minimization_successful', False)
        warnings_text = "\n".join([f"- {w}" for w in warnings]) if warnings else "None"

        # Extract current ADVAN from code
        current_advan = "Unknown"
        if "ADVAN1" in current_code:
            current_advan = "ADVAN1 (IV, 1-compartment)"
        elif "ADVAN2" in current_code:
            current_advan = "ADVAN2 (Oral, 1-compartment)"
        elif "ADVAN3" in current_code:
            current_advan = "ADVAN3 (IV, 2-compartment)"
        elif "ADVAN4" in current_code:
            current_advan = "ADVAN4 (Oral, 2-compartment)"

        prompt = f"""You are diagnosing structural model adequacy.

═══════════════════════════════════════════════════════════════════
PHASE 2: DIAGNOSE STRUCTURAL MODEL
═══════════════════════════════════════════════════════════════════

ITERATION: {iteration}
DATASET: N={n_subjects} subjects
CURRENT STRUCTURE: {current_advan}
MINIMIZATION: {'✓ Successful' if minimization_ok else '✗ Failed'}
OFV: {ofv}

CURRENT MODEL:
```
{current_code}
```

NONMEM WARNINGS:
{warnings_text}

═══════════════════════════════════════════════════════════════════
DIAGNOSTIC STRATEGY: Is the Structure Adequate?
═══════════════════════════════════════════════════════════════════

**Check 1: Residual Pattern Analysis**

Look for these patterns in NONMEM warnings/output:

1. **U-shaped or Systematic Residual Trends**
   - Indicates: Wrong number of compartments
   - U-shape in CWRES vs TIME → Need more compartments
   - Action: Consider ADVAN2→ADVAN4 or ADVAN1→ADVAN3

2. **Bi-phasic or Multi-phasic Decline**
   - Indicates: Distribution phase present
   - 1-compartment insufficient for drug with distribution
   - Action: Upgrade to 2-compartment model

3. **Funnel-shaped Residuals (CWRES vs PRED)**
   - Indicates: Heteroscedastic errors
   - Narrow at low concentrations, wide at high
   - Action: Ensure proportional error component in error model

**Check 2: OFV Magnitude**

- OFV > 0 and reasonable: Likely OK
- OFV extremely negative (<-50): Possible structural issue or overfitting
- OFV extremely high (>5000 for N<50): Poor fit, structure may be wrong

**Check 3: Warnings Analysis**

Critical warnings suggesting structural issues:
- "PARAMETER ESTIMATE IS NEAR ITS BOUNDARY" → May need more compartments
- "PRED.LE.0" or "NUMERICAL HESSIAN" → Structural misspecification
- Systematic mentions of residual patterns → Structure problem

**Check 4: Error Model Adequacy**

Current error model type (check $ERROR block):
- Proportional only: Y = IPRED * (1 + EPS(1))
- Combined (better): W = SQRT(THETA^2 + (THETA*IPRED)^2), Y = IPRED + W*EPS(1)
- Additive only: Y = IPRED + EPS(1)

If proportional-only and low concentrations exist → Consider combined model

═══════════════════════════════════════════════════════════════════
DECISION CRITERIA
═══════════════════════════════════════════════════════════════════

**Keep Current Structure IF:**
✓ Minimization successful
✓ No systematic residual trends mentioned
✓ OFV reasonable (not extremely negative)
✓ No warnings about structural misspecification
→ ACTION: Make minor refinements only (bounds, error model)

**Change to 2-Compartment IF:**
- Clear bi-phasic decline mentioned
- U-shaped residuals noted
- Drug known to have distribution phase (e.g., IV antibiotics)
- N > 30 (enough data to support 2-cmt)
→ ACTION: ADVAN2→ADVAN4 or ADVAN1→ADVAN3

**Improve Error Model IF:**
- Funnel-shaped residuals
- Heteroscedastic errors mentioned
- Proportional-only model failing at low concentrations
→ ACTION: Switch to combined error model

**DO NOT Change Structure IF:**
- N < 20 (insufficient data for complex models)
- Current model already working well
- Only minor issues with parameters
→ ACTION: Proceed to optimization phases

═══════════════════════════════════════════════════════════════════
STRUCTURAL MODEL UPGRADE TEMPLATES
═══════════════════════════════════════════════════════════════════

**Oral 1-cmt → 2-cmt (ADVAN2 → ADVAN4):**
```
$SUBROUTINE ADVAN4 TRANS4

$PK
CL = THETA(1) * EXP(ETA(1))
V1 = THETA(2) * EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
KA = THETA(5) * EXP(ETA(3))
S2 = V1

$THETA
(0.1, 3, 100)    ; CL (L/h)
(1, 30, 200)     ; V1 (L)
(0.1, 5, 50)     ; Q (L/h) - inter-compartmental clearance
(1, 50, 500)     ; V2 (L) - peripheral volume
(0.1, 1, 10)     ; KA (1/h)
```

**IV 1-cmt → 2-cmt (ADVAN1 → ADVAN3):**
```
$SUBROUTINE ADVAN3 TRANS4

$PK
CL = THETA(1) * EXP(ETA(1))
V1 = THETA(2) * EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
S1 = V1

$THETA
(0.1, 3, 100)    ; CL
(1, 30, 200)     ; V1
(0.1, 5, 50)     ; Q
(1, 50, 500)     ; V2
```

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

STRUCTURAL DIAGNOSIS:
[Is the current compartment model adequate? Cite specific evidence from output]

RECOMMENDATION:
[Keep current structure OR Upgrade to 2-compartment OR Improve error model]

IMPROVED CODE:
```
[Complete NONMEM control stream - with structural changes if needed]
```

CHANGES MADE:
[List specific modifications and rationale]

EXPECTED OUTCOME:
[Better fit if structure changed, or confirmation that structure is adequate]
"""

        return prompt
