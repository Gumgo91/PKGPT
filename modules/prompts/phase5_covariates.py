"""
Phase 5: Covariate Analysis
Focus: Add statistically significant covariates to explain variability
"""

from typing import Dict, List, Optional


class Phase5Covariates:
    """Phase 5 prompts for systematic covariate analysis"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        parsed_results: Dict,
        shrinkage_data: List[Dict],
        available_covariates: List[Dict],
        current_covariates_in_model: List[str],
        n_subjects: int
    ) -> str:
        """
        Generate Phase 5 prompt focused on covariate analysis

        Goals:
        - Screen covariates for relationships with ETAs
        - Add covariates using forward selection (ΔOFV > 3.84)
        - Validate covariates using backward elimination (ΔOFV > 6.63)
        - Explain inter-individual variability
        """

        ofv = parsed_results.get('objective_function', 'N/A')
        minimization_ok = parsed_results.get('minimization_successful', False)

        # Format available covariates
        cov_text = ""
        if available_covariates:
            cov_text = "Available Covariates:\n"
            for cov in available_covariates:
                name = cov.get('name', '?')
                cov_type = cov.get('type', '?')
                median = cov.get('median', '?')
                cov_range = f"{cov.get('min', '?')}-{cov.get('max', '?')}"
                model_type = cov.get('suggested_model', 'linear')
                cov_text += f"  {name}: {cov_type}, median={median}, range={cov_range}, model={model_type}\n"
        else:
            cov_text = "No covariates available in dataset"

        # Current covariates already in model
        current_cov_text = ""
        if current_covariates_in_model:
            current_cov_text = f"\nCovariates already in model: {', '.join(current_covariates_in_model)}"
        else:
            current_cov_text = "\nNo covariates currently in model (base model)"

        # Shrinkage
        shrinkage_text = ""
        if shrinkage_data:
            avg_shrink = sum([s['shrinkage'] for s in shrinkage_data]) / len(shrinkage_data)
            shrinkage_text = f"Average Shrinkage: {avg_shrink:.1f}% "
            shrinkage_text += ("✓ Good for covariate analysis" if avg_shrink < 50 else "⚠ High - covariates may not help")

        prompt = f"""You are conducting systematic covariate analysis.

═══════════════════════════════════════════════════════════════════
PHASE 5: COVARIATE ANALYSIS - Explain Variability
═══════════════════════════════════════════════════════════════════

ITERATION: {iteration}
DATASET: N={n_subjects} subjects
MINIMIZATION: {'✓ Success' if minimization_ok else '✗ Failed'}
OFV: {ofv}
{shrinkage_text}
{current_cov_text}

{cov_text}

CURRENT MODEL:
```
{current_code}
```

═══════════════════════════════════════════════════════════════════
PREREQUISITES CHECK (Must ALL be TRUE)
═══════════════════════════════════════════════════════════════════

Before adding covariates, verify:
✓ Minimization successful
✓ Average shrinkage <50% (ETAs are informative)
✓ Base model stable (OFV consistent across iterations)
✓ No major structural issues remaining
✓ Covariates available in dataset

If ANY prerequisite fails → Return to earlier phase, DO NOT add covariates

═══════════════════════════════════════════════════════════════════
COVARIATE ANALYSIS PROCEDURE
═══════════════════════════════════════════════════════════════════

**Step 1: Screening (Identify Candidates)**

For each available covariate, evaluate biological plausibility:

Body Size (WT, BSA):
- Expected on: CL, V (both clear and distribute in body)
- Strong candidate - physiologically sound

Age (AGE):
- Expected on: CL (metabolism changes with age), possibly V
- Moderate candidate - check data range

Sex (SEX):
- Expected on: CL, V (body composition differences)
- Consider if sex distribution balanced

Renal Function (CRCL, SCR):
- Expected on: CL (renal clearance)
- Strong candidate for renally eliminated drugs

**Step 2: Forward Selection (Add One at a Time)**

Statistical threshold: ΔOFV > 3.84 (χ² with df=1, p<0.05)

Procedure:
1. Start with base model (no covariates)
2. Add one candidate covariate
3. Run model, compare OFV to base
4. If ΔOFV > 3.84 → Keep covariate, update base
5. Repeat with remaining covariates
6. Stop when no more significant covariates

Functional form for each covariate–parameter test:
- For continuous covariates (covariate type reported as "continuous" in the metadata):
  - Use the provided median M to center the covariate.
    * For additive/linear form, work with (COV - M).
    * For power/log-normal forms, work with the ratio (COV/M).
  - For each candidate covariate–parameter pair, test the following three forms and record ΔOFV, convergence, and shrinkage:
    1. Additive / linear:
       - Example: TVCL = THETA(CL) * (1 + THETA(COV) * (COV - M))
    2. Power (allometric-type):
       - Example: TVCL = THETA(CL) * (COV/M)**THETA(COV)
    3. Log-normal (log-linear):
       - Example: TVCL = THETA(CL) * EXP(THETA(COV) * LOG(COV/M))
  - Among forms that converge without pathological shrinkage and satisfy ΔOFV > 3.84, prefer the form that appears most consistently across models; aim for consistent structure across similar covariates (e.g., size-related covariates → power form).

- For categorical covariates (covariate type reported as "categorical"):
  - Do not center.
  - Use indicator/factor coding with one reference category and multiplicative effects on the parameter for each non-reference level (e.g., SEX, treatment group).

After testing all forms for a given covariate–parameter pair, choose the best-supported combination based on ΔOFV, shrinkage, and clinical plausibility, and use that in the forward selection step.

Example iteration:
```
Base OFV = 150.2
Try WT on CL → OFV = 138.5 → ΔOFV = 11.7 > 3.84 ✓ KEEP
New base OFV = 138.5
Try AGE on CL → OFV = 136.8 → ΔOFV = 1.7 < 3.84 ✗ REJECT
Try SEX on V → OFV = 135.0 → ΔOFV = 3.5 < 3.84 ✗ REJECT
```

**Step 3: Backward Elimination (Remove Non-significant)**

Statistical threshold: ΔOFV > 6.63 (χ² with df=1, p<0.01) to STAY

More stringent than forward selection to avoid overfitting.

Procedure:
1. Start with full model (all covariates from forward selection)
2. Remove one covariate
3. If OFV increase < 6.63 → Covariate not significant, remove it
4. If OFV increase > 6.63 → Covariate significant, keep it
5. Repeat for all covariates

**Step 4: Validation**

Final covariate model must satisfy:
✓ All covariates statistically significant (p<0.01)
✓ Covariate effects physiologically plausible
✓ IIV reduced compared to base model (smaller OMEGAs)
✓ Shrinkage still acceptable (<50%)
✓ Minimization successful

═══════════════════════════════════════════════════════════════════
COVARIATE IMPLEMENTATION TEMPLATES
═══════════════════════════════════════════════════════════════════

**Template 1: Body Size on CL/V (Power Model - RECOMMENDED)**

```
$PK
; Covariate effect - WT on CL
TVCL = THETA(1) * (WT/70)**THETA(4)  ; Power model, centered at 70 kg
CL = TVCL * EXP(ETA(1))

; Covariate effect - WT on V
TVV = THETA(2) * (WT/70)**THETA(5)   ; Power model
V = TVV * EXP(ETA(2))

KA = THETA(3) * EXP(ETA(3))

$THETA
(0.1, 3, 100)    ; CL at 70 kg
(1, 30, 200)     ; V at 70 kg
(0.1, 1, 10)     ; KA
(0.5, 0.75, 1.5) ; WT exponent on CL (physiological ~0.75)
(0.5, 1.0, 1.5)  ; WT exponent on V (physiological ~1.0)
```

**Template 2: Continuous Covariate (Linear Model)**

```
$PK
; Covariate effect - AGE on CL
TVCL = THETA(1) * (1 + THETA(4) * (AGE - 45))  ; Linear, centered at median
CL = TVCL * EXP(ETA(1))

$THETA
(0.1, 3, 100)       ; CL at AGE=45
(-0.05, 0, 0.05)    ; AGE effect on CL (fractional change per year)
```

**Template 3: Categorical Covariate (Sex)**

```
$PK
; Covariate effect - SEX on CL
TVCL = THETA(1)
IF (SEX.EQ.1) TVCL = TVCL * THETA(4)  ; Male factor (1=male, 0=female)
CL = TVCL * EXP(ETA(1))

$THETA
(0.1, 3, 100)    ; CL for females (reference)
(0.5, 1.2, 2.0)  ; Male factor (typically 1.0-1.3 for CL)
```

**Template 4: Renal Function on CL**

```
$PK
; Covariate effect - CRCL on CL
TVCL = THETA(1) * (CRCL/80)**THETA(4)  ; Power model, centered at typical CRCL
CL = TVCL * EXP(ETA(1))

$THETA
(0.1, 3, 100)     ; CL at CRCL=80 mL/min
(0.3, 0.5, 1.0)   ; CRCL exponent (0.5-0.7 typical for renal drugs)
```

═══════════════════════════════════════════════════════════════════
IMPORTANT CONSIDERATIONS
═══════════════════════════════════════════════════════════════════

**Add ONE Covariate Per Iteration:**
- Don't add multiple covariates at once
- Allows clear assessment of each covariate's impact
- Follow forward selection procedure

**Physiological Plausibility:**
- WT exponent on CL: ~0.75 (allometric scaling)
- WT exponent on V: ~1.0 (volume scales with size)
- Age effect: usually modest (±20% over lifespan)
- Sex effect: 10-30% difference typical

**Avoid Overfitting:**
- Smaller datasets can reliably support fewer covariates; larger datasets can support more, but there is no fixed maximum number.
- Use the statistical thresholds (ΔOFV > 3.84 for inclusion, ΔOFV > 6.63 for retention), shrinkage, and physiological plausibility as the primary decision criteria, rather than a hard cap on the number of covariates.
- Always use backward elimination to confirm that each retained covariate remains statistically significant (ΔOFV > 6.63) and clinically reasonable.

**Centering:**
- Always center continuous covariates at their median when defining continuous relationships.
- This makes THETA for the typical value interpretable and stabilizes numerical estimation.
- Example: (WT/70) or (WT/median(WT)) rather than raw WT.

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

COVARIATE SCREENING:
[Which covariate(s) to test? Why biologically plausible?]

IMPLEMENTATION PLAN:
[Which parameter (CL/V/KA)? Which model type (power/linear/categorical)?]

IMPROVED CODE:
```
[Complete NONMEM control stream with ONE new covariate added]
```

CHANGES MADE:
- Added covariate: {name} on {parameter}
- Model type: {power/linear/categorical}
- New THETA(s): {describe}
- Centering value: {median}

HYPOTHESIS:
[Expected OFV reduction if covariate is significant: ΔOFV should be > 3.84]

STATISTICAL TEST:
[Will compare new OFV to base OFV={ofv}]
"""

        return prompt
