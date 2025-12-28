"""
PKGPT Prompt Templates - Comprehensive NONMEM Code Generation
Systematic model development with best practices from pharmacometrics

Key features:
- Comprehensive ADVAN selection (ADVAN1-13)
- Phase-separated prompts for focused improvements
- Enhanced estimation method selection (FO/FOCE/FOCE-I)
- Covariate analysis methodology
- Evidence-based model diagnostic criteria
"""

from typing import Dict, List, Optional
from enum import Enum

# Import phase-specific prompts
from .prompts import (
    Phase1Establish,
    Phase2Diagnose,
    Phase3Reduce,
    Phase4Optimize,
    Phase5Covariates
)

# Import ModelPhase enum (needed for routing)
import sys
import os
# Add parent directory to path to import ModelPhase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ModelPhase(Enum):
    """Model development phases - duplicated here to avoid circular import"""
    ESTABLISH_BASE = 1
    DIAGNOSE_STRUCTURE = 2
    REDUCE_OVERFITTING = 3
    OPTIMIZE_IIV = 4
    COVARIATE_ANALYSIS = 5


class PromptTemplates:
    """Templates for generating prompts for Gemini API"""

    @staticmethod
    def _get_advan_guidance(route: str = "oral", compartments: int = 1) -> str:
        """
        Comprehensive ADVAN selection guidance

        Special ADVANs: ADVAN1-4, 10-12 (predefined structures)
        General ADVANs: ADVAN5-8, 9, 13 (user-defined via $DES)
        """

        advan_guide = {
            # 1-compartment models
            ("iv", 1): {
                "advan": "ADVAN1",
                "trans": "TRANS2",
                "params": ["CL", "V"],
                "description": "IV bolus, 1-compartment",
                "pk_code": """$PK
CL = THETA(1) * EXP(ETA(1))
V  = THETA(2) * EXP(ETA(2))
K  = CL/V
S1 = V""",
                "theta": """$THETA
(0.1, 3, 100)   ; CL (L/h)
(1, 30, 200)    ; V (L)"""
            },
            ("oral", 1): {
                "advan": "ADVAN2",
                "trans": "TRANS2",
                "params": ["CL", "V", "KA"],
                "description": "1st-order absorption, 1-compartment",
                "pk_code": """$PK
CL = THETA(1) * EXP(ETA(1))
V  = THETA(2) * EXP(ETA(2))
KA = THETA(3) * EXP(ETA(3))
K  = CL/V
S2 = V""",
                "theta": """$THETA
(0.1, 3, 100)   ; CL (L/h)
(1, 30, 200)    ; V (L)
(0.1, 1, 10)    ; KA (1/h)"""
            },
            # 2-compartment models
            ("iv", 2): {
                "advan": "ADVAN3",
                "trans": "TRANS4",
                "params": ["CL", "V1", "Q", "V2"],
                "description": "IV bolus, 2-compartment",
                "pk_code": """$PK
CL = THETA(1) * EXP(ETA(1))
V1 = THETA(2) * EXP(ETA(2))
Q  = THETA(3)
V2 = THETA(4)
S1 = V1""",
                "theta": """$THETA
(0.1, 3, 100)   ; CL (L/h)
(1, 30, 200)    ; V1 (L)
(0.1, 5, 50)    ; Q (L/h)
(1, 50, 500)    ; V2 (L)"""
            },
            ("oral", 2): {
                "advan": "ADVAN4",
                "trans": "TRANS4",
                "params": ["CL", "V2", "Q", "V3", "KA"],
                "description": "1st-order absorption, 2-compartment",
                "pk_code": """$PK
KA = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
V2 = THETA(3) * EXP(ETA(3))
Q  = THETA(4)
V3 = THETA(5)
S2 = V2""",
                "theta": """$THETA
(0.1, 1, 10)    ; KA (1/h)
(0.1, 3, 100)   ; CL (L/h)
(1, 30, 200)    ; V2 (L)
(0.1, 5, 50)    ; Q (L/h)
(1, 50, 500)    ; V3 (L)"""
            },
            # 3-compartment models
            ("iv", 3): {
                "advan": "ADVAN11",
                "trans": "TRANS4",
                "params": ["CL", "V1", "Q2", "V2", "Q3", "V3"],
                "description": "IV bolus, 3-compartment",
                "pk_code": """$PK
CL = THETA(1) * EXP(ETA(1))
V1 = THETA(2) * EXP(ETA(2))
Q2 = THETA(3)
V2 = THETA(4)
Q3 = THETA(5)
V3 = THETA(6)
S1 = V1""",
                "theta": """$THETA
(0.1, 3, 100)   ; CL (L/h)
(1, 30, 200)    ; V1 (L)
(0.1, 5, 50)    ; Q2 (L/h)
(1, 50, 500)    ; V2 (L)
(0.01, 2, 30)   ; Q3 (L/h)
(1, 100, 1000)  ; V3 (L)"""
            },
            ("oral", 3): {
                "advan": "ADVAN12",
                "trans": "TRANS4",
                "params": ["CL", "V2", "Q3", "V3", "Q4", "V4", "KA"],
                "description": "1st-order absorption, 3-compartment",
                "pk_code": """$PK
KA = THETA(1) * EXP(ETA(1))
CL = THETA(2) * EXP(ETA(2))
V2 = THETA(3) * EXP(ETA(3))
Q3 = THETA(4)
V3 = THETA(5)
Q4 = THETA(6)
V4 = THETA(7)
S2 = V2""",
                "theta": """$THETA
(0.1, 1, 10)    ; KA (1/h)
(0.1, 3, 100)   ; CL (L/h)
(1, 30, 200)    ; V2 (L)
(0.1, 5, 50)    ; Q3 (L/h)
(1, 50, 500)    ; V3 (L)
(0.01, 2, 30)   ; Q4 (L/h)
(1, 100, 1000)  ; V4 (L)"""
            }
        }

        key = (route.lower(), compartments)
        if key in advan_guide:
            info = advan_guide[key]
            return f"""
**RECOMMENDED MODEL: {info['advan']} {info['trans']} ({info['description']})**

Parameters: {', '.join(info['params'])}

{info['theta']}

{info['pk_code']}
"""
        else:
            return f"""
**GENERAL ADVAN APPROACH:**
For complex models not covered by special ADVANs, use ADVAN6/13 with $DES
"""

    @staticmethod
    def _get_estimation_method_guidance(n_subjects: int, omega_count: int) -> str:
        """
        Estimation method selection based on dataset size

        Methods:
        - METHOD=ZERO (FO): First Order
        - METHOD=1 (FOCE): First Order Conditional Estimation
        - METHOD=1 INTER (FOCE-I): FOCE with Interaction
        """

        if n_subjects < 15:
            return """
**METHOD=ZERO (First Order) - REQUIRED for N<15**

```
$ESTIMATION METHOD=ZERO MAXEVAL=9999 PRINT=5 POSTHOC
```

Why FO for small datasets:
- More robust and stable
- Faster convergence
- Less sensitive to poor initial estimates
- Lower computational burden
- Better for sparse data

Trade-off: Less accurate for models with high IIV, but stability is critical for N<15
"""
        elif n_subjects < 30:
            return """
**METHOD=1 INTER (FOCE-I) - STANDARD for moderate datasets**

```
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC
```

Why FOCE-I for N=15-30:
- Accounts for correlation between IIV and residual error
- More accurate parameter estimates than FO
- Standard method for population PK
- Good balance between accuracy and stability

Note: Can fallback to METHOD=ZERO if convergence issues occur
"""
        else:
            return """
**METHOD=1 INTER (FOCE-I) - STANDARD for large datasets**

```
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC NOABORT
$COV PRINT=E
```

Why FOCE-I for N>30:
- Most accurate parameter estimates
- Standard for population PK analysis
- Can handle complex models
- Sufficient data to support interaction term
- Include $COV for standard errors

Alternative: METHOD=IMP (Importance Sampling) or METHOD=SAEM for difficult models
"""

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
        Based on systematic pharmacometric model development approach
        """

        # Extract number of subjects from data_summary (DYNAMIC)
        import re
        n_subjects = 50  # Conservative default if extraction fails
        match = re.search(r'Number of subjects:\s*(\d+)', data_summary)
        if match:
            n_subjects = int(match.group(1))  # Use actual N from data

        # Detect route and compartment structure from data
        # DataLoader에서 제공하는 PKGPT_STRUCT_HINT를 우선 사용
        route = "oral"  # default
        compartments = 1  # default

        # e.g. "PKGPT_STRUCT_HINT: ROUTE=IV", "PKGPT_STRUCT_HINT: COMPARTMENTS=2"
        m_route = re.search(r"PKGPT_STRUCT_HINT:\s*ROUTE\s*=\s*(IV|ORAL)", data_summary, re.IGNORECASE)
        if m_route:
            route = m_route.group(1).lower()

        m_comp = re.search(r"PKGPT_STRUCT_HINT:\s*COMPARTMENTS\s*=\s*(\d+)", data_summary)
        if m_comp:
            try:
                c_val = int(m_comp.group(1))
                if c_val >= 1:
                    compartments = c_val
            except ValueError:
                pass

        # Get ADVAN guidance
        advan_guidance = PromptTemplates._get_advan_guidance(route, compartments)

        # Get estimation method guidance
        estimation_guidance = PromptTemplates._get_estimation_method_guidance(n_subjects, omega_count=3)

        # Determine appropriate IIV structure based on dataset size (Ch 3.8, Ch 9)
        if n_subjects < 15:
            iiv_guidance = f"""
**SMALL DATASET (N={n_subjects}<15) - USE DIAGONAL OMEGA WITH ALL PARAMETERS**

For N<15, use diagonal OMEGA on ALL structural parameters (CL, V, KA):
```
$OMEGA
0.2  ; IIV on CL
0.2  ; IIV on V
0.2  ; IIV on KA

$PK
CL = TVCL * EXP(ETA(1))  ; All have IIV
V  = TVV * EXP(ETA(2))
KA = TVKA * EXP(ETA(3))
```

**CRITICAL**: Use METHOD=ZERO (First Order) instead of METHOD=1 INTER:
```
$ESTIMATION METHOD=ZERO MAXEVAL=9999 PRINT=5 POSTHOC
```

**WHY METHOD=ZERO for small datasets:**
- More robust and stable
- Faster convergence
- Less sensitive to poor initial estimates
- Better for N<20

**WHY diagonal OMEGA (not BLOCK):**
- Simpler to estimate (no correlations)
- Fewer parameters
- More stable for small N

**Initial OMEGA values:**
- 0.2 = ~45% CV (reasonable for PK parameters)
- Larger than 0.1 to avoid numerical collapse
"""
        elif n_subjects < 30:
            iiv_guidance = f"""
**MODERATE DATASET (N={n_subjects}, 15-30) - USE 3 DIAGONAL OMEGA**

For N=15-30, use 3 diagonal OMEGA:
```
$OMEGA
0.2  ; IIV on CL
0.2  ; IIV on V
0.2  ; IIV on KA

$PK
CL = TVCL * EXP(ETA(1))
V  = TVV * EXP(ETA(2))
KA = TVKA * EXP(ETA(3))
```

**Use METHOD=1 INTER** (can handle more complexity than FO):
```
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC
```
"""
        else:
            iiv_guidance = f"""
**LARGE DATASET (N={n_subjects}>30) - CAN USE BLOCK OMEGA**

For N>30, can use BLOCK OMEGA (allows correlation):
```
$OMEGA BLOCK(3)
0.2
0.1 0.2
0.1 0.1 0.2

$PK
CL = TVCL * EXP(ETA(1))
V  = TVV * EXP(ETA(2))
KA = TVKA * EXP(ETA(3))
```

Or simpler diagonal:
```
$OMEGA
0.2  ; IIV on CL
0.2  ; IIV on V
0.2  ; IIV on KA
```

**Use METHOD=1 INTER** (standard for rich data):
```
$ESTIMATION METHOD=1 INTER MAXEVAL=9999 PRINT=5 POSTHOC
```
"""

        prompt = f"""You are an expert pharmacometrician developing a population PK model using NONMEM.
Follow the systematic approach for model development.

═══════════════════════════════════════════════════════════════════
DATASET INFORMATION (N={n_subjects} subjects)
═══════════════════════════════════════════════════════════════════
{dataset_info}

{data_summary}

Available columns: {', '.join(columns)}
NONMEM columns: {_format_dict(nonmem_columns)}
Covariates: {', '.join(covariates) if covariates else 'None'}

═══════════════════════════════════════════════════════════════════
SYSTEMATIC MODEL DEVELOPMENT APPROACH
═══════════════════════════════════════════════════════════════════

**PHASE 1: ESTABLISH BASE STRUCTURAL MODEL**

Goal: Build simplest adequate structural model
Strategy: Start with 1-compartment, increase complexity only if justified

{advan_guidance}

**ADVAN Selection Hierarchy:**
1. Start with ADVAN1/2 (1-compartment)
2. If bi-phasic decline observed → ADVAN3/4 (2-compartment)
3. If tri-phasic decline observed → ADVAN11/12 (3-compartment)
4. For complex kinetics → ADVAN6/13 with $DES

**Critical NONMEM Syntax Requirements:**
- List ALL columns in $INPUT in exact order from CSV
- NEVER rename DV or MDV in $INPUT
- For ADVAN2 TRANS2: MUST define K=CL/V and S2=V in $PK
- For ADVAN1 TRANS2: MUST define K=CL/V and S1=V in $PK
- NO $END statement (not required in modern NONMEM)

**PHASE 2: PARAMETER BOUNDS (Initial Estimates)**

Use 3-value format: (lower, initial, upper)
```
$THETA
(0.1, 3, 100)   ; CL (L/h) - lower prevents collapse, upper prevents explosion
(1, 30, 200)    ; V (L)
(0.1, 1, 10)    ; KA (1/h)
```

**Initial value strategy:**
- Start with literature values or physiologically plausible values
- Lower bound: ~1/30 of initial (prevents numerical collapse)
- Upper bound: ~30x initial (prevents unrealistic values)
- Can use FIX keyword to fix parameters: THETA(1) FIX

**PHASE 3: INTER-INDIVIDUAL VARIABILITY (IIV) - OMEGA**

{iiv_guidance}

**OMEGA Structure Types:**
1. **DIAGONAL** - No correlations (simplest, recommended for N<30)
```
$OMEGA
0.2  ; IIV on CL
0.2  ; IIV on V
0.2  ; IIV on KA
```

2. **BLOCK** - With correlations (requires N>30, more parameters)
```
$OMEGA BLOCK(3)
0.2           ; VAR(ETA1) for CL
0.1 0.2       ; COV(ETA1,ETA2), VAR(ETA2) for V
0.1 0.1 0.2   ; COV(ETA1,ETA3), COV(ETA2,ETA3), VAR(ETA3) for KA
```

**IIV Implementation in $PK:**
- Exponential model (recommended for CL, V, KA - ensures positivity):
  ```
  CL = TVCL * EXP(ETA(1))  ; Typical value × random effect
  ```
- Additive model (for already-constrained parameters):
  ```
  ALAG1 = TVLAG + ETA(4)
  ```

**PHASE 4: RESIDUAL ERROR MODEL - SIGMA**

**COMBINED ERROR MODEL (Additive + Proportional) - RECOMMENDED:**
Most robust across concentration ranges
```
$ERROR
IPRED = F
W = SQRT(THETA(X)**2 + (THETA(Y) * IPRED)**2)  ; X, Y = last THETA indices
Y = IPRED + W * EPS(1)

$THETA  ; Add to existing THETA block
(0.01, 0.1, 10)    ; Additive error (same units as DV)
(0.01, 0.1, 0.5)   ; Proportional error (fraction)

$SIGMA
1 FIX  ; Variance fixed at 1 (already in W)
```

**Alternative: Proportional-only** (if combined is too complex for small N):
```
$ERROR
IPRED = F
Y = IPRED * (1 + EPS(1))

$SIGMA
0.04  ; Proportional error variance (~20% CV)
```

**PHASE 5: ESTIMATION METHOD**

{estimation_guidance}

**Troubleshooting convergence issues:**
- If FOCE-I fails → Try METHOD=1 (FOCE without interaction)
- If FOCE fails → Fallback to METHOD=ZERO (FO)
- If still fails → Check initial estimates, simplify OMEGA structure

**PHASE 6: OUTPUT AND DIAGNOSTICS**

**Required $TABLE statements:**
```
$TABLE ID TIME DV PRED IPRED CWRES
       ONEHEADER NOPRINT FILE=sdtab001
```

**Additional tables for diagnostics:**
```
$TABLE ID CL V KA ETA(1) ETA(2) ETA(3)
       ONEHEADER NOPRINT FILE=patab001
```

═══════════════════════════════════════════════════════════════════
COMPLETE CONTROL STREAM STRUCTURE
═══════════════════════════════════════════════════════════════════

Required record order:
1. $PROBLEM - Description
2. $DATA - Input file path
3. $INPUT - Column names (exact order from CSV)
4. $SUBROUTINE - ADVAN and TRANS
5. $PK - Parameter definitions
6. $ERROR - Observation model
7. $THETA - Fixed effects
8. $OMEGA - Random effects (IIV)
9. $SIGMA - Residual error
10. $ESTIMATION - Estimation method
11. $COVARIANCE - Optional, for SE calculation
12. $TABLE - Output tables

═══════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════

Generate a complete NONMEM control stream following the systematic approach above.

**Critical checklist:**
✓ Dataset has N={n_subjects} subjects
✓ Use appropriate ADVAN for route of administration
✓ Use appropriate OMEGA structure for N={n_subjects}
✓ Use appropriate estimation METHOD for N={n_subjects}
✓ Include K=CL/V and S1/S2=V definitions
✓ Use 3-value THETA bounds
✓ Use exponential IIV model (EXP(ETA))
✓ Include $TABLE for diagnostics
✓ Add brief comments for clarity

Generate complete NONMEM control stream now:
"""
        return prompt

    @staticmethod
    def improvement_prompt(
        iteration: int,
        dataset_info: str,
        current_code: str,
        nonmem_output: str,
        previous_improvements: List[Dict],
        previous_models: Optional[List[Dict]] = None,
        issues_found: List[str] = None,
        mandatory_simplification: bool = False,
        simplification_reason: Optional[str] = None,
        current_omega_count: int = 0
    ) -> str:
        """Generate prompt for iterative improvement"""

        previous_models = previous_models or []
        issues_found = issues_found or []
        history_text = _format_improvement_history(previous_improvements)
        issues_text = "\n".join([f"- {issue}" for issue in issues_found]) if issues_found else "None"
        has_ofv = any(h.get('ofv') is not None for h in previous_improvements)

        # Extract dataset size from history (DYNAMIC)
        n_subjects = 50  # Conservative default
        for h in previous_improvements:
            if 'n_subjects' in h:
                n_subjects = h['n_subjects']
                break

        # CRITICAL: 1 OMEGA protection
        one_omega_warning = ""
        if current_omega_count == 1:
            one_omega_warning = f"""
{'='*70}
🛑 CRITICAL: ONLY 1 OMEGA REMAINS - PROTECT IT! 🛑
{'='*70}

Current model has ONLY 1 OMEGA parameter.

**YOU MUST NOT REMOVE IT!**

Why:
- Removing it → non-population model
- METHOD=1 INTER will fail ("SINGLE-SUBJECT DATA" error)
- For N={n_subjects}: shrinkage 50-75% with 1 OMEGA is ACCEPTABLE

What you CAN do:
- Fix syntax errors ✓
- Adjust THETA bounds/initials ✓
- Fix structural issues (K=CL/V) ✓

What you CANNOT do:
- Remove last OMEGA ✗
- Delete $OMEGA block ✗

{'='*70}

"""

        # Simplification section (if mandatory)
        simplification_section = ""
        if mandatory_simplification and current_omega_count > 1:
            simplification_section = f"""
{'🚨'*35}
MANDATORY SIMPLIFICATION REQUIRED
{'🚨'*35}

{simplification_reason}

REQUIRED ACTION (choose one):

1. Remove 1 random effect (e.g., from {current_omega_count} → {current_omega_count-1} OMEGA)
   Example: Remove IIV on KA
   ```
   $PK
   CL = TVCL * EXP(ETA(1))
   V  = TVV * EXP(ETA(2))
   KA = TVKA  ; Fixed (no ETA)

   $OMEGA
   0.1  ; CL only
   0.1  ; V only
   ```

2. Fix problematic parameter to typical value
   Example: Fix KA=1.0, remove its THETA and OMEGA

PROHIBITED:
❌ Add complexity
❌ Just adjust bounds without simplifying

ALLOWED:
✓ Change METHOD (e.g., METHOD=1 → METHOD=ZERO for stability)
✓ Simplify error model if needed
✓ Remove OMEGA (but keep at least 1)

VERIFICATION:
✓ OMEGA count decreased ({current_omega_count} → {current_omega_count-1})
✓ At least 1 OMEGA remains

{'🚨'*35}

"""

        # Format previous models section
        previous_models_section = ""
        if previous_models:
            previous_models_section = "\n" + "="*70 + "\n"
            previous_models_section += "PREVIOUS MODELS (for context to avoid cycling)\n"
            previous_models_section += "="*70 + "\n"
            for i, model_entry in enumerate(previous_models, 1):
                iter_num = model_entry.get('iteration', '?')
                desc = model_entry.get('description', 'No description')
                code = model_entry.get('code', '')

                previous_models_section += f"\n--- Previous Model (Iteration {iter_num}) ---\n"
                previous_models_section += f"Changes made: {desc}\n"
                previous_models_section += f"```\n{code}\n```\n"
            previous_models_section += "="*70 + "\n"
            previous_models_section += "IMPORTANT: Review these previous attempts to avoid repeating failed strategies.\n"
            previous_models_section += "If the current model is cycling back to a previous configuration, try a different approach.\n"
            previous_models_section += "="*70 + "\n\n"

        prompt = f"""Diagnose and fix the NONMEM model.

═══════════════════════════════════════════════════════════════════
ITERATION {iteration} - MODEL REFINEMENT
═══════════════════════════════════════════════════════════════════

DATASET: N={n_subjects} subjects
CURRENT OMEGA COUNT: {current_omega_count}

{one_omega_warning}
{simplification_section}
{previous_models_section}

CURRENT MODEL:
```
{current_code}
```

NONMEM OUTPUT:
```
{nonmem_output}
```

ISSUES IDENTIFIED:
{issues_text}

HISTORY:
{history_text}

═══════════════════════════════════════════════════════════════════
SYSTEMATIC MODEL REFINEMENT STRATEGY
═══════════════════════════════════════════════════════════════════

**CURRENT DEVELOPMENT PHASE:**
{'PHASE 1: ESTABLISH BASE MODEL' if not has_ofv else 'PHASE 2-5: DIAGNOSE & IMPROVE MODEL'}

{'Priority: Fix syntax/structural errors to get stable base model' if not has_ofv else 'Priority: Diagnose quality → Improve structure/variability → Add covariates if stable'}

**PHASE 1: ESTABLISH BASE MODEL - Get it running**
Goal: Stable minimization with reasonable OFV
Actions:
1. Fix syntax errors
   - Missing K=CL/V or S1/S2=V definitions
   - Wrong $INPUT column order
   - Missing semicolons or comments
2. Fix parameter boundaries
   - THETA out of bounds → Adjust (lower, initial, upper)
   - Parameters hitting bounds → Widen or FIX to typical value
3. Fix estimation issues
   - Convergence failure → Try METHOD=ZERO instead of METHOD=1 INTER
   - "SINGLE-SUBJECT DATA" error → Check OMEGA count ≥1
4. Ensure minimization successful (MINIMIZATION SUCCESSFUL in output)

**PHASE 2: DIAGNOSE MODEL QUALITY - Is structure adequate?**
Goal: Check if structural model fits data well
Actions:
1. Check residual patterns
   - Plot CWRES vs TIME
     * Flat random scatter → Structure OK
     * U-shape or systematic trend → Wrong compartment structure
     * Multi-phasic pattern → Need more compartments
   - Plot CWRES vs PRED
     * Random scatter → Model fits across concentration range
     * Funnel shape → Need proportional error component
     * Systematic bias → Structural issue
2. Structural model changes if needed
   - 1-cmt insufficient (bi-phasic decline) → ADVAN2→ADVAN4 or ADVAN1→ADVAN3
   - 2-cmt insufficient (tri-phasic decline) → ADVAN4→ADVAN12 or ADVAN3→ADVAN11
3. Error model adequacy
   - Proportional-only fails at low concentrations → Add additive component
   - Combined error unstable for small N → Simplify to proportional-only

**PHASE 3: REDUCE OVERFITTING - Simplify if needed**
Goal: Ensure model fits signal, not noise
Overfitting indicators:
1. OFV < -50 (extremely negative) → Likely fitting noise
2. Shrinkage > 90-95% → IIV collapsed, ETAs uninformative
3. OMEGA estimates consistently near zero (< 0.001) with very high shrinkage
4. Covariance step failure + high shrinkage → Model overparameterized

Actions when overfitting detected:
1. First review the structural and residual error model.
   - Simplify the number of compartments or the absorption model only if clearly over‑complex.
   - Consider simplifying the residual error model (e.g. combined → proportional) only when diagnostics support it.
2. Only if one or more OMEGA are repeatedly near zero with very high shrinkage and their removal does not meaningfully worsen OFV, consider removing those specific IIV terms (but keep ≥1 OMEGA for a population model).
   - Start with the parameter that has the highest shrinkage.
   - Example: If ETA(KA) shrinkage≈98% and OMEGA(KA) is near zero in multiple runs → try a model without IIV on KA.
3. Fix problematic THETA to a reasonable typical value when it is poorly estimated or stuck on bounds.
   - For example, fix maturation or peripheral volumes if data do not inform them.
4. Change estimation method
   - METHOD=1 INTER → METHOD=ZERO (more robust for very small N)
**CRITICAL: OMEGA=1 PROTECTION**
If only 1 OMEGA remains:
- DO NOT remove it (would become non-population model)
- For N<20: Shrinkage <75% with OMEGA=1 is ACCEPTABLE
- For N>20: Shrinkage <60% with OMEGA=1 is ACCEPTABLE
- Focus on other refinements (THETA bounds, error model, estimation method)

**PHASE 4: OPTIMIZE RANDOM EFFECTS**
Goal: Appropriate IIV structure for dataset size
Actions:
1. Check shrinkage for each ETA
   - Shrinkage <30%: Excellent, retain IIV
   - Shrinkage 30-50%: Good, retain IIV
   - Shrinkage 50-70%: Acceptable, consider retaining
   - Shrinkage 70-90%: Poor, consider removing
   - Shrinkage >90%: Critical, should remove (but keep ≥1 OMEGA)
2. OMEGA structure optimization
   - For N<30: Use DIAGONAL (simpler, more stable)
   - For N>30 with good data: Can consider BLOCK for correlations
3. Check OMEGA estimates
   - Very small (<0.001) → Consider removing
   - Very large (>1.0) → May indicate structural issue

**PHASE 5: COVARIATE ANALYSIS - Explain variability**
Prerequisites (must ALL be true):
- Minimization successful
- Average ETA shrinkage <50% (ETAs are informative)
- No structural misspecification detected
- Base model stable (consistent OFV across iterations)
- Covariates available in dataset

Covariate analysis procedure:
1. **Screening phase** - Identify candidate covariates
   - Plot ETA(CL) vs WT, AGE, SEX, etc.
   - Look for trends or relationships
   - Use graphical or statistical tests (p<0.1)

2. **Forward selection** - Add covariates one at a time
   - Add covariate that reduces OFV most
   - Significance threshold: ΔOFV > 3.84 (p<0.05, df=1)
   - Repeat until no more significant covariates

   Example covariate implementations:
   ```
   ; Weight on CL (power model)
   TVCL = THETA(1) * (WT/70)**THETA(4)
   CL = TVCL * EXP(ETA(1))

   ; Age on V (linear)
   TVV = THETA(2) * (1 + THETA(5) * (AGE-40))
   V = TVV * EXP(ETA(2))

   ; Sex on CL (categorical)
   TVCL = THETA(1)
   IF (SEX.EQ.1) TVCL = TVCL * THETA(6)  ; Male factor
   CL = TVCL * EXP(ETA(1))
   ```

3. **Backward elimination** - Remove non-significant covariates
   - Remove least significant covariate
   - Significance threshold: ΔOFV > 6.63 (p<0.01, df=1) to stay
   - More stringent than forward selection to avoid overfitting
   - Repeat until all remaining covariates are significant

4. **Validation** - Check final covariate model
   - All covariates significant (p<0.01)
   - Covariate effects physiologically plausible
   - IIV reduced (OMEGA estimates smaller than base model)
   - Shrinkage still acceptable (<50%)

**COMMON ISSUES BY PHASE:**
Phase 1: Missing K=CL/V, wrong $INPUT order, boundary issues, convergence failure
Phase 2: U-shaped CWRES (wrong structure), funnel residuals (wrong error model)
Phase 3: Shrinkage >90%, OFV<-50, collapsed OMEGA, covariance failure
Phase 4: Inappropriate OMEGA structure for N, poor shrinkage
Phase 5: Non-significant covariates, covariate overfitting, implausible effects

**MODEL QUALITY INDICATORS:**
Excellent model (Grade A):
- Minimization successful + Covariance OK
- OFV reasonable (not extremely negative)
- Shrinkage <30% average
- CWRES randomly scattered
- Stable parameter estimates

Good model (Grade B):
- Minimization successful
- Shrinkage 30-50%
- Minor residual patterns
- Covariance may have minor issues

Acceptable model (Grade C):
- Minimization successful
- Shrinkage 50-70%
- Some systematic residual patterns
- For small N (N<20): Shrinkage <75% with OMEGA=1 is acceptable

Poor model (Grade D/F):
- Minimization failure or boundary issues
- Shrinkage >70-90%
- Clear structural misspecification
- OFV extremely negative (<-50)

═══════════════════════════════════════════════════════════════════
RESPONSE
═══════════════════════════════════════════════════════════════════

ANALYSIS:
[2-3 sentences: Issue + Root cause + Solution]

IMPROVED CODE:
```
[Complete corrected NONMEM control stream]
```

CHANGES MADE:
[Specific modifications + rationale]

EXPECTED OUTCOME:
[What this should achieve]
"""
        return prompt

    @staticmethod
    def improvement_prompt_v2(
        iteration: int,
        current_code: str,
        nonmem_output: str,
        parsed_results: Dict,
        current_phase: ModelPhase,
        metadata: Dict,
        previous_improvements: List[Dict],
        issues_found: List[str] = None
    ) -> str:
        """
        Generate phase-specific improvement prompt (V2 - Phase-aware)

        This is the new version that routes to specialized phase prompts.
        Much shorter and focused than the legacy improvement_prompt().

        Args:
            iteration: Current iteration number
            current_code: Current NONMEM control stream
            nonmem_output: NONMEM output (.lst file content)
            parsed_results: Parsed NONMEM results (OFV, shrinkage, etc.)
            current_phase: Current ModelPhase enum
            metadata: Dataset metadata (n_subjects, covariates, etc.)
            previous_improvements: History of previous iterations
            issues_found: List of issues identified

        Returns:
            Phase-specific prompt string
        """
        issues_found = issues_found or []
        n_subjects = metadata.get('n_subjects', 50)

        # Extract relevant data from parsed_results
        ofv = parsed_results.get('objective_function')
        minimization_ok = parsed_results.get('minimization_successful', False)
        warnings = parsed_results.get('warnings', [])
        shrinkage_data = parsed_results.get('eta_shrinkage', [])

        # Count current OMEGAs in code
        import re
        omega_count = len(re.findall(r'^\s*\$OMEGA', current_code, re.MULTILINE))
        if omega_count == 0:
            # Count individual OMEGA statements
            omega_count = len(re.findall(r'^\s*[0-9.]+\s*;.*(?:IIV|OMEGA|ETA)', current_code, re.MULTILINE))

        # Extract available covariates and normalize to dict list format
        raw_covariates = metadata.get('covariates', [])
        available_covariates = []

        for cov in raw_covariates:
            if isinstance(cov, dict):
                # Already in dict format
                available_covariates.append(cov)
            elif isinstance(cov, str):
                # Convert string to minimal dict format
                available_covariates.append({
                    'name': cov,
                    'type': 'unknown',
                    'median': 'N/A',
                    'min': 'N/A',
                    'max': 'N/A',
                    'suggested_model': 'linear'
                })

        # Extract current covariates in model (simplified detection)
        current_covariates_in_model = []
        for cov in available_covariates:
            cov_name = cov.get('name', '')
            if cov_name and cov_name in current_code:
                current_covariates_in_model.append(cov_name)

        # Route to appropriate phase prompt
        if current_phase == ModelPhase.ESTABLISH_BASE:
            return Phase1Establish.generate_prompt(
                iteration=iteration,
                current_code=current_code,
                nonmem_output=nonmem_output,
                issues_found=issues_found,
                history=previous_improvements,
                n_subjects=n_subjects
            )

        elif current_phase == ModelPhase.DIAGNOSE_STRUCTURE:
            return Phase2Diagnose.generate_prompt(
                iteration=iteration,
                current_code=current_code,
                nonmem_output=nonmem_output,
                parsed_results=parsed_results,
                warnings=warnings,
                n_subjects=n_subjects
            )

        elif current_phase == ModelPhase.REDUCE_OVERFITTING:
            return Phase3Reduce.generate_prompt(
                iteration=iteration,
                current_code=current_code,
                parsed_results=parsed_results,
                shrinkage_data=shrinkage_data,
                current_omega_count=omega_count,
                n_subjects=n_subjects
            )

        elif current_phase == ModelPhase.OPTIMIZE_IIV:
            return Phase4Optimize.generate_prompt(
                iteration=iteration,
                current_code=current_code,
                parsed_results=parsed_results,
                shrinkage_data=shrinkage_data,
                current_omega_count=omega_count,
                n_subjects=n_subjects
            )

        elif current_phase == ModelPhase.COVARIATE_ANALYSIS:
            return Phase5Covariates.generate_prompt(
                iteration=iteration,
                current_code=current_code,
                parsed_results=parsed_results,
                shrinkage_data=shrinkage_data,
                available_covariates=available_covariates,
                current_covariates_in_model=current_covariates_in_model,
                n_subjects=n_subjects
            )

        else:
            # Fallback to Phase 1 if unknown phase
            print(f"[WARNING] Unknown phase {current_phase}, using Phase 1 prompt")
            return Phase1Establish.generate_prompt(
                iteration=iteration,
                current_code=current_code,
                nonmem_output=nonmem_output,
                issues_found=issues_found,
                history=previous_improvements,
                n_subjects=n_subjects
            )

    @staticmethod
    def quality_evaluation_prompt(
        iteration: int,
        parsed_data: Dict,
        previous_improvements: List[Dict],
        overfitting_warnings: Optional[List[str]] = None
    ) -> str:
        """Generate prompt for model quality evaluation"""

        overfitting_warnings = overfitting_warnings or []

        # Extract metrics
        ofv = parsed_data.get('objective_function')
        minimization_ok = parsed_data.get('minimization_successful', False)

        eta_shrinkage = parsed_data.get('eta_shrinkage', [])
        avg_eta_shrink = sum([s['shrinkage'] for s in eta_shrinkage]) / len(eta_shrinkage) if eta_shrinkage else None

        cov_successful = parsed_data.get('covariance_step', {}).get('successful', False)

        params = parsed_data.get('parameter_estimates', {})
        omega_count = len(params.get('omega', []))

        history_text = _format_improvement_history(previous_improvements)

        # Determine dataset size from history (DYNAMIC)
        n_subjects = 50  # Conservative default
        for h in previous_improvements:
            if 'n_subjects' in h:
                n_subjects = h['n_subjects']
                break

        # Overfitting warnings
        overfitting_section = ""
        if overfitting_warnings:
            overfitting_section = f"""
{'!'*70}
OVERFITTING DETECTED
{'!'*70}
{chr(10).join([f'  - {w}' for w in overfitting_warnings])}
{'!'*70}

These indicate the model is fitting NOISE, not signal.
Simplification is MANDATORY.

"""

        # OMEGA count warning
        omega_warning = ""
        if omega_count == 1:
            omega_warning = f"\n⚠️ **ONLY 1 OMEGA REMAINS - DO NOT recommend removing it!**\n"

        prompt = f"""You are an expert pharmacometrician trained in systematic model development.
Evaluate this NONMEM model using comprehensive diagnostic criteria and decide if optimization should continue.

═══════════════════════════════════════════════════════════════════
ITERATION {iteration} - MODEL QUALITY EVALUATION
═══════════════════════════════════════════════════════════════════
{overfitting_section}

**CURRENT MODEL METRICS:**
- OFV (Objective Function Value): {f'{ofv:.2f}' if ofv is not None else 'N/A'}
- Minimization Status: {'SUCCESS' if minimization_ok else 'FAILED'}
- Covariance Step: {'SUCCESS' if cov_successful else 'FAILED'}
- Average ETA Shrinkage: {f'{avg_eta_shrink:.1f}%' if avg_eta_shrink is not None else 'N/A'}
- OMEGA Count: {omega_count}{omega_warning}
- Dataset Size: N={n_subjects}

**ITERATION HISTORY:**
{history_text}

═══════════════════════════════════════════════════════════════════
MODEL EVALUATION FRAMEWORK
═══════════════════════════════════════════════════════════════════

**CRITICAL DIAGNOSTIC CRITERIA:**

1. **CONVERGENCE ASSESSMENT**
   - Minimization successful: Required for reliable estimates
   - Covariance step successful: Indicates stable curvature at minimum
   - No boundary issues: Parameters not hitting limits
   - Gradient values near zero: True minimum achieved

2. **OBJECTIVE FUNCTION VALUE**
   - OFV magnitude relative to data:
     * OFV > 0: Normal for real data
     * OFV slightly negative (-50 to 0): Usually acceptable
     * OFV < -50: RED FLAG - Likely overfitting noise
   - OFV trend across iterations:
     * Decreasing then stabilizing: Good convergence
     * Still rapidly decreasing: May need more iterations
     * Negative and decreasing: Overfitting developing

3. **SHRINKAGE ASSESSMENT**
   Critical for evaluating IIV informativeness

   **Shrinkage interpretation for N={n_subjects}:**
   - <20%: Excellent - ETAs highly informative
   - 20-30%: Very good - Retain all IIV
   - 30-40%: Good - Retain all IIV
   - 40-50%: Acceptable - Retain IIV
   - 50-60%: Marginal - Consider simplifying if OMEGA>1
   - 60-70%: Poor - Should simplify if OMEGA>1
   - 70-80%: Very poor - Simplification needed if OMEGA>1
   - 80-90%: Critical - Simplification mandatory if OMEGA>1
   - >90%: Collapsed - IIV not estimable, remove if OMEGA>1

   **Context-dependent shrinkage thresholds:**
   - For N<15: Shrinkage <80% acceptable (limited data)
   - For N=15-30: Shrinkage <70% acceptable
   - For N=30-50: Shrinkage <60% target
   - For N>50: Shrinkage <50% expected

   **OMEGA=1 special case:**
   - For N<20 with OMEGA=1: Shrinkage <75% is ACCEPTABLE
   - For N≥20 with OMEGA=1: Shrinkage <60% is ACCEPTABLE
   - Cannot remove last OMEGA (would become non-population model)

4. **PARAMETER ESTIMATE QUALITY**
   - THETA estimates: Within physiologically plausible ranges
   - OMEGA estimates:
     * Too small (<0.001): Parameter collapsed, remove if OMEGA>1
     * Very large (>1.0): May indicate structural issue or poor initial estimates
     * Typical PK: 0.05-0.5 (16-52% CV)
   - SIGMA estimates: Reasonable relative to data variability
   - Standard errors: <50% of estimate (good precision)

5. **MODEL COMPLEXITY vs DATASET SIZE**
   Dataset size appropriateness:
   - N<15: Maximum 1-2 OMEGA, METHOD=ZERO, simple error model
   - N=15-30: Maximum 2-3 OMEGA, METHOD=1 INTER, combined/proportional error
   - N=30-50: 3-4 OMEGA possible, METHOD=1 INTER, can try BLOCK OMEGA
   - N>50: More complex models supported

   **Rule of thumb:** ~5-10 subjects per estimated random effect

6. **OVERFITTING DETECTION**
   Red flags indicating overfitting:
   - OFV extremely negative (<-50)
   - Shrinkage >90% (IIV collapsed)
   - OMEGA estimates <0.001 (numerically zero)
   - Covariance failure despite successful minimization
   - Parameter estimates unstable across similar models

   **Action if overfitting detected:** Simplify model (reduce OMEGA count, simplify error model)

7. **MODEL STABILITY ASSESSMENT**
   - Consistent results across iterations
   - No erratic changes in parameter estimates
   - Gradual OFV improvement
   - No cycling between different solutions

═══════════════════════════════════════════════════════════════════
DECISION RULES - WHEN TO STOP OPTIMIZATION
═══════════════════════════════════════════════════════════════════

**STOP OPTIMIZATION (Model adequate) if ANY of the following:**

1. **Excellent model achieved:**
   - Shrinkage < 40% AND Minimization successful
   - Quality score ≥ 75

2. **Good model for dataset size:**
   - For N<15: Shrinkage <80% AND Minimization OK
   - For N=15-30: Shrinkage <70% AND Minimization OK
   - For N>30: Shrinkage <60% AND Minimization OK

3. **Minimum viable model with OMEGA=1:**
   - OMEGA count = 1 AND Shrinkage <75% (N<20) or <60% (N≥20)
   - Minimization successful
   - This is acceptable for small datasets

4. **Model quality sufficient:**
   - Quality score ≥ 65
   - Minimization successful
   - No critical overfitting signs
   - Stable across last 2-3 iterations

5. **Further improvement unlikely:**
   - OFV stable across last 2-3 iterations (change <5)
   - Shrinkage not improving
   - Already at minimum complexity (OMEGA=1)

**CONTINUE OPTIMIZATION (Model needs improvement) if ANY:**

1. **Critical failures:**
   - Minimization failed
   - Syntax errors or execution errors
   - Parameter boundary violations

2. **Overfitting detected:**
   - OFV < -50 (fitting noise)
   - Shrinkage > 90-95% with OMEGA>1 (should simplify)
   - OMEGA estimates < 0.001 (near-collapsed; consider simplifying IIV/structure, but avoid blindly fixing OMEGA to zero)

3. **Poor quality for dataset size:**
   - For N<20: Shrinkage >80% with OMEGA>1
   - For N=20-50: Shrinkage >70% with OMEGA>1
   - For N>50: Shrinkage >65% with OMEGA>1

4. **Model complexity mismatch:**
   - Too complex for N (e.g., OMEGA=3 for N=10)
   - Should simplify

5. **Unstable or improving:**
   - OFV still rapidly improving (change >10 between iterations)
   - Recent changes showing promise
   - Not yet converged to stable solution

**SPECIAL CASE - OMEGA=1 PROTECTION:**
When OMEGA count = 1:
- DO NOT recommend further OMEGA reduction
- If quality poor: Recommend THETA adjustments, error model changes, or METHOD change
- Shrinkage thresholds relaxed: <75% for N<20, <60% for N≥20 is acceptable
- This is minimum viable population model

═══════════════════════════════════════════════════════════════════
QUALITY SCORE BREAKDOWN (0-100 scale)
═══════════════════════════════════════════════════════════════════

Your evaluation must include detailed scoring across 5 dimensions:

1. **Convergence Score (0-100):**
   - 100: Minimization successful + Covariance OK + No boundary issues
   - 80-99: Minimization successful + Minor covariance issues
   - 60-79: Minimization successful + Covariance failed
   - 40-59: Minimization borderline (rounding errors, near boundary)
   - 0-39: Minimization failed

2. **Precision Score (0-100):**
   - 100: All parameters well-estimated (SE <30% of estimate)
   - 80-99: Most parameters precise (SE 30-50% of estimate)
   - 60-79: Acceptable precision (SE 50-100% of estimate)
   - 40-59: Poor precision (SE >100% of estimate)
   - 0-39: Very poor precision or estimates not available

3. **Shrinkage Score (0-100):**
   Based on average ETA shrinkage and dataset size N={n_subjects}:
   - For N<15: 100-(shrinkage-80) if shrinkage<80, else 100-(shrinkage-80)*2
   - For N=15-30: 100-(shrinkage-70) if shrinkage<70, else 100-(shrinkage-70)*2
   - For N>30: 100-(shrinkage-50) if shrinkage<50, else 100-(shrinkage-50)*2

   Quick reference for N={n_subjects}:
   - Shrinkage <30%: Score 100
   - Shrinkage 30-40%: Score 90-100
   - Shrinkage 40-50%: Score 80-90
   - Shrinkage 50-60%: Score 60-80
   - Shrinkage 60-70%: Score 40-60
   - Shrinkage 70-80%: Score 20-40
   - Shrinkage >80%: Score 0-20

4. **Stability Score (0-100):**
   - 100: OFV reasonable (>-50), parameters plausible, no overfitting signs
   - 80-99: OFV slightly negative (-50 to 0), minor issues
   - 60-79: OFV concerning (<-50) OR some parameter instability
   - 40-59: Multiple stability concerns
   - 0-39: Severe overfitting (OFV <<-50, collapsed parameters)

5. **Utility Score (0-100):**
   Model complexity appropriate for dataset:
   - 100: Complexity perfect for N (e.g., OMEGA=1-2 for N<15, OMEGA=2-3 for N=15-30)
   - 80-99: Slightly suboptimal but acceptable
   - 60-79: Some mismatch (too simple or too complex)
   - 40-59: Clear mismatch (e.g., OMEGA=3 for N=10)
   - 0-39: Severe mismatch or model not useful

**Overall Quality Score:** Weighted average
- Convergence: 30%
- Precision: 15%
- Shrinkage: 25%
- Stability: 20%
- Utility: 10%

**Model Grading Scale:**
- A (90-100): Excellent model - Publication ready
- B (75-89): Good model - Minor improvements possible
- C (60-74): Acceptable model - Some issues remain
- D (45-59): Poor model - Major issues need addressing
- F (0-44): Failed model - Requires substantial rework

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT (Return ONLY valid JSON)
═══════════════════════════════════════════════════════════════════

{{
  "should_continue": true or false,
  "quality_score": 0-100,
  "score_breakdown": {{
    "convergence": 0-100,
    "precision": 0-100,
    "shrinkage": 0-100,
    "stability": 0-100,
    "utility": 0-100
  }},
  "decision_reason": "Brief explanation based on evaluation criteria (2-3 sentences)",
  "critical_issues": ["List specific issues based on diagnostic criteria"],
  "model_grade": "A/B/C/D/F",
  "recommendations_if_continuing": [
    "Specific actionable recommendations based on phase analysis"
  ]
}}

**Important notes for decision_reason:**
- Explain why STOP or CONTINUE based on criteria above
- If OMEGA=1, acknowledge this is minimum viable model
- Be specific about which phase the model is in (Phase 1-5)

**Important notes for recommendations_if_continuing:**
- Only provide if should_continue=true
- Be specific and actionable (not generic advice)
- Prioritize based on current phase
- If OMEGA=1, DO NOT recommend removing it

Evaluate the model now using these comprehensive criteria:
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
        lines.append(f"Iteration {i}: {item.get('status', 'unknown')}")
        if 'ofv' in item and item['ofv'] is not None:
            lines.append(f"  OFV: {item['ofv']:.2f}")
        shrink = item.get('avg_eta_shrinkage')
        if shrink is not None:
            lines.append(f"  Shrinkage: {shrink:.1f}%")
        lines.append("")

    return "\n".join(lines)
