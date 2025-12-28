"""
Phase 4: Optimize Inter-Individual Variability (IIV)
Focus: Fine-tune OMEGA structure for optimal balance
"""

from typing import Dict, List, Optional


class Phase4Optimize:
    """Phase 4 prompts for optimizing IIV structure"""

    @staticmethod
    def generate_prompt(
        iteration: int,
        current_code: str,
        parsed_results: Dict,
        shrinkage_data: List[Dict],
        current_omega_count: int,
        n_subjects: int
    ) -> str:
        """
        Generate Phase 4 prompt focused on IIV optimization

        Goals:
        - Fine-tune OMEGA structure (DIAGONAL vs BLOCK)
        - Optimize OMEGA initial values
        - Balance complexity vs informativeness
        - Achieve shrinkage <50% if possible
        """

        ofv = parsed_results.get('objective_function', 'N/A')
        minimization_ok = parsed_results.get('minimization_successful', False)

        # Format shrinkage
        shrinkage_text = ""
        if shrinkage_data:
            avg_shrink = sum([s['shrinkage'] for s in shrinkage_data]) / len(shrinkage_data)
            shrinkage_text = f"Average Shrinkage: {avg_shrink:.1f}%\n\n"
            for s in shrinkage_data:
                eta_num = s.get('eta_number', '?')
                shrink = s.get('shrinkage', 0)
                quality = ("Excellent" if shrink < 30 else
                          "Good" if shrink < 50 else
                          "Acceptable" if shrink < 70 else "Poor")
                shrinkage_text += f"  ETA({eta_num}): {shrink:.1f}% ({quality})\n"

        prompt = f"""You are optimizing the inter-individual variability (IIV) structure.

═══════════════════════════════════════════════════════════════════
PHASE 4: OPTIMIZE IIV STRUCTURE
═══════════════════════════════════════════════════════════════════

ITERATION: {iteration}
DATASET: N={n_subjects} subjects
OMEGA COUNT: {current_omega_count}
MINIMIZATION: {'✓ Success' if minimization_ok else '✗ Failed'}
OFV: {ofv}

{shrinkage_text}

CURRENT MODEL:
```
{current_code}
```

═══════════════════════════════════════════════════════════════════
OPTIMIZATION STRATEGY
═══════════════════════════════════════════════════════════════════

**Goal: Achieve Shrinkage <50% if Possible**

**Check 1: OMEGA Structure Type**

Current structure (check $OMEGA block):
- DIAGONAL: Separate OMEGAs, no correlations
- BLOCK: Correlated OMEGAs, more complex

Recommendation by dataset size:
- N < 20: Use DIAGONAL only (insufficient data for correlations)
- N = 20-50: DIAGONAL preferred (BLOCK if strong rationale)
- N > 50: Can consider BLOCK if improves fit significantly

**Check 2: OMEGA Initial Values**

OMEGA represents variance of IIV:
- Too small (0.01): May underestimate variability
- Good range (0.05-0.3): Typical for most PK parameters
- Too large (>0.5): May indicate structural issue

Typical values by parameter type:
- CL, V: 0.1-0.3 (10-30% CV)
- KA: 0.2-0.5 (absorption more variable)
- Bioavailability: 0.1-0.2

**Check 3: Shrinkage Quality**

Interpretation:
- <30%: Excellent - ETAs well-informed by data
- 30-50%: Good - Acceptable information content
- 50-70%: Acceptable - Limited information
- >70%: Poor - Consider removing in Phase 3

Target for this phase: All ETAs <50% if possible

═══════════════════════════════════════════════════════════════════
OPTIMIZATION ACTIONS
═══════════════════════════════════════════════════════════════════

**Action 1: Adjust OMEGA Initials**

If shrinkage is moderate (50-70%):
```
; Try larger OMEGA initial (allows more variability)
$OMEGA
0.2  ; Instead of 0.1 - allows model to explore more IIV
```

If shrinkage is very low (<20%) but OFV poor:
```
; Try smaller OMEGA initial (less variability)
$OMEGA
0.05  ; More constrained IIV
```

**Action 2: BLOCK vs DIAGONAL Decision**

Only consider BLOCK if:
- N > 50 subjects
- Clinical rationale for correlation (e.g., CL and V often correlated)
- DIAGONAL model stable and working

Example BLOCK(2) for CL and V:
```
$OMEGA BLOCK(2)
0.2           ; VAR(ETA1) for CL
0.1 0.2       ; COV(ETA1,ETA2), VAR(ETA2) for V
```

**Action 3: Bounds on OMEGA**

If OMEGA estimates hitting boundaries:
```
$OMEGA
(0.01, 0.1, 0.5)  ; (lower, initial, upper) - prevent collapse or explosion
```

**Action 4: Fine-tune Estimation**

If using METHOD=ZERO, try upgrading:
```
; More accurate for IIV estimation
$ESTIMATION METHOD=1 INTER MAXEVAL=9999
```

Only if N > 20 and model stable.

═══════════════════════════════════════════════════════════════════
DATASET-SPECIFIC RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════

For N < 20 (Very Small Dataset):
- Keep OMEGA count ≤ 2
- Use DIAGONAL only
- Accept shrinkage <70%
- Focus on stability over perfect shrinkage

For N = 20-50 (Medium Dataset):
- OMEGA count 2-3 is reasonable
- DIAGONAL preferred
- Target shrinkage <50%
- Can use METHOD=1 INTER

For N > 50 (Large Dataset):
- Can support 3-4 OMEGAs
- BLOCK may be beneficial
- Target shrinkage <30%
- Use METHOD=1 INTER

═══════════════════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════════════════

IIV ASSESSMENT:
[Is the current IIV structure appropriate for N={n_subjects}? Are shrinkages acceptable?]

OPTIMIZATION PLAN:
[What adjustments to make: OMEGA values, structure type, estimation method?]

IMPROVED CODE:
```
[Complete NONMEM control stream with optimized IIV]
```

CHANGES MADE:
[List specific OMEGA adjustments and rationale]

EXPECTED OUTCOME:
[Improved shrinkage or confirmation that structure is optimal]
"""

        return prompt
