# PKGPT Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

Or export it:
```bash
export GEMINI_API_KEY='your_api_key_here'
```

### 3. Run Optimization

Basic usage:
```bash
python pkgpt_optimizer.py dataset/theophylline_nonmem.csv output_theo
```

With custom settings:
```bash
python pkgpt_optimizer.py dataset/warfarin_nonmem.csv output_warf \
    --min-iter 3 \
    --max-iter 10 \
    --nmfe nmfe75
```

## Example Workflow

### Step 1: Prepare Your Data

Ensure your CSV file has NONMEM-standard columns:
- `ID` - Subject identifier
- `TIME` - Time points
- `AMT` - Dose amounts
- `DV` - Observations (concentrations)
- `EVID` - Event ID (1=dose, 0=observation)
- `MDV` - Missing DV flag
- Covariates: `WT`, `AGE`, `SEX`, etc.

Example:
```csv
ID,TIME,AMT,DV,EVID,MDV,WT,SEX
1,0.0,100,,1,1,75,M
1,0.5,,5.2,0,0,75,M
1,1.0,,8.1,0,0,75,M
...
```

### Step 2: Run Optimizer

```bash
python pkgpt_optimizer.py mydata.csv output_model
```

This will:
1. Analyze your dataset structure
2. Generate initial NONMEM model
3. Run NONMEM (or create mock output if not installed)
4. Parse results and identify issues
5. Iteratively improve the model (3-10 iterations)
6. Save the best model

### Step 3: Review Results

Output files:
```
output_model_iter0.txt    <- Initial model
output_model_iter1.txt    <- Iteration 1
output_model_iter1.lst    <- NONMEM output 1
output_model_iter2.txt    <- Iteration 2
output_model_iter2.lst    <- NONMEM output 2
...
output_model_final.txt    <- Best model (COPY THIS!)
```

### Step 4: Transfer to NONMEM System

If NONMEM is on another computer:

1. Copy `output_model_final.txt` to NONMEM system
2. Copy your dataset CSV file
3. Run NONMEM:
   ```bash
   nmfe75 output_model_final.txt output_model_final.lst
   ```

## Command Line Options

```
python pkgpt_optimizer.py [OPTIONS] <data_file> <output_base>

Required Arguments:
  data_file          Path to CSV dataset (NONMEM format)
  output_base        Base name for output files (no extension)

Optional Arguments:
  --min-iter N       Minimum iterations (default: 3)
  --max-iter N       Maximum iterations (default: 10)
  --nmfe CMD         NONMEM command (default: nmfe75)
  --api-key KEY      Override GEMINI_API_KEY env var
  -h, --help         Show help message
  --version          Show version
```

## Examples by Drug Type

### Oral Administration (e.g., Theophylline)

```bash
python pkgpt_optimizer.py dataset/theophylline_nonmem.csv theo_model
```

Expected model features:
- 1-compartment with absorption
- KA (absorption rate constant)
- Weight effects on CL and V

### IV Administration (e.g., some antibiotics)

Prepare data with RATE column or RATE=0 for bolus:
```bash
python pkgpt_optimizer.py mydata_iv.csv iv_model
```

### Multiple Compartments

For drugs with complex distribution:
```bash
python pkgpt_optimizer.py mydata_complex.csv complex_model --max-iter 15
```

System may generate 2-compartment model if data supports it.

## Troubleshooting

### Issue: API Rate Limits

**Error:**
```
[WARNING] Error generating response: 429 Too Many Requests
```

**Solution:**
- Wait a few minutes between runs
- The system automatically retries with exponential backoff

### Issue: NONMEM Not Found

**Message:**
```
[WARNING] NONMEM command 'nmfe75' not found
```

**This is OK!** The system will:
- Generate NONMEM code anyway
- Create mock output for testing
- You can transfer files to NONMEM system later

**If you have NONMEM but it's not found:**
- Add NONMEM to PATH, or
- Use `--nmfe` with full path:
  ```bash
  python pkgpt_optimizer.py data.csv output --nmfe /path/to/nmfe75
  ```

### Issue: Model Not Converging

If all iterations fail:

1. **Check your data:**
   - Are there enough observations?
   - Is DV column properly formatted?
   - Are there outliers?

2. **Simplify the model:**
   - The system will automatically try this
   - But you can guide it by editing prompts in `modules/prompt_templates.py`

3. **Increase iterations:**
   ```bash
   python pkgpt_optimizer.py data.csv output --max-iter 20
   ```

### Issue: Unicode/Encoding Errors

If you see encoding errors:
- Files use UTF-8 encoding
- Check your terminal supports UTF-8
- On Windows, this should now be fixed (using [OK]/[ERROR] instead of symbols)

## Advanced Customization

### Custom Initial Estimates

Edit the generated `output_model_iter0.txt` file before running NONMEM.

### Custom Prompts

Modify prompt templates in `modules/prompt_templates.py`:
- `initial_generation_prompt()` - Initial model generation
- `improvement_prompt()` - Iterative improvements
- `convergence_check_prompt()` - Convergence assessment

### Custom Parsing

Extend `modules/nonmem_parser.py` to extract:
- Additional statistics
- Custom warnings
- Model diagnostics

## Best Practices

1. **Data Quality First:**
   - Clean your data before optimization
   - Check for missing values
   - Verify column names match NONMEM standards

2. **Start Simple:**
   - Use default settings first
   - Let the system learn from your data
   - Review generated code before using in production

3. **Iterate and Refine:**
   - Review iteration history
   - Understand why changes were made
   - Use insights for future models

4. **Validate Results:**
   - Always review generated NONMEM code
   - Check parameter estimates are reasonable
   - Perform model diagnostics
   - Never use AI-generated models in production without expert review

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review example files in `examples/` directory
- Check generated code for comments explaining decisions
- Contact: Hyunseung Kong (hskong@snu.ac.kr)

## Citation

If you use PKGPT in your research:
```
PKGPT - Pharmacokinetic NONMEM Optimizer using Google Gemini AI
Author: Hyunseung Kong (hskong@snu.ac.kr)
Seoul National University
```

---

**Happy Modeling!**
