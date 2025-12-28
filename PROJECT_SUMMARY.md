# PKGPT Project Summary

## Project Overview

**PKGPT** (Pharmacokinetic GPT) is an AI-powered recursive optimizer for NONMEM population pharmacokinetic models using Google Gemini API.

## Authors

**Hyunseung Kong¹*** and **Hoyoung Kwack²*** (Co-first authors, contributed equally)

¹ Seoul National University (hskong@snu.ac.kr)
² Yonsei University (hoyoung0104@yonsei.ac.kr)

\* These authors contributed equally to this work.

## Key Features

1. **Automated NONMEM Code Generation**
   - Analyzes dataset structure automatically
   - Generates complete, executable NONMEM control streams
   - Includes appropriate structural models, covariates, and error models

2. **Recursive Model Optimization**
   - 3-10 iterations of automatic improvement
   - Parses NONMEM output and identifies issues
   - Suggests and implements fixes iteratively
   - Converges when no further improvement is possible

3. **Multi-Model AI System**
   - Uses 3 Gemini models in rotation:
     - gemini-2.5-pro (initial generation, complex analysis)
     - gemini-flash-latest (balanced performance)
     - gemini-flash-lite-latest (quick iterations)

4. **Smart Data Analysis**
   - Automatic detection of NONMEM standard columns
   - Identification of subject-level covariates
   - Data quality assessment
   - Summary statistics generation

5. **Robust Error Handling**
   - Works without NONMEM installed (mock mode)
   - Automatic retry with exponential backoff
   - Windows-compatible (no Unicode issues)
   - Environment variable management for API keys

## Project Structure

```
PKGPT/
├── pkgpt_optimizer.py          # Main CLI script
├── requirements.txt             # Python dependencies
├── README.md                    # Complete documentation
├── USAGE.md                     # Usage guide
├── PROJECT_SUMMARY.md           # This file
├── .env                         # API key (gitignored)
├── .env.example                 # API key template
├── .gitignore                   # Git ignore rules
│
├── modules/                     # Core modules
│   ├── __init__.py
│   ├── data_loader.py          # Data loading & analysis
│   ├── gemini_client.py        # Gemini API client
│   ├── prompt_templates.py     # AI prompt templates
│   ├── nonmem_parser.py        # NONMEM output parser
│   └── optimizer.py            # Recursive optimizer engine
│
├── dataset/                     # Example datasets
│   ├── theophylline_nonmem.csv
│   ├── warfarin_nonmem.csv
│   ├── pkpdRemifentanil_nonmem.csv
│   └── pheno_sd.csv
│
└── examples/                    # Reference files
    ├── example.txt             # NONMEM control stream
    └── example.lst             # NONMEM output
```

## Technical Stack

**Language:** Python 3.8+

**Core Dependencies:**
- pandas >= 2.0.0 (data manipulation)
- google-generativeai >= 0.3.0 (Gemini API)
- python-dotenv >= 1.0.0 (environment management)

**AI Models:**
- Google Gemini 2.5 Pro
- Google Gemini Flash
- Google Gemini Flash Lite

## Installation & Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Run optimizer
python pkgpt_optimizer.py dataset/theophylline_nonmem.csv output
```

## Usage Examples

### Basic Usage
```bash
python pkgpt_optimizer.py dataset/theophylline_nonmem.csv theo_output
```

### Advanced Usage
```bash
python pkgpt_optimizer.py dataset/warfarin_nonmem.csv warf_output \
    --min-iter 5 \
    --max-iter 15 \
    --nmfe nmfe75
```

### With Custom API Key
```bash
python pkgpt_optimizer.py data.csv output --api-key YOUR_API_KEY
```

## Output Files

For each run:
- `<output>_iter0.txt` - Initial model
- `<output>_iter1.txt` - First iteration
- `<output>_iter1.lst` - NONMEM output 1
- `<output>_iter2.txt` - Second iteration
- `<output>_iter2.lst` - NONMEM output 2
- ... (continues)
- `<output>_final.txt` - **Best model**

## Git Configuration

The project includes comprehensive `.gitignore` rules to protect:

### Excluded from Git:
- `.env` (API keys and secrets)
- `output_*`, `test_*` (temporary files)
- `*.lst`, `*.ext`, etc. (NONMEM outputs)
- `.claude/`, `.vscode/`, `.idea/` (IDE files)
- `__pycache__/`, `*.pyc` (Python cache)
- `.DS_Store`, `Thumbs.db` (OS files)

### Included in Git:
- `.env.example` (template without secrets)
- Source code (`.py` files)
- Documentation (`.md` files)
- Example datasets (`.csv` files)
- Example NONMEM input (`.txt` files)

## Key Algorithms

### 1. Data Analysis Pipeline
```python
Load CSV → Detect columns → Identify covariates → Generate statistics
```

### 2. Optimization Loop
```python
Generate initial model
↓
FOR iteration 1 to max_iterations:
    Execute NONMEM
    ↓
    Parse output (.lst file)
    ↓
    Identify issues (OFV, parameters, warnings)
    ↓
    IF converged AND iteration >= min_iterations:
        BREAK
    ↓
    Generate improved model (using AI)
    ↓
    NEXT iteration
↓
Select best model (lowest OFV, successful minimization)
```

### 3. AI Prompt Strategy

**Initial Generation:**
- Dataset structure analysis
- Column mapping
- Covariate identification
- Generate complete NONMEM code

**Iterative Improvement:**
- Current code + NONMEM output
- Error/warning analysis
- History of previous changes
- Generate single focused improvement

**Convergence Check:**
- OFV trajectory
- Parameter stability
- Error/warning status
- Recommend continue/stop

## Performance Characteristics

- **Initial generation:** 30-60 seconds (Gemini Pro)
- **Each iteration:** 10-30 seconds (varies by model)
- **Total runtime:** 2-5 minutes for 3 iterations (without NONMEM)
- **With NONMEM:** Add 1-10 minutes per NONMEM run

## Testing Status

✅ **Tested and Working:**
- Data loading and analysis
- Gemini API integration
- NONMEM code generation
- Iterative improvement (3 iterations)
- Output file management
- Windows compatibility
- API key environment management
- .gitignore protection

⚠️ **Requires NONMEM System:**
- Actual NONMEM execution
- Real convergence assessment
- Full optimization cycle with feedback

## Future Enhancements

Potential improvements:
1. Support for additional PK models (PKPD, absorption models)
2. Graphical diagnostic plots
3. Model comparison metrics
4. Batch processing multiple datasets
5. Web interface
6. Integration with other PK tools

## License

MIT License - See LICENSE file for details

## Citation

```
PKGPT - Pharmacokinetic NONMEM Optimizer using Google Gemini AI
Authors: Hyunseung Kong* and Hoyoung Kwack* (*equal contribution)
Seoul National University & Yonsei University
Contact: hskong@snu.ac.kr, hoyoung0104@yonsei.ac.kr
```

## Acknowledgments

- Google Gemini API for AI capabilities
- NONMEM community for pharmacometric standards
- Open-source Python community

## Contact & Support

**Primary Contact:**
- Hyunseung Kong (hskong@snu.ac.kr)
- Hoyoung Kwack (hoyoung0104@yonsei.ac.kr)

**Documentation:**
- README.md - Complete documentation
- USAGE.md - Usage guide
- Examples in examples/ directory

---

**Version:** 1.0.0
**Last Updated:** October 2025
**Status:** Production Ready (requires NONMEM for full functionality)
