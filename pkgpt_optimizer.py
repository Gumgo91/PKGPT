#!/usr/bin/env python
"""
PKGPT - Pharmacokinetic NONMEM Optimizer using Google Gemini
Recursive optimization of NONMEM control stream files
"""

import sys
import os
import argparse
from dotenv import load_dotenv

# Add modules directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.optimizer import NONMEMOptimizer


def main():
    """Main CLI entry point"""

    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description='PKGPT - Recursive NONMEM Model Optimizer using Google Gemini AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pkgpt_optimizer.py dataset/theophylline_nonmem.csv output_theo

  # Specify iterations
  python pkgpt_optimizer.py dataset/warfarin_nonmem.csv output_warf --min-iter 5 --max-iter 15

  # Custom NONMEM command
  python pkgpt_optimizer.py data.csv output --nmfe nmfe74

Environment Variables:
  GEMINI_API_KEY    Google Gemini API key (required)

The optimizer will:
  1. Analyze your pharmacokinetic dataset
  2. Generate an initial NONMEM control stream
  3. Execute NONMEM and parse results
  4. Recursively improve the model (3-10 iterations)
  5. Save the best model as <output>_final.txt

Output files:
  <output>_iter0.txt       Initial model
  <output>_iter1.txt       First iteration
  <output>_iter1.lst       NONMEM output for iteration 1
  ...
  <output>_final.txt       Best model (copy of best iteration)
        """
    )

    parser.add_argument(
        'data_file',
        help='Path to input CSV dataset (NONMEM format)'
    )

    parser.add_argument(
        'output_base',
        help='Base name for output files (without extension)'
    )

    parser.add_argument(
        '--min-iter',
        type=int,
        default=3,
        metavar='N',
        help='Minimum number of iterations (default: 3)'
    )

    parser.add_argument(
        '--max-iter',
        type=int,
        default=10,
        metavar='N',
        help='Maximum number of iterations (default: 10)'
    )

    parser.add_argument(
        '--nmfe',
        default='nmfe75',
        metavar='CMD',
        help='NONMEM execution command (default: nmfe75)'
    )

    parser.add_argument(
        '--model',
        choices=['flash', 'flash-lite', 'pro'],
        default='flash',
        help='Gemini model to use: flash (default), flash-lite, or pro'
    )

    parser.add_argument(
        '--api-key',
        help='Google Gemini API key (overrides GEMINI_API_KEY env var)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='PKGPT v1.0.0'
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    if args.min_iter < 1:
        print("Error: Minimum iterations must be >= 1")
        sys.exit(1)

    if args.max_iter < args.min_iter:
        print("Error: Maximum iterations must be >= minimum iterations")
        sys.exit(1)

    # Check for API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it in your environment or .env file, or use --api-key")
        print("\nExample:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        print("  python pkgpt_optimizer.py data.csv output")
        sys.exit(1)

    try:
        # Create optimizer
        optimizer = NONMEMOptimizer(
            data_file=args.data_file,
            output_base=args.output_base,
            api_key=api_key,
            min_iterations=args.min_iter,
            max_iterations=args.max_iter,
            nmfe_command=args.nmfe,
            model=args.model
        )

        # Run optimization
        results = optimizer.run()

        # Success
        if results['final_file']:
            print(f"\n[OK] Optimization completed successfully!")
            print(f"[OK] Best model saved to: {results['final_file']}")
            return 0
        else:
            print("\n[WARNING] Optimization completed with issues")
            print("[WARNING] Check iteration files for details")
            return 1

    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user")
        return 130

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
