"""Extract text-only features, evaluate frozen Ling models, verify, and report."""
import argparse
import subprocess
import sys

from ling_common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    extraction_args = ["--reuse"] if args.reuse_features else []
    for script, extra in [("extract_features.py", extraction_args), ("evaluate_intel_ling.py", []),
                          ("verify_results.py", []), ("build_comparison.py", [])]:
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *extra], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
