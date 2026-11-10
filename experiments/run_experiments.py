"""Entry point wrapper; the canonical driver is ../run_experiments.py."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_experiments import main
if __name__ == "__main__":
    main()
