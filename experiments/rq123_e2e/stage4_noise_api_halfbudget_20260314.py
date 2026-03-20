"""
Half-budget Stage4 API runner.

Purpose:
- Keep the same representative sampled logic as stage4_noise_api_sampled_20260313.py
- Cut DeepSeek calls roughly in half by using 5 cases per dataset

Call count:
- 3 datasets × 5 cases × 3 noise levels × 3 methods = 135 API calls
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq123_e2e.stage4_noise_api_sampled_20260313 import run_stage4_noise_api_sampled  # type: ignore


if __name__ == "__main__":
    run_stage4_noise_api_sampled(cases_per_dataset=5, noise_levels=[0.0, 0.6, 1.0])
