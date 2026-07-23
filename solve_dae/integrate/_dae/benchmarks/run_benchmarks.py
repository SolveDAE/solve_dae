import os
import sys

# These benchmark scripts are dev-only tools that are not installed as part
# of the `solve_dae` package (see benchmarks/meson.build, which only installs
# __init__.py and common.py) and the per-problem subdirectories are plain
# directories, not packages (no __init__.py). Make the imports below work
# regardless of the caller's current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arevalo.arevalo import run_arevalo
from brenan.brenan import run_brenan
from knife_edge.knife_edge import run_knife_edge
from kvaerno.kvaerno import run_kvaerno
from particle.particle import run_particle
from robertson.robertson import run_robertson
from weissinger.weissinger import run_weissinger


if __name__ == "__main__":
    run_arevalo()
    run_brenan()
    run_knife_edge()
    run_kvaerno()
    run_particle()
    run_robertson()
    run_weissinger()
