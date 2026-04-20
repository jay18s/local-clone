#!/usr/bin/env python3
"""
ROX Proven Edge Engine — main_production.py  [RETIRED]
=======================================================
This file was the v3.0 entry point.  It has been RETIRED as of v4.0.

Migration guide
---------------
All functionality has been consolidated into ``main.py`` (v4.0 Unified).

  Old command                          New equivalent
  ──────────────────────────────────── ──────────────────────────────────────
  python main_production.py plan        python main.py --mode paper
  python main_production.py status      python main.py --mode demo
  python main_production.py test        python main.py --mode demo
  python main_production.py weekly      python main.py --mode demo
  python main_production.py backtest    python main.py --mode backtest \\
                                            --start-date YYYY-MM-DD \\
                                            --end-date   YYYY-MM-DD

This file re-invokes ``main.py`` with translated arguments so any existing
scripts / cron jobs keep working.  Update them to use main.py directly to
silence this warning.
"""

import sys
import warnings
import subprocess
from pathlib import Path

_BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║  ⚠  DEPRECATION WARNING                                         ║
║  main_production.py is RETIRED — use main.py instead.           ║
║  See the file header for a command migration guide.              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Map v3 positional commands → v4 --mode flags
_CMD_MAP = {
    "plan":     ["--mode", "paper"],
    "demo":     ["--mode", "demo"],
    "test":     ["--mode", "demo"],
    "status":   ["--mode", "demo"],
    "weekly":   ["--mode", "demo"],
    "backtest": ["--mode", "backtest"],
}


def main():
    warnings.warn(
        "main_production.py is retired. Use 'python main.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print(_BANNER, file=sys.stderr)

    project_root = Path(__file__).parent
    main_py      = project_root / "main.py"

    # Translate first positional arg (v3 command) if present
    v3_args  = sys.argv[1:]
    v4_args: list = []

    if v3_args:
        cmd = v3_args[0].lower()
        if cmd in _CMD_MAP:
            v4_args = list(_CMD_MAP[cmd])
            # Carry through known compatible flags
            for arg in v3_args[1:]:
                if arg.startswith("--portfolio"):
                    # v3: --portfolio / -p   v4: --portfolio-value
                    v4_args += ["--portfolio-value"] + arg.split("=")[1:]
                elif arg.startswith("--start-date") or arg.startswith("--end-date"):
                    v4_args.append(arg)
                elif arg.startswith("--watchlist") or arg.startswith("-w"):
                    v4_args.append(arg)
                # other v3-only flags are silently dropped
        else:
            # Unknown command — pass through; main.py will error cleanly
            v4_args = list(v3_args)

    cmd_line = [sys.executable, str(main_py)] + v4_args
    print(f"[main_production] Forwarding to: {' '.join(cmd_line)}", file=sys.stderr)
    sys.exit(subprocess.call(cmd_line))


if __name__ == "__main__":
    main()
