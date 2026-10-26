"""Train the severity, connection, or final-adaptation stage."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--stage', choices=['severity', 'connection', 'final'], required=True)
    if any(arg in sys.argv for arg in ['-h', '--help']) and '--stage' not in sys.argv:
        parser.print_help()
        print('\nUse --stage STAGE --help for training options.')
        return
    args, remaining = parser.parse_known_args()
    if not any(arg == '--config' or arg.startswith('--config=') for arg in remaining):
        config = Path(__file__).resolve().parent / 'configs' / f'train_{args.stage}.yaml'
        remaining = ['--config', str(config), *remaining]
    sys.argv = [sys.argv[0], *remaining]
    module = 'severity' if args.stage == 'severity' else 'restoration'
    importlib.import_module(f'training.{module}').main()


if __name__ == '__main__':
    main()
