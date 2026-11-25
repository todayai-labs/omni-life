"""Command line entry point for Omni-Life evaluation."""
import argparse
from importlib import import_module
from pathlib import Path
import runpy
import sys


SCORERS = {
    'reproduce': ('reproduce', 'Verify the results shipped in this repository'),
    'qa': ('qa', 'Build a QA prompt or score QA answers'),
    'streaming': ('score_streaming', 'Score adjudicated Streaming captures'),
    'ablation': ('ablation', 'Score ablations paired with a Streaming baseline'),
}
NATIVE = {
    'native-run': ('stream-runtime/run_native.py', 'Verify or run a local Streaming worker'),
    'native-ablation': ('ablations/runner/native/run_arm.py', 'Verify or run a local ablation'),
    'api-run': ('stream-runtime/api/run.py', 'Verify or run an API Streaming session'),
    'api-ablation': ('ablations/runner/api/run.py', 'Verify or run an API ablation'),
    'export-streaming': ('stream-runtime/export_native.py', 'Export local Streaming captures for adjudication'),
    'export-ablation': ('ablations/runner/native/export_arm.py', 'Export local ablation captures for adjudication'),
}


def main():
    commands = SCORERS | NATIVE
    parser = argparse.ArgumentParser(prog='python -m evaluation', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Commands\n'+'\n'.join(f'  {name:<18} {entry[1]}' for name,entry in commands.items()))
    parser.add_argument('command', choices=commands, nargs='?')
    argv = sys.argv[1:]
    command = parser.parse_args(argv[:1]).command
    if not command:
        parser.print_help()
        return
    sys.argv = [f'python -m evaluation {command}', *argv[1:]]
    if command in SCORERS:
        import_module(f'evaluation.{SCORERS[command][0]}').main()
    else:
        script = Path(__file__).resolve().parent/'native'/NATIVE[command][0]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
