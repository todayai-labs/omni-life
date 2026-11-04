"""One entry point for reproduction, new inference, and release checks."""
import argparse
from importlib import import_module
from pathlib import Path
import sys
import unittest


COMMANDS = {
    "score": ("score", "Recompute saved scores offline; --check verifies the published reference"),
    "validate": ("validate_release", "Validate dataset schemas, coverage, and reference results offline"),
    "check": ("check_release", "Verify frozen file hashes, release hygiene, and local links offline"),
    "media": ("check_media_rights", "Check media review records; never grants data access or usage rights"),
    "prepare": ("prepare_media", "Prepare separately obtained source media with ffmpeg"),
    "run": ("run_minicpm", "Run a new local MiniCPM inference job on prepared media"),
    "export": ("export_run", "Normalize new inference outputs into the portable result schema"),
    "judge": ("run_judge", "Judge new text pairs through an explicitly configured HTTPS API"),
    "test": (None, "Run the offline regression suite with Python unittest"),
}


def test_main(argv):
    parser = argparse.ArgumentParser(
        prog="python -m evaluation test",
        description=COMMANDS["test"][1],
    )
    parser.add_argument("--pattern", default="test*.py", help="Unittest discovery file pattern")
    parser.add_argument("--verbose", action="store_true", help="Print each test result")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(
        str(root / "evaluation/tests"), pattern=args.pattern, top_level_dir=str(root)
    )
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    descriptions = "\n".join(f"  {name:<10} {description}" for name, (_, description) in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Omni-Life evaluation: reproduce reference scores or run and evaluate a new model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Commands:\n" + descriptions + "\n\n"
            "Start offline: python -m evaluation score --check\n"
            "New inference: prepare -> run -> export -> judge -> score\n"
            "Offline checks use only Python 3.10+ standard library. New inference requires\n"
            "separately obtained source videos, a local model, and inference dependencies.\n"
            "Only judge makes API calls; judge --dry-run performs no network requests.\n"
            "Use python -m evaluation <command> --help for command options."
        ),
    )
    parser.add_argument("command", choices=COMMANDS, metavar="command", nargs="?")
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv[:1])
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "test":
        return test_main(argv[1:])
    module = import_module("evaluation." + COMMANDS[args.command][0])
    previous = sys.argv
    try:
        sys.argv = [f"python -m evaluation {args.command}", *argv[1:]]
        return module.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    sys.exit(main())
