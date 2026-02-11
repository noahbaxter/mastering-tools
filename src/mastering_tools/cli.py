"""Unified mastering CLI — dispatches to subcommands."""

import argparse
import sys

from mastering_tools import loudness, declick, spectrum, crest, stereo, dynamics, check


def main():
    parser = argparse.ArgumentParser(
        prog="mastering",
        description="Audio mastering analysis toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    loudness.register_subcommand(subparsers)
    declick.register_subcommand(subparsers)
    spectrum.register_subcommand(subparsers)
    crest.register_subcommand(subparsers)
    stereo.register_subcommand(subparsers)
    dynamics.register_subcommand(subparsers)
    check.register_subcommand(subparsers)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
