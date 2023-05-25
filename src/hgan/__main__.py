"""
The hgan package can be run as a module by invoking it as:
python -m hgan <command> <arguments> ..
"""

import sys
import hgan

from hgan.demo import main as demo
from hgan.run import main as run
from hgan.configuration import show_config

commands = {"demo": demo, "run": run, "show-config": show_config}


def print_usage():
    print("hgan " + hgan.__version__)
    print("Usage: hgan <command> <arguments ..>")
    print("\nThe following commands are supported:\n " + "\n ".join(commands))


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(0)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command not in commands:
        print_usage()
        sys.exit(1)

    commands[command](*args)


if __name__ == "__main__":
    sys.exit(main())
