import sys


import kpi.models
from kpi.cli import cli


assert sys.flags.utf8_mode, "Please run this script with -X utf8"


if __name__ == "__main__":
    cli()
