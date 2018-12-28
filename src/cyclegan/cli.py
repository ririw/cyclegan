# -*- coding: utf-8 -*-

"""Console script for cyclegan."""
import sys
import click


@click.command()
def main() -> int:
    """Console script for cyclegan."""
    click.echo("Replace this message by putting your code into "
               "cyclegan.cli.main")
    click.echo("See click documentation at http://click.pocoo.org/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
