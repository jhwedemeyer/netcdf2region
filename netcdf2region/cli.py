"""Command-line helpers shared by ``compute_grid_weights``, ``netcdf2region`` and
``validate_weights``.

Options are spelled with hyphens (``--shift-longitudes``). Options that used to be
spelled with underscores keep that spelling as an alias hidden from ``--help``, so
existing scripts keep working.
"""

import argparse
import logging

#: Keyword arguments that only concern how an option is shown, not how it is parsed.
_DISPLAY_KWARGS = ("help", "metavar")


def add_option(parser, flag, *, legacy=None, **kwargs):
    """Add ``flag`` to ``parser`` (or an argument group), plus ``legacy`` as a hidden alias.

    Both spellings write to the same destination, so the rest of the code only ever
    sees one of them.
    """
    if legacy is not None and kwargs.get("required"):
        # argparse would demand the canonical spelling and reject the alias.
        raise ValueError(f"{flag}: a required option cannot have a legacy alias")
    action = parser.add_argument(flag, **kwargs)
    if legacy is not None:
        alias_kwargs = {k: v for k, v in kwargs.items() if k not in _DISPLAY_KWARGS}
        alias_kwargs["dest"] = action.dest
        parser.add_argument(legacy, help=argparse.SUPPRESS, **alias_kwargs)
    return action


def add_shift_longitudes(parser):
    add_option(
        parser, "--shift-longitudes", legacy="--shift_longitudes", action="store_true",
        help="Shift longitudes from [0, 360] to [-180, 180], the convention the weights use",
    )


def add_grid_options(parser):
    """Options describing the template grid the weights are computed on."""
    add_option(
        parser, "--coordinate-origin", legacy="--coordinate_origin",
        choices=["centered", "top-left"], default="centered",
        help="Whether the grid coordinates are cell centres or top-left cell edges (default: centered)",
    )
    add_shift_longitudes(parser)


def add_region_options(parser):
    """Options selecting the regions in a shapefile."""
    add_option(
        parser, "--shapefile-layer", legacy="--shapefile_layer", type=int, default=None,
        help="Layer index in the shapefile (default: the first layer)",
    )
    add_option(
        parser, "--gid", default="GID_1",
        help="Column with the region identifier (default: GID_1)",
    )


def setup_logging():
    """Show the library's progress messages and warnings on the console."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
