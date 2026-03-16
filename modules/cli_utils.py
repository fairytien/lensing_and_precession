"""Small argparse helpers shared by repo entry points."""

from argparse import ArgumentParser
from typing import Iterable


def add_orientation_args(parser: ArgumentParser) -> ArgumentParser:
    """Attach the common orientation arguments used by pipeline scripts."""
    parser.add_argument("--theta_J", type=float, default=None)
    parser.add_argument("--phi_J", type=float, default=None)
    parser.add_argument("--theta_S", type=float, default=None)
    parser.add_argument("--phi_S", type=float, default=None)
    parser.add_argument(
        "--orient_preset",
        type=str,
        default=None,
        help=(
            "Optional orientation preset to use for both params and tag."
            "If not provided, angles (theta_J, phi_J, theta_S, phi_S) form the tag."
        ),
    )
    return parser


def set_argument_choices(
    parser: ArgumentParser,
    dest: str,
    choices: Iterable[str],
) -> None:
    """Update an existing argparse action with dynamic choices."""
    for action in parser._actions:
        if getattr(action, "dest", None) == dest:
            action.choices = sorted(choices)
            return
