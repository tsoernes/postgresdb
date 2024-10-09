import importlib
from typing import Callable

from betterpathlib import Path


def git_root() -> Path:
    try:
        current = Path(__file__).parent
    except NameError:
        # Running in REPL, where __file__ is not defined.
        current = Path.cwd()

    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            raise ValueError("Reached root, could not find '.git' folder")
        current = parent


def confirm_action(
    desc="Really execute?",
    yes_func: Callable | None = None,
    no_func: Callable | None = None,
    enter_is_yes=False,
) -> bool:
    """
    Return True if user confirms with 'Y' input
    """
    if desc == "Really execute?" and yes_func:
        desc = f"Really execute {yes_func.__name__}?"
    inp = None
    yes_inputs = ["y", "yes"]
    allowed_inputs = ["n", "no"] + yes_inputs
    if enter_is_yes:
        allowed_inputs.append("")
        yes_inputs.append("")
    while inp not in allowed_inputs:
        inp = input(desc + " Y/N: ").lower()
    yes = inp in yes_inputs
    if yes and yes_func:
        return yes_func()
    if not yes and no_func:
        return no_func()
    return yes
