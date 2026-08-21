"""Small checkpoint scheduling helpers with behavior-level tests."""

from __future__ import annotations

from collections.abc import Callable


def save_checkpoint_if_needed(
    *,
    should_save: bool,
    already_saved: bool,
    save: Callable[[], None],
) -> bool:
    """Execute at most one save for a trainer step and return updated state."""

    if should_save and not already_saved:
        save()
        return True
    return already_saved
