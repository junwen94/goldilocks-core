"""Query the status of AiiDA workchain nodes.

Requires: ``pip install goldilocks-core[aiida]``
"""

from __future__ import annotations


def get_status(pk: int) -> dict[str, str]:
    """Return a status summary for an AiiDA process node.

    Args:
        pk: Primary key of the AiiDA workchain node.

    Returns:
        Dict with keys ``"state"``, ``"process_state"``, ``"exit_status"``,
        ``"exit_message"``.
    """
    try:
        from aiida import orm
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required. Install with: pip install goldilocks-core[aiida]"
        ) from exc

    node = orm.load_node(pk)
    return {
        "state": str(node.process_state.value if node.process_state else "unknown"),
        "process_state": str(node.process_state.value if node.process_state else ""),
        "exit_status": str(node.exit_status or ""),
        "exit_message": str(node.exit_message or ""),
    }


def is_finished(pk: int) -> bool:
    """Return True when the workchain has reached a terminal state."""
    try:
        from aiida import orm
        from aiida.engine import ProcessState
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required. Install with: pip install goldilocks-core[aiida]"
        ) from exc

    node = orm.load_node(pk)
    terminal = {ProcessState.FINISHED, ProcessState.KILLED, ProcessState.EXCEPTED}
    return node.process_state in terminal
