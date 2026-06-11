"""Submit AiiDA workchains and return node identifiers.

Requires: ``pip install goldilocks-core[aiida]``
"""

from __future__ import annotations

from typing import Any


def submit_pw(builder: Any) -> tuple[int, str]:
    """Submit a pw.x workchain builder and return (pk, uuid).

    Args:
        builder: Populated ``PwBaseWorkChain`` or ``PwRelaxWorkChain`` builder.

    Returns:
        ``(pk, uuid)`` of the submitted workchain node.
    """
    try:
        from aiida.engine import submit
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required. Install with: pip install goldilocks-core[aiida]"
        ) from exc

    node = submit(builder)
    return node.pk, str(node.uuid)
