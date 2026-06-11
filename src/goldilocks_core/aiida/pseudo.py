"""aiida-pseudo integration: family lookup and UpfData loading.

Requires: ``pip install goldilocks-core[aiida]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from goldilocks_core.advise.types import QEParameterSet


def load_pseudos(
    params: "QEParameterSet",
    pseudo_family: str | None = None,
) -> dict[str, Any]:
    """Return a dict of ``{element: UpfData}`` nodes for pw.x.

    Tries two strategies in order:
    1. If *pseudo_family* is given, queries the aiida-pseudo family by element.
    2. Falls back to loading ``UpfData`` from the local file paths in *params*.

    Args:
        params: QEParameterSet with per-element pseudo selections.
        pseudo_family: aiida-pseudo family label (e.g.
            ``"PseudoDojo/0.4/PBEsol/SR/standard/upf"``).

    Returns:
        Dict mapping element symbol → ``UpfData`` node.
    """
    try:
        import aiida  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "aiida-core is required. Install with: pip install goldilocks-core[aiida]"
        ) from exc

    from aiida import orm

    pseudos: dict[str, Any] = {}

    if pseudo_family is not None:
        try:
            from aiida_pseudo.groups.family import PseudoPotentialFamily

            family = PseudoPotentialFamily.objects.get(label=pseudo_family)
            elements = [ps.element for ps in params.pseudos]
            return dict(family.get_pseudos(elements=elements))
        except Exception:
            pass  # Fall through to local file loading

    for ps in params.pseudos:
        if ps.path is not None and ps.path.exists():
            pseudos[ps.element] = orm.UpfData(file=str(ps.path))
        else:
            raise FileNotFoundError(
                f"Pseudo file not found for {ps.element}: {ps.path}. "
                "Provide a pseudo_family label or a local pseudo directory."
            )

    return pseudos
