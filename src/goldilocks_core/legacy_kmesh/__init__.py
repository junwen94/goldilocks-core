"""v1's k-point mesh math and resolution -- renamed from `kmesh/` (v2 epic 6,
#6), content and behaviour unchanged. Freed the `kmesh` name for the new
top-level `kmesh.py` module ported verbatim from goldilocks-data, which v2
uses instead (v1 core's own ladder has a per-axis enumeration cap that
goldilocks-data's distance-floor approach replaces -- see `kmesh.py`'s
module docstring). Deleted along with the rest of v1 at cutover (v2 epic 9);
nothing here should gain new callers in the meantime.
"""
