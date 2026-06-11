from goldilocks_core.pseudo.pp_metadata import PseudoMetadata
from goldilocks_core.pseudo.pp_policy import PseudoPolicy, apply_pseudo_policy


def test_apply_pseudo_policy_filters_metadata_list() -> None:
    """Apply a simple pseudo policy to a metadata list."""
    metadata_list = [
        PseudoMetadata(
            filepath="a.UPF",
            filename="a.UPF",
            header_format="attr",
            library="pslibrary",
            element="Hg",
            pseudo_type="USPP",
            functional="PBE",
            relativistic="scalar",
            z_valence=12.0,
        ),
        PseudoMetadata(
            filepath="b.UPF",
            filename="b.UPF",
            header_format="attr",
            library="SSSP",
            element="Hg",
            pseudo_type="PAW",
            functional="PBE",
            relativistic="full",
            z_valence=12.0,
        ),
    ]

    policy = PseudoPolicy(
        relativistic_mode="full",
        preferred_functional="PBE",
        allowed_sources=("SSSP",),
        allowed_pseudo_types=("PAW",),
    )

    selected = apply_pseudo_policy(metadata_list, policy)

    assert len(selected) == 1
    assert selected[0].filename == "b.UPF"
