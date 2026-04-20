from goldilocks_core.cli.cli_kmesh import build_parser


def test_build_parser_parses_required_arguments() -> None:
    """Parse the required CLI arguments for k-mesh recommendation."""
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--model", "model.joblib"])

    assert args.structure == "example.cif"
    assert args.model == "model.joblib"
