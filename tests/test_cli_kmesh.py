from goldilocks_core.cli.cli_kmesh import build_parser


def test_build_parser_parses_required_arguments() -> None:
    """Parse the required CLI arguments for k-mesh recommendation."""
    parser = build_parser()
    args = parser.parse_args(["example.cif"])

    assert args.structure == "example.cif"
    assert args.accuracy == "accurate"


def test_build_parser_accepts_accuracy_flag() -> None:
    """Accept an explicit accuracy flag."""
    parser = build_parser()
    args = parser.parse_args(["example.cif", "--accuracy", "balanced"])

    assert args.accuracy == "balanced"
