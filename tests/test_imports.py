def test_project_and_jacobian_lens_import() -> None:
    import jlens

    import jlens_reasoning

    assert jlens_reasoning.__version__ == "0.1.0"
    assert jlens is not None
