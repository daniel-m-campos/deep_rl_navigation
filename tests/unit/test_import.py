def test_version():
    import python_template
    from pkg_resources import parse_version
    assert parse_version(python_template.__version__) >= parse_version("0.0.0")
