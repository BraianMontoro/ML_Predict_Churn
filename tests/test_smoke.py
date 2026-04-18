def test_import_app():
    from src.api.main import app
    assert app is not None
