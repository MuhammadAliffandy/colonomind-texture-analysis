"""
app/__init__.py
---------------
Package initialiser for the Colonomind web application module.

create_app is imported lazily to avoid loading Flask (and TensorFlow) when
only the texture_extractor submodule is needed (e.g., in the DGX pipeline).
"""


def create_app(test_config=None):
    """Lazy wrapper around routes.create_app to defer Flask import."""
    from app.routes import create_app as _create_app  # noqa: PLC0415

    return _create_app(test_config=test_config)


__all__ = ["create_app"]
