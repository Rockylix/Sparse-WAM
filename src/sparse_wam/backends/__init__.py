"""Backbone-specific attention execution, sharing the same Sparse-WAM algorithm."""


def make_backend(config):
    if config.backbone == "edge":
        from .edge import EdgeBackend

        return EdgeBackend(config)
    from .nano import NanoBackend

    return NanoBackend(config)
