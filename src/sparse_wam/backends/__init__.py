"""Backbone-specific attention execution, sharing the same Sparse-WAM algorithm."""


def make_backend(config):
    if config.backbone == "edge":
        from .edge import EdgeBackend

        return EdgeBackend(config)
    if config.backbone == "nano":
        from .nano import NanoBackend

        return NanoBackend(config)
    raise ValueError(f"{config.backbone} uses its own native controller via create_service")
