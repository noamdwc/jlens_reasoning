"""Experiment-tracking authentication."""

from __future__ import annotations

from collections.abc import Callable


def authenticate_wandb(
    *,
    api_key: str | None,
    enabled: bool = True,
    login: Callable[..., bool] | None = None,
) -> bool:
    """Authenticate W&B when enabled and fail on every authentication error."""

    if not enabled:
        return False

    if not api_key:
        raise RuntimeError("W&B is enabled but WANDB_API_KEY is missing")

    if login is None:
        import wandb

        login = wandb.login

    try:
        authenticated = login(
            key=api_key,
            relogin=True,
            verify=True,
        )
    except Exception:
        raise RuntimeError("W&B authentication failed") from None

    if not authenticated:
        raise RuntimeError("W&B authentication failed")

    return True
