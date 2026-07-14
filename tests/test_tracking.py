from collections.abc import Callable

import pytest

from jlens_reasoning.tracking import authenticate_wandb


def test_disabled_wandb_does_not_call_login() -> None:
    def unexpected_login(**_: object) -> bool:
        raise AssertionError("login must not run")

    assert (
        authenticate_wandb(
            api_key=None,
            enabled=False,
            login=unexpected_login,
        )
        is False
    )


def test_enabled_wandb_requires_a_key() -> None:
    with pytest.raises(RuntimeError, match="WANDB_API_KEY is missing"):
        authenticate_wandb(api_key=None, enabled=True)


def test_enabled_wandb_verifies_server_authentication() -> None:
    calls: list[dict[str, object]] = []

    def successful_login(**kwargs: object) -> bool:
        calls.append(kwargs)
        return True

    assert authenticate_wandb(
        api_key="secret-value",
        enabled=True,
        login=successful_login,
    )
    assert calls == [
        {
            "key": "secret-value",
            "relogin": True,
            "verify": True,
        }
    ]


@pytest.mark.parametrize(
    "login",
    [
        lambda **_: False,
        lambda **_: (_ for _ in ()).throw(ConnectionError("secret-value")),
    ],
)
def test_enabled_wandb_failure_is_fatal_and_redacted(
    login: Callable[..., bool],
) -> None:
    with pytest.raises(RuntimeError, match="W&B authentication failed") as error:
        authenticate_wandb(
            api_key="secret-value",
            enabled=True,
            login=login,
        )

    assert "secret-value" not in str(error.value)
    assert error.value.__cause__ is None
