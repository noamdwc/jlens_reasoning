from __future__ import annotations

import configparser
import json

import pytest

from jlens_reasoning.environments import colab_drive

TOKEN = json.dumps({"access_token": "fake-access", "refresh_token": "fake-refresh"})
USER_CONFIG = f"[jlens]\ntype = drive\ntoken = {TOKEN}\nscope = drive.file\n"


def stage(tmp_path, content=USER_CONFIG, **env):
    source = tmp_path / "source.conf"
    source.write_text(content)
    destination = tmp_path / "staged"
    mode = colab_drive.stage_colab_credentials(
        destination,
        environ={"HOME": str(tmp_path), "JLENS_RCLONE_CONFIG": str(source), **env},
    )
    return mode, destination, source


def test_stages_only_jlens_and_preserves_oauth_settings_without_mutating_source(
    tmp_path,
):
    content = (
        USER_CONFIG
        + "client_id = fake-client\nclient_secret = fake-secret\nroot_folder_id = project-root\n\n[personal]\ntype = drive\ntoken = other-secret\n"
    )
    mode, directory, source = stage(tmp_path, content, WANDB_API_KEY="fake-wandb")
    assert mode == "rclone"
    staged = directory / "rclone.conf"
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(staged)
    assert parser.sections() == ["jlens"]
    assert dict(parser["jlens"]) == {
        "type": "drive",
        "token": TOKEN,
        "scope": "drive.file",
        "client_id": "fake-client",
        "client_secret": "fake-secret",
        "root_folder_id": "project-root",
    }
    assert source.read_text() == content
    assert directory.stat().st_mode & 0o777 == 0o700
    assert staged.stat().st_mode & 0o777 == 0o600
    env_file = directory / "jlens.env"
    assert env_file.stat().st_mode & 0o777 == 0o600
    assert colab_drive.load_env_file(env_file) == {
        "JLENS_DRIVE_AUTH": "rclone",
        "WANDB_API_KEY": "fake-wandb",
    }
    assert not (directory / "drive-sa.json").exists()


@pytest.mark.parametrize(
    "content",
    [
        "[jlens]\ntype = drive\ntoken = secret-not-json\n",
        "[jlens]\ntype = s3\nsecret = do-not-print\n",
        "[jlens]\ntype = drive\ntoken = {}\n",
        "[jlens]\ntoken = fake\ntoken = do-not-print\n",
        "[elsewhere]\ntype = drive\ntoken = do-not-print\n",
        "[DEFAULT]\ntoken = do-not-print\n" + USER_CONFIG,
        USER_CONFIG + "service_account_file = /private/key.json\n",
    ],
)
def test_invalid_explicit_config_fails_redacted_without_staging_credentials(
    tmp_path, content
):
    with pytest.raises(RuntimeError) as error:
        stage(tmp_path, content)
    assert "do-not-print" not in str(error.value)
    assert "secret-not-json" not in str(error.value)
    assert not (tmp_path / "staged" / "rclone.conf").exists()


def test_explicit_missing_config_does_not_fall_back_to_service_account(tmp_path):
    sa = tmp_path / "drive-sa.json"
    sa.write_text("{}")
    with pytest.raises(RuntimeError, match="rclone"):
        colab_drive.stage_colab_credentials(
            tmp_path / "staged",
            environ={
                "JLENS_RCLONE_CONFIG": str(tmp_path / "missing.conf"),
                "JLENS_DRIVE_SA_JSON": str(sa),
            },
        )


@pytest.mark.parametrize(
    "variable", ["JLENS_RCLONE_CONFIG", "RCLONE_CONFIG", "XDG_CONFIG_HOME", "HOME"]
)
def test_discovers_config_from_supported_locations(tmp_path, variable):
    config = tmp_path / "rclone.conf"
    env = {"HOME": str(tmp_path / "empty")}
    if variable == "XDG_CONFIG_HOME":
        config = tmp_path / "rclone" / "rclone.conf"
        env[variable] = str(tmp_path)
    elif variable == "HOME":
        config = tmp_path / ".config" / "rclone" / "rclone.conf"
        env[variable] = str(tmp_path)
    else:
        env[variable] = str(config)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(USER_CONFIG)
    assert (
        colab_drive.stage_colab_credentials(tmp_path / "staged", environ=env)
        == "rclone"
    )


@pytest.mark.parametrize(
    "override,expected",
    [
        ("auto", "rclone"),
        ("rclone", "rclone"),
        ("service_account", "service_account"),
        ("interactive", "interactive"),
    ],
)
def test_auth_precedence_and_override_when_both_credentials_exist(
    tmp_path, override, expected
):
    sa = tmp_path / "drive-sa.json"
    sa.write_text(
        json.dumps(
            {
                "type": "service_account",
                "client_email": "sa@example.com",
                "private_key": "synthetic-key",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )
    mode, directory, _ = stage(
        tmp_path,
        JLENS_DRIVE_AUTH=override,
        JLENS_DRIVE_SA_JSON=str(sa),
        JLENS_DRIVE_SHARED_DRIVE_ID="shared",
        JLENS_DRIVE_ROOT_FOLDER_ID="project",
    )
    assert mode == expected
    assert (directory / "rclone.conf").exists() == (expected == "rclone")
    assert (directory / "drive-sa.json").exists() == (expected == "service_account")


def test_folder_overrides_are_applied_only_to_staged_remote(tmp_path):
    _, directory, source = stage(tmp_path, JLENS_DRIVE_ROOT_FOLDER_ID="project")
    parser = configparser.ConfigParser()
    parser.read(directory / "rclone.conf")
    assert parser["jlens"]["root_folder_id"] == "project"
    assert "root_folder_id" not in source.read_text()


def test_no_credentials_requires_explicit_interactive_opt_in(tmp_path):
    env = {"HOME": str(tmp_path)}
    with pytest.raises(RuntimeError, match="unattended"):
        colab_drive.stage_colab_credentials(tmp_path / "staged", environ=env)
    env["JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT"] = "1"
    assert (
        colab_drive.stage_colab_credentials(tmp_path / "staged", environ=env)
        == "interactive"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "private-invalid-json",
        "[]",
        '{"type":"authorized_user"}',
        '{"type":"service_account"}',
    ],
)
def test_malformed_service_account_fails_before_staging_and_redacts_details(
    tmp_path, payload
):
    sa = tmp_path / "drive-sa.json"
    sa.write_text(payload)
    with pytest.raises(RuntimeError, match="service-account") as error:
        colab_drive.stage_colab_credentials(
            tmp_path / "staged",
            environ={
                "HOME": str(tmp_path),
                "JLENS_DRIVE_AUTH": "service_account",
                "JLENS_DRIVE_SA_JSON": str(sa),
                "JLENS_DRIVE_ROOT_FOLDER_ID": "project",
                "JLENS_DRIVE_SHARED_DRIVE_ID": "shared",
            },
        )
    assert "private-invalid-json" not in str(error.value)
    assert not (tmp_path / "staged" / "drive-sa.json").exists()
