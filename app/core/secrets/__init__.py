from .config import Secrets


def secret_manager_setup() -> Secrets:
    return Secrets()


config_secrets = secret_manager_setup()
