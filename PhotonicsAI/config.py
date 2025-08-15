"""Store configuration."""

import pathlib

from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv_path = find_dotenv(usecwd=True)

__all__ = ["PATH"]


home = pathlib.Path.home()
cwd = pathlib.Path.cwd()

module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent


class Path:
    module = module_path
    repo = repo_path
    cells = module / "cells"
    photon = module / "Photon"
    prompts = photon / "prompts.yaml"
    templates = photon / "templates.yaml"
    logs = module / "log"
    pdk = module / "KnowledgeBase" / "DesignLibrary"
    build = repo / "build"


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=dotenv_path, extra="ignore")
    openai_api_key: str = ""


PATH = Path()
PATH.build.mkdir(parents=True, exist_ok=True)
CONF = Config()
