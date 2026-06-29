import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


def _load_from_streamlit_secrets():
    try:
        import streamlit as st
        value = st.secrets.get("GEMINI_API_KEY")
        if value:
            return str(value)
    except Exception:
        return None


def _load_from_toml():
    candidates = [
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / "secrets.toml",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
            for key in ("GEMINI_API_KEY", "gemini_api_key", "api_key"):
                value = data.get(key)
                if value:
                    return str(value)
            if isinstance(data, dict):
                for nested in data.values():
                    if isinstance(nested, dict):
                        for key in ("GEMINI_API_KEY", "gemini_api_key", "api_key"):
                            value = nested.get(key)
                            if value:
                                return str(value)
        except Exception:
            continue
    return None


GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or _load_from_streamlit_secrets()
    or _load_from_toml()
    or ""
)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")