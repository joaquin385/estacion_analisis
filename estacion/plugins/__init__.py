"""Plugins de dominio: reglas + checks semánticos + diccionarios."""

from .registry import list_plugins, get_plugin, run_plugin_checks
from .dictionaries import (
    load_dictionary,
    save_dictionary,
    validate_against_dictionary,
    infer_dictionary_stub,
    DataDictionary,
)

__all__ = [
    "list_plugins",
    "get_plugin",
    "run_plugin_checks",
    "load_dictionary",
    "save_dictionary",
    "validate_against_dictionary",
    "infer_dictionary_stub",
    "DataDictionary",
]
