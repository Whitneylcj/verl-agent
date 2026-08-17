"""Strict one-call AppWorld JSON action compiler."""

from __future__ import annotations

import json
import re


def extract_tagged_payload(action: str, start_tag: str, end_tag: str) -> str:
    start_index = action.find(start_tag)
    end_index = action.find(end_tag, start_index + len(start_tag))
    if start_index == -1 or end_index == -1:
        raise ValueError(f"missing {start_tag}...{end_tag} block")
    if action.find(start_tag, start_index + len(start_tag)) != -1:
        raise ValueError(f"multiple {start_tag} blocks are not allowed")
    return action[start_index + len(start_tag):end_index].strip()


def compile_json_api_action(action: str) -> str:
    """Compile one tagged JSON object into a fixed-shape AppWorld API call."""

    extract_tagged_payload(action, "<think>", "</think>")
    payload = json.loads(extract_tagged_payload(action, "<action>", "</action>"))
    if not isinstance(payload, dict) or set(payload) != {"app", "api", "arguments"}:
        raise ValueError("JSON action must contain exactly app, api, and arguments")
    app_name = payload["app"]
    api_name = payload["api"]
    arguments = payload["arguments"]
    identifier = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
    if not isinstance(app_name, str) or identifier.fullmatch(app_name) is None:
        raise ValueError("app must be a public Python identifier")
    if not isinstance(api_name, str) or identifier.fullmatch(api_name) is None:
        raise ValueError("api must be a public Python identifier")
    if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
        raise ValueError("arguments must be a JSON object with string keys")
    # repr() turns JSON booleans/null into valid Python literals while all code
    # structure remains fixed by this adapter.
    return f"print(apis.{app_name}.{api_name}(**{arguments!r}))"
