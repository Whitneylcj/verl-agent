"""Strict one-call AppWorld JSON action compiler."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedAppWorldAction:
    """Validated one-call action plus the first prefix-safe argument offset."""

    app: str
    api: str
    arguments: dict[str, Any]
    arguments_char_start: int | None


def extract_tagged_payload(action: str, start_tag: str, end_tag: str) -> str:
    start_index = action.find(start_tag)
    end_index = action.find(end_tag, start_index + len(start_tag))
    if start_index == -1 or end_index == -1:
        raise ValueError(f"missing {start_tag}...{end_tag} block")
    if action.find(start_tag, start_index + len(start_tag)) != -1:
        raise ValueError(f"multiple {start_tag} blocks are not allowed")
    return action[start_index + len(start_tag) : end_index].strip()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_api_action(action: str) -> ParsedAppWorldAction:
    """Validate one tagged JSON action without executing or compiling it."""

    match = re.fullmatch(
        r"\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*",
        action,
        flags=re.DOTALL,
    )
    if match is None or not match.group(1).strip():
        raise ValueError("action must contain exactly one non-empty think block and one action block")
    payload = json.loads(match.group(2).strip(), object_pairs_hook=_unique_object)
    expected_keys = ("app", "api", "arguments")
    if not isinstance(payload, dict) or tuple(payload) != expected_keys:
        raise ValueError("JSON action keys must be ordered exactly as app, api, arguments")
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
    # The canonical root key is the first possible occurrence because app/api
    # values are restricted identifiers. Escaped root keys remain valid JSON,
    # but do not expose a trustworthy character boundary and therefore resolve
    # to the conservative whole-response selector span.
    payload_start = match.start(2)
    arguments_key_start = match.group(2).find('"arguments"')
    arguments_char_start = payload_start + arguments_key_start if arguments_key_start >= 0 else None
    return ParsedAppWorldAction(
        app=app_name,
        api=api_name,
        arguments=arguments,
        arguments_char_start=arguments_char_start,
    )


def compile_json_api_action(action: str) -> str:
    """Compile one tagged JSON object into a fixed-shape AppWorld API call."""

    parsed = parse_json_api_action(action)
    # repr() turns JSON booleans/null into valid Python literals while all code
    # structure remains fixed by this adapter.
    return f"print(apis.{parsed.app}.{parsed.api}(**{parsed.arguments!r}))"
