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


def _trimmed_segment(text: str, start: int, end: int) -> tuple[str, int]:
    segment = text[start:end]
    left = len(segment) - len(segment.lstrip())
    return segment.strip(), start + left


def _unwrap_json_fence(action: str) -> tuple[str, int]:
    """Return one optional JSON-fenced payload and its source offset."""

    fenced = re.fullmatch(
        r"\s*```(?:json)?[ \t]*(?:\r?\n)?(.*?)(?:\r?\n)?```\s*",
        action,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced is None:
        return _trimmed_segment(action, 0, len(action))
    return _trimmed_segment(action, fenced.start(1), fenced.end(1))


def parse_json_api_action(action: str) -> ParsedAppWorldAction:
    """Validate one fixed-shape JSON action without executing model text.

    The policy-facing contract is a bare ``app/api/arguments`` object.  A
    single JSON fence, a single ``<action>`` wrapper, and the original
    ``<think>...<action>`` form remain accepted because small instruction
    models commonly emit those harmless wrappers.  Extra prose, multiple
    calls, and arbitrary code remain invalid.
    """

    if not isinstance(action, str):
        raise TypeError("action must be a string")
    content, content_start = _unwrap_json_fence(action)
    canonical = re.fullmatch(
        r"<think>(.*?)</think>\s*<action>(.*?)</action>",
        content,
        flags=re.DOTALL,
    )
    action_only = re.fullmatch(r"<action>(.*?)</action>", content, flags=re.DOTALL)
    payload_text = content
    payload_start = content_start
    nested_payload = False
    if canonical is not None:
        if not canonical.group(1).strip():
            raise ValueError("think block must be non-empty")
        payload_text, relative_start = _trimmed_segment(
            content,
            canonical.start(2),
            canonical.end(2),
        )
        payload_start = content_start + relative_start
    elif action_only is not None:
        payload_text, relative_start = _trimmed_segment(
            content,
            action_only.start(1),
            action_only.end(1),
        )
        payload_start = content_start + relative_start

    payload = json.loads(payload_text, object_pairs_hook=_unique_object)
    if isinstance(payload, dict) and tuple(payload) == ("think", "action"):
        thought = payload["think"]
        if not isinstance(thought, str) or not thought.strip():
            raise ValueError("nested think field must be a non-empty string")
        payload = payload["action"]
        nested_payload = True
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
    if app_name == "app_name" or api_name == "api_name":
        raise ValueError("app and api must not use literal placeholder names")
    if not isinstance(arguments, dict) or not all(isinstance(key, str) for key in arguments):
        raise ValueError("arguments must be a JSON object with string keys")
    # Escaped root keys remain valid JSON, but do not expose a trustworthy
    # character boundary and therefore resolve conservatively to the whole
    # response.  In the nested compatibility form, begin after the action key
    # so a quoted word inside the thought cannot be mistaken for the boundary.
    arguments_search_start = payload_text.find('"action"') if nested_payload else 0
    arguments_key_start = payload_text.find('"arguments"', max(arguments_search_start, 0))
    arguments_char_start = payload_start + arguments_key_start if arguments_key_start >= 0 else None
    return ParsedAppWorldAction(
        app=app_name,
        api=api_name,
        arguments=arguments,
        arguments_char_start=arguments_char_start,
    )


def compile_json_api_action(action: str) -> str:
    """Compile one validated JSON object into a fixed-shape AppWorld API call."""

    parsed = parse_json_api_action(action)
    # repr() turns JSON booleans/null into valid Python literals while all code
    # structure remains fixed by this adapter.
    return f"print(apis.{parsed.app}.{parsed.api}(**{parsed.arguments!r}))"
