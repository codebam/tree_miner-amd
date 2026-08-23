"""Every file in proto/examples/ must validate against its JSON Schema.

WHY this exists: the schemas in ``proto/`` are the written contract between the
platform and the miner, and nothing was checking them. They had drifted --
``set_config`` and the ``auth`` envelope were implemented on both sides and
documented in neither, and ``assign_task.json`` carried an all-lowercase
``consumer_address`` that both the miner and the server now reject. A schema
nobody checks drifts again, so this test checks it on every run.

There is no ``jsonschema`` dependency in this project's test environment, so the
small subset of Draft 2020-12 the two schema files actually use is implemented
below. It is deliberately strict: an unknown keyword raises rather than being
skipped, so extending a schema with a keyword this validator does not understand
fails loudly here instead of silently validating nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROTO_DIR = Path(__file__).resolve().parents[2] / "proto"
EXAMPLES_DIR = PROTO_DIR / "examples"

#: MQTT topic suffix -> the schema that owns messages on that topic.
SCHEMA_FOR_SUFFIX = {
    "register": "worker_to_platform.json",
    "heartbeat": "worker_to_platform.json",
    "status": "worker_to_platform.json",
    "block": "worker_to_platform.json",
    "task": "platform_to_worker.json",
    "control": "platform_to_worker.json",
}

#: Keywords that carry no validation semantics for our subset.
ANNOTATIONS = {"$schema", "$id", "title", "description", "default", "$comment"}


class SchemaError(AssertionError):
    """An example did not validate, with the JSON path that failed."""


def _resolve(ref: str, root: dict) -> dict:
    assert ref.startswith("#/"), f"only local refs are supported, got {ref!r}"
    node = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _type_ok(value, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        # bool is an int in Python but not in JSON.
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "null":
        return value is None
    raise AssertionError(f"unsupported type {expected!r}")


def _matches(instance, schema: dict, root: dict) -> bool:
    """True iff ``instance`` validates. Used for oneOf branch selection."""
    try:
        _validate(instance, schema, root, "$")
    except SchemaError:
        return False
    return True


def _validate(instance, schema: dict, root: dict, path: str) -> None:
    if "$ref" in schema:
        _validate(instance, _resolve(schema["$ref"], root), root, path)
        remaining = set(schema) - {"$ref"} - ANNOTATIONS
        assert not remaining, f"$ref combined with {remaining} is not supported"
        return

    for keyword, expected in schema.items():
        if keyword in ANNOTATIONS or keyword in {"$defs", "then", "else"}:
            continue

        if keyword == "type":
            if not _type_ok(instance, expected):
                raise SchemaError(f"{path}: expected type {expected}, got {type(instance).__name__}")
        elif keyword == "const":
            if instance != expected:
                raise SchemaError(f"{path}: expected const {expected!r}, got {instance!r}")
        elif keyword == "enum":
            if instance not in expected:
                raise SchemaError(f"{path}: {instance!r} is not one of {expected}")
        elif keyword == "pattern":
            if isinstance(instance, str) and not re.search(expected, instance):
                raise SchemaError(f"{path}: {instance!r} does not match {expected!r}")
        elif keyword == "minimum":
            if isinstance(instance, (int, float)) and not isinstance(instance, bool) and instance < expected:
                raise SchemaError(f"{path}: {instance} < minimum {expected}")
        elif keyword == "maximum":
            if isinstance(instance, (int, float)) and not isinstance(instance, bool) and instance > expected:
                raise SchemaError(f"{path}: {instance} > maximum {expected}")
        elif keyword == "required":
            if isinstance(instance, dict):
                missing = [k for k in expected if k not in instance]
                if missing:
                    raise SchemaError(f"{path}: missing required {missing}")
        elif keyword == "properties":
            if isinstance(instance, dict):
                for key, subschema in expected.items():
                    if key in instance:
                        _validate(instance[key], subschema, root, f"{path}.{key}")
        elif keyword == "additionalProperties":
            if isinstance(instance, dict) and expected is False:
                extra = set(instance) - set(schema.get("properties", {}))
                if extra:
                    raise SchemaError(f"{path}: unexpected properties {sorted(extra)}")
        elif keyword == "items":
            if isinstance(instance, list):
                for i, item in enumerate(instance):
                    _validate(item, expected, root, f"{path}[{i}]")
        elif keyword == "oneOf":
            matched = [i for i, sub in enumerate(expected) if _matches(instance, sub, root)]
            if len(matched) != 1:
                raise SchemaError(
                    f"{path}: matched {len(matched)} oneOf branches {matched}, expected exactly 1"
                )
        elif keyword == "if":
            branch = "then" if _matches(instance, expected, root) else "else"
            if branch in schema:
                _validate(instance, schema[branch], root, path)
        else:
            raise AssertionError(f"{path}: unsupported schema keyword {keyword!r}")


def load_schema(name: str) -> dict:
    return json.loads((PROTO_DIR / name).read_text())


def wire_message(raw: dict) -> dict:
    """Strip the ``_``-prefixed documentation keys; they are not on the wire."""
    return {k: v for k, v in raw.items() if not k.startswith("_")}


EXAMPLE_FILES = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))


def test_examples_directory_is_not_empty():
    assert EXAMPLE_FILES, "no examples found; the glob or the directory moved"


@pytest.mark.parametrize("filename", EXAMPLE_FILES)
def test_example_validates_against_its_schema(filename):
    raw = json.loads((EXAMPLES_DIR / filename).read_text())

    topic = raw.get("_topic", "")
    suffix = topic.rsplit("/", 1)[-1]
    assert suffix in SCHEMA_FOR_SUFFIX, (
        f"{filename}: _topic {topic!r} does not end in a known suffix "
        f"{sorted(SCHEMA_FOR_SUFFIX)}"
    )

    schema = load_schema(SCHEMA_FOR_SUFFIX[suffix])
    _validate(wire_message(raw), schema, schema, "$")


def test_signed_examples_carry_a_verifiable_signature():
    """The auth envelopes in the examples are real, not illustrative filler."""
    from server.command_signing import verify_signature

    secret = "example-shared-secret"
    signed = [
        f for f in EXAMPLE_FILES
        if "auth" in wire_message(json.loads((EXAMPLES_DIR / f).read_text()))
    ]
    assert signed, "no signed example demonstrates the auth envelope"

    for filename in signed:
        message = wire_message(json.loads((EXAMPLES_DIR / filename).read_text()))
        worker_id = message["auth"]["worker_id"]
        assert verify_signature(message, secret, worker_id), (
            f"{filename}: auth.sig does not verify under the documentation secret; "
            "regenerate the example rather than hand-editing it"
        )


def test_commands_that_can_move_money_require_an_envelope():
    """assign_task / shutdown / set_config must be unrepresentable unsigned."""
    schema = load_schema("platform_to_worker.json")
    for def_name in ("assign_task", "shutdown", "set_config"):
        assert "auth" in schema["$defs"][def_name]["required"], (
            f"{def_name} must require the auth envelope: the miner refuses it "
            "unsigned"
        )
    for def_name in ("register_ack", "release", "control"):
        assert "auth" not in schema["$defs"][def_name]["required"], (
            f"{def_name} is non-mutating and is accepted unsigned by a worker "
            "with no shared secret"
        )
