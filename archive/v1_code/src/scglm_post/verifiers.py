"""Strict parsing shared by training labels and task evaluations."""
import json
import re


def object_value(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if type(value) is not dict:
        raise ValueError("Require a JSON object")
    return value


def exact_integer_object(text, expected):
    try:
        value = object_value(text)
        return (value.keys() == expected.keys()
                and all(type(value[k]) is int and value[k] == v for k, v in expected.items()))
    except (ValueError, TypeError, RecursionError):
        return False


def integer_list(text):
    if not re.fullmatch(r"\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)*\s*", text):
        raise ValueError("Require a comma-separated integer list")
    return [int(part.strip()) for part in text.split(",")]
