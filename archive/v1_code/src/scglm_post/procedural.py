"""Small verified tasks with separate templates and input ranges per split."""
from __future__ import annotations

import json
import random
from .common import identity

FAMILIES = ("arithmetic", "sorting", "extraction", "classification", "json")


def generate(split, per_family, seed=20260923):
    offset = {"train": 0, "development": 1, "confirmation": 2}[split]
    rng = random.Random(seed + offset)
    low, high = ((1, 100), (101, 200), (201, 300))[offset]
    templates = {
        "arithmetic": ("Add {a} and {b}. Return only the integer.",
                       "What is {a} + {b}? Give just the integer.",
                       "Compute the sum of {a} with {b}; output only its integer value."),
        "sorting": ("Sort these integers ascending. Output a comma-separated list: {values}",
                    "Put the numbers {values} in increasing order, separated by commas.",
                    "Arrange {values} from smallest to largest. Reply with comma-separated integers."),
        "extraction": ("Read this record and return only its code: name={name}; code={code}; color={color}",
                       "Record: color={color}, code={code}, name={name}. What is the code? Output it alone.",
                       "Extract the code value, without explanation: {{name: {name}, color: {color}, code: {code}}}"),
        "classification": ("Is {a} even or odd? Reply with even or odd only.",
                           "Classify {a} by parity. Output exactly even or odd.",
                           "For the integer {a}, return the label even if divisible by two; otherwise odd."),
        "json": ("Return only a JSON object with key number and integer value {a}.",
                 "Encode {a} as a JSON object under the key number. No other text.",
                 "Produce valid JSON containing one field named number whose integer value is {a}."),
    }
    rows = []
    for family in FAMILIES:
        seen = set()
        attempts = 0
        while len(seen) < per_family:
            attempts += 1
            if attempts > per_family * 100:
                raise ValueError(f"Requested more unique {family}/{split} cases than the generator can supply")
            # Parity/JSON use a wider disjoint range to avoid oversampling 100 cases.
            a = rng.randrange(low, high + 1) if family == "arithmetic" else rng.randrange(offset * 100000 + 1, (offset + 1) * 100000)
            b = rng.randrange(low, high + 1)
            values = rng.sample(range(low, high + 1), 3 + offset)
            code = f"{('TR', 'DV', 'CF')[offset]}{rng.randrange(100000):05d}"
            fields = {"a": a, "b": b, "values": ", ".join(map(str, values)),
                      "name": f"item{a}", "code": code, "color": rng.choice(("red", "blue", "green"))}
            prompt = templates[family][offset].format(**fields)
            if prompt in seen:
                continue
            seen.add(prompt)
            answer = {
                "arithmetic": str(a + b), "sorting": ", ".join(map(str, sorted(values))),
                "extraction": code, "classification": "even" if a % 2 == 0 else "odd",
                "json": json.dumps({"number": a}),
            }[family]
            rows.append({"id": "procedural:" + identity(prompt), "source": "procedural",
                         "group": f"procedural:{split}:{family}:{identity(prompt)}", "split": split,
                         "category": family, "license": "project-generated",
                         "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                         "verification": {"family": family, "fields": fields},
                         "provenance": {"generator": "scglm_post.procedural", "seed": seed, "template": offset}})
    return rows


def correct(row, response):
    """Check full outputs; an answer embedded in extra prose does not pass."""
    if "system_variant" in row["verification"]:
        from .system_data import correct_system
        return correct_system(row, response)
    fields, family = row["verification"]["fields"], row["verification"]["family"]
    text = response.strip()
    try:
        if family == "arithmetic":
            return text == str(fields["a"] + fields["b"])
        if family == "sorting":
            from .verifiers import integer_list
            return integer_list(text) == sorted(integer_list(fields["values"]))
        if family == "extraction":
            return text == fields["code"]
        if family == "classification":
            return text == ("even" if fields["a"] % 2 == 0 else "odd")
        if family == "json":
            from .verifiers import exact_integer_object
            return exact_integer_object(text, {"number": fields["a"]})
    except (ValueError, TypeError, IndexError):
        return False
    return False
