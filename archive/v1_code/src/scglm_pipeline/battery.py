"""Controlled grounded tasks plus human-task diagnostics. No teacher model.

These are narrow checks using fictional records, not general reasoning benchmarks.
Development and confirmation have different entities, facts, and prompt wrappers.
"""
from __future__ import annotations
import random
import re
from collections import Counter
from .common import digest

FAMILIES = ("grounded_qa", "unsupported_qa", "rewrite", "summary", "grounded_choice")
WRAPPERS = {
    "development": (
        "Use only the passage below.", "Answer from the supplied note.",
        "The following fictional record is the complete evidence.",
        "Read the report before answering.", "Restrict your answer to these facts.",
        "Treat the text below as the source for this task.",
    ),
    "confirmation": (
        "Consult only this short account.", "Base the response on the enclosed description.",
        "Use the information in this fictional entry.",
        "Here is the entire record needed for the question.",
        "According to the provided account, answer the request.",
        "Use this source and supply no external details.",
    ),
}


def generate(phase, per_family=120, seed=20260923):
    if phase not in WRAPPERS or per_family < 12:
        raise ValueError("Require development/confirmation and >=12 cases/family")
    rng = random.Random(seed + (0 if phase == "development" else 10000))
    code = "DV" if phase == "development" else "CF"
    colors = ("blue", "red", "green", "yellow", "orange", "purple")
    days = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    rows = []
    for family in FAMILIES:
        for i in range(per_family):
            person = f"Mira-{code}{i:04d}"
            place = f"Ridge-{code}{i:04d}"
            color = rng.choice(colors)
            amount = rng.randrange(11, 89)
            opening, inspection = rng.sample(days, 2)
            variant = (i // 6) % 3
            context = f"{person} maintains the depot at {place}. The depot stores {amount} {color} crates. It opens on {opening}. Its inspection is on {inspection}."
            if family == "grounded_qa":
                questions = (f"Who maintains the depot at {place}? Return the name only.",
                             f"How many crates are stored at {place}? Return the integer only.",
                             f"What color are the crates at {place}? Return the color only.")
                answers = (person, str(amount), color)
                question, answer = questions[variant], answers[variant]
            elif family == "unsupported_qa":
                field = ("telephone number", "street address", "founding year")[variant]
                question = f"What is the depot's {field}? If the passage does not say, reply exactly UNKNOWN."
                answer = "UNKNOWN"
            elif family == "rewrite":
                question = f'Rewrite "{amount} {color} crates are stored by {person}." in active voice. Return only the rewritten sentence.'
                answer = f"{person} stores {amount} {color} crates."
            elif family == "summary":
                question = 'Summarize only the stored quantity and color, using exactly "Stock: NUMBER COLOR crates."'
                answer = f"Stock: {amount} {color} crates."
            else:
                question = "On which day does the depot open? Reply with the day only."
                answer = opening
            prompt = f"{WRAPPERS[phase][i % 6]}\nPassage: {context}\nTask: {question}"
            choices = [answer]
            if family == "grounded_choice":
                choices += [inspection, *rng.sample([day for day in days if day not in (opening, inspection)], 2)]
            elif family == "grounded_qa":
                choices += ([f"Lena-{code}{i:04d}", place, "UNKNOWN"] if variant == 0 else
                            [str(amount+1), str(amount-1), "UNKNOWN"] if variant == 1 else
                            [c for c in colors if c != color][:3])
            else:
                choices = []
            if choices:
                # Explicit, balanced answer positions, with seeded distractor order.
                other = choices[1:]; rng.shuffle(other)
                other.insert(i % 4, answer); choices = other
            group = f"{phase}:{family}:wrapper{i % 6}:variant{variant}"
            rows.append({"id": digest([phase, family, i, prompt]), "group": group,
                         "family": family, "phase": phase, "context": context, "prompt": prompt,
                         "answer": answer, "choices": choices,
                         "answer_index": choices.index(answer) if choices else None,
                         "scoring": "normalized_exact", "provenance": "local deterministic fictional records; no LLM generation"})
    return rows


def normalized_answer(text):
    return " ".join(text.casefold().split()).rstrip(".")


def correct(row, response):
    return normalized_answer(response) == normalized_answer(row["answer"])


def token_f1(reference, hypothesis):
    a, b = Counter(re.findall(r"\w+", reference.casefold())), Counter(re.findall(r"\w+", hypothesis.casefold()))
    shared = sum((a & b).values())
    return 2 * shared / max(1, sum(a.values()) + sum(b.values()))


def repetition(text):
    words = text.split()
    return any(words[i:i+4] * 4 == words[i:i+16] for i in range(max(0, len(words)-15)))
