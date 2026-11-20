#!/usr/bin/env python3
"""Score model responses against weighted yes/no criteria.

Files are JSON Lines.

Dataset, one item per line:
    {"id": "q1", "prompt": "...", "response": "...",
     "criteria": [{"id": "c1", "text": "States the unit", "weight": 2}]}

Grades, one judgement per line:
    {"item": "q1", "criterion": "c1", "rater": "alice", "passed": true}

Commands:
    python rubric_eval.py check dataset.jsonl
    python rubric_eval.py score dataset.jsonl grades.jsonl
    python rubric_eval.py agree dataset.jsonl grades.jsonl alice bob
"""
import argparse
import json
import sys
from collections import defaultdict


MAX_WEIGHT = 1_000_000


class DataError(Exception):
    pass


def _read_jsonl(path):
    rows = []
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, RecursionError) as exc:
                    reason = getattr(exc, "msg", "too deeply nested")
                    raise DataError(f"{path}:{number}: invalid JSON ({reason})") from None
    except UnicodeDecodeError:
        raise DataError(f"{path}: not a UTF-8 text file") from None
    return rows


def _has_surrogate(text):
    return any("\ud800" <= ch <= "\udfff" for ch in text)


def validate_dataset(items):
    """Return a list of problems (empty if the dataset is fine)."""
    problems = []
    seen = set()
    for i, item in enumerate(items, 1):
        where = f"item {i}"
        if not isinstance(item, dict):
            problems.append(f"{where}: must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            problems.append(f"{where}: 'id' must be a non-empty string")
        elif not item_id.strip() or _has_surrogate(item_id):
            problems.append(f"{where}: 'id' must be readable text")
        elif item_id in seen:
            problems.append(f"{where}: id '{item_id}' is used twice")
        else:
            seen.add(item_id)
        for key in ("prompt", "response"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                problems.append(f"{where}: '{key}' must be a non-empty string")
        criteria = item.get("criteria")
        if not isinstance(criteria, list) or not criteria:
            problems.append(f"{where}: needs at least one criterion")
            continue
        ids = set()
        positive = 0
        for criterion in criteria:
            cid = criterion.get("id") if isinstance(criterion, dict) else None
            if not isinstance(cid, str) or not cid:
                problems.append(f"{where}: every criterion needs an 'id'")
                continue
            if cid in ids:
                problems.append(f"{where}: criterion '{cid}' is used twice")
            ids.add(cid)
            if not isinstance(criterion.get("text"), str) or not criterion["text"].strip():
                problems.append(f"{where}/{cid}: 'text' must be a non-empty string")
            weight = criterion.get("weight", 1)
            if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                    or weight == 0 or not abs(weight) <= MAX_WEIGHT):  # also rejects nan and inf
                problems.append(
                    f"{where}/{cid}: 'weight' must be a non-zero number, at most {MAX_WEIGHT:g} in size")
            elif weight > 0:
                positive += weight
        if positive == 0:
            problems.append(f"{where}: needs at least one criterion with a positive weight")
    return problems


def load_dataset(path):
    items = _read_jsonl(path)
    problems = validate_dataset(items)
    if problems:
        raise DataError("\n".join(problems))
    return {item["id"]: item for item in items}


def load_grades(path, dataset):
    """Return {rater: {(item, criterion): passed}}.

    The same judgement twice is allowed, a conflicting one is an error.
    """
    grades = defaultdict(dict)
    for number, row in enumerate(_read_jsonl(path), 1):
        where = f"{path}: grade {number}"
        if not isinstance(row, dict):
            raise DataError(f"{where}: must be an object")
        item, criterion, rater, passed = (row.get(k) for k in ("item", "criterion", "rater", "passed"))
        if not isinstance(item, str) or item not in dataset:
            raise DataError(f"{where}: unknown item {item!r}")
        if not isinstance(criterion, str) or criterion not in {c["id"] for c in dataset[item]["criteria"]}:
            raise DataError(f"{where}: unknown criterion {criterion!r} for item '{item}'")
        if not isinstance(rater, str) or not rater:
            raise DataError(f"{where}: 'rater' must be a non-empty string")
        if not isinstance(passed, bool):
            raise DataError(f"{where}: 'passed' must be true or false")
        key = (item, criterion)
        if key in grades[rater] and grades[rater][key] != passed:
            raise DataError(f"{where}: rater '{rater}' already graded {item}/{criterion} differently")
        grades[rater][key] = passed
    return dict(grades)


def score_item(item, rater_grades):
    # None until the rater has graded every criterion of the item.
    # Negative weights are penalties: "passed" means the flaw is there.
    total = 0.0
    for criterion in item["criteria"]:
        key = (item["id"], criterion["id"])
        if key not in rater_grades:
            return None
        if rater_grades[key]:
            total += criterion.get("weight", 1)
    best = sum(c.get("weight", 1) for c in item["criteria"] if c.get("weight", 1) > 0)
    return max(0.0, min(1.0, total / best))


def score_all(dataset, grades):
    result = {}
    for rater, rater_grades in grades.items():
        scores = {}
        for item_id, item in dataset.items():
            value = score_item(item, rater_grades)
            if value is not None:
                scores[item_id] = value
        result[rater] = scores
    return result


def cohen_kappa(a, b):
    """Cohen's kappa for two lists of booleans, or None where it is undefined
    (no data, or both raters always give the same one answer)."""
    if len(a) != len(b):
        raise ValueError("lists must have the same length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def agreement(grades, rater_a, rater_b):
    if rater_a == rater_b:
        raise DataError("choose two different raters")
    for rater in (rater_a, rater_b):
        if rater not in grades:
            raise DataError(f"no grades from rater '{rater}'")
    shared = sorted(set(grades[rater_a]) & set(grades[rater_b]))
    a = [grades[rater_a][k] for k in shared]
    b = [grades[rater_b][k] for k in shared]
    same = sum(x == y for x, y in zip(a, b))
    return {
        "shared": len(shared),
        "agreed": same,
        "percent": (same / len(shared)) if shared else None,
        "kappa": cohen_kappa(a, b),
        "disagreements": [k for k, x, y in zip(shared, a, b) if x != y],
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rubric_eval", description="Score responses against weighted criteria.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate a dataset").add_argument("dataset")
    s = sub.add_parser("score", help="score each rater's graded items")
    s.add_argument("dataset")
    s.add_argument("grades")
    g = sub.add_parser("agree", help="compare two raters")
    g.add_argument("dataset")
    g.add_argument("grades")
    g.add_argument("rater_a")
    g.add_argument("rater_b")
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
        if args.command == "check":
            criteria = sum(len(i["criteria"]) for i in dataset.values())
            print(f"ok: {len(dataset)} items, {criteria} criteria")
        elif args.command == "score":
            grades = load_grades(args.grades, dataset)
            for rater, scores in sorted(score_all(dataset, grades).items()):
                mean = sum(scores.values()) / len(scores) if scores else None
                print(f"{rater}: {len(scores)} of {len(dataset)} items fully graded, mean {_fmt(mean)}")
                for item_id, value in sorted(scores.items()):
                    print(f"  {item_id}: {value:.2f}")
        elif args.command == "agree":
            grades = load_grades(args.grades, dataset)
            r = agreement(grades, args.rater_a, args.rater_b)
            print(f"{r['shared']} shared judgements, {r['agreed']} agreed ({_fmt(r['percent'])}), kappa {_fmt(r['kappa'])}")
            for item, criterion in r["disagreements"]:
                print(f"  differ: {item} / {criterion}")
        return 0
    except (DataError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
