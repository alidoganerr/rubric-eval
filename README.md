# rubric-eval

Score model answers with a weighted yes/no checklist, and see how far two reviewers agree. One Python file, standard library only (3.9+).

It's just a weighted checklist. You write a few checks per prompt, a reviewer marks each one passed or not, and the marks become a score between 0 and 1. The agreement part is there because a score from one reviewer is hard to trust until someone else has graded the same answers.

## Files

Both inputs are JSON Lines. A dataset item:

```json
{"id": "unit-conversion",
 "prompt": "A car travels 90 km in 1.5 hours. What is its average speed in metres per second?",
 "response": "Speed is 90 / 1.5 = 60 km/h. That is about 16.7 m/s.",
 "criteria": [
   {"id": "speed", "text": "Computes 60 km/h", "weight": 2},
   {"id": "converts", "text": "Converts to metres per second correctly", "weight": 3},
   {"id": "extra", "text": "Adds information nobody asked for", "weight": -1}
 ]}
```

A grade:

```json
{"item": "unit-conversion", "criterion": "speed", "rater": "alice", "passed": true}
```

## Commands

```
python3 rubric_eval.py check example_dataset.jsonl
python3 rubric_eval.py score example_dataset.jsonl example_grades.jsonl
python3 rubric_eval.py agree example_dataset.jsonl example_grades.jsonl alice bob
```

`check` looks for missing fields, repeated ids and bad weights. `score` prints each rater's score per item and their mean; an item only counts once that rater has graded all of its criteria. `agree` compares two raters on the criteria they both graded and lists where they differ, with percent agreement and Cohen's kappa.

From Python:

```python
import rubric_eval as ev

dataset = ev.load_dataset("example_dataset.jsonl")
grades = ev.load_grades("example_grades.jsonl", dataset)
print(ev.score_all(dataset, grades))
print(ev.agreement(grades, "alice", "bob")["kappa"])
```

## Scoring details

Passed criteria add their weight, and the total is divided by the sum of the positive weights. A negative weight is a penalty: marking it as passed means the flaw is there, so it subtracts. The result is clipped to 0..1.

If a rater grades the same criterion twice with different answers, that's an error. The same answer twice is fine.

Kappa has no value when there's nothing to agree on beyond chance, for example when both raters pass everything. You get `n/a` then.

`python3 -m unittest` runs the tests. MIT license.
