import json
import os
import tempfile
import unittest

import rubric_eval as ev


def item(criteria):
    return {"id": "q", "prompt": "p", "response": "r", "criteria": criteria}


class Scoring(unittest.TestCase):
    def test_weighted_score(self):
        it = item([{"id": "a", "text": "a", "weight": 3}, {"id": "b", "text": "b", "weight": 1}])
        self.assertAlmostEqual(ev.score_item(it, {("q", "a"): True, ("q", "b"): False}), 0.75)

    def test_default_weight_is_one(self):
        it = item([{"id": "a", "text": "a"}, {"id": "b", "text": "b"}])
        self.assertAlmostEqual(ev.score_item(it, {("q", "a"): True, ("q", "b"): False}), 0.5)

    def test_penalty_and_clipping(self):
        it = item([{"id": "a", "text": "a", "weight": 1}, {"id": "bad", "text": "bad", "weight": -2}])
        self.assertAlmostEqual(ev.score_item(it, {("q", "a"): True, ("q", "bad"): False}), 1.0)
        self.assertEqual(ev.score_item(it, {("q", "a"): False, ("q", "bad"): True}), 0.0)

    def test_incomplete(self):
        it = item([{"id": "a", "text": "a"}, {"id": "b", "text": "b"}])
        self.assertIsNone(ev.score_item(it, {("q", "a"): True}))


class Validation(unittest.TestCase):
    def test_example_dataset(self):
        self.assertEqual(ev.validate_dataset(ev._read_jsonl("example_dataset.jsonl")), [])

    def test_problems(self):
        bad = [
            {"id": "x", "prompt": "p", "response": "", "criteria": []},
            {"id": "x", "prompt": "p", "response": "r",
             "criteria": [{"id": "a", "text": "t", "weight": 0}]},
        ]
        text = "\n".join(ev.validate_dataset(bad))
        self.assertIn("'response' must be a non-empty string", text)
        self.assertIn("needs at least one criterion", text)
        self.assertIn("used twice", text)
        self.assertIn("non-zero number", text)

    def test_bad_weights(self):
        for weight in (float("inf"), float("nan"), 1e308, 10 ** 400, True, "2"):
            with self.subTest(weight=weight):
                bad = [item([{"id": "a", "text": "t", "weight": weight}])]
                self.assertTrue(ev.validate_dataset(bad))

    def test_unreadable_ids(self):
        for bad_id in ("  ", "q\ud800"):
            with self.subTest(bad_id=bad_id):
                bad = [{"id": bad_id, "prompt": "p", "response": "r",
                        "criteria": [{"id": "a", "text": "t"}]}]
                self.assertTrue(ev.validate_dataset(bad))

    def test_no_positive_weight(self):
        bad = [item([{"id": "a", "text": "t", "weight": -1}])]
        self.assertTrue(any("positive weight" in p for p in ev.validate_dataset(bad)))


class Agreement(unittest.TestCase):
    def test_kappa(self):
        self.assertAlmostEqual(ev.cohen_kappa([True, False, True, False], [True, False, True, False]), 1.0)
        self.assertAlmostEqual(ev.cohen_kappa([True, True, False, False], [True, False, True, False]), 0.0)

    def test_kappa_undefined(self):
        self.assertIsNone(ev.cohen_kappa([], []))
        self.assertIsNone(ev.cohen_kappa([True, True], [True, True]))

    def test_kappa_length_mismatch(self):
        with self.assertRaises(ValueError):
            ev.cohen_kappa([True], [True, False])

    def test_agreement(self):
        dataset = ev.load_dataset("example_dataset.jsonl")
        grades = ev.load_grades("example_grades.jsonl", dataset)
        result = ev.agreement(grades, "alice", "bob")
        self.assertEqual(result["shared"], 8)
        self.assertEqual(result["agreed"], 7)
        self.assertEqual(result["disagreements"], [("unit-conversion", "unit")])


class Files(unittest.TestCase):
    def write(self, folder, name, text):
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return path

    def test_unknown_item(self):
        dataset = ev.load_dataset("example_dataset.jsonl")
        row = {"item": "nope", "criterion": "a", "rater": "r", "passed": True}
        with tempfile.TemporaryDirectory() as folder:
            path = self.write(folder, "g.jsonl", json.dumps(row) + "\n")
            with self.assertRaises(ev.DataError):
                ev.load_grades(path, dataset)

    def test_malformed_grades(self):
        dataset = ev.load_dataset("example_dataset.jsonl")
        rows = ["5", "[]", "null", '{"item": ["a"], "criterion": "x", "rater": "r", "passed": true}',
                '{"item": "unit-conversion", "criterion": ["a"], "rater": "r", "passed": true}',
                '{"item": "unit-conversion", "criterion": "speed", "rater": "", "passed": true}',
                '{"item": "unit-conversion", "criterion": "speed", "rater": "r", "passed": "yes"}']
        with tempfile.TemporaryDirectory() as folder:
            for row in rows:
                with self.subTest(row=row):
                    path = self.write(folder, "g.jsonl", row + "\n")
                    with self.assertRaises(ev.DataError):
                        ev.load_grades(path, dataset)

    def test_duplicate_grades(self):
        dataset = ev.load_dataset("example_dataset.jsonl")
        row = '{"item": "unit-conversion", "criterion": "speed", "rater": "r", "passed": %s}\n'
        with tempfile.TemporaryDirectory() as folder:
            same = self.write(folder, "a.jsonl", row % "true" + row % "true")
            self.assertEqual(len(ev.load_grades(same, dataset)["r"]), 1)
            conflict = self.write(folder, "b.jsonl", row % "true" + row % "false")
            with self.assertRaises(ev.DataError):
                ev.load_grades(conflict, dataset)

    def test_unreadable_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bad.jsonl")
            with open(path, "wb") as handle:
                handle.write(b"\xff\xfe\x00")
            with self.assertRaises(ev.DataError):
                ev.load_dataset(path)
            deep = self.write(folder, "deep.jsonl", "[" * 100000 + "]" * 100000 + "\n")
            with self.assertRaises(ev.DataError):
                ev.load_dataset(deep)

    def test_cli(self):
        self.assertEqual(ev.main(["check", "example_dataset.jsonl"]), 0)
        self.assertEqual(ev.main(["score", "example_dataset.jsonl", "example_grades.jsonl"]), 0)
        self.assertEqual(ev.main(["agree", "example_dataset.jsonl", "example_grades.jsonl", "alice", "bob"]), 0)
        self.assertEqual(ev.main(["agree", "example_dataset.jsonl", "example_grades.jsonl", "alice", "zed"]), 1)
        self.assertEqual(ev.main(["agree", "example_dataset.jsonl", "example_grades.jsonl", "alice", "alice"]), 1)
        self.assertEqual(ev.main(["check", "missing.jsonl"]), 1)


if __name__ == "__main__":
    unittest.main()
