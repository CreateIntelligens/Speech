"""Offline regression tests for the QA judgments: failures must never count as passes.

    uv run --no-project --with jiwer --with opencc-python-reimplemented --with soundfile python -m unittest qa/test_checks.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402
import run_suite  # noqa: E402


def answer(text="hello world", status=200, finish="stop"):
    return {"status": status, "text": text, "finish": finish}


def with_words(words, text="hello world"):
    return dict(answer(text), raw={"ctc_timestamps": {"words": words, "error": None}})


GOOD_WORDS = [{"word": "hello", "start": 0.1, "end": 0.5}, {"word": "world", "start": 0.6, "end": 1.0}]


class ConsistencyTest(unittest.TestCase):
    def test_identical_failures_do_not_pass(self):
        for failed in (answer(None, 500, None), answer(None, None, None), answer(None), answer("  ")):
            self.assertFalse(run_suite.consistent([failed, dict(failed)]), failed)

    def test_truncated_outputs_do_not_pass(self):
        self.assertFalse(run_suite.consistent([answer(finish="length"), answer(finish="length")]))

    def test_stream_cut_off_does_not_pass(self):
        chunk = {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}
        r = run_suite.Client._read_stream([f"data: {json.dumps(chunk)}\n".encode()], 0.0)
        self.assertFalse(run_suite.response_ok(r))

    def test_stream_error_chunk_does_not_pass(self):
        lines = [b'data: {"error": {"message": "boom"}}\n', b"data: [DONE]\n"]
        r = run_suite.Client._read_stream(lines, 0.0)
        self.assertFalse(run_suite.response_ok(r))
        self.assertIn("boom", r["error"])

    def test_same_good_answers_pass_and_different_fail(self):
        self.assertTrue(run_suite.consistent([answer(), answer()]))
        self.assertFalse(run_suite.consistent([answer("a"), answer("b")]))


class TimestampTest(unittest.TestCase):
    def check(self, words, text="hello world", duration=2.0):
        return run_suite.timestamp_check(with_words(words, text), duration)

    def test_valid_alignment_passes(self):
        self.assertTrue(self.check(GOOD_WORDS)["ok"])

    def test_speaker_tags_and_punctuation_do_not_break_text_match(self):
        self.assertTrue(self.check(GOOD_WORDS, "<spk:0> Hello, world.")["ok"])

    def test_long_identical_transcript_matches_fully(self):
        text = " ".join(f"word{i % 7} common" for i in range(3000))
        words = [{"word": w, "start": i * 0.1, "end": i * 0.1 + 0.05} for i, w in enumerate(text.split())]
        self.assertEqual(self.check(words, text, duration=1000.0)["text_match"], 1.0)

    def test_reordered_words_fail(self):
        words = [{"word": "world", "start": 0.1, "end": 0.5}, {"word": "hello", "start": 0.6, "end": 1.0}]
        self.assertFalse(self.check(words)["ok"])

    def test_bad_ranges_fail(self):
        for bad in (
            [{"word": "hello", "start": -0.5, "end": 0.5}, GOOD_WORDS[1]],
            [{"word": "hello", "start": 0.5, "end": 0.1}, GOOD_WORDS[1]],
            [GOOD_WORDS[0], {"word": "world", "start": 0.6, "end": 99.0}],
            [{"word": "hello", "start": 1.0, "end": 1.2}, {"word": "world", "start": 0.2, "end": 0.4}],
            [{"word": "hello", "start": None, "end": 0.5}, GOOD_WORDS[1]],
        ):
            self.assertFalse(self.check(bad)["ok"], bad)

    def test_words_that_do_not_match_the_transcript_fail(self):
        self.assertFalse(self.check(GOOD_WORDS, "something completely different")["ok"])

    def test_alignment_error_and_no_words_fail(self):
        failed = dict(answer(), raw={"ctc_timestamps": {"words": [], "error": "alignment_failed"}})
        self.assertFalse(run_suite.timestamp_check(failed, 2.0)["ok"])

    def test_failed_request_fails(self):
        self.assertFalse(run_suite.timestamp_check(answer(None, 500, None), 2.0)["ok"])


class VerdictTest(unittest.TestCase):
    def reject(self, status, error):
        return {
            "id": "edge/corrupt_bytes",
            "expect": "reject",
            "status": status,
            "text": None,
            "error": error,
            "lang": "en_us",
            "ref": None,
        }

    def test_server_error_is_not_a_correct_rejection(self):
        for status in (500, 503, None):
            self.assertFalse(report._edge_verdict(self.reject(status, "Invalid or corrupted audio"))["pass"])

    def test_client_error_needs_the_expected_message(self):
        self.assertFalse(report._edge_verdict(self.reject(400, "model not found"))["pass"])
        self.assertTrue(report._edge_verdict(self.reject(400, "Invalid or corrupted audio data"))["pass"])

    def test_transcript_and_no_speech_need_a_clean_finish(self):
        base = {"id": "edge/x", "lang": "en_us", "ref": None}
        self.assertFalse(report._edge_verdict(dict(base, expect="transcript", **answer(finish="length")))["pass"])
        self.assertFalse(report._edge_verdict(dict(base, expect="no_speech", **answer(None, 500, None)))["pass"])
        self.assertTrue(report._edge_verdict(dict(base, expect="no_speech", **answer(None)))["pass"])


class ScoringTest(unittest.TestCase):
    def row(self, text, status=200, finish="stop"):
        return {
            "lang": "cmn_hans_cn",
            "ref": "今天天气",
            "text": text,
            "status": status,
            "finish": finish,
            "latency_s": 1.0,
        }

    def test_empty_answers_stay_in_the_score(self):
        rate = report._error_rate([self.row("今天天气"), self.row(None)])
        self.assertEqual(rate["rate"], 50.0)
        self.assertEqual(rate["empty"], 1)

    def test_failed_requests_are_reported_and_scored_in_the_task_rate(self):
        rate = report._error_rate([self.row("今天天气"), self.row(None, status=500)])
        self.assertEqual(rate["rate"], 0.0)
        self.assertEqual(rate["rate_all_requests"], 50.0)
        self.assertEqual(rate["failed"], 1)

    def test_all_failed_has_no_answered_rate(self):
        rate = report._error_rate([self.row(None, status=500)])
        self.assertIsNone(rate["rate"])
        self.assertEqual(rate["rate_all_requests"], 100.0)

    def test_rounds_keep_every_result(self):
        rows = [dict(self.row("今天天气"), round=0), dict(self.row("今天"), round=1)]
        scored = report._rounds(rows)
        self.assertEqual(scored["per_round"], [0.0, 50.0])
        self.assertEqual(scored["rate"], 25.0)

    def test_throughput_counts_only_completed_audio(self):
        rows = [
            dict(answer(), path="a", group="g", burst_wall_s=10.0, latency_s=1.0),
            dict(answer(None, 500, None), path="b", group="g", burst_wall_s=10.0, latency_s=1.0),
        ]
        with mock.patch.object(report, "_duration", return_value=20.0):
            burst = report._burst(rows)
        self.assertEqual(burst["effective_rtfx"], 2.0)
        self.assertEqual(burst["completed"], 1)

    def test_unreadable_duration_raises(self):
        with self.assertRaises(Exception):
            report._duration("/nonexistent.wav")
        with self.assertRaises(RuntimeError):
            run_suite.audio_duration("/nonexistent.wav")


class EvidenceTest(unittest.TestCase):
    def make_run(self, root):
        with mock.patch.object(run_suite, "_provenance", return_value={"manifest_sha256": "x"}):
            return run_suite.Run("variant", "http://x", "suite", root=Path(root))

    def test_runs_never_share_a_directory_and_suites_never_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            first, second = self.make_run(root), self.make_run(root)
            self.assertNotEqual(first.dir, second.dir)
            first.save("api", [{"a": 1}])
            with self.assertRaises(FileExistsError):
                first.save("api", [{"a": 2}])
            self.assertEqual(json.loads((first.dir / "api.jsonl").read_text()), {"a": 1})
            self.assertEqual(json.loads((first.dir / "run.json").read_text())["suites"], {"api": 1})

    def run_data(self, sha="a", finished=True, suites=report.SUITES):
        meta = {
            "finished_at": "t" if finished else None,
            "provenance": {
                "git": {"sha": sha, "tracked_diff_sha256": "d"},
                "manifest_sha256": "m",
                "qa_files_sha256": {},
            },
        }
        return dict({s: [] for s in suites}, _meta=meta)

    def validate(self, data, partial=False, mismatch=False):
        manifest = '{"id": "c1"}\n'
        with mock.patch("builtins.open", mock.mock_open(read_data=manifest)):
            return report._validate(data, partial, mismatch)

    def complete(self, **kw):
        d = self.run_data(**kw)
        d["accuracy"] = [{"id": "c1"}]
        return d

    def test_missing_suite_or_unfinished_run_is_refused(self):
        with self.assertRaises(SystemExit):
            self.validate({"r": self.run_data(suites=["api"])})
        with self.assertRaises(SystemExit):
            self.validate({"r": self.complete(finished=False)})

    def test_different_versions_are_refused(self):
        with self.assertRaises(SystemExit):
            self.validate({"a": self.complete(sha="a"), "b": self.complete(sha="b")})

    def test_results_without_versions_are_refused_unless_waived(self):
        legacy = dict(self.complete(), _meta=None)
        with self.assertRaises(SystemExit):
            self.validate({"r": legacy})
        self.assertTrue(self.validate({"r": legacy}, mismatch=True))

    def test_matching_complete_runs_pass(self):
        self.assertEqual(self.validate({"a": self.complete(), "b": self.complete()}), [])


if __name__ == "__main__":
    unittest.main()
