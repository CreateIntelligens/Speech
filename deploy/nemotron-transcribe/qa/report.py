"""Score QA runs and compare them.

    uv run --no-project --with jiwer --with opencc-python-reimplemented --with soundfile \\
        python qa/report.py nvfp4_mtp8/<run id> fp8_mtp8/<run id>

Each argument is a directory under qa/results. Refuses incomplete runs and runs of different code or data unless told
otherwise; --allow-partial and --allow-mismatch print what was waived. Directories without run.json (the 2026-10-30
results) count as a mismatch because their versions are unknown. Prints markdown; --json also writes a new summary file.
"""

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path

import jiwer
import opencc
import soundfile

from run_suite import SUITES, response_ok

QA = Path(__file__).resolve().parent
T2S = opencc.OpenCC("t2s")
S2T = opencc.OpenCC("s2t")
# 這些語言用字元錯誤率：大多沒有可靠的空白斷詞，泰文與越南文則照模型卡的算法
CER_LANGS = {"cmn_hans_cn", "yue_hant_hk", "ja_jp", "ko_kr", "th_th", "vi_vn"}
# 預期被拒絕的案例要回 4xx 且訊息符合，5xx 或其他錯誤不算正確拒絕
REJECT_PATTERNS = {
    "edge/corrupt_bytes": "Invalid or corrupted audio",
    "edge/empty_file": "Invalid or corrupted audio",
    "edge/long_62min": "exceeds maximum allowed duration",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="result directories under qa/results, e.g. nvfp4_mtp8/<run id>")
    parser.add_argument("--allow-partial", action="store_true", help="accept runs missing suites or manifest cases")
    parser.add_argument("--allow-mismatch", action="store_true", help="accept runs of different or unknown versions")
    parser.add_argument("--json", type=Path, help="also write the summary to this new file")
    args = parser.parse_args()

    data = {run: _load(run) for run in args.runs}
    waived = _validate(data, args.allow_partial, args.allow_mismatch)
    summary = {run: _summarize(d) for run, d in data.items()}
    if args.json:
        with open(args.json, "x") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
    for line in waived:
        print(f"> **Waived:** {line}")
    if waived:
        print()
    _print_markdown(args.runs, summary)


def _load(run):
    directory = QA / "results" / run
    if not directory.is_dir():
        raise SystemExit(f"no such result directory: {directory}")
    out = {path.stem: [json.loads(line) for line in open(path)] for path in directory.glob("*.jsonl")}
    meta = directory / "run.json"
    out["_meta"] = json.loads(meta.read_text()) if meta.is_file() else None
    return out


def _validate(data, allow_partial, allow_mismatch):
    """Return the problems the caller chose to waive; stop on any other."""
    partial, mismatch = [], []
    cases = [json.loads(line) for line in open(QA / "data" / "manifest.jsonl")]
    expected = {c["id"] for c in cases if not c["id"].startswith("edge/long")}
    for run, d in data.items():
        missing = [s for s in SUITES if s not in d]
        if missing:
            partial.append(f"{run} has no {missing}")
        seen = {r["id"] for r in d.get("accuracy", [])}
        if "accuracy" in d and expected - seen:
            partial.append(f"{run} accuracy misses {len(expected - seen)} manifest cases")
        if d["_meta"] is None:
            mismatch.append(f"{run} has no run.json, so its code, service and data versions are unknown")
        elif not d["_meta"].get("finished_at"):
            partial.append(f"{run} did not finish")
    known = {run: d["_meta"]["provenance"] for run, d in data.items() if d["_meta"]}
    for key, read in {
        "git commit": lambda p: p.get("git", {}).get("sha"),
        "uncommitted diff": lambda p: p.get("git", {}).get("tracked_diff_sha256"),
        "dataset manifest": lambda p: p.get("manifest_sha256"),
        "QA and deploy files": lambda p: json.dumps(p.get("qa_files_sha256"), sort_keys=True),
    }.items():
        if len({read(p) for p in known.values()}) > 1:
            mismatch.append(f"runs differ in {key}")
    if partial and not allow_partial:
        raise SystemExit("incomplete runs (use --allow-partial to accept):\n  " + "\n  ".join(partial))
    if mismatch and not allow_mismatch:
        raise SystemExit("runs are not comparable (use --allow-mismatch to accept):\n  " + "\n  ".join(mismatch))
    return partial + mismatch


def _summarize(d):
    s = {"accuracy": {}, "edge": {}, "versions": (d["_meta"] or {}).get("provenance")}
    groups = {}
    for r in d.get("accuracy", []):
        groups.setdefault(r["group"], []).append(r)
    clean = {r["id"]: r for r in d.get("accuracy", []) if r["group"].startswith("fleurs_") and r.get("round", 0) == 0}
    for group, rows in sorted(groups.items()):
        if group == "edge":
            for r in rows:
                s["edge"][r["id"]] = _edge_verdict(r)
            continue
        s["accuracy"][group] = _rounds(rows)
        if group.startswith("degrade_"):
            # 乾淨列有 40 句、劣化只有其中 20 句，比較要用同一批句子
            ids = {r["id"].split("/", 1)[1] for r in rows}
            paired = [clean[i] for i in ids if i in clean]
            if paired:
                s["accuracy"][group]["same_utts_clean_rate"] = _error_rate(paired)["rate_all_requests"]
    if d.get("zh_tw"):
        rows = d["zh_tw"]
        s["zh_tw"] = _error_rate(rows)
        texts = [r["text"] or "" for r in rows]
        han = [ch for t in texts for ch in t if _is_han(ch)]
        simplified = [ch for ch in han if S2T.convert(ch) != ch]
        s["zh_tw"]["simplified_char_ratio"] = round(len(simplified) / max(1, len(han)), 4)
        s["zh_tw"]["utts_with_simplified"] = sum(any(S2T.convert(ch) != ch for ch in t if _is_han(ch)) for t in texts)
    if d.get("api"):
        checks = {}
        for r in d["api"]:
            c = checks.setdefault(r["check"], {"pass": 0, "total": 0, "notes": []})
            c["total"] += 1
            c["pass"] += r.get("ok") is True
            if r["check"] in ("timestamps_with_stream", "audio_only_no_prompt") or (
                r["check"] == "timestamps" and r.get("ok") is not True
            ):
                c["notes"].append({k: r.get(k) for k in ("id", "status", "error", "problems", "text")})
        s["api"] = checks
    if d.get("latency"):
        s["latency"] = _latency(d["latency"])
    if d.get("burst"):
        s["burst"] = _burst(d["burst"])
    if d.get("long"):
        s["long"] = {
            r["id"]: dict(
                {k: r.get(k) for k in ("status", "latency_s", "n_words", "last_end", "speakers", "finish", "error")},
                audio_s=round(_duration(r["path"]), 1),
                timestamps_ok=(r.get("timestamps") or {}).get("ok"),
                verdict=_edge_verdict(r),
            )
            for r in d["long"]
        }
    return s


def _rounds(rows):
    """Score each repeat separately: NVFP4 is not deterministic, so one round says little."""
    by_round = {}
    for r in rows:
        by_round.setdefault(r.get("round", 0), []).append(r)
    scored = [_error_rate(by_round[k]) for k in sorted(by_round)]
    if len(scored) == 1:
        return dict(scored[0], rounds=1)
    keys = ("rate", "rate_all_requests")
    out = dict(scored[0], rounds=len(scored), per_round=[x["rate"] for x in scored])
    out.update({k: round(statistics.mean(x[k] for x in scored), 2) for k in keys})
    out.update(rate_min=min(out["per_round"]), rate_max=max(out["per_round"]))
    for k in ("n", "failed", "empty", "truncated"):
        out[k] = sum(x[k] for x in scored)
    return out


def _error_rate(rows):
    """Errors over every answered request; a request that failed outright is reported, and also scored as all-missed.

    Empty and truncated answers stay in the score. Dropping them would reward a variant for saying nothing.
    """
    lang = rows[0]["lang"]
    answered = [r for r in rows if r["status"] == 200]
    refs = [_normalize(r["ref"], lang) for r in rows]
    hyps = [_normalize(r["text"] if r["status"] == 200 else "", lang) for r in rows]
    score = jiwer.cer if lang in CER_LANGS else jiwer.wer
    answered_refs = [_normalize(r["ref"], lang) for r in answered]
    answered_hyps = [_normalize(r["text"], lang) for r in answered]
    latency = [r["latency_s"] for r in answered]
    return {
        "metric": "CER" if lang in CER_LANGS else "WER",
        "rate": round(score(answered_refs, answered_hyps) * 100, 2) if answered else None,
        "rate_all_requests": round(score(refs, hyps) * 100, 2),
        "n": len(rows),
        "failed": len(rows) - len(answered),
        "empty": sum(not (r.get("text") or "").strip() for r in answered),
        "truncated": sum(r.get("finish") == "length" for r in answered),
        "latency_p50_s": round(_pct(latency, 50), 2) if latency else None,
    }


def _latency(rows):
    good = [r for r in rows if response_ok(r)]
    return {
        "n": len(rows),
        "completed": len(good),
        "ttft_p50_ms": _pct([r["ttft_s"] for r in good], 50) * 1000,
        "ttft_p95_ms": _pct([r["ttft_s"] for r in good], 95) * 1000,
        "total_p50_ms": _pct([r["latency_s"] for r in good], 50) * 1000,
        "total_p95_ms": _pct([r["latency_s"] for r in good], 95) * 1000,
    }


def _burst(rows):
    good = [r for r in rows if response_ok(r)]
    wall = rows[0]["burst_wall_s"]
    audio = sum(_duration(r["path"]) for r in good)
    return {
        "requests": len(rows),
        "completed": len(good),
        "groups": sorted({r["group"] for r in rows}),
        "completed_audio_s": round(audio, 1),
        "wall_s": round(wall, 1),
        "effective_rtfx": round(audio / wall, 1),
        "latency_p95_s": round(_pct([r["latency_s"] for r in rows], 95), 2),
    }


def _edge_verdict(r):
    text = r.get("text")
    verdict = {
        "expect": r["expect"],
        "status": r["status"],
        "finish": r.get("finish"),
        "text": (text or "")[:120],
        "error": (r.get("error") or "")[:160],
    }
    if r["expect"] == "reject":
        pattern = REJECT_PATTERNS.get(r["id"])
        verdict["pass"] = (
            isinstance(r["status"], int)
            and 400 <= r["status"] < 500
            and bool(pattern)
            and pattern in (r.get("error") or "")
        )
    elif r["expect"] == "no_speech":
        # 靜音或雜訊應該幾乎沒有輸出；吐出整句就是幻覺
        verdict["pass"] = (
            r["status"] == 200 and r.get("finish") == "stop" and len(_normalize(text or "", "cmn_hans_cn")) <= 3
        )
    else:
        verdict["pass"] = response_ok(r)
        if r.get("ref"):
            verdict["wer"] = round(
                jiwer.wer(_normalize(r["ref"], r["lang"]), _normalize(text or "", r["lang"])) * 100, 1
            )
    return verdict


def _normalize(text, lang):
    text = re.sub(r"<spk:\d+>", " ", text or "")
    if lang in ("cmn_hans_cn", "yue_hant_hk"):
        text = T2S.convert(text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    if lang in CER_LANGS:
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_han(ch):
    return "一" <= ch <= "鿿"


def _duration(path):
    """Raises on an unreadable file: counting it as zero seconds would quietly change every throughput figure."""
    return soundfile.info(path).duration


def _pct(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def _print_markdown(variants, summary):
    print("## Versions\n")
    for v in variants:
        p = summary[v]["versions"]
        if not p:
            print(f"- {v}: unknown (no run.json)")
            continue
        service = p["service"]
        print(
            f"- {v}: git {p.get('git', {}).get('sha', '?')[:10]} dirty={p.get('git', {}).get('dirty')}, "
            f"model {Path(service.get('model_root') or '?').name}@{(service.get('model_revision') or '?')[:8]}, "
            f"image {(service.get('image_id') or '?')[:19]}"
        )
    print("\n## Accuracy (lower is better)\n")
    print(
        "`rate` scores answered requests, including empty and truncated answers. `all` also counts failed requests "
        "as fully missed. `clean` is the same utterances without degradation.\n"
    )
    print("| group | metric | " + " | ".join(variants) + " |")
    print("|---|---|" + "---|" * len(variants))
    groups = sorted({g for v in variants for g in summary[v]["accuracy"]})
    for g in groups:
        cells = [_accuracy_cell(summary[v]["accuracy"].get(g)) for v in variants]
        metric = next(summary[v]["accuracy"][g]["metric"] for v in variants if g in summary[v]["accuracy"])
        print(f"| {g} | {metric} | " + " | ".join(cells) + " |")
    print("\n## zh-TW prompt\n")
    for v in variants:
        print(f"- {v}: {summary[v].get('zh_tw')}")
    print("\n## API checks\n")
    for v in variants:
        print(f"- {v}: " + ", ".join(f"{k} {c['pass']}/{c['total']}" for k, c in summary[v].get("api", {}).items()))
    print("\n## Latency / burst\n")
    for v in variants:
        print(f"- {v}: latency {summary[v].get('latency')} | burst {summary[v].get('burst')}")
    print("\n## Long audio\n")
    for v in variants:
        print(f"- {v}: {json.dumps(summary[v].get('long'), ensure_ascii=False)}")
    print("\n## Edge cases\n")
    print("| case | expect | " + " | ".join(variants) + " |")
    print("|---|---|" + "---|" * len(variants))
    for case in sorted({c for v in variants for c in summary[v]["edge"]}):
        cells = []
        for v in variants:
            e = summary[v]["edge"].get(case)
            cells.append("—" if not e else ("PASS" if e["pass"] else "FAIL") + f" [{e['status']}] {e['text'][:50]!r}")
        expect = next(summary[v]["edge"][case]["expect"] for v in variants if case in summary[v]["edge"])
        print(f"| {case} | {expect} | " + " | ".join(cells) + " |")


def _accuracy_cell(a):
    if not a:
        return "—"
    cell = f"{a['rate']}"
    if a["rounds"] > 1:
        cell += f" (mean of {a['rounds']}: {a['rate_min']}–{a['rate_max']})"
    if a["failed"]:
        cell += f" all={a['rate_all_requests']} failed={a['failed']}"
    if a["empty"]:
        cell += f" empty={a['empty']}"
    if a["truncated"]:
        cell += f" truncated={a['truncated']}"
    if "same_utts_clean_rate" in a:
        cell += f" clean={a['same_utts_clean_rate']}"
    return cell


if __name__ == "__main__":
    main()
