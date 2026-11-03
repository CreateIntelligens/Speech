"""Run the QA suite against a running server and save raw results for one variant.

    python3 qa/run_suite.py --variant nvfp4
    python3 qa/run_suite.py --variant nvfp4 --only accuracy --groups fleurs_cmn_hans_cn --rounds 5

Every invocation creates qa/results/<variant>/<run id>/ holding run.json (code, service, model and data versions) and
one JSONL per suite, and refuses to overwrite anything. Score with qa/report.py <variant>/<run id>. Needs ffprobe;
otherwise standard library only.
"""

import argparse
import base64
import concurrent.futures as cf
import datetime
import difflib
import hashlib
import json
import math
import mimetypes
import re
import secrets
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

QA = Path(__file__).resolve().parent
REPO = QA.parents[2]
DEPLOY = QA.parent
API_CONTAINER = "nemotron-transcribe-api"
# 模型卡 FLEURS benchmark 的 prompt
BENCH_PROMPT = "Audio transcription. Language: as spoken, in its standard writing system. Wording: verbatim. Output only the transcript."
ZH_TW_PROMPT = "請將這段音訊逐字轉寫成繁體中文，只輸出逐字稿。"
SPEAKER_PROMPT = BENCH_PROMPT + " Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
SUITES = ["accuracy", "zh_tw", "api", "latency", "burst", "long"]
RECORDED_ENV = (
    "MODEL_VARIANT",
    "NUM_SPECULATIVE_TOKENS",
    "GPU_MEMORY_UTILIZATION",
    "CACHE_GIB",
    "MAX_NUM_SEQS",
    "MAX_MODEL_LEN",
    "VLLM_DISABLED_KERNELS",
)
# 時間戳可以略超過檔尾（幀對齊），文字則要幾乎逐字對上轉寫結果
TIMESTAMP_TOLERANCE_S = 0.25
TIMESTAMP_MIN_TEXT_MATCH = 0.98


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="label for this server config, e.g. nvfp4 / fp8")
    parser.add_argument("--url", default="http://127.0.0.1:8804")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", help=f"comma list of {SUITES}")
    parser.add_argument("--groups", help="accuracy suite only: comma list of manifest groups (default: all)")
    parser.add_argument("--rounds", type=int, default=1, help="accuracy suite only: repeat each case this many times")
    args = parser.parse_args()

    cases = [json.loads(line) for line in open(QA / "data" / "manifest.jsonl")]
    client = Client(args.url)
    run = Run(args.variant, args.url, label="suite")
    for suite in args.only.split(",") if args.only else SUITES:
        start = time.perf_counter()
        if suite == "accuracy":
            rows = suite_accuracy(client, cases, args.concurrency, args.groups, args.rounds)
        else:
            rows = globals()[f"suite_{suite}"](client, cases, args.concurrency)
        run.save(suite, rows)
        print(f"[{run.name}] {suite}: {len(rows)} rows in {time.perf_counter() - start:.0f}s", flush=True)
    run.finish()


def suite_accuracy(client, cases, concurrency, groups=None, rounds=1):
    wanted = set(groups.split(",")) if groups else None
    todo = [c for c in cases if not c["id"].startswith("edge/long") and (wanted is None or c["group"] in wanted)]
    if wanted and wanted - {c["group"] for c in todo}:
        raise SystemExit(f"unknown groups: {sorted(wanted - {c['group'] for c in todo})}")
    rows = []
    for round_index in range(rounds):
        results = _parallel(lambda c: dict(c, **client.chat(c["path"], BENCH_PROMPT)), todo, concurrency)
        rows += [dict(r, round=round_index) for r in results]
    return rows


def suite_zh_tw(client, cases, concurrency):
    todo = [c for c in cases if c["group"] == "fleurs_cmn_hans_cn"]
    return _parallel(lambda c: dict(c, **client.chat(c["path"], ZH_TW_PROMPT)), todo, concurrency)


def suite_api(client, cases, concurrency):
    en = [c for c in cases if c["group"] == "fleurs_en_us"]
    zh = [c for c in cases if c["group"] == "fleurs_cmn_hans_cn"]
    rows = []
    # 串流與非串流的輸出應一致（temperature 0）
    for c in en[:10] + zh[:10]:
        plain = client.chat(c["path"], BENCH_PROMPT)
        streamed = client.chat(c["path"], BENCH_PROMPT, stream=True)
        rows.append(
            {
                "check": "stream_equals_nonstream",
                "id": c["id"],
                "ok": consistent([plain, streamed]),
                "plain": plain["text"],
                "stream": streamed["text"],
                "responses": _brief([plain, streamed]),
            }
        )
    # 同一請求重複三次應得到相同輸出
    for c in en[10:15] + zh[10:15]:
        responses = [client.chat(c["path"], BENCH_PROMPT) for _ in range(3)]
        rows.append(
            {
                "check": "deterministic",
                "id": c["id"],
                "ok": consistent(responses),
                "texts": [r["text"] for r in responses],
                "responses": _brief(responses),
            }
        )
    for c in en[:5] + zh[:5]:
        r = client.chat(c["path"], BENCH_PROMPT, timestamps=True)
        check = timestamp_check(r, audio_duration(c["path"]))
        rows.append(dict(check, check="timestamps", id=c["id"], text=r["text"], responses=_brief([r])))
    # 模型卡說時間戳只支援非串流：串流時應回明確錯誤而不是靜默丟掉
    r = client.chat(en[0]["path"], BENCH_PROMPT, timestamps=True, stream=True)
    rows.append(
        {
            "check": "timestamps_with_stream",
            "id": en[0]["id"],
            "status": r["status"],
            "error": r.get("error"),
            "text": r["text"],
            "ok": _is_client_error(r["status"]),
        }
    )
    # 不給文字 prompt、只有音訊
    r = client.chat(en[1]["path"], None)
    rows.append(
        {
            "check": "audio_only_no_prompt",
            "id": en[1]["id"],
            "status": r["status"],
            "text": r["text"],
            "ref": en[1]["ref"],
            "ok": response_ok(r),
        }
    )
    # 錯的 model 名稱應 404
    r = client.chat(en[1]["path"], BENCH_PROMPT, model="no-such-model")
    rows.append(
        {"check": "wrong_model_name", "status": r["status"], "error": r.get("error"), "ok": r["status"] == 404}
    )
    return rows


def suite_latency(client, cases, concurrency):
    # 單一請求、依序送，量首字與完成時間（短句，最貼近即時字幕的用法）
    todo = [c for c in cases if c["group"] in ("fleurs_en_us", "fleurs_cmn_hans_cn")][::4]
    return [dict(c, **client.chat(c["path"], BENCH_PROMPT, stream=True)) for c in todo]


def suite_burst(client, cases, concurrency):
    # 一次湧入 32 個請求，英文與華語各半：看總吞吐、尾端延遲與有無失敗
    en = [c for c in cases if c["group"] == "fleurs_en_us"][:16]
    zh = [c for c in cases if c["group"] == "fleurs_cmn_hans_cn"][:16]
    todo = [c for pair in zip(en, zh) for c in pair]
    start = time.perf_counter()
    rows = _parallel(lambda c: dict(c, **client.chat(c["path"], BENCH_PROMPT)), todo, 32)
    wall = time.perf_counter() - start
    return [dict(r, burst_wall_s=wall) for r in rows]


def suite_long(client, cases, concurrency):
    rows = []
    for c in [c for c in cases if c["id"].startswith("edge/long")]:
        r = client.chat(c["path"], SPEAKER_PROMPT, timestamps=True)
        check = timestamp_check(r, audio_duration(c["path"])) if r["status"] == 200 else {}
        words = ((r.get("raw") or {}).get("ctc_timestamps") or {}).get("words") or []
        rows.append(
            dict(
                c,
                **r,
                timestamps=check,
                n_words=len(words),
                last_end=words[-1]["end"] if words else None,
                speakers=sorted({str(w.get("speaker")) for w in words} - {"None"}),
            )
        )
    return rows


def response_ok(r):
    """A request counts as answered only when it succeeded, finished normally and produced text."""
    text = r.get("text")
    return r.get("status") == 200 and r.get("finish") == "stop" and isinstance(text, str) and bool(text.strip())


def consistent(responses):
    """Identical failures must not pass as identical outputs."""
    return all(response_ok(r) for r in responses) and len({r["text"] for r in responses}) == 1


def timestamp_check(response, duration_s):
    """Validate CTC word timestamps against the audio length and the transcript they claim to align."""
    alignment = (response.get("raw") or {}).get("ctc_timestamps") or {}
    words = alignment.get("words") or []
    problems = []
    if not response_ok(response):
        problems.append("request did not complete")
    if alignment.get("error") is not None:
        problems.append(f"alignment error: {alignment['error']}")
    if not words:
        problems.append("no words")
    previous_start = -math.inf
    for word in words:
        start, end = word.get("start"), word.get("end")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (start, end)):
            problems.append(f"non-numeric time on {word.get('word')!r}")
            break
        if start < 0 or end < start or end > duration_s + TIMESTAMP_TOLERANCE_S:
            problems.append(f"bad range {start}-{end} of {duration_s:.2f}s on {word.get('word')!r}")
            break
        if start < previous_start:
            problems.append(f"start goes backwards at {word.get('word')!r}")
            break
        previous_start = start
    match = _text_match("".join(str(w.get("word")) for w in words), response.get("text") or "")
    if words and match < TIMESTAMP_MIN_TEXT_MATCH:
        problems.append(f"words match the transcript only {match:.3f}")
    return {
        "ok": not problems,
        "problems": problems,
        "error": alignment.get("error"),
        "n_words": len(words),
        "text_match": round(match, 4),
        "last_end": words[-1].get("end") if words else None,
        "duration_s": duration_s,
    }


def audio_duration(path):
    """Audio length in seconds from ffprobe; raises rather than guessing, since scores and limits depend on it."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError(f"cannot read duration of {path}: {result.stderr.strip()[:200]}") from None


class Run:
    """One immutable result directory with the versions it ran against."""

    def __init__(self, variant, url, label, root=None):
        started = datetime.datetime.now(datetime.timezone.utc)
        self.run_id = f"{started:%Y%m%dT%H%M%SZ}-{secrets.token_hex(2)}"
        self.name = f"{variant}/{self.run_id}"
        self.dir = (root or QA / "results") / variant / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.meta = {
            "schema_version": 1,
            "run_id": self.run_id,
            "variant": variant,
            "label": label,
            "url": url,
            "argv": sys.argv,
            "started_at": started.isoformat(),
            "finished_at": None,
            "suites": {},
            "provenance": _provenance(url),
        }
        self._write_meta()

    def save(self, suite, rows):
        with open(self.dir / f"{suite}.jsonl", "x") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.meta["suites"][suite] = len(rows)
        self._write_meta()

    def finish(self):
        self.meta["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._write_meta()

    def _write_meta(self):
        (self.dir / "run.json").write_text(json.dumps(self.meta, ensure_ascii=False, indent=1))


class Client:
    def __init__(self, url):
        self.url = url

    def chat(self, path, prompt, stream=False, timestamps=False, model="nemotron-transcribe"):
        path = Path(path)
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
        content = [
            {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}
        ]
        if prompt is not None:
            content.insert(0, {"type": "text", "text": prompt})
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            # 重複輸出（聽不懂的語言、雜訊）會一路生成到 context 用完，照時長給上限
            "max_tokens": _max_tokens(path, data),
            "chat_template_kwargs": {"enable_thinking": False},
            "skip_special_tokens": False,
            "stream": stream,
        }
        if timestamps:
            body["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=3600) as resp:
                if stream:
                    return self._read_stream(resp, start)
                raw = json.load(resp)
                choice = raw["choices"][0]
                return {
                    "status": 200,
                    "text": choice["message"]["content"],
                    "finish": choice["finish_reason"],
                    "completion_tokens": raw["usage"]["completion_tokens"],
                    "latency_s": time.perf_counter() - start,
                    "raw": raw,
                }
        except urllib.error.HTTPError as e:
            return {
                "status": e.code,
                "text": None,
                "finish": None,
                "error": e.read().decode()[:500],
                "latency_s": time.perf_counter() - start,
            }
        except Exception as e:  # 連線中斷、逾時：記下來讓報告看得到，而不是讓整輪測試停掉
            return {
                "status": None,
                "text": None,
                "finish": None,
                "error": repr(e)[:500],
                "latency_s": time.perf_counter() - start,
            }

    @staticmethod
    def _read_stream(resp, start):
        text, finish, ttft, done, error = "", None, None, False, None
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            if line.endswith("[DONE]"):
                done = True
                continue
            chunk = json.loads(line[5:])
            error = chunk.get("error") or error
            if not chunk.get("choices"):
                continue
            piece = chunk["choices"][0]["delta"].get("content") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - start
            text += piece
            finish = chunk["choices"][0].get("finish_reason") or finish
        # 串流的 HTTP 狀態在第一個位元組就定了，半途出錯或被切斷都要另外判斷
        return {
            "status": 200 if done and not error else "stream_incomplete",
            "text": text or None,
            "finish": finish if done else None,
            "ttft_s": ttft,
            "error": json.dumps(error) if error else None,
            "latency_s": time.perf_counter() - start,
        }


def _parallel(fn, items, workers):
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def _brief(responses):
    return [{"status": r.get("status"), "finish": r.get("finish"), "error": r.get("error")} for r in responses]


def _is_client_error(status):
    return isinstance(status, int) and 400 <= status < 500


def _max_tokens(path, data):
    try:
        seconds = audio_duration(path)
    except RuntimeError:  # 故意送壞檔的案例讀不到時長，請求仍要送出去看伺服器怎麼回
        seconds = len(data) / 32000
    return min(30000, int(seconds * 8) + 256)


def _text_match(a, b):
    # autojunk 會把長文字裡的常見字元當雜訊丟掉，3 萬字的逐字稿明明一樣也只得到 0.48
    return difflib.SequenceMatcher(None, _comparable(a), _comparable(b), autojunk=False).ratio()


def _comparable(text):
    text = unicodedata.normalize("NFKC", re.sub(r"<spk:\d+>", " ", text)).lower()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in "PSZC")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def _provenance(url):
    """What this run tested. A missing piece is recorded as an error so the report can show the gap."""
    out = {"manifest_sha256": _sha256(QA / "data" / "manifest.jsonl")}
    out["qa_files_sha256"] = {
        str(p.relative_to(DEPLOY)): _sha256(p)
        for p in [*QA.glob("*.py"), DEPLOY / "entrypoint.sh", DEPLOY / "compose.yaml", DEPLOY / "Dockerfile"]
    }
    try:
        diff = subprocess.run(["git", "-C", str(REPO), "diff", "HEAD"], capture_output=True, check=True).stdout
        status = _command("git", "-C", str(REPO), "status", "--short").splitlines()
        out["git"] = {
            "sha": _command("git", "-C", str(REPO), "rev-parse", "HEAD"),
            "dirty": bool(status),
            "status": status,
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        out["git"] = {"error": repr(e)[:200]}
    with urllib.request.urlopen(f"{url}/v1/models", timeout=30) as resp:
        model = json.load(resp)["data"][0]
    out["service"] = {"model_root": model.get("root"), "max_model_len": model.get("max_model_len")}
    name = Path(model.get("root") or "").name
    metadata = DEPLOY / "models" / name / ".cache" / "huggingface" / "download" / "config.json.metadata"
    out["service"]["model_revision"] = metadata.read_text().splitlines()[0] if metadata.is_file() else None
    try:
        info = json.loads(_command("docker", "inspect", API_CONTAINER))[0]
        env = dict(item.split("=", 1) for item in info["Config"]["Env"] if "=" in item)
        out["service"]["image_id"] = info["Image"]
        out["service"]["env"] = {k: env[k] for k in RECORDED_ENV if k in env}
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        out["service"]["docker_error"] = repr(e)[:200]
    meminfo = Path("/proc/meminfo").read_text()
    out["mem_available_gib_at_start"] = round(int(re.search(r"MemAvailable:\s+(\d+)", meminfo)[1]) / 1048576, 1)
    return out


if __name__ == "__main__":
    main()
