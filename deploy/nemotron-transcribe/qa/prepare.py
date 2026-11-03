"""Build the QA audio set: FLEURS subsets, degraded copies, and edge cases.

    uv run --with numpy --with soundfile python qa/prepare.py

Writes qa/data/cases/<group>/*.wav (or other formats) and qa/data/manifest.jsonl, one case per line:
{"id", "group", "path", "lang", "ref", "expect"}. `expect` is "transcript", "reject" or "no_speech".
"""

import csv
import json
import shutil
import subprocess
import tarfile
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

QA = Path(__file__).resolve().parent
RAW = QA / "data" / "fleurs_raw"
CASES = QA / "data" / "cases"
TESTDATA = QA.parent / "testdata"

LANGS = ["en_us", "cmn_hans_cn", "ja_jp", "ko_kr", "yue_hant_hk"]
PER_LANG = 40
# 劣化測試只做英文與華語，各取前 20 句
DEGRADE_LANGS = ["en_us", "cmn_hans_cn"]
PER_DEGRADE = 20
SR = 16000


def main():
    if CASES.exists():
        shutil.rmtree(CASES)
    cases = []
    clean = {}
    for lang in LANGS:
        clean[lang] = _extract_fleurs(lang)
        cases += [dict(c, group=f"fleurs_{lang}") for c in clean[lang]]
    babble = np.concatenate([sf.read(c["path"])[0] for c in clean["ko_kr"][:6]])
    for lang in DEGRADE_LANGS:
        for c in clean[lang][:PER_DEGRADE]:
            cases += _degrade(c, lang, babble)
    cases += _edge_cases(clean["en_us"][0])

    with open(QA / "data" / "manifest.jsonl", "w") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    groups = {}
    for c in cases:
        groups[c["group"]] = groups.get(c["group"], 0) + 1
    print(json.dumps(groups, indent=1))


def _extract_fleurs(lang):
    rows = list(csv.reader(open(RAW / f"{lang}.test.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    # 同一句常有多位朗讀者，每個句子 id 只取一個錄音，避免同句重複
    seen, picked = set(), []
    for row in rows:
        if row[0] not in seen:
            seen.add(row[0])
            picked.append(row)
        if len(picked) == PER_LANG:
            break
    wanted = {row[1]: row for row in picked}
    out = CASES / f"fleurs_{lang}"
    out.mkdir(parents=True)
    cases = []
    with tarfile.open(RAW / f"{lang}.test.tar.gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if name not in wanted or not member.isfile():
                continue
            dest = out / name
            dest.write_bytes(tar.extractfile(member).read())
            row = wanted[name]
            cases.append(
                {
                    "id": f"{lang}/{Path(name).stem}",
                    "path": str(dest),
                    "lang": lang,
                    "ref": row[2],
                    "expect": "transcript",
                }
            )
    return sorted(cases, key=lambda c: c["id"])


def _degrade(case, lang, babble):
    audio, sr = sf.read(case["path"])
    assert sr == SR
    # FLEURS 有些錄音峰值只有 -44 dBFS；音量類劣化要從同一基準出發才有意義
    audio = _peak_normalize(audio)
    rng = np.random.default_rng(zlib.crc32(case["id"].encode()))
    variants = {
        "noise_snr10": _mix(audio, rng.standard_normal(len(audio)), 10),
        "noise_snr0": _mix(audio, rng.standard_normal(len(audio)), 0),
        "babble_snr5": _mix(audio, np.resize(babble, len(audio)), 5),
        "quiet_-30db": audio * 10 ** (-30 / 20),
        "clipped": np.clip(audio * 10 ** (25 / 20), -0.3, 0.3) / 0.3 * 0.9,
    }
    out = []
    for name, signal in variants.items():
        dest = CASES / f"degrade_{name}" / f"{Path(case['path']).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dest, signal.astype(np.float32), SR, subtype="PCM_16")
        out.append(dict(case, id=f"{name}/{case['id']}", group=f"degrade_{name}_{lang}", path=str(dest)))
    # 電話音質與語速用 ffmpeg，濾波與變速比自己寫可靠
    for name, af in {
        "telephone_8k": "highpass=f=300,lowpass=f=3400,aresample=8000,aresample=16000",
        "fast_1.4x": "atempo=1.4",
    }.items():
        dest = CASES / f"degrade_{name}" / f"{Path(case['path']).name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        _ffmpeg("-i", case["path"], "-af", af, "-ar", str(SR), str(dest))
        out.append(dict(case, id=f"{name}/{case['id']}", group=f"degrade_{name}_{lang}", path=str(dest)))
    return out


def _edge_cases(sample):
    out_dir = CASES / "edge"
    out_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    cases = []

    def add(name, path, expect, ref=None, lang=None):
        cases.append(
            {
                "id": f"edge/{name}",
                "group": "edge",
                "path": str(path),
                "lang": lang or sample["lang"],
                "ref": ref,
                "expect": expect,
            }
        )

    for name, signal in {
        "silence_2s": np.zeros(2 * SR),
        "silence_30s": np.zeros(30 * SR),
        "white_noise_30s": rng.standard_normal(30 * SR) * 0.1,
        "tone_440hz_10s": 0.3 * np.sin(2 * np.pi * 440 * np.arange(10 * SR) / SR),
    }.items():
        sf.write(out_dir / f"{name}.wav", signal.astype(np.float32), SR, subtype="PCM_16")
        add(name, out_dir / f"{name}.wav", "no_speech")

    audio, _ = sf.read(sample["path"])
    audio = _peak_normalize(audio)
    normalized = out_dir / "source_normalized.wav"
    sf.write(normalized, audio.astype(np.float32), SR, subtype="PCM_16")
    sf.write(out_dir / "speech_0.4s.wav", audio[: int(0.4 * SR)].astype(np.float32), SR, subtype="PCM_16")
    add("speech_0.4s", out_dir / "speech_0.4s.wav", "transcript")
    # 同一句換成各種容器與取樣率，結果應該一樣
    for name, args in {
        "fmt_mp3": ["-c:a", "libmp3lame", "-b:a", "64k"],
        "fmt_m4a": ["-c:a", "aac", "-b:a", "64k"],
        "fmt_flac": [],
        "fmt_ogg": ["-c:a", "libvorbis"],
        "fmt_stereo_44k": ["-ac", "2", "-ar", "44100"],
        "fmt_8bit_8k": ["-ar", "8000", "-c:a", "pcm_u8"],
    }.items():
        ext = {"fmt_mp3": "mp3", "fmt_m4a": "m4a", "fmt_flac": "flac", "fmt_ogg": "ogg"}.get(name, "wav")
        dest = out_dir / f"{name}.{ext}"
        _ffmpeg("-i", str(normalized), *args, str(dest))
        add(name, dest, "transcript", ref=sample["ref"])

    (out_dir / "corrupt.wav").write_bytes(b"RIFF" + rng.bytes(4096))
    add("corrupt_bytes", out_dir / "corrupt.wav", "reject")
    (out_dir / "empty.wav").write_bytes(b"")
    add("empty_file", out_dir / "empty.wav", "reject")

    # 長度上限：模型卡說 1 小時，伺服器設 VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
    long_src = [TESTDATA / "en_npr_34min.wav", TESTDATA / "en_ted_full12min.wav"]
    _ffmpeg(
        *sum([["-i", str(p)] for p in long_src], []),
        "-filter_complex",
        "concat=n=2:v=0:a=1",
        "-ar",
        str(SR),
        "-ac",
        "1",
        str(out_dir / "long_46min.wav"),
    )
    add("long_46min", out_dir / "long_46min.wav", "transcript", lang="en_us")
    _ffmpeg(
        "-stream_loop",
        "1",
        "-i",
        str(TESTDATA / "en_npr_34min.wav"),
        "-t",
        "3720",
        "-ar",
        str(SR),
        "-ac",
        "1",
        str(out_dir / "long_62min.wav"),
    )
    add("long_62min", out_dir / "long_62min.wav", "reject", lang="en_us")
    return cases


def _peak_normalize(audio, peak=0.9):
    return audio * (peak / max(1e-9, np.max(np.abs(audio))))


def _mix(speech, noise, snr_db):
    p_speech = np.mean(speech**2)
    p_noise = np.mean(noise**2)
    mixed = speech + noise * np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    return mixed / max(1.0, np.max(np.abs(mixed)) / 0.95)


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


if __name__ == "__main__":
    main()
