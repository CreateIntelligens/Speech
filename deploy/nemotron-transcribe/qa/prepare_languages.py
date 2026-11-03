"""Extract 40 FLEURS test lines for every language the model card lists, plus Cantonese and Taiwan Mandarin.

    uv run --no-project --with numpy --with soundfile python qa/prepare_languages.py

Writes qa/data/cases_languages/ and qa/data/manifest_languages.jsonl. Download each language's test.tsv and
audio/test.tar.gz to qa/data/fleurs_raw/ first. Taiwan Mandarin is 40 lines from Common Voice 17 (qa/data/cv_zh_tw/test.tsv
and test.tar, from the fsicoli/common_voice_17_0 mirror), one per speaker. Uses its own directory so a comparison reading
qa/data/cases/ keeps running, and refuses to replace an existing one.
"""

import csv
import json
import random
import subprocess
import tarfile
from pathlib import Path

import prepare
from prepare import QA, RAW, SR, _extract_fleurs

CASES = QA / "data" / "cases_languages"

# (FLEURS code, name, model card tier)
LANGUAGES = [
    ("cs_cz", "Czech", "Leading"),
    ("nl_nl", "Dutch", "Leading"),
    ("en_us", "English", "Leading"),
    ("fr_fr", "French", "Leading"),
    ("de_de", "German", "Leading"),
    ("hi_in", "Hindi", "Leading"),
    ("it_it", "Italian", "Leading"),
    ("ko_kr", "Korean", "Leading"),
    ("pl_pl", "Polish", "Leading"),
    ("pt_br", "Portuguese", "Leading"),
    ("ru_ru", "Russian", "Leading"),
    ("sk_sk", "Slovak", "Leading"),
    ("es_419", "Spanish", "Leading"),
    ("bg_bg", "Bulgarian", "Production-ready"),
    ("hr_hr", "Croatian", "Production-ready"),
    ("da_dk", "Danish", "Production-ready"),
    ("et_ee", "Estonian", "Production-ready"),
    ("fi_fi", "Finnish", "Production-ready"),
    ("hu_hu", "Hungarian", "Production-ready"),
    ("ja_jp", "Japanese", "Production-ready"),
    ("lv_lv", "Latvian", "Production-ready"),
    ("lt_lt", "Lithuanian", "Production-ready"),
    ("ro_ro", "Romanian", "Production-ready"),
    ("sl_si", "Slovenian", "Production-ready"),
    ("sv_se", "Swedish", "Production-ready"),
    ("th_th", "Thai", "Production-ready"),
    ("tr_tr", "Turkish", "Production-ready"),
    ("uk_ua", "Ukrainian", "Production-ready"),
    ("ar_eg", "Arabic", "Supported"),
    ("bn_in", "Bengali", "Supported"),
    ("cmn_hans_cn", "Mandarin", "Supported"),
    ("el_gr", "Greek", "Supported"),
    ("he_il", "Hebrew", "Supported"),
    ("id_id", "Indonesian", "Supported"),
    ("mt_mt", "Maltese", "Supported"),
    ("mr_in", "Marathi", "Supported"),
    ("nb_no", "Norwegian", "Supported"),
    ("fil_ph", "Tagalog", "Supported"),
    ("ta_in", "Tamil", "Supported"),
    ("te_in", "Telugu", "Supported"),
    ("vi_vn", "Vietnamese", "Supported"),
    ("yue_hant_hk", "Cantonese", "Not listed"),
]


def main():
    missing = [code for code, _, _ in LANGUAGES if not (RAW / f"{code}.test.tar.gz").exists()]
    if missing:
        raise SystemExit(f"download these first: {missing}")
    CASES.mkdir(parents=True, exist_ok=False)
    prepare.CASES = CASES
    rows = []
    for code, name, tier in LANGUAGES:
        for case in _extract_fleurs(code):
            rows.append(dict(case, group=f"fleurs_{code}", name=name, tier=tier))
    rows += _common_voice_taiwan()
    with open(QA / "data" / "manifest_languages.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} lines, {len(LANGUAGES) + 1} groups")


def _common_voice_taiwan(lines=40, seed=42):
    source = QA / "data" / "cv_zh_tw"
    out = CASES / "cv_zh_tw"
    out.mkdir(parents=True)
    rows = list(csv.DictReader(open(source / "test.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE))
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked = {}
    for row in rows:
        if row["sentence"].strip():
            picked.setdefault(row["client_id"], row)
    picked = dict(list(picked.items())[:lines])
    wanted = {row["path"]: row for row in picked.values()}
    cases = []
    with tarfile.open(source / "test.tar") as tar:
        for member in tar:
            row = wanted.get(Path(member.name).name)
            if row is None or not member.isfile():
                continue
            mp3 = out / Path(member.name).name
            mp3.write_bytes(tar.extractfile(member).read())
            wav = mp3.with_suffix(".wav")
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ac", "1", "-ar", str(SR), str(wav)],
                check=True,
            )
            mp3.unlink()
            cases.append(
                {
                    "id": f"cv_zh_tw/{wav.stem}",
                    "path": str(wav),
                    "lang": "cmn_hans_cn",
                    "ref": row["sentence"],
                    "expect": "transcript",
                    "group": "cv_zh_tw",
                    "name": "Mandarin (Taiwan, Common Voice)",
                    "tier": "Supported",
                }
            )
    return sorted(cases, key=lambda c: c["id"])


if __name__ == "__main__":
    main()
