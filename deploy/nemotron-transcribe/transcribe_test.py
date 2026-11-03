#!/usr/bin/env python3
"""Send audio files to the Nemotron Transcribe server and print transcripts with timing.

Standard library only, so it runs on the host without the serving image:
    python3 transcribe_test.py testdata/en_ted.wav
    python3 transcribe_test.py --speakers --timestamps testdata/en_npr_2spk.wav
"""
import argparse
import base64
import json
import time
import urllib.request
import wave
from pathlib import Path

DEFAULT_PROMPT = "Transcribe the following audio."
# 模型卡的多講者寫法：要求每段換人時標 <spk:N>
SPEAKER_PROMPT = (
    "Transcribe the following audio. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8804")
    parser.add_argument("--model", default="nemotron-transcribe")
    parser.add_argument("--prompt", help=f"default: {DEFAULT_PROMPT!r}")
    parser.add_argument("--speakers", action="store_true", help="ask for <spk:N> speaker tags")
    parser.add_argument("--timestamps", action="store_true", help="request CTC word timestamps")
    parser.add_argument("--out", type=Path, help="directory to save each raw JSON response")
    args = parser.parse_args()

    prompt = args.prompt or (SPEAKER_PROMPT if args.speakers else DEFAULT_PROMPT)
    for path in args.audio:
        duration = _wav_duration(path)
        start = time.perf_counter()
        response = _transcribe(args, path, prompt)
        elapsed = time.perf_counter() - start

        text = response["choices"][0]["message"]["content"]
        print(f"=== {path.name}  audio {duration:.1f}s  took {elapsed:.1f}s  RTFx {duration / elapsed:.1f}")
        print(text.strip())
        if args.timestamps:
            _print_timestamps(response.get("ctc_timestamps"))
        print()
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{path.stem}.json").write_text(json.dumps(response, ensure_ascii=False, indent=2))


def _transcribe(args, path, prompt):
    audio = base64.b64encode(path.read_bytes()).decode()
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
                ],
            }
        ],
        "temperature": 0,
        # 聽不懂的語言會陷入重複迴圈直到 context 用完；照時長給上限，
        # 模型卡說一小時密集對話約兩萬 token
        "max_tokens": int(_wav_duration(path) * 8) + 256,
        "chat_template_kwargs": {"enable_thinking": False},
        # 時間戳要靠特殊 token 對齊，模型卡要求保留
        "skip_special_tokens": False,
    }
    if args.timestamps:
        body["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
    request = urllib.request.Request(
        f"{args.url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3600) as resp:
        return json.load(resp)


def _print_timestamps(alignment):
    if not alignment:
        print("[no ctc_timestamps in response]")
        return
    if alignment.get("error") is not None:
        print(f"[timestamp error] {alignment['error']}")
        return
    words = alignment["words"]
    print(f"[{len(words)} words with timestamps; first 15]")
    for word in words[:15]:
        print(f"  {word['start']:7.2f}-{word['end']:7.2f}  spk={word.get('speaker')}  {word['word']}")


def _wav_duration(path):
    with wave.open(str(path)) as wav:
        return wav.getnframes() / wav.getframerate()


if __name__ == "__main__":
    main()
