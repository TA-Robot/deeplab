#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone


def safe_write_session(session_file: str, tid: str) -> None:
    os.makedirs(os.path.dirname(session_file), exist_ok=True)
    tmp = session_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(tid)
    os.chmod(tmp, 0o600)
    os.replace(tmp, session_file)


def main() -> int:
    if len(sys.argv) < 5:
        print(
            "usage: codex-second-agent-filter.py <session_file> <raw_json:0|1> <prompt_file> <transcript_log>",
            file=sys.stderr,
        )
        return 2

    session_file = sys.argv[1]
    raw_json = sys.argv[2] == "1"
    prompt_file = sys.argv[3]
    transcript_log = sys.argv[4]
    agent_name = sys.argv[5] if len(sys.argv) >= 6 else None
    effective_cd = sys.argv[6] if len(sys.argv) >= 7 else None

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    except Exception:
        prompt = ""

    thread_id = None
    agent_texts: list[str] = []

    for line in sys.stdin:
        line = line.rstrip("\n")
        if raw_json:
            print(line, flush=True)

        try:
            obj = json.loads(line)
        except Exception:
            continue

        t = obj.get("type")
        if t == "thread.started":
            tid = obj.get("thread_id")
            if isinstance(tid, str) and tid:
                thread_id = tid
                try:
                    safe_write_session(session_file, tid)
                except Exception:
                    pass

        if raw_json:
            continue

        if t == "item.completed":
            item = obj.get("item") or {}
            if item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    agent_texts.append(text)
                    print(text, flush=True)

    # raw_json=0 で agent_message が無いケースでもエラーにはしない
    try:
        os.makedirs(os.path.dirname(transcript_log), exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "thread_id": thread_id,
            "agent": agent_name,
            "cd": effective_cd,
            "prompt": prompt,
            "response": "\n\n".join(agent_texts).strip(),
        }
        with open(transcript_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


