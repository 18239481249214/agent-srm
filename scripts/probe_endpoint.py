from __future__ import annotations
import argparse, json, os, sys
import requests

RATING = """The chat has ended. Thinking about the person you just spoke with, please rate
the degree to which this person expresses the following characteristics, where
1 means "not at all" and 5 means "very much".

warm
kind
capable
effective

Reply with only a JSON object mapping each word to a whole number from
1 to 5. No other text."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args()

    key = os.environ.get(a.key_env, "")
    if not key:
        sys.exit("environment variable " + a.key_env + " is not set")

    payload = {
        "model": a.model,
        "messages": [
            {"role": "system", "content": "You are a friendly person in a chat."},
            {"role": "user", "content": "Send your first message to introduce yourself."},
            {"role": "assistant", "content": "Hey! Nice to meet you. Slow week here."},
            {"role": "user", "content": "Hi there, likewise.\n\n" + RATING},
        ],
        "max_tokens": a.max_tokens,
        "temperature": 0.0,
    }

    print("POST " + a.base_url + "/chat/completions  model=" + a.model)
    r = requests.post(a.base_url + "/chat/completions",
                      headers={"Content-Type": "application/json",
                               "Authorization": "Bearer " + key},
                      data=json.dumps(payload), timeout=180)
    print("HTTP " + str(r.status_code))
    if r.status_code >= 400:
        print(r.text[:2000])
        return

    d = r.json()
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    print("finish_reason :", ch.get("finish_reason"))
    print("content       :", repr(msg.get("content"))[:400])
    print("reasoning     :", repr(msg.get("reasoning_content") or msg.get("reasoning"))[:300])
    print("model         :", d.get("model"))
    print("usage         :", json.dumps(d.get("usage"), indent=2))
    print("--- keys in message ---")
    print(sorted(msg.keys()))


main()
