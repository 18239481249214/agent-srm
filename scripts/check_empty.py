import collections, glob, json, os, sys

def scan(path):
    n = empty = truncated = max_out = 0
    stops = collections.Counter(); empty_stops = collections.Counter()
    bad = set()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        t = json.loads(line)
        for m in t["messages"]:
            n += 1
            sr = m.get("stop_reason") or "?"
            stops[sr] += 1
            max_out = max(max_out, m.get("output_tokens") or 0)
            if sr == "length":
                truncated += 1
            if not (m.get("content") or "").strip():
                empty += 1; empty_stops[sr] += 1; bad.add(t["dyad_id"])
    return dict(messages=n, empty=empty, stops=stops, empty_stops=empty_stops,
                bad=bad, truncated=truncated, max_out=max_out)

targets = sys.argv[1:] or sorted(os.path.dirname(p) for p in glob.glob("runs/*/transcripts.jsonl"))
print("run".ljust(46), "msgs".rjust(6), "empty".rjust(12), "trunc".rjust(7), "dyads".rjust(7), "maxout".rjust(8))
print("-" * 92)
flagged = []
for d in targets:
    p = os.path.join(d, "transcripts.jsonl")
    if not os.path.exists(p):
        continue
    r = scan(p)
    pct = r["empty"] / max(r["messages"], 1)
    print(os.path.basename(d)[:46].ljust(46),
          str(r["messages"]).rjust(6),
          (str(r["empty"]) + " " + format(pct, ".1%")).rjust(12),
          str(r["truncated"]).rjust(7),
          str(len(r["bad"])).rjust(7),
          str(r["max_out"]).rjust(8))
    if r["empty"]:
        flagged.append((os.path.basename(d), r))

if not flagged:
    print("\nNo empty messages in any run.")
else:
    print("\nDETAIL")
    for name, r in flagged:
        print("\n  " + name)
        print("    stop_reason on empty messages:", dict(r["empty_stops"]))
        print("    affected dyads:", len(r["bad"]))
        if r["empty_stops"].get("length"):
            print("    -> truncation: reasoning consumed max_tokens before any content.")
