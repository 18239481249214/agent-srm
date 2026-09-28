import json, sys, os, glob

path = sys.argv[1] if len(sys.argv) > 1 else glob.glob("runs/*/transcripts.jsonl")[0]
tag = os.path.basename(os.path.dirname(path))
outpath = "conversations_" + tag + ".txt"

out = []
for line in open(path, encoding="utf-8"):
    t = json.loads(line)
    out.append("=" * 70)
    out.append("run " + str(t["run"]) + "   dyad " + t["dyad_id"] +
               "   k=" + str(t["k"]))
    out.append("=" * 70)
    for m in t["messages"]:
        out.append("")
        out.append("[" + str(m["sender_pid"]) + "]  " + m["content"])
        if m.get("flags"):
            out.append("      FLAGS: " + ", ".join(m["flags"]))
    out.append("")
    for r in t["ratings"]:
        p = r["parsed"]
        out.append("  rating " + str(r["rater"]) + " -> " + str(r["target"]) +
                   ":  " + ", ".join(k + "=" + str(v) for k, v in p.items()) +
                   "   (retries " + str(r["json_retries"]) + ")")
    out.append("")
    out.append("")

open(outpath, "w", encoding="utf-8").write("\n".join(out))
print("wrote " + outpath + "  (" + str(len(out)) + " lines)")
