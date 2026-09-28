import glob, json, os, sys

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/prune_truncated.py runs/<folder> [--delete]")

d = sys.argv[1].rstrip("/\\")
delete = "--delete" in sys.argv
ckpt = os.path.join(d, "checkpoints")

bad, total = {}, set()
for line in open(os.path.join(d, "transcripts.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    t = json.loads(line)
    total.add(t["run"])
    for m in t["messages"]:
        if m.get("stop_reason") == "length":
            bad.setdefault(t["run"], []).append(
                (t["dyad_id"], m.get("output_tokens"), m.get("reasoning_tokens")))

print(d)
print("blocks total:", len(total), "  blocks with truncation:", len(bad))
for run in sorted(bad):
    print("  block", run, "-", bad[run])

files = [os.path.join(ckpt, "block_%05d.json" % r) for r in sorted(bad)]
present = [f for f in files if os.path.exists(f)]
print("\ncheckpoint files to remove:", len(present), "of", len(files))

if not delete:
    print("\nDRY RUN. Re-run with --delete to remove them.")
else:
    for f in present:
        os.remove(f)
    print("\nremoved", len(present), "checkpoints. Re-run the config.")
