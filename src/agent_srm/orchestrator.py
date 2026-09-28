"""Run loop: blocks -> dyads -> turns -> ratings, with the completeness gate.

A block reaches disk only when every directed rating parsed. Anything else is
quarantined whole, with per-pid attribution (§9.3).
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import conversation as conv
from . import design as dsg
from . import outputs as out
from .rating import RatingParseError, rate

log = logging.getLogger("agent_srm")


@dataclass
class ArmRunner:
    cfg: object
    arm: object
    prompts: object
    personas: dict
    client: object
    names: dict | None = None

    # -- helpers ---------------------------------------------------------
    def _system(self, pid: str, other_pid: str | None = None) -> str:
        # mode "none": no persona text reaches the model. The parquet is still
        # loaded, so blocks, dyads and personas.csv are unchanged — this is a
        # paired null arm, not a different study.
        summary = (None if self.cfg.personas.mode == "none"
                   else self.personas[pid].summary)
        if not self.cfg.naming.enabled or self.names is None:
            return self.prompts.system_for(summary)
        me = self.names[pid]
        them = self.names[other_pid] if other_pid else {"first": ""}
        return self.prompts.system_for(
            summary, self_first=me["first"], self_last=me["last"],
            other_first=them["first"])

    def _provenance(self, snapshot: str) -> dict:
        return {
            "provider": self.arm.provider,
            "model_id": self.arm.model_id,
            "model_snapshot": snapshot or self.arm.model_id,
            "reasoning_mode": self.arm.reasoning_mode,
            "persona_mode": self.cfg.personas.mode,
            "temperature": self.cfg.model.rating.temperature,
            "seed": self.cfg.seed,
        }

    # -- intros ----------------------------------------------------------
    def _intros(self, block) -> dict:
        """k=2. shared -> one intro per persona, reused verbatim for all partners.
        unique -> a fresh intro per partner, still blind to who the partner is."""
        cc, rz = self.cfg.model.conversation, self.arm.reasoning
        if self.cfg.design.intro_mode == "shared":
            store = {}
            for pid in block.pids:
                text, c = conv.generate_intro(
                    self.client, self._system(pid), self.prompts, cc, rz)
                store[pid] = (text, c)
            return {"mode": "shared", "by_pid": store}
        store = {}
        for d in block.dyads:
            for pid, other in ((d.a, d.b), (d.b, d.a)):
                text, c = conv.generate_intro(
                    self.client, self._system(pid, other), self.prompts, cc, rz)
                store[(d.dyad_id, pid)] = (text, c)
        return {"mode": "unique", "by_dyad": store}

    def _intro_for(self, intros, dyad, pid):
        if intros["mode"] == "shared":
            return intros["by_pid"][pid]
        return intros["by_dyad"][(dyad.dyad_id, pid)]

    # -- one dyad --------------------------------------------------------
    def _process_dyad(self, block, d, intros) -> dict:
        """Everything for one dyad. Independent of every other dyad by design —
        each persona is a fresh instance here and carries no memory across
        dyads — which is what makes the parallel map in run_block safe."""
        cfg, items = self.cfg, self.cfg.items
        long_rows, wide_rows = [], []
        if True:
            turns, tx_msgs = [], []
            if cfg.design.k == 2:
                for idx, pid in enumerate((d.a, d.b), start=1):
                    text, c = self._intro_for(intros, d, pid)
                    turns.append(conv.Turn(
                        turn=idx, sender_pid=pid, content=text,
                        reasoning_text=c.reasoning_text,
                        reasoning_tokens=c.reasoning_tokens,
                        input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                        latency_ms=c.latency_ms, stop_reason=c.stop_reason,
                        flags=conv.flag_message(text), timestamp=out.now()))
            else:
                turns = conv.run_sequential_dyad(
                    self.client, self.prompts, cfg.model.conversation,
                    self.arm.reasoning, k=cfg.design.k,
                    systems={d.a: self._system(d.a, d.b), d.b: self._system(d.b, d.a)},
                    initiator=d.initiator, responder=d.other(d.initiator))
            tx_msgs = [t.record() for t in turns]

            ratings = []
            for rater, target in ((d.a, d.b), (d.b, d.a)):
                order = dsg.item_order(items.items, cfg.seed, block.run, rater, target,
                                       items.randomize_item_order)
                rendered = self.prompts.rating_prompt(
                    order, items.scale_min, items.scale_max)
                if cfg.design.k == 2:
                    own = self._intro_for(intros, d, rater)[0]
                    partner = self._intro_for(intros, d, target)[0]
                    msgs = conv.blind_thread(self.prompts, own_intro=own,
                                             partner_intro=partner,
                                             rating_prompt=rendered)
                else:
                    msgs = conv.rating_thread(rater, turns, self.prompts, rendered)

                res = rate(
                    self.client, self.prompts, items, cfg.model.rating,
                    self.arm.reasoning,
                    system=self._system(rater, target), messages=msgs, order=order,
                    rendered_prompt=rendered, rater=rater, target=target,
                    max_retries=cfg.model.max_json_retries)
                ratings.append(res)

                prov = self._provenance(res.model_snapshot)
                base = {
                    "study_id": cfg.study_id, "arm": self.arm.slug, "run": block.run,
                    "rater": rater, "target": target,
                    "k": cfg.design.k, "intro_mode": cfg.design.intro_mode,
                    "rater_role": d.role(rater),
                    "rater_name": (self.names or {}).get(rater, {}).get("first"),
                    "target_name": (self.names or {}).get(target, {}).get("first"),
                    **prov, "timestamp": res.timestamp,
                }
                for pos, item in enumerate(order, start=1):
                    long_rows.append({**base, "construct": items.item_to_construct[item],
                                      "item": item, "score": res.parsed[item],
                                      "item_position": pos})
                wide_rows.append({**base, **res.parsed,
                                  "reasoning_tokens": res.reasoning_tokens})

            transcript = {
                "study_id": cfg.study_id, "arm": self.arm.slug, "run": block.run,
                "dyad_id": d.dyad_id, "k": cfg.design.k,
                "intro_mode": cfg.design.intro_mode,
                "persona_a": {"pid": d.a, "name": (self.names or {}).get(d.a, {}).get("first"),
                              "role": d.role(d.a)},
                "persona_b": {"pid": d.b, "name": (self.names or {}).get(d.b, {}).get("first"),
                              "role": d.role(d.b)},
                "system_prompt_a": self._system(d.a, d.b),
                "system_prompt_b": self._system(d.b, d.a),
                "messages": tx_msgs,
                "ratings": [r.record() for r in ratings],
                "model": {"arm": self.arm.slug, "model_id": self.arm.model_id,
                          "model_snapshot": ratings[0].model_snapshot,
                          "provider": self.arm.provider,
                          "reasoning_mode": self.arm.reasoning_mode,
                          "conversation_temperature": cfg.model.conversation.temperature,
                          "rating_temperature": cfg.model.rating.temperature},
            }
        return {"long_rows": long_rows, "wide_rows": wide_rows,
                "transcript": transcript}

    # -- one block -------------------------------------------------------
    def run_block(self, block) -> dict:
        cfg = self.cfg
        intros = self._intros(block) if cfg.design.k == 2 else None
        workers = max(1, int(cfg.execution.max_concurrent_dyads))

        if workers == 1:
            results = [self._process_dyad(block, d, intros) for d in block.dyads]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # executor.map preserves input order and re-raises the first
                # exception on iteration, so a failed dyad still quarantines
                # the whole block.
                results = list(pool.map(
                    lambda d: self._process_dyad(block, d, intros), block.dyads))

        long_rows, wide_rows, transcripts = [], [], []
        for r in results:
            long_rows.extend(r["long_rows"])
            wide_rows.extend(r["wide_rows"])
            transcripts.append(r["transcript"])

        n = len(block.pids)
        expected = n * (n - 1)
        got = len(wide_rows)
        if got != expected:
            raise RatingParseError(
                f"block {block.run} incomplete: {got} directed ratings, expected {expected}")

        return {"run": block.run, "pids": block.pids, "long_rows": long_rows,
                "wide_rows": wide_rows, "transcripts": transcripts}


def run_arm(cfg, arm, prompts, personas, client, blocks, names=None,
            resume: bool = True) -> dict:
    """Sequential across blocks. Returns a summary for the manifest."""
    out_dir, ckpt = cfg.out_for(arm.slug), cfg.ckpt_for(arm.slug)
    done = out.completed_runs(ckpt) if resume else set()
    runner = ArmRunner(cfg=cfg, arm=arm, prompts=prompts, personas=personas,
                       client=client, names=names)

    quarantined, flag_counts = [], Counter()
    todo = [b for b in blocks if b.run not in done]
    finished, t_start = 0, time.time()

    for block in blocks:
        if block.run in done:
            log.info("arm=%s run=%d skipped (checkpoint)", arm.slug, block.run)
            continue
        try:
            shard = runner.run_block(block)
        except Exception as e:                      # noqa: BLE001 - gate is deliberate
            pid = _attribute(e, block)
            log.error("arm=%s run=%d QUARANTINED (%s): %s",
                      arm.slug, block.run, pid or "unattributed", e)
            quarantined.append({"run": block.run, "pids": block.pids,
                                "failure_class": type(e).__name__,
                                "attributed_pid": pid, "reason": str(e)})
            out.write_quarantine(out_dir, block.run,
                                 {"run": block.run, "pids": block.pids,
                                  "failure_class": type(e).__name__,
                                  "attributed_pid": pid, "reason": str(e)})
            continue
        for tx in shard["transcripts"]:
            for m in tx["messages"]:
                flag_counts.update(m["flags"])
        out.write_shard(ckpt, block.run, shard)
        finished += 1
        # Rebuild the CSVs after every block, not just at the end. If the run
        # dies mid-way (power cut, dropped connection), the files on disk must
        # describe the blocks actually collected — a stale CSV from an earlier
        # run silently misreports n to the analysis.
        out.assemble(out_dir, out.load_shards(ckpt), cfg.items.items)
        log.info("arm=%s run=%d complete (%d wide rows) | %s",
                 arm.slug, block.run, len(shard["wide_rows"]),
                 _progress(finished, len(todo), time.time() - t_start))

    shards = out.load_shards(ckpt)
    counts = out.assemble(out_dir, shards, cfg.items.items)
    return {"counts": counts, "blocks_completed": len(shards),
            "quarantined": quarantined, "flag_counts": dict(flag_counts)}


def _progress(done_n: int, total: int, elapsed_s: float) -> str:
    """Blocks done, rate, and a remaining estimate — a 50-block k=10 run is a
    multi-hour job and 'is this going to finish tonight' should not require
    arithmetic on log timestamps."""
    per = elapsed_s / max(done_n, 1)
    left = per * max(total - done_n, 0)
    return (f"{done_n}/{total} blocks | {per / 60:.1f} min/block | "
            f"~{int(left // 3600)}h{int((left % 3600) // 60):02d}m remaining")


def _attribute(exc: Exception, block) -> str | None:
    """Pull the offending pid out of a rating failure so quarantine is per-pid."""
    msg = str(exc)
    for pid in block.pids:
        if f"{pid}->" in msg:
            return pid
    return None
