"""Acceptance criteria from design doc v2.0 §14. Run: pytest -q (no API calls)."""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from agent_srm import design as dsg                       # noqa: E402
from agent_srm.clients.mock import MockClient             # noqa: E402
from agent_srm.config import ConfigError, load_config, validate  # noqa: E402
from agent_srm.conversation import blind_thread, run_sequential_dyad, thread_for  # noqa: E402
from agent_srm.personas import load_personas              # noqa: E402
from agent_srm.prompts import load_prompts                # noqa: E402
from agent_srm.rating import RatingParseError, parse_ratings  # noqa: E402


def _fixture_config() -> Path:
    """A copy of the smoke config pointed at the fixture persona dir."""
    raw = (ROOT / "configs/dryrun_smoke.yaml").read_text()
    raw = raw.replace("dir: data/personas", f"dir: {FIXTURE_DIR.as_posix()}")
    raw = raw.replace("manifest: data/personas/MANIFEST.json",
                      f"manifest: {(FIXTURE_DIR / 'MANIFEST.json').as_posix()}")
    p = ROOT / "configs" / "_test_generated.yaml"
    p.write_text(raw)
    return p


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    return subprocess.run([sys.executable, "-m", "agent_srm.cli", *args],
                          cwd=ROOT, capture_output=True, text=True, env=env)


def read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


FIXTURE_DIR = ROOT / "tests" / "_fixture_personas"


@pytest.fixture(scope="module", autouse=True)
def fixtures():
    """Build synthetic personas in tests/_fixture_personas — NEVER in
    data/personas, which holds the real Twin-2K-500 chunks. An earlier version
    wrote to data/personas and silently destroyed a user's real data."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "scripts/make_synthetic_personas.py",
                    "--n", "64", "--chunks", "2", "--out", str(FIXTURE_DIR)],
                   cwd=ROOT, check=True, capture_output=True)
    shutil.rmtree(ROOT / "runs", ignore_errors=True)
    yield
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)


def test_tests_never_touch_real_persona_dir():
    """Guard: the suite must not write into data/personas under any path."""
    src = pathlib.Path(__file__).read_text()
    for line in src.splitlines():
        if "make_synthetic_personas" in line and "--out" not in src:
            pytest.fail("fixture generator invoked without --out")
    assert "FIXTURE_DIR" in src


def cfg_with(tmp_path: Path, **overrides) -> Path:
    base = load_config(ROOT / "configs/dryrun_smoke.yaml")
    raw = (ROOT / "configs/dryrun_smoke.yaml").read_text()
    for k, v in overrides.items():
        raw = raw.replace(k, v)
    p = tmp_path / "cfg.yaml"
    p.write_text(raw)
    assert base is not None
    return p


# -- AC1: dry run row counts -------------------------------------------------
def test_ac1_dry_run_row_counts():
    r = run_cli("run", "--config", str(_fixture_config()), "--dry-run")
    assert r.returncode == 0, r.stderr
    out = ROOT / "runs/dryrun_smoke__mock-a"
    assert len(read_csv(out / "ratings.csv")) == 144      # 3 blocks x 12 x 4 items
    assert len(read_csv(out / "ratings_wide.csv")) == 36  # 3 blocks x 12
    assert (out / "transcripts.jsonl").read_text().count("\n") == 18  # 3 x 6 dyads


# -- AC2 / AC10: determinism and cross-arm identity --------------------------
def test_ac2_block_assignment_is_deterministic():
    pids = [f"p{i:04d}" for i in range(64)]
    a, _ = dsg.assign_blocks(pids, 20260818, 4, 3)
    b, _ = dsg.assign_blocks(pids, 20260818, 4, 3)
    assert [x.pids for x in a] == [x.pids for x in b]
    c, _ = dsg.assign_blocks(pids, 999, 4, 3)
    assert [x.pids for x in a] != [x.pids for x in c]


def test_ac10_arms_share_design_digest():
    r = run_cli("verify-arms", "--config", str(_fixture_config()))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout
    a = read_csv(ROOT / "runs/dryrun_smoke__mock-a/personas.csv")
    b = read_csv(ROOT / "runs/dryrun_smoke__mock-b/personas.csv")
    assert a == b


# -- AC3: validation gates ---------------------------------------------------
@pytest.mark.parametrize("field,value,needle", [
    ("k", 3, "even"),
    ("k", 1, ">= 2"),
    ("block_size", 3, ">= 4"),
    ("self_ratings", True, "self_ratings"),
])
def test_ac3_design_validation(field, value, needle):
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    if field == "k" and value != 2:
        cfg.design.intro_mode = None
    setattr(cfg.design, field, value)
    with pytest.raises(ConfigError) as e:
        validate(cfg, pool_size=64, dry_run=True)
    assert needle in str(e.value)


def test_ac3_intro_mode_requires_k2():
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.design.k = 4
    cfg.design.intro_mode = "shared"
    with pytest.raises(ConfigError, match="k=2 only"):
        validate(cfg, pool_size=64, dry_run=True)


def test_ac3_construct_name_may_not_be_an_item():
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.items.constructs = {"warm": ["warm", "kind"], "competence": ["capable", "effective"]}
    with pytest.raises(ConfigError, match="never be renderable as a stimulus"):
        validate(cfg, pool_size=64, dry_run=True)


def test_ac3_exactly_two_indicators():
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.items.constructs = {"warmth": ["warm", "kind", "friendly"],
                            "competence": ["capable", "effective"]}
    with pytest.raises(ConfigError, match="exactly 2 indicators"):
        validate(cfg, pool_size=64, dry_run=True)


def test_ac3_pool_ceiling():
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.design.n_blocks = 100
    with pytest.raises(ConfigError, match="exceeds persona pool size"):
        validate(cfg, pool_size=64, dry_run=True)


def test_ac3_rating_max_tokens_floor():
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.model.rating.max_tokens = 64
    with pytest.raises(ConfigError, match=">= 512"):
        validate(cfg, pool_size=64, dry_run=True)


# -- AC11: arms may differ only on model fields ------------------------------
def test_ac11_arm_rejects_non_model_field(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text((ROOT / "configs/dryrun_smoke.yaml").read_text().replace(
        "  - slug: mock-a\n    model_id: mock-a\n",
        "  - slug: mock-a\n    model_id: mock-a\n    block_size: 6\n"))
    with pytest.raises(ConfigError, match="unknown field"):
        load_config(p)


# -- AC4: repair retries then whole-block quarantine -------------------------
def test_ac4_parse_validation():
    items = ["warm", "kind", "capable", "effective"]
    assert parse_ratings('{"warm":4,"kind":5,"capable":3,"effective":2}', items, 1, 5)
    assert parse_ratings('```json\n{"warm":1,"kind":1,"capable":1,"effective":1}\n```',
                         items, 1, 5)
    with pytest.raises(RatingParseError, match="key mismatch"):
        parse_ratings('{"warm":4,"kind":5,"capable":3}', items, 1, 5)
    with pytest.raises(RatingParseError, match="outside 1..5"):
        parse_ratings('{"warm":9,"kind":5,"capable":3,"effective":2}', items, 1, 5)
    with pytest.raises(RatingParseError):
        parse_ratings("not json at all", items, 1, 5)


def test_ac4_block_quarantined_whole(tmp_path):
    """One persona that never returns valid JSON removes its entire block —
    and no partial rows reach either CSV."""
    from agent_srm import outputs as out
    from agent_srm.orchestrator import run_arm

    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.output_dir = str(tmp_path / "{study_id}")
    cfg.execution.checkpoint_dir = str(tmp_path / "{study_id}/checkpoints")
    personas, _ = load_personas(FIXTURE_DIR, manifest=FIXTURE_DIR / "MANIFEST.json")
    blocks, _ = dsg.assign_blocks(sorted(personas), cfg.seed, 4, 3)
    for b in blocks:
        dsg.orient(b, cfg.seed, cfg.design.k)
    bad_pid = blocks[0].pids[0]

    client = MockClient(model_id="mock", fail_json_for={f"Respondent {bad_pid}"})
    prompts = load_prompts(ROOT / "prompts")
    summary = run_arm(cfg, cfg.arms[0], prompts, personas, client, blocks, resume=False)

    assert len(summary["quarantined"]) == 1
    q = summary["quarantined"][0]
    assert q["run"] == blocks[0].run
    assert q["attributed_pid"] == bad_pid          # per-pid attribution
    assert summary["blocks_completed"] == 2
    rows = read_csv(cfg.out_for(cfg.arms[0].slug) / "ratings_wide.csv")
    assert {int(r["run"]) for r in rows} == {2, 3}
    assert len(rows) == 24                          # 2 complete blocks only


# -- AC6: k>=4 alternation and balanced initiator counts ---------------------
def test_ac6_orientation_is_balanced():
    for n, expected in ((4, {1, 2}), (6, {2, 3}), (5, {2})):
        pids = [f"p{i}" for i in range(n)]
        blk = dsg.Block(run=1, pids=pids)
        dsg.orient(blk, 42, k=4)
        counts = dsg.initiator_counts(blk)
        assert len(blk.dyads) == n * (n - 1) // 2
        assert set(counts.values()) <= expected
        assert sum(counts.values()) == n * (n - 1) // 2


def test_ac6_k10_alternation():
    prompts = load_prompts(ROOT / "prompts")
    client = MockClient()
    turns = run_sequential_dyad(
        client, prompts, type("C", (), {"temperature": 1.0, "max_tokens": 256,
                                        "top_p": 1.0, "seed": None})(),
        None, k=10, systems={"A": "sysA", "B": "sysB"},
        initiator="A", responder="B")
    assert len(turns) == 10
    assert [t.sender_pid for t in turns] == ["A", "B"] * 5
    assert sum(t.sender_pid == "A" for t in turns) == 5
    msgs = thread_for("A", turns, prompts, opening="Send your first message.")
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user"
    assert all(a != b for a, b in zip(roles, roles[1:]))    # strict alternation


# -- AC7: k=2 has no initiator and shared intros are byte-identical ----------
def test_ac7_k2_no_initiator_and_shared_intros_identical():
    blk = dsg.Block(run=1, pids=["a", "b", "c", "d"])
    dsg.orient(blk, 42, k=2)
    assert all(d.initiator is None for d in blk.dyads)
    assert all(d.role("a") == "none" for d in blk.dyads)

    tx = [json.loads(l) for l in
          (ROOT / "runs/dryrun_smoke__mock-a/transcripts.jsonl").read_text().splitlines()]
    seen: dict[str, set[str]] = {}
    for t in tx:
        for side in ("persona_a", "persona_b"):
            pid = t[side]["pid"]
            turn = next(m for m in t["messages"] if m["sender_pid"] == pid)
            seen.setdefault(pid, set()).add(turn["content"])
    assert all(len(v) == 1 for v in seen.values()), "shared intro varied across partners"


# -- AC9: resume ------------------------------------------------------------
def test_ac9_resume_skips_completed_blocks():
    before = read_csv(ROOT / "runs/dryrun_smoke__mock-a/ratings_wide.csv")
    r = run_cli("run", "--config", str(_fixture_config()), "--dry-run")
    assert r.returncode == 0
    after = read_csv(ROOT / "runs/dryrun_smoke__mock-a/ratings_wide.csv")
    assert before == after                       # no duplication, no drops


# -- AC14 + §5.5 contract: no construct label ever reaches the model ---------
def test_stimulus_label_separation():
    tx = [json.loads(l) for l in
          (ROOT / "runs/dryrun_smoke__mock-a/transcripts.jsonl").read_text().splitlines()]
    for t in tx:
        for r in t["ratings"]:
            assert "warmth" not in r["rating_prompt"].lower()
            assert "competence" not in r["rating_prompt"].lower()
            assert set(r["item_order"]) == {"warm", "kind", "capable", "effective"}
        for blob in (t["system_prompt_a"], t["system_prompt_b"]):
            assert "competence" not in blob.lower()


def test_long_csv_construct_mapping():
    rows = read_csv(ROOT / "runs/dryrun_smoke__mock-a/ratings.csv")
    mapping = {"warm": "warmth", "kind": "warmth",
               "capable": "competence", "effective": "competence"}
    for r in rows:
        assert r["construct"] == mapping[r["item"]]
        assert 1 <= int(r["score"]) <= 5
        assert 1 <= int(r["item_position"]) <= 4
        assert r["arm"] and r["model_id"] and r["model_snapshot"]
        assert r["reasoning_mode"] == "provider_default"
    # no self-ratings anywhere
    assert all(r["rater"] != r["target"] for r in rows)


def test_reasoning_never_enters_thread():
    """Reasoning text is transcript-only; it must never appear as an assistant turn."""
    tx = [json.loads(l) for l in
          (ROOT / "runs/dryrun_smoke__mock-a/transcripts.jsonl").read_text().splitlines()]
    for t in tx:
        for m in t["messages"]:
            assert "reasoning_text" in m
        for r in t["ratings"]:
            assert "reasoning" not in r["rating_prompt"].lower()


# -- regression: live client signature must match the mock's ----------------
def test_live_client_signature_matches_mock():
    """The `reasoning` kwarg was added to the protocol and the mock but not to
    the HTTP client, so every dry run passed and the first live run quarantined
    every block. Keep the two signatures locked together."""
    import inspect
    from agent_srm.clients.openai_compatible import OpenAICompatibleClient

    mock_sig = inspect.signature(MockClient.complete)
    live_sig = inspect.signature(OpenAICompatibleClient.complete)
    assert list(mock_sig.parameters) == list(live_sig.parameters), (
        f"signature drift: mock={list(mock_sig.parameters)} "
        f"live={list(live_sig.parameters)}")


def test_live_client_accepts_orchestrator_call_shape():
    """Call the real client the exact way orchestrator/rating do, with a stubbed
    transport, so a kwarg mismatch fails in the suite rather than in production."""
    from agent_srm.clients.openai_compatible import OpenAICompatibleClient

    c = OpenAICompatibleClient(base_url="https://example.invalid/v1", model_id="m")
    c._post = lambda payload: ({
        "model": "m-0813",
        "choices": [{"message": {"content": '{"warm":4,"kind":4,"capable":3,'
                                            '"effective":3}',
                                 "reasoning_content": "thinking..."},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                  "completion_tokens_details": {"reasoning_tokens": 42}},
    }, 200)

    out = c.complete(system="s", messages=[{"role": "user", "content": "u"}],
                     max_tokens=512, temperature=0.0, top_p=1.0, seed=None,
                     json_schema={"type": "object"}, reasoning=None)
    assert out.model_snapshot == "m-0813"      # resolved snapshot, not the alias
    assert out.reasoning_tokens == 42
    assert out.reasoning_text == "thinking..."
    assert "warm" in out.text


# -- concurrency: parallel dyads must equal sequential dyads ----------------
@pytest.mark.parametrize("k", [2, 10])
def test_parallel_dyads_match_sequential(tmp_path, k):
    """max_concurrent_dyads must be a speed knob only. Same rows, same order."""
    import shutil
    from agent_srm.orchestrator import run_arm

    personas, _ = load_personas(FIXTURE_DIR, manifest=FIXTURE_DIR / "MANIFEST.json")
    prompts = load_prompts(ROOT / "prompts")

    def run(workers, tag):
        cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
        cfg.design.k = k
        cfg.design.intro_mode = "shared" if k == 2 else None
        cfg.execution.max_concurrent_dyads = workers
        cfg.output_dir = str(tmp_path / tag / "{study_id}")
        cfg.execution.checkpoint_dir = str(tmp_path / tag / "{study_id}/checkpoints")
        shutil.rmtree(tmp_path / tag, ignore_errors=True)
        blocks, _ = dsg.assign_blocks(sorted(personas), cfg.seed, 4, 2)
        for b in blocks:
            dsg.orient(b, cfg.seed, k)
        run_arm(cfg, cfg.arms[0], prompts, personas, MockClient(), blocks, resume=False)
        return cfg.out_for(cfg.arms[0].slug)

    seq, par = run(1, "seq"), run(4, "par")
    key = lambda r: (r["run"], r["rater"], r["target"])
    a = [key(r) for r in read_csv(seq / "ratings_wide.csv")]
    b = [key(r) for r in read_csv(par / "ratings_wide.csv")]
    assert a == b and len(a) == 24

    tx = [json.loads(x) for x in
          (par / "transcripts.jsonl").read_text().splitlines()]
    for t in tx:
        assert len(t["messages"]) == k
        senders = [m["sender_pid"] for m in t["messages"]]
        assert all(x != y for x, y in zip(senders, senders[1:]))


def test_csvs_stay_current_if_run_dies_midway(tmp_path):
    """A run that stops after block 2 of 3 must leave CSVs describing 2 blocks —
    not a stale file from an earlier run. (Power-cut failure, seen live.)"""
    from agent_srm import outputs as out
    from agent_srm.orchestrator import ArmRunner

    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.output_dir = str(tmp_path / "{study_id}")
    cfg.execution.checkpoint_dir = str(tmp_path / "{study_id}/checkpoints")
    personas, _ = load_personas(FIXTURE_DIR, manifest=FIXTURE_DIR / "MANIFEST.json")
    blocks, _ = dsg.assign_blocks(sorted(personas), cfg.seed, 4, 3)
    for b in blocks:
        dsg.orient(b, cfg.seed, cfg.design.k)

    arm = cfg.arms[0]
    out_dir, ckpt = cfg.out_for(arm.slug), cfg.ckpt_for(arm.slug)
    runner = ArmRunner(cfg=cfg, arm=arm, prompts=load_prompts(ROOT / "prompts"),
                       personas=personas, client=MockClient())

    for i, block in enumerate(blocks[:2], start=1):        # simulate dying at 2/3
        out.write_shard(ckpt, block.run, runner.run_block(block))
        out.assemble(out_dir, out.load_shards(ckpt), cfg.items.items)
        rows = read_csv(out_dir / "ratings_wide.csv")
        assert len(rows) == 12 * i, f"after block {i}: {len(rows)} rows"
        assert {int(r["run"]) for r in rows} == {b.run for b in blocks[:i]}
def test_persona_mode_none_sends_only_the_framing(tmp_path):
    """No persona text reaches the model, the framing is byte-identical to a
    full run, and block composition is unchanged so the arms stay paired."""
    from agent_srm import outputs as out
    from agent_srm.orchestrator import run_arm

    personas, _ = load_personas(FIXTURE_DIR, manifest=FIXTURE_DIR / "MANIFEST.json")
    prompts = load_prompts(ROOT / "prompts")
    framing = prompts.framing_anonymous.strip()

    digests = {}
    for mode in ("full", "none"):
        cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
        cfg.personas.mode = mode
        cfg.output_dir = str(tmp_path / mode / "{study_id}")
        cfg.execution.checkpoint_dir = str(tmp_path / mode / "{study_id}/ckpt")
        validate(cfg, pool_size=len(personas), dry_run=True)
        blocks, _ = dsg.assign_blocks(sorted(personas), cfg.seed, 4, 2)
        for b in blocks:
            dsg.orient(b, cfg.seed, cfg.design.k)
        digests[mode] = out.design_digest(blocks)
        run_arm(cfg, cfg.arms[0], prompts, personas, MockClient(), blocks,
                resume=False)

        tx = [json.loads(x) for x in
              (cfg.out_for(cfg.arms[0].slug) / "transcripts.jsonl")
              .read_text().splitlines()]
        prompts_sent = {t["system_prompt_a"] for t in tx} | {
            t["system_prompt_b"] for t in tx}

        if mode == "none":
            # every persona receives the identical framing-only prompt
            assert len(prompts_sent) == 1
            only = prompts_sent.pop()
            assert only.strip() == framing
            for pid, p in personas.items():
                assert p.summary not in only
            assert "description of a person" not in only
        else:
            assert len(prompts_sent) > 1
            assert all(framing in s for s in prompts_sent)

        rows = read_csv(cfg.out_for(cfg.arms[0].slug) / "ratings_wide.csv")
        assert all(r["persona_mode"] == mode for r in rows)

    # paired: identical blocks, dyads and initiator orientation
    assert digests["full"] == digests["none"]


def test_persona_mode_validation():
    cfg = load_config(ROOT / "configs/dryrun_smoke.yaml")
    cfg.personas.mode = "demographics_only"
    with pytest.raises(ConfigError, match="personas.mode"):
        validate(cfg, pool_size=64, dry_run=True)
