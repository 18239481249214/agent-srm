"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import design as dsg
from . import outputs as out
from .config import ConfigError, load_config, validate
from .clients.mock import MockClient
from .naming import assign_names
from .orchestrator import run_arm
from .personas import WHITELIST, load_personas
from .prompts import load_prompts

log = logging.getLogger("agent_srm")


def _setup_logging(out_dir: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers, force=True,
    )


def _build_design(cfg, personas):
    blocks, dropped = dsg.assign_blocks(
        sorted(personas), cfg.seed, cfg.design.block_size, cfg.design.n_blocks)
    for b in blocks:
        dsg.orient(b, cfg.seed, cfg.design.k)
    return blocks, dropped


def _client_for(cfg, arm, dry_run: bool):
    if dry_run or arm.base_url is None:
        return MockClient(model_id=arm.model_id,
                          scale_min=cfg.items.scale_min, scale_max=cfg.items.scale_max)
    from .clients.openai_compatible import OpenAICompatibleClient
    return OpenAICompatibleClient(
        base_url=arm.base_url, model_id=arm.model_id, api_key_env=arm.api_key_env,
        timeout_s=cfg.execution.request_timeout_s,
        max_retries=cfg.execution.max_retries,
        backoff_base_s=cfg.execution.backoff_base_s,
        json_strategy=arm.json_strategy)


def cmd_validate(args) -> int:
    cfg = load_config(args.config)
    personas, _ = load_personas(
        cfg.personas.dir, cfg.personas.field, cfg.personas.manifest,
        verify=cfg.personas.verify_checksums, naming_enabled=cfg.naming.enabled)
    validate(cfg, pool_size=len(personas), dry_run=True)
    print(f"config OK — {len(personas)} personas, {len(cfg.arms)} arm(s): "
          f"{[a.slug for a in cfg.arms]}")
    return 0


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    personas, pmeta = load_personas(
        cfg.personas.dir, cfg.personas.field, cfg.personas.manifest,
        verify=cfg.personas.verify_checksums, naming_enabled=cfg.naming.enabled)
    validate(cfg, pool_size=len(personas), dry_run=args.dry_run)

    if getattr(args, "workers", None):
        cfg.execution.max_concurrent_dyads = args.workers
    prompts = load_prompts(cfg.prompts_dir)
    blocks, dropped = _build_design(cfg, personas)
    digest = out.design_digest(blocks)
    run_of = {pid: b.run for b in blocks for pid in b.pids}
    names = (assign_names(personas, list(run_of), cfg.seed,
                          cfg.naming.first_names, cfg.naming.surnames,
                          cfg.naming.match_gender) if cfg.naming.enabled else None)

    arms = [cfg.arm(args.arm)] if args.arm else cfg.arms
    for arm in arms:
        out_dir = cfg.out_for(arm.slug)
        _setup_logging(out_dir, args.verbose)
        client = _client_for(cfg, arm, args.dry_run)
        if hasattr(client, "smoke_test") and not args.dry_run:
            client.smoke_test()

        log.info("arm=%s model=%s blocks=%d k=%d intro_mode=%s workers=%d digest=%s",
                 arm.slug, arm.model_id, len(blocks), cfg.design.k,
                 cfg.design.intro_mode, cfg.execution.max_concurrent_dyads,
                 digest[:12])

        summary = run_arm(cfg, arm, prompts, personas, client, blocks,
                          names=names, resume=not args.no_resume)
        n_personas = out.write_personas_csv(out_dir, personas, run_of, names, WHITELIST)

        out.write_manifest(out_dir, {
            "study_id": cfg.study_id_for(arm.slug),
            "base_study_id": cfg.study_id,
            "arm": arm.slug,
            "config": cfg.to_dict(),
            "arm_config": {"slug": arm.slug, "model_id": arm.model_id,
                           "provider": arm.provider, "base_url": arm.base_url,
                           "json_strategy": arm.json_strategy,
                           "reasoning_mode": arm.reasoning_mode},
            "seed": cfg.seed,
            "design_digest": digest,
            "prompt_hashes": prompts.hashes,
            "personas": {"n_pool": len(personas), "n_used": n_personas,
                         "dropped_pids": dropped, "files": pmeta["files"],
                         "manifest": pmeta["manifest"],
                         "exceptions": pmeta["exceptions"][:200]},
            "blocks_completed": summary["blocks_completed"],
            "quarantined": summary["quarantined"],
            "flag_counts": summary["flag_counts"],
            "counts": summary["counts"],
            "dry_run": args.dry_run,
            "finished": out.now(),
        })
        print(f"[{arm.slug}] {summary['blocks_completed']} blocks | "
              f"{summary['counts']['long_rows']} long rows | "
              f"{summary['counts']['wide_rows']} wide rows | "
              f"{len(summary['quarantined'])} quarantined -> {out_dir}")
    return 0


def cmd_verify_arms(args) -> int:
    cfg = load_config(args.config)
    digests = {}
    for arm in cfg.arms:
        mpath = cfg.out_for(arm.slug) / "manifest.json"
        if not mpath.exists():
            print(f"missing manifest for arm {arm.slug}: {mpath}")
            return 1
        digests[arm.slug] = json.loads(mpath.read_text(encoding="utf-8"))["design_digest"]
    unique = set(digests.values())
    for slug, d in digests.items():
        print(f"{slug:20s} {d[:16]}")
    if len(unique) != 1:
        print("FAIL — arms do not share identical block composition")
        return 1
    print("OK — all arms share an identical design digest")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent_srm")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("validate", cmd_validate), ("run", cmd_run),
                     ("verify-arms", cmd_verify_arms)):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--arm", default=None, help="run a single arm by slug")
            p.add_argument("--no-resume", action="store_true")
            p.add_argument("--workers", type=int, default=None,
                           help="override execution.max_concurrent_dyads")
        p.set_defaults(func=fn)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
