"""Config schema, loading, and the §7.1 validation gates.

Every hard failure here fires *before* any API call is made.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

JSON_STRATEGIES = {"none", "response_format", "guided_json", "ollama_format"}


class ConfigError(ValueError):
    """Raised for any §7.1 hard failure. Message names the offending field."""


PERSONA_MODES = {"full", "none"}


@dataclass
class PersonaCfg:
    dir: str = "data/personas"
    field: str = "persona_summary"
    manifest: str | None = None
    verify_checksums: bool = True
    # full -> the entire persona_summary is rendered into the system prompt.
    # none -> no persona text at all; the system prompt is the chat framing
    #         only. The parquet is still loaded, so block composition, dyads,
    #         seeds and personas.csv are IDENTICAL to a full run — which makes
    #         a `none` arm a paired null model: any partner variance it shows
    #         is generation noise, not persona signal.
    mode: str = "full"


@dataclass
class DesignCfg:
    block_size: int = 4
    n_blocks: int = 20
    k: int = 2
    intro_mode: str | None = "shared"   # shared | unique ; k == 2 only
    self_ratings: bool = False


@dataclass
class NamingCfg:
    enabled: bool = False
    first_names: str | None = None
    surnames: str | None = None
    match_gender: bool = True


@dataclass
class ItemsCfg:
    constructs: dict[str, list[str]] = field(
        default_factory=lambda: {
            "warmth": ["warm", "kind"],
            "competence": ["capable", "effective"],
        }
    )
    scale_min: int = 1
    scale_max: int = 5
    randomize_item_order: bool = True

    @property
    def items(self) -> list[str]:
        out: list[str] = []
        for names in self.constructs.values():
            out.extend(names)
        return out

    @property
    def item_to_construct(self) -> dict[str, str]:
        return {i: c for c, names in self.constructs.items() for i in names}

    def json_schema(self, order: list[str]) -> dict:
        """Property order follows the *presentation* order for this call."""
        return {
            "type": "object",
            "properties": {i: {"type": "integer"} for i in order},
            "required": list(order),
            "additionalProperties": False,
        }


@dataclass
class CallCfg:
    temperature: float = 1.0
    max_tokens: int = 1024
    top_p: float = 1.0
    seed: int | None = None


@dataclass
class ArmCfg:
    """One model's pass over the study. Arms differ only in these fields."""
    slug: str
    model_id: str = "mock"
    base_url: str | None = None
    api_key_env: str = "LLM_API_KEY"
    json_strategy: str = "none"
    # None == provider default (the v2.0 setting). A dict is passed through as
    # extra request fields so reasoning can be adjusted per arm later.
    reasoning: dict | None = None

    @property
    def provider(self) -> str:
        if not self.base_url:
            return "mock"
        return urlparse(self.base_url).netloc or self.base_url

    @property
    def reasoning_mode(self) -> str:
        return "provider_default" if self.reasoning is None else repr(self.reasoning)


@dataclass
class ModelCfg:
    """Shared call parameters. Arms carry the endpoint-specific fields."""
    conversation: CallCfg = field(default_factory=lambda: CallCfg(1.0, 1024))
    rating: CallCfg = field(default_factory=lambda: CallCfg(0.0, 512))
    max_json_retries: int = 3


@dataclass
class ExecCfg:
    max_concurrent_dyads: int = 4
    request_timeout_s: int = 120
    max_retries: int = 5
    backoff_base_s: float = 2.0
    checkpoint_dir: str = "runs/{study_id}/checkpoints"


@dataclass
class ReplayCfg:
    source_run: str
    source_manifest_sha256: str | None = None
    replay_id: str = "r1"


@dataclass
class Config:
    study_id: str
    seed: int
    personas: PersonaCfg = field(default_factory=PersonaCfg)
    design: DesignCfg = field(default_factory=DesignCfg)
    naming: NamingCfg = field(default_factory=NamingCfg)
    items: ItemsCfg = field(default_factory=ItemsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    arms: list[ArmCfg] = field(default_factory=lambda: [ArmCfg(slug="mock")])
    execution: ExecCfg = field(default_factory=ExecCfg)
    output_dir: str = "runs/{study_id}"
    prompts_dir: str = "prompts"
    replay: ReplayCfg | None = None

    # -- derived ---------------------------------------------------------
    def arm(self, slug: str) -> ArmCfg:
        for a in self.arms:
            if a.slug == slug:
                return a
        raise ConfigError(f"no arm with slug {slug!r}; have {[a.slug for a in self.arms]}")

    def study_id_for(self, arm_slug: str) -> str:
        return f"{self.study_id}__{arm_slug}"

    def out_for(self, arm_slug: str) -> Path:
        return Path(self.output_dir.format(study_id=self.study_id_for(arm_slug)))

    def ckpt_for(self, arm_slug: str) -> Path:
        return Path(self.execution.checkpoint_dir.format(
            study_id=self.study_id_for(arm_slug)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"]["items"] = self.items.items
        return d


_CALL_KEYS = {"temperature", "max_tokens", "top_p", "seed"}
_ARM_KEYS = {"slug", "model_id", "base_url", "api_key_env", "json_strategy", "reasoning"}


def _call(d: Any, default: CallCfg) -> CallCfg:
    if not d:
        return default
    return CallCfg(**{k: v for k, v in d.items() if k in _CALL_KEYS})


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items_raw = raw.get("items", {}) or {}
    reserved = {"scale_min", "scale_max", "randomize_item_order"}
    constructs = {k: v for k, v in items_raw.items() if k not in reserved}
    model_raw = raw.get("model", {}) or {}

    arms_raw = raw.get("arms") or []
    arms = []
    for a in arms_raw:
        unknown = set(a) - _ARM_KEYS
        if unknown:
            raise ConfigError(
                f"arm {a.get('slug')!r} has unknown field(s) {sorted(unknown)}; arms may "
                "only differ on model fields"
            )
        arms.append(ArmCfg(**a))
    if not arms:
        arms = [ArmCfg(slug="mock")]

    return Config(
        study_id=raw["study_id"],
        seed=int(raw["seed"]),
        personas=PersonaCfg(**(raw.get("personas") or {})),
        design=DesignCfg(**(raw.get("design") or {})),
        naming=NamingCfg(**(raw.get("naming") or {})),
        items=ItemsCfg(
            constructs=constructs or ItemsCfg().constructs,
            scale_min=items_raw.get("scale_min", 1),
            scale_max=items_raw.get("scale_max", 5),
            randomize_item_order=items_raw.get("randomize_item_order", True),
        ),
        model=ModelCfg(
            conversation=_call(model_raw.get("conversation"), CallCfg(1.0, 1024)),
            rating=_call(model_raw.get("rating"), CallCfg(0.0, 512)),
            max_json_retries=model_raw.get("max_json_retries", 3),
        ),
        arms=arms,
        execution=ExecCfg(**(raw.get("execution") or {})),
        output_dir=(raw.get("output") or {}).get("dir", "runs/{study_id}"),
        prompts_dir=raw.get("prompts_dir", "prompts"),
        replay=ReplayCfg(**raw["replay"]) if raw.get("replay") else None,
    )


def _is_local(base_url: str) -> bool:
    return any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def validate(cfg: Config, pool_size: int | None = None, *,
             require_key: bool = True, dry_run: bool = False) -> None:
    """§7.1 hard failures. Raises ConfigError naming the offending field."""
    d, it = cfg.design, cfg.items

    if d.k < 2:
        raise ConfigError(f"design.k must be >= 2 (got {d.k})")
    if d.k % 2 != 0:
        raise ConfigError(
            f"design.k must be even (got {d.k}); at odd k the initiator sends one more "
            "message than the responder and holds the last word, confounding exposure "
            "with initiator assignment"
        )
    if d.intro_mode is not None and d.k != 2:
        raise ConfigError(
            f"design.intro_mode is k=2 only (got k={d.k}, intro_mode={d.intro_mode!r}); "
            "at k>=4 the responder's message is dyad-specific by construction"
        )
    if d.k == 2 and d.intro_mode not in {"shared", "unique"}:
        raise ConfigError("design.intro_mode must be 'shared' or 'unique' when k == 2")
    if d.self_ratings:
        raise ConfigError("design.self_ratings must remain false")
    if d.block_size < 4:
        raise ConfigError(
            f"design.block_size must be >= 4 (got {d.block_size}); a block of 3 yields 6 "
            "directed ratings against 3 actor and 3 partner effects, leaving no degrees "
            "of freedom for relationship variance"
        )
    if d.n_blocks < 1:
        raise ConfigError(f"design.n_blocks must be >= 1 (got {d.n_blocks})")

    overlap = set(it.constructs) & set(it.items)
    if overlap:
        raise ConfigError(
            f"items: construct name(s) {sorted(overlap)} also appear as items; a construct "
            "label must never be renderable as a stimulus"
        )
    if len(it.items) != len(set(it.items)):
        raise ConfigError("items: duplicate item names across constructs")
    for c, names in it.constructs.items():
        if len(names) != 2:
            raise ConfigError(
                f"items.{c} must have exactly 2 indicators (got {len(names)}); TripleR "
                "accepts a maximum of two indicators per latent construct"
            )
    if it.scale_min >= it.scale_max:
        raise ConfigError("items.scale_min must be < items.scale_max")

    if cfg.personas.mode not in PERSONA_MODES:
        raise ConfigError(
            f"personas.mode must be one of {sorted(PERSONA_MODES)} "
            f"(got {cfg.personas.mode!r})")

    if cfg.model.rating.max_tokens < 512:
        raise ConfigError(
            f"model.rating.max_tokens must be >= 512 (got {cfg.model.rating.max_tokens}); "
            "the Study 2 N=10 truncation bug came from this"
        )

    if pool_size is not None and d.n_blocks * d.block_size > pool_size:
        raise ConfigError(
            f"design.n_blocks * design.block_size = {d.n_blocks * d.block_size} exceeds "
            f"persona pool size {pool_size} (ceiling is {pool_size // d.block_size} blocks "
            f"at block_size {d.block_size})"
        )

    if cfg.naming.enabled and not (cfg.naming.first_names and cfg.naming.surnames):
        raise ConfigError("naming.enabled requires naming.first_names and naming.surnames")

    slugs = [a.slug for a in cfg.arms]
    if len(slugs) != len(set(slugs)):
        raise ConfigError(f"duplicate arm slug(s) in {slugs}")

    for a in cfg.arms:
        if a.json_strategy not in JSON_STRATEGIES:
            raise ConfigError(
                f"arm {a.slug!r}: unknown json_strategy {a.json_strategy!r}; "
                f"expected one of {sorted(JSON_STRATEGIES)}"
            )
        if dry_run or a.base_url is None:
            continue
        if require_key and a.api_key_env not in os.environ and not _is_local(a.base_url):
            raise ConfigError(
                f"arm {a.slug!r}: missing API credentials — environment variable "
                f"{a.api_key_env} is not set"
            )


def validate_replay(replay_cfg: Config, source_manifest: dict) -> None:
    """Anything that determined the *conversations* is frozen; measurement is free."""
    src = source_manifest["config"]
    frozen = [
        ("design.block_size", replay_cfg.design.block_size, src["design"]["block_size"]),
        ("design.n_blocks", replay_cfg.design.n_blocks, src["design"]["n_blocks"]),
        ("design.k", replay_cfg.design.k, src["design"]["k"]),
        ("design.intro_mode", replay_cfg.design.intro_mode, src["design"]["intro_mode"]),
        ("design.self_ratings", replay_cfg.design.self_ratings,
         src["design"]["self_ratings"]),
        ("seed", replay_cfg.seed, src["seed"]),
        ("naming.enabled", replay_cfg.naming.enabled, src["naming"]["enabled"]),
        ("personas.field", replay_cfg.personas.field, src["personas"]["field"]),
    ]
    bad = [(n, a, b) for n, a, b in frozen if a != b]
    if bad:
        lines = "; ".join(f"{n}: replay={a!r} source={b!r}" for n, a, b in bad)
        raise ConfigError(f"replay may not alter conversation-determining fields — {lines}")
