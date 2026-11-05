"""JSON Schema of :class:`qts.core.config.SystemConfig` and strict validation of
dotted-path overrides.

Schema
------
Generated from the config dataclasses (``typing.get_type_hints``): every section is
an ``object``; scalar fields become ``number`` / ``integer`` / ``boolean`` /
``string`` leaves with the default value, a title derived from the field name, a
description taken from the field's comments in ``qts/core/config.py`` (the comment
block directly above it plus its inline comment) and, where curated below, enums
and bounds (``minimum`` / ``maximum``, or ``exclusiveMinimum`` / ``exclusiveMaximum``
for open ends).  The ``cost`` and ``execution`` bounds are the ones
``CostConfig`` / ``ExecutionConfig`` enforce on construction
(``qts.core.config.COST_CONFIG_BOUNDS`` / ``EXECUTION_CONFIG_BOUNDS``); every other
section's bounds are enforced here only (API layer), e.g. ``alpha.lgbm.n_estimators
<= 10000``.  Optional scalars (``float | None``) are typed by their scalar type and
have no ``default`` when the default is ``None``.  Lists, tuples and dicts (style
lists, theme weights, the PBO grid) are not form-editable and are left out; they
are still settable through a config file.

Overrides
---------
:func:`validate_overrides` checks every ``"section.field"`` path against the same
introspection (unknown path, non-scalar field, wrong type, out of bounds, not in
enum -> a :class:`ConfigError` with one issue per key), then builds the nested
dict and passes it through ``SystemConfig.from_dict`` (strict) and ``validate()``.
"""

from __future__ import annotations

import inspect
import io
import re
import tokenize
import types
import typing
from dataclasses import MISSING, fields, is_dataclass
from functools import cache, lru_cache
from typing import Any, Union, get_args, get_origin, get_type_hints

from qts.core import config as C

#: Section titles shown by the form generator.
SECTION_TITLES: dict[str, str] = {
    "universe": "Universe",
    "alpha": "Alpha model",
    "alpha.lgbm": "Alpha model: LightGBM",
    "alpha.mlp": "Alpha model: MLP",
    "risk": "Risk model",
    "cost": "Transaction costs",
    "portfolio": "Portfolio construction",
    "execution": "Execution",
    "validation": "Validation",
}

#: Closed sets of string values.
ENUMS: dict[str, list[str]] = {
    "portfolio.book": ["long_short", "long_only", "130_30"],
    "portfolio.vol_target_mode": ["projection", "constraint"],
    "portfolio.risk_aversion_mode": ["target_vol", "fixed"],
    "portfolio.solver": ["CLARABEL", "SCS", "OSQP", "ECOS"],
    "cost.tca_reversion_reference": ["next_close", "next_open"],
    "risk.exposure_dtype": ["float64", "float32"],
}

#: Curated (minimum, maximum); ``None`` = unbounded on that side.
BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "universe.top_n_by_mcap": (10, 10_000),
    "universe.min_price": (0, None),
    "universe.min_adv_usd": (0, None),
    "universe.adv_window": (1, 504),
    "universe.exclude_bottom_size_quantile": (0, 0.9),
    "universe.refresh_days": (1, 252),
    "universe.exit_after_n_fails": (1, 50),
    "alpha.horizon_days": (1, 252),
    "alpha.winsor_pct": (0, 0.25),
    "alpha.factor_momentum_max_tilt": (0, 1),
    "alpha.ic_shrink": (0, 1),
    "alpha.default_ic": (-0.2, 0.2),
    "alpha.composite_weight": (0, 1),
    "alpha.model_seeds": (1, 20),
    "risk.min_history_days": (21, 2520),
    "risk.alignment_nu_scale": (0, None),
    "risk.max_statistical_factors": (0, 50),
    "portfolio.gross_leverage_max": (0, 10),
    "portfolio.net_exposure_target": (-1, 1),
    "portfolio.net_exposure_band": (0, 1),
    "portfolio.max_position_weight": (0.001, 0.1),
    "portfolio.max_adv_participation": (0, 1),
    "portfolio.industry_band": (0, 1),
    "portfolio.style_band_sd": (0, 5),
    "portfolio.risk_aversion": (0, 100),
    "portfolio.trade_aversion": (0, 10),
    "portfolio.hold_aversion": (0, 10),
    "portfolio.turnover_penalty": (0, None),
    "portfolio.target_vol_annual": (0.01, 0.4),
    "portfolio.beta_band": (0, 1),
    "validation.n_splits": (2, 12),
    "validation.embargo_days": (0, 120),
    "validation.min_fold_days": (1, 504),
    "validation.dsr_threshold": (0, 1),
    "validation.pbo_threshold": (0, 1),
    "validation.min_fold_pass_fraction": (0, 1),
    "validation.max_failed_solve_frac": (0, 1),
    "trading_days_per_year": (200, 366),
    # API-layer guards against absurd sizes (sections without semantic validation in
    # qts.core.config): a typo must not launch a job that runs for days.
    "alpha.lgbm.n_estimators": (1, 10_000),
    "alpha.lgbm.num_leaves": (2, 4096),
    "alpha.lgbm.min_data_in_leaf": (1, 1_000_000),
    "alpha.lgbm.learning_rate": (1e-6, 1),
    "alpha.lgbm.feature_fraction": (0.01, 1),
    "alpha.lgbm.bagging_fraction": (0.01, 1),
    "alpha.lgbm.bagging_freq": (0, 1000),
    "alpha.lgbm.lambda_l2": (0, 1e6),
    "alpha.lgbm.early_stopping_rounds": (1, 10_000),
    "alpha.lgbm.n_jobs": (1, 256),
    "alpha.mlp.alpha": (0, 1e3),
    "alpha.mlp.max_iter": (1, 10_000),
    "alpha.mlp.batch_size": (1, 1_000_000),
    "alpha.mlp.learning_rate_init": (1e-8, 1),
    "alpha.mlp.max_train_rows": (1_000, 10_000_000),
    "alpha.mlp.n_iter_no_change": (1, 1000),
    "alpha.mlp.valid_fraction": (0.01, 0.5),
    "alpha.ic_window": (20, 2520),
    "alpha.ic_min_obs": (1, 2520),
    "alpha.valid_days": (1, 2520),
    "alpha.refit_frequency_days": (1, 2520),
    "alpha.train_min_days": (1, 10_000),
    "alpha.embargo_days": (0, 252),
    "risk.n_eigen_sims": (0, 10_000),
    "risk.statistical_window": (21, 2520),
    "risk.statistical_refit_every": (1, 2520),
    "portfolio.exit_ineligible_days": (1, 252),
    "portfolio.effective_holding_days_target": (1, 2520),
}

#: Open bounds (``exclusiveMinimum`` / ``exclusiveMaximum``) by path.
EXCLUSIVE: dict[str, tuple[bool, bool]] = {}


def _semantic_bounds() -> None:
    """Bounds of the sections that ``qts.core.config`` validates itself (``CostConfig``,
    ``ExecutionConfig``): published from the same tables the dataclasses enforce, so the
    form, this module and ``SystemConfig.from_dict`` agree."""
    for section, table in (("cost", C.COST_CONFIG_BOUNDS), ("execution", C.EXECUTION_CONFIG_BOUNDS)):
        for name, (lo, hi, lo_inc, hi_inc) in table.items():
            BOUNDS[f"{section}.{name}"] = (lo, hi)
            if not (lo_inc and hi_inc):
                EXCLUSIVE[f"{section}.{name}"] = (not lo_inc, not hi_inc)


_semantic_bounds()

#: Paths the launch request sets through its own fields (rejected as overrides).
RESERVED: dict[str, str] = {
    "seed": "use the request's `seed` field",
    "execution.signal_lag_days": "use the request's `signal_lag_days` field",
}

MAX_DESCRIPTION = 320

#: Hand-written descriptions where the source comment does not read well on its own
#: (key: ``<ConfigClass>.<field>``).
DESCRIPTIONS: dict[str, str] = {
    "PortfolioConfig.book": "Dollar-neutral long/short, long-only, or 130/30.",
    "PortfolioConfig.trade_aversion": "Multiplier on the transaction-cost term. Raise it when "
    "effective holding is below target.",
    "PortfolioConfig.risk_aversion": "Annual risk aversion gamma on portfolio variance "
    "(objective alpha'w - gamma*H*w'Sigma_d*w). Used as-is when risk_aversion_mode is "
    '"fixed" or no vol target is set; in "target_vol" mode gamma is searched per solve '
    "from the closed form IR/(2*target) and this value is only the fallback when that "
    "start is not finite.",
    "PortfolioConfig.risk_aversion_mode": '"target_vol": gamma is searched per solve so the '
    "book's ex-ante vol hits target_vol_annual (capped books accepted; vol_target_mode "
    'unused). "fixed": gamma = risk_aversion, target enforced by vol_target_mode.',
    "PortfolioConfig.target_vol_annual": "Annualised volatility target, as a fraction (0.10 = 10%).",
    "PortfolioConfig.max_position_weight": "Cap on |weight| per name, as a fraction of NAV.",
    "PortfolioConfig.beta_neutral": "Constrain the book's beta exposure to the band around zero.",
    "ValidationConfig.n_splits": "Purged walk-forward test blocks for the fold-Sharpe gate.",
    "ValidationConfig.embargo_days": "Days dropped after each test block; at least the label horizon.",
}


class ConfigError(ValueError):
    """Invalid overrides; ``issues`` = [(loc, msg, type)]."""

    def __init__(self, issues: list[tuple[list[str | int], str, str]]):
        self.issues = issues
        super().__init__("; ".join(f"{'.'.join(map(str, loc))}: {msg}" for loc, msg, _ in issues))


# ----------------------------------------------------------------------------- introspection
def _title(name: str) -> str:
    s = name.replace("_", " ").strip()
    s = re.sub(r"\badv\b", "ADV", s)
    s = re.sub(r"\bic\b", "IC", s)
    s = re.sub(r"\bdsr\b", "DSR", s)
    s = re.sub(r"\bpbo\b", "PBO", s)
    s = re.sub(r"\bsd\b", "SD", s)
    s = re.sub(r"\busd\b", "USD", s)
    s = re.sub(r"\bmcap\b", "market cap", s)
    s = re.sub(r"\btaf\b", "TAF", s)
    s = re.sub(r"\bsec\b", "SEC", s)
    s = re.sub(r"\bfinra\b", "FINRA", s)
    s = re.sub(r"\bcat\b", "CAT", s)
    s = re.sub(r"\bhtb\b", "HTB", s)
    s = re.sub(r"\bsi\b", "SI", s)
    s = re.sub(r"\btca\b", "TCA", s)
    s = re.sub(r"\bvra\b", "VRA", s)
    s = re.sub(r"\bnu\b", "nu", s)
    s = re.sub(r"\bl2\b", "L2", s)
    s = re.sub(r"\bgp\b", "GP", s)
    s = re.sub(r"\bloc\b", "LOC", s)
    s = re.sub(r"\bpct\b", "%", s)
    return s[:1].upper() + s[1:]


def _clean_comment(lines: list[str]) -> str:
    text = " ".join(ln.strip().lstrip("-").strip() for ln in lines if ln.strip())
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_DESCRIPTION:
        cut = text[:MAX_DESCRIPTION]
        dot = cut.rfind(". ")
        text = (cut[: dot + 1] if dot > 80 else cut.rstrip() + "…").strip()
    return text


@cache
def field_comments(cls: type) -> dict[str, str]:
    """``{field: description}`` from the comments around each field in the class source."""
    try:
        src = inspect.getsource(cls)
    except (OSError, TypeError):
        return {}
    comments_by_line: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                comments_by_line[tok.start[0]] = tok.string[1:].strip()
    except (tokenize.TokenError, IndentationError):
        return {}
    out: dict[str, str] = {}
    pending: list[str] = []
    field_re = re.compile(r"^    (\w+)\s*:\s*[^=]+")
    for i, line in enumerate(src.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            pending = []
            continue
        if stripped.startswith("#"):
            pending.append(comments_by_line.get(i, stripped.lstrip("#").strip()))
            continue
        m = field_re.match(line)
        if m and not stripped.startswith(("def ", "class ", "@", '"""', "return")):
            block = " ".join(pending).strip()
            parts = [] if block.startswith("(") and block.endswith(")") else list(pending)
            if i in comments_by_line:
                parts.append(comments_by_line[i])
            text = DESCRIPTIONS.get(f"{cls.__name__}.{m.group(1)}") or _clean_comment(parts)
            if text:
                out[m.group(1)] = text
        pending = []
    return out


def _doc_summary(cls: type) -> str | None:
    doc = inspect.getdoc(cls)
    if not doc or cls.__doc__ is None:
        return None
    first = doc.split("\n\n")[0]
    return _clean_comment(first.splitlines()) or None


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _scalar_kind(tp: Any) -> str | None:
    tp, _ = _unwrap_optional(tp)
    if tp is bool:
        return "boolean"
    if tp is int:
        return "integer"
    if tp is float:
        return "number"
    if tp is str:
        return "string"
    return None


def _default(f: Any) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return None


def _leaf(path: str, kind: str, default: Any, description: str | None) -> dict[str, Any]:
    leaf: dict[str, Any] = {"type": kind, "title": _title(path.rsplit(".", 1)[-1])}
    if description:
        leaf["description"] = description
    if default is not None:
        leaf["default"] = float(default) if kind == "number" else default
    if path in ENUMS:
        leaf["enum"] = list(ENUMS[path])
    if kind in ("number", "integer") and path in BOUNDS:
        lo, hi = BOUNDS[path]
        lo_open, hi_open = EXCLUSIVE.get(path, (False, False))
        if lo is not None:
            leaf["exclusiveMinimum" if lo_open else "minimum"] = lo
        if hi is not None:
            leaf["exclusiveMaximum" if hi_open else "maximum"] = hi
    return leaf


def _object_schema(cls: type, prefix: str) -> dict[str, Any]:
    hints = get_type_hints(cls, vars(C))
    comments = field_comments(cls)
    props: dict[str, Any] = {}
    for f in fields(cls):
        path = f"{prefix}{f.name}"
        default = _default(f)
        if is_dataclass(default):
            props[f.name] = _object_schema(type(default), f"{path}.")
            continue
        kind = _scalar_kind(hints.get(f.name, Any))
        if kind is None:
            continue
        props[f.name] = _leaf(path, kind, default, comments.get(f.name))
    obj: dict[str, Any] = {"type": "object", "properties": props}
    key = prefix.rstrip(".")
    title = SECTION_TITLES.get(key) or (cls.__name__ if not key else _title(key.rsplit(".", 1)[-1]))
    obj["title"] = title
    desc = _doc_summary(cls)
    if desc:
        obj["description"] = desc
    return obj


@lru_cache(maxsize=1)
def config_json_schema() -> dict[str, Any]:
    """The ``/api/config/schema`` document."""
    schema = _object_schema(C.SystemConfig, "")
    schema["title"] = "SystemConfig"
    return schema


@lru_cache(maxsize=1)
def leaf_index() -> dict[str, dict[str, Any]]:
    """``{"section.field": leaf schema}`` for every overridable scalar."""
    out: dict[str, dict[str, Any]] = {}

    def walk(node: dict[str, Any], prefix: str) -> None:
        for k, v in node["properties"].items():
            if v["type"] == "object":
                walk(v, f"{prefix}{k}.")
            else:
                out[f"{prefix}{k}"] = v

    walk(config_json_schema(), "")
    return out


# ----------------------------------------------------------------------------- overrides
@lru_cache(maxsize=1)
def _all_fields() -> list[Any]:
    out: list[Any] = []

    def walk(cls: type) -> None:
        for f in fields(cls):
            out.append(f)
            d = _default(f)
            if is_dataclass(d):
                walk(type(d))

    walk(C.SystemConfig)
    return out


def _check_value(path: str, leaf: dict[str, Any], v: Any) -> tuple[Any, str | None]:
    kind = leaf["type"]
    if kind == "boolean":
        return (v, None) if isinstance(v, bool) else (v, "must be true or false")
    if kind == "string":
        if not isinstance(v, str):
            return v, "must be a string"
        if "enum" in leaf and v not in leaf["enum"]:
            return v, f"must be one of {leaf['enum']}"
        return v, None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v, "must be a number"
    if kind == "integer":
        if isinstance(v, float):
            if not v.is_integer():
                return v, "must be a whole number"
            v = int(v)
    else:
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return v, "must be finite"
    lo, hi = leaf.get("minimum"), leaf.get("maximum")
    if lo is not None and v < lo:
        return v, f"must be at least {lo}"
    if hi is not None and v > hi:
        return v, f"must be at most {hi}"
    xlo, xhi = leaf.get("exclusiveMinimum"), leaf.get("exclusiveMaximum")
    if xlo is not None and v <= xlo:
        return v, f"must be greater than {xlo}"
    if xhi is not None and v >= xhi:
        return v, f"must be less than {xhi}"
    return v, None


def validate_overrides(overrides: typing.Mapping[str, Any], base: C.SystemConfig | None = None) -> C.SystemConfig:
    """``SystemConfig`` with the dotted-path ``overrides`` applied, strictly validated."""
    index = leaf_index()
    issues: list[tuple[list[str | int], str, str]] = []
    clean: dict[str, Any] = {}
    for key, v in overrides.items():
        loc: list[str | int] = ["body", "config_overrides", key]
        if key in RESERVED:
            issues.append((loc, f"{key} cannot be overridden here: {RESERVED[key]}", "value_error.reserved"))
            continue
        leaf = index.get(key)
        if leaf is None:
            section = key.split(".")[0]
            props = config_json_schema()["properties"]
            if "." in key and section not in props:
                msg = f"unknown config section {section!r} in {key!r}"
            elif key.split(".")[-1] in {f.name for f in _all_fields()}:
                msg = f"{key!r} is not a scalar field (lists, dicts and sections need a config file)"
            else:
                msg = f"unknown config field {key!r}"
            issues.append((loc, msg, "value_error.unknown_key"))
            continue
        val, err = _check_value(key, leaf, v)
        if err:
            issues.append((loc, f"{leaf.get('title', key)} {err}", "value_error.invalid"))
            continue
        clean[key] = val
    if issues:
        raise ConfigError(issues)
    d = (base or C.SystemConfig()).to_dict()
    for key, val in clean.items():
        parts = key.split(".")
        node = d
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    try:
        return C.SystemConfig.from_dict(d).validate()
    except (KeyError, TypeError, ValueError) as e:
        raise ConfigError([(["body", "config_overrides"], str(e), "value_error.config")]) from e
