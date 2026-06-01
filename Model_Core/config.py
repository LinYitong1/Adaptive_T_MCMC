"""Project paths and model parameter defaults."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

ModelId = Literal[
    "M0_frequency",
    "M1_crw",
    "M2_crw_jump",
    "M3_fixed_T",
    "M4_adaptive_T",
    "M4b_adaptive_T_global",
]


@dataclass
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def vocab(self) -> Path:
        return self.data / "vocab"

    @property
    def embeddings(self) -> Path:
        return self.data / "embeddings"

    @property
    def llm_scores(self) -> Path:
        return self.data / "llm_scores"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def simulations(self) -> Path:
        return self.outputs / "simulations"

    @property
    def fits(self) -> Path:
        return self.outputs / "fits"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def tables(self) -> Path:
        return self.outputs / "tables"


@dataclass
class ModelParams:
    """Core sampler parameters (V1 defaults)."""

    model_id: ModelId = "M4_adaptive_T"
    eta: float = 0.5
    T_min: float = 0.5
    T_max: float = 3.0
    a: float = 0.0
    b: float = 2.0
    sigma_q: float = 0.35
    p_global: float = 0.25
    gamma0: float = -1.0  # logit intercept for p_global(T) in M4b / M6
    gamma1: float = 1.0  # logit slope for p_global(T) in M4b / M6
    fixed_T: float = 1.5  # M3
    p_jump: float = 0.15  # M2
    k_neighbors: int = 20
    switch_quantile: float = 0.80
    max_steps: int = 500
    use_latency_cap: bool = True  # stop when latency_sum exceeds cap (turns)
    latency_cap: int = 1000  # task-level cap, same semantics as social LLM nested sim
    allowed_categories: tuple[str, ...] = ("mammal", "bird", "water", "insect", "reptile")
    human_fluency_cohort: str = "oaf"  # primary: oaf | appendix: zhu | combined
    prompt: str = "Name animals"
    category: str = "animals"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "gpt2"
    history_dependent_llm: bool = False
    random_seed: Optional[int] = 42
    # Social-LLM shared probe assets (20260519_github_minimal/01_probe_assets)
    probe_json: Optional[str] = None
    embedding_cache: Optional[str] = None
    category_map: Optional[str] = None
    max_candidates: int = 5000
    use_probe_assets: bool = False  # True = read probe from external repo at runtime

    def validate(self) -> None:
        if self.eta < 0:
            raise ValueError("eta must be >= 0")
        if self.T_min <= 0:
            raise ValueError("T_min must be > 0")
        if self.T_max <= self.T_min:
            raise ValueError("T_max must be > T_min")
        if self.model_id == "M4b_adaptive_T_global" and self.b <= 0:
            raise ValueError("M4b requires b > 0 for monotone ΔE → T")
        if self.sigma_q <= 0:
            raise ValueError("sigma_q must be > 0")
        if not 0.0 <= self.p_global <= 1.0:
            raise ValueError("p_global must be in [0, 1]")


@dataclass
class FitConfig:
    n_simulations_per_theta: int = 50
    n_param_samples: int = 200
    summary_weights: dict = field(
        default_factory=lambda: {
            "mean_adjacent_distance": 1.0,
            "switch_rate_distance": 1.5,
            "bigram_relevance": 1.0,
            "mu_latency_pooled": 1.0,
            "delta_E_before_switch": 1.5,
            "slope_delta_E_before_switch": 1.0,
        }
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def standalone_data_files(paths: ProjectPaths, category: str = "animals") -> dict[str, Path]:
    """Paths required for fully standalone runs (no external probe repo)."""
    return {
        "vocab": paths.vocab / f"{category}_vocab.json",
        "llm_scores": paths.llm_scores / f"{category}_prompt_scores.parquet",
        "embeddings": paths.embeddings / f"{category}_embeddings.npy",
        # distance_matrix is derived on the fly from distance_sq (= sqrt) to avoid
        # shipping a redundant 200 MB array; we only require distance_sq here.
        "distance_sq": paths.embeddings / f"{category}_distance_sq.npy",
        "knn_indices": paths.embeddings / f"{category}_knn_indices.npy",
        "word_index": paths.embeddings / f"{category}_word_index.json",
        "categories": paths.vocab / f"{category}_categories.json",
    }


def is_standalone_ready(paths: ProjectPaths, category: str = "animals") -> tuple[bool, list[str]]:
    """Return (ready, missing_relative_paths)."""
    missing = []
    for name, p in standalone_data_files(paths, category).items():
        if not p.exists():
            missing.append(f"data/... ({name})")
    return len(missing) == 0, missing


def require_standalone_data(paths: ProjectPaths, category: str = "animals") -> None:
    ready, missing = is_standalone_ready(paths, category)
    if ready:
        return
    raise FileNotFoundError(
        "Standalone data bundle incomplete. Missing:\n  - "
        + "\n  - ".join(missing)
        + "\n\nBuild once (while still next to 20260519_github_minimal):\n"
        "  python scripts/00_import_probe_assets.py\n"
        "Or copy the full data/ folder from a machine that already ran import.\n"
        "See STANDALONE.md for the file checklist."
    )


def _apply_probe_defaults(params: ModelParams) -> None:
    """Fill probe paths from github_minimal bundle when use_probe_assets=True."""
    if not params.use_probe_assets:
        return
    from .probe_assets import default_probe_paths

    defaults = default_probe_paths()
    if not params.probe_json:
        params.probe_json = str(defaults["probe_json"])
    if not params.embedding_cache:
        params.embedding_cache = str(defaults["embedding_cache"])
    if not params.category_map:
        params.category_map = str(defaults["category_map"])


def _migrate_legacy_latency_params(loaded: dict, params: ModelParams) -> None:
    """Map old sec-based budget keys from saved configs."""
    if loaded.get("use_latency_budget") is False:
        params.use_latency_cap = False
    if "latency_cap" not in loaded and loaded.get("latency_budget_sec"):
        params.use_latency_cap = True
    if params.use_latency_cap and int(getattr(params, "latency_cap", 0) or 0) <= 0:
        params.latency_cap = 1000


def load_config(config_path: Optional[Path] = None) -> tuple[ProjectPaths, ModelParams]:
    paths = ProjectPaths(project_root())
    params = ModelParams()
    loaded: dict = {}
    if config_path is None:
        default = paths.root / "config" / "default_params.json"
        if default.exists():
            config_path = default
    if config_path and config_path.exists():
        with open(config_path) as f:
            raw = json.load(f)
        loaded = raw.get("params", {})
        for key, val in loaded.items():
            if hasattr(params, key):
                setattr(params, key, val)
    _apply_probe_defaults(params)
    _migrate_legacy_latency_params(loaded, params)
    if params.use_latency_cap:
        from .latency_budget import apply_latency_cap_defaults

        apply_latency_cap_defaults(params)
    params.validate()
    return paths, params


def save_params(params: ModelParams, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=2)
