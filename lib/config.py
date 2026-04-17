from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class ModelCfg:
    name: str
    hf_id: str
    dtype: str
    num_layers: int
    lens_layers: List[int]
    pre_think_tag: str
    think_close_tag: str
    max_position_embeddings: int | None = None
    rope_scaling: dict | None = None


@dataclass
class GenerationCfg:
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int


@dataclass
class DirectAnswerCfg:
    max_new_tokens: int
    num_samples: int
    temperature: float
    top_p: float


@dataclass
class LensCfg:
    type: str
    train_data: str
    train_steps: int
    train_batch_size: int
    train_lr: float


@dataclass
class DecompositionCfg:
    tau: float
    boundary_tokens: List[str]


@dataclass
class PathsCfg:
    lens_dir: str
    lib_dir: str
    cot_dir: str
    direct_dir: str


@dataclass
class Cfg:
    model: ModelCfg
    generation: GenerationCfg
    direct_answer: DirectAnswerCfg
    lens: LensCfg
    decomposition: DecompositionCfg
    paths: PathsCfg


def load_cfg(path: str | Path) -> Cfg:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Cfg(
        model=ModelCfg(**raw["model"]),
        generation=GenerationCfg(**raw["generation"]),
        direct_answer=DirectAnswerCfg(**raw["direct_answer"]),
        lens=LensCfg(**raw["lens"]),
        decomposition=DecompositionCfg(**raw["decomposition"]),
        paths=PathsCfg(**raw["paths"]),
    )
