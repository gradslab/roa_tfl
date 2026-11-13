"""Configuration of the unicycle TFL closed loop and its constructor, used by problem.py."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from tfl_system import TFLClosedLoop


@dataclass
class Config:
    path: str = "x^2 + y^2 - 1 = 0"
    pi: str = "atan2(y, x)"
    target: str = "z"
    eta_c: float = 0.5
    eta_c_hi: float = float("inf")
    vbar: float = 1.0
    k1: float = -1.0
    k2: float = -2.0
    k3: float = -1.0
    eta2_ref: float = 1.0
    box_xy: float = 2.0
    box_x4: tuple = (-1.5, 1.5)
    periodic_x1: bool = False
    normalize_field: bool = True
    z_quantile: float = 0.5
    z_sat: float = 10.0
    init_Q_diag: tuple = (5.941, 1.092, 19.378)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_system(cfg: Config) -> TFLClosedLoop:
    return TFLClosedLoop(cfg.path, cfg.pi, cfg.vbar, cfg.k1, cfg.k2, cfg.k3, cfg.eta2_ref,
                         cfg.box_xy, tuple(cfg.box_x4), target=cfg.target,
                         normalize_field=cfg.normalize_field, device=cfg.device,
                         z_quantile=cfg.z_quantile, z_sat=cfg.z_sat, periodic_x1=cfg.periodic_x1,
                         eta_c=cfg.eta_c, eta_c_hi=cfg.eta_c_hi)
