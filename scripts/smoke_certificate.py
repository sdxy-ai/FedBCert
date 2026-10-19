#!/usr/bin/env python3
"""Small offline check of the analytic FedBCert certificate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openfgl.flcore.fedbcert import (
    PolynomialSpectralClassifier,
    cone_pairwise_certificate_statistics,
    retained_mass_uncertainty_basis,
)


def main() -> None:
    torch.manual_seed(18)
    model = PolynomialSpectralClassifier(
        input_dim=3,
        hidden_dim=4,
        num_classes=3,
        order_count=3,
        dropout=0.0,
        encoder_type="dual_feature_spectral",
        encoder_depth=2,
        spectral_fusion_scale=0.25,
        certificate_margin_mode="pairwise_difference",
    ).eval()
    features = torch.randn(5, 3)
    shift = torch.eye(5).to_sparse_coo().coalesce()
    row_budget = torch.tensor([0.0, 0.4, 1.0, 1.5, 2.0])
    basis = retained_mass_uncertainty_basis(shift, row_budget, 3)

    with torch.no_grad():
        logits = model(features, shift)
        cone = cone_pairwise_certificate_statistics(
            model,
            features,
            shift,
            row_budget,
            logits,
            basis,
            combine_symmetric=False,
        )
        hybrid = cone_pairwise_certificate_statistics(
            model,
            features,
            shift,
            row_budget,
            logits,
            basis,
            combine_symmetric=True,
        )

    if not torch.all(~cone.certified | hybrid.certified):
        raise RuntimeError("the hybrid certificate did not contain the cone certificate")
    if not torch.isfinite(hybrid.worst_slack).all():
        raise RuntimeError("non-finite certificate slack")

    print(
        json.dumps(
            {
                "complete": True,
                "nodes": int(features.size(0)),
                "cone_certified": int(cone.certified.sum()),
                "hybrid_certified": int(hybrid.certified.sum()),
                "hybrid_contains_cone": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
