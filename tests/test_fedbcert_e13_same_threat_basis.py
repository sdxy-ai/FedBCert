import torch

from openfgl.flcore.fedbcert import completion_uncertainty_basis


def identity_shift(node_count: int) -> torch.Tensor:
    indices = torch.arange(node_count, dtype=torch.long).repeat(2, 1)
    return torch.sparse_coo_tensor(
        indices, torch.ones(node_count), (node_count, node_count)
    ).coalesce()


def test_same_threat_bases_are_sound_supersets_of_exact_rows():
    shift = identity_shift(3)
    exact = torch.tensor([0.0, 0.6, 1.5])
    missing = torch.tensor([0.0, 1.0, 3.0])
    exact_basis = completion_uncertainty_basis(
        shift, exact, missing, 3, information="nodewise_exact"
    )
    presence = completion_uncertainty_basis(
        shift, exact, missing, 3, information="boundary_presence"
    )
    global_max = completion_uncertainty_basis(
        shift, exact, missing, 3, information="global_max",
        global_max_row_budget=float(exact.max()),
    )
    universal = completion_uncertainty_basis(
        shift, exact, missing, 3, information="universal"
    )
    for order in range(3):
        assert torch.all(exact_basis[order] <= presence[order])
        assert torch.all(exact_basis[order] <= global_max[order])
        assert torch.all(exact_basis[order] <= universal[order])
    assert torch.equal(universal[0], torch.zeros(3))
    assert torch.equal(universal[1], torch.full((3,), 2.0))


def test_global_max_requires_an_explicit_valid_bound():
    shift = identity_shift(1)
    exact = torch.tensor([0.5])
    missing = torch.tensor([1.0])
    try:
        completion_uncertainty_basis(
            shift, exact, missing, 2, information="global_max"
        )
    except ValueError as error:
        assert "global_max_row_budget" in str(error)
    else:
        raise AssertionError("missing global maximum must fail")
