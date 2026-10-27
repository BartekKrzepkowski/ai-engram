"""CPU parity tests for replication/scripts/recast_core.py (RECAST_PORTING_CONTRACT.md, section
"Required parity tests for every port"). Run on a compute node: they import torch."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import recast_core as R  # noqa: E402


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)


class Toy(nn.Module):
    """Mimics the HF layout: targets under model.layers.*, an excluded lm_head."""

    def __init__(self, d=6, n=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.model.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, d, bias=False)

    def forward(self, input_ids, labels, attention_mask=None):
        h = input_ids
        for blk in self.model.layers:
            h = blk.down_proj(torch.tanh(blk.q_proj(h)))
        out = self.lm_head(self.model.norm(h))
        return SimpleNamespace(loss=((out - labels) ** 2).mean(dim=1).mean())  # mean over examples


def loader(x, y, sizes):
    i = 0
    for s in sizes:
        yield {"input_ids": x[i:i + s], "labels": y[i:i + s]}
        i += s


def test_target_profiles():
    m = Toy()
    blocks = lambda ps: {f"model.layers.{i}.{p}" for i in range(2) for p in ps}  # noqa: E731
    assert set(R.select_targets(m, "decoder_linear")) == blocks(("q_proj", "down_proj"))
    assert set(R.select_targets(m, "mlp")) == blocks(("down_proj",))
    assert set(R.select_targets(m, "mlp_head")) == blocks(("down_proj",)) | {"lm_head"}
    assert set(R.select_targets(m, "mlp_down")) == blocks(("down_proj",))
    assert set(R.select_targets(m, "mlp_gate_down")) == blocks(("down_proj",))  # the toy has no gate_proj
    assert set(R.select_targets(m, "down_up_q")) == blocks(("down_proj", "q_proj"))  # the toy has no up_proj
    assert set(R.select_targets(m, "mlp_o")) == blocks(("down_proj",))  # the toy has no gate/up/o_proj
    with pytest.raises(ValueError):  # the toy has neither up_proj nor gate_proj
        R.select_targets(m, "mlp_up_gate")
    with pytest.raises(ValueError):  # the toy has no up_proj
        R.select_targets(m, "mlp_up")


def test_mean_gradient_is_example_weighted_and_state_restored():
    torch.manual_seed(0)
    m = Toy()
    m.train()
    m.model.norm.weight.requires_grad_(False)
    x, y = torch.randn(7, 6), torch.randn(7, 6)
    targets = R.select_targets(m, "decoder_linear")
    got, n = R.collect_mean_gradients(m, loader(x, y, [3, 3, 1]), targets, lambda b: b)
    assert n == 7
    weights = [targets[k].weight for k in targets]
    full = torch.autograd.grad(m(x, y).loss, weights)  # one batch = exact dataset mean
    for k, g in zip(targets, full):
        assert torch.allclose(got[k], g, atol=1e-6)
    assert m.training and not m.model.norm.weight.requires_grad and m.lm_head.weight.requires_grad
    assert all(p.grad is None for p in m.parameters())


def test_row_alignment_and_fixed_floor():
    w = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    g = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    a = R.row_alignment(w, g)
    assert torch.allclose(a, torch.tensor([1.0, -1.0, 0.0]), atol=1e-6)
    assert torch.allclose(R.shrink_factors(a, 0.1), torch.tensor([1.0, 0.1, 0.1]))
    with pytest.raises(ValueError):
        R.shrink_factors(a, 1.5)


def test_svd_rank_follows_evr_and_zero_energy():
    rank1 = torch.outer(torch.arange(1.0, 5.0), torch.arange(1.0, 4.0))
    assert R.svd_basis(rank1, 0.98).rank == 1
    diag = torch.diag(torch.tensor([3.0, 2.0, 1.0]))  # energies 9, 4, 1 -> EVR .643, .929, 1
    assert R.svd_basis(diag, 0.90).rank == 2
    assert R.svd_basis(diag, 0.93).rank == 3
    assert R.svd_basis(diag, 1.0).rank == 3
    zero = R.svd_basis(torch.zeros(4, 3), 0.98)
    assert not zero.active and zero.reason == "zero_energy"


def test_two_sided_projection_matches_direct_formula():
    torch.manual_seed(1)
    b = R.svd_basis(torch.randn(5, 2) @ torch.randn(2, 4), 0.999)
    d = torch.randn(5, 4)
    direct = b.u @ b.u.T @ d @ b.vh.T @ b.vh
    p = R.project(d, b.u, b.vh)
    assert torch.allclose(p, direct, atol=1e-6)
    assert torch.allclose(R.project(p, b.u, b.vh), p, atol=1e-6)  # idempotent


def test_unrestricted_warm_start_scales_rows_of_targets_only():
    torch.manual_seed(2)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    head = m.lm_head.weight.detach().clone()
    g = {k: torch.randn_like(v.weight) for k, v in t.items()}
    restr, rep = R.warm_start(t, g, 0.05)
    assert restr == {}
    for k, v in t.items():
        s = R.shrink_factors(R.row_alignment(before[k], g[k]), 0.05)
        assert torch.allclose(v.weight, s[:, None] * before[k], atol=1e-6)
        # regression (pilot 24955489): the report compared the float32 weight with itself
        expected = (v.weight.detach() - before[k]).norm() / before[k].norm()
        assert rep[k]["weight_rel_change"] == pytest.approx(float(expected), rel=1e-5) and expected > 0
    assert torch.equal(m.lm_head.weight, head)


def test_floor_one_is_exact_noop():
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    R.warm_start(t, {k: torch.randn_like(v.weight) for k, v in t.items()}, 1.0)
    assert all(torch.equal(t[k].weight, before[k]) for k in t)


def test_restricted_warm_start_delta_lies_in_subspace_and_zero_energy_skips():
    torch.manual_seed(3)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 2) @ torch.randn(2, 6) for k in t}
    zero_key = "model.layers.1.down_proj"
    gf[zero_key] = torch.zeros(6, 6)
    restr, rep = R.warm_start(t, gr, 0.05, gf, 0.98)
    for k, r in restr.items():
        delta = t[k].weight.detach() - before[k]
        if k == zero_key:
            assert not r.basis.active
            s = R.shrink_factors(R.row_alignment(before[k], gr[k]), 0.05)
            assert torch.allclose(t[k].weight, s[:, None] * before[k], atol=1e-6)  # skip_projection
        else:
            assert r.basis.active and r.basis.rank <= 2
            assert torch.allclose(R.project(delta, r.basis.u, r.basis.vh), delta, atol=1e-5)
            assert torch.equal(r.baseline, t[k].weight.detach())


def test_semu_adapter_starts_at_baseline_and_training_stays_in_subspace():
    torch.manual_seed(4)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, _ = R.warm_start(t, gr, 0.05, gf, 0.98)
    rep = R.apply_semu_adapters(m, restr)
    assert rep["adapters"] == 4 and rep["max_rel_error_initial_weight"] < 1e-5
    mods = dict(m.named_modules())
    for p in m.parameters():
        p.requires_grad_(False)
    cores = [mods[k].weight for k in restr]
    for c in cores:
        c.requires_grad_(True)
    checks = R.frozen_checksums(m)
    snap = R.snapshot_trainable(m)
    opt = torch.optim.AdamW(cores, lr=1e-2, weight_decay=0.01)
    x, y = torch.randn(8, 6), torch.randn(8, 6)
    for _ in range(5):
        opt.zero_grad()
        m(x, y).loss.backward()
        opt.step()
    for k, r in restr.items():
        d = mods[k].effective_weight().detach() - r.baseline
        assert d.norm() > 0
        assert torch.allclose(R.project(d, r.basis.u, r.basis.vh), d, atol=1e-5)
    report = R.change_report(m, snap, checks)
    assert report["frozen_tensors_moved_count"] == 0 and report["trainable_entries_changed_fraction"] > 0.9


def test_change_report_flags_a_moved_frozen_tensor():
    m = Toy()
    for p in m.parameters():
        p.requires_grad_(False)
    checks, snap = R.frozen_checksums(m), R.snapshot_trainable(m)
    with torch.no_grad():
        m.lm_head.weight.add_(1.0)
    assert R.change_report(m, snap, checks)["frozen_tensors_moved"] == ["lm_head.weight"]



class TiedToy(Toy):
    """Toy whose head shares the input-embedding matrix, as Llama-3.2-1B."""

    def __init__(self, d=6, vocab=6):
        super().__init__(d)
        self.embed = nn.Embedding(vocab, d)
        self.lm_head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


def test_untie_gives_the_head_an_equal_separate_copy():
    m = TiedToy()
    rep = R.untie_output_embedding(m)
    assert rep == {"was_tied": True, "tied_after": False, "equal_values": True}
    assert m.config.tie_word_embeddings is False
    emb = m.embed.weight.detach().clone()
    with torch.no_grad():
        m.lm_head.weight.mul_(0.5)
    assert torch.equal(m.embed.weight, emb)


def test_restricted_head_outside_restriction_gets_raw_shrink_and_no_adapter():
    torch.manual_seed(5)
    m = Toy()
    t = R.select_targets(m, "mlp_head")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 2) @ torch.randn(2, 6) for k in t if k != "lm_head"}
    restr, rep = R.warm_start(t, gr, 0.1, gf, 0.97, unrestricted={"lm_head"})
    s = R.shrink_factors(R.row_alignment(before["lm_head"], gr["lm_head"]), 0.1)
    assert torch.allclose(m.lm_head.weight, s[:, None] * before["lm_head"], atol=1e-6)
    assert not restr["lm_head"].basis.active and rep["lm_head"]["reason"] == "parameter_unrestricted"
    out = R.apply_semu_adapters(m, restr)
    assert out["adapters"] == 2 and type(m.lm_head) is nn.Linear


def test_piecewise_schedule_points():
    pts = R.parse_points("0:0,1:3e-5,5:1e-5,10:1e-5")
    assert pts == [(0.0, 0.0), (1.0, 3e-5), (5.0, 1e-5), (10.0, 1e-5)]
    f = lambda x: R.piecewise_linear(pts, x)  # noqa: E731
    assert f(0.0) == 0.0 and f(0.5) == pytest.approx(1.5e-5) and f(1.0) == pytest.approx(3e-5)
    assert f(3.0) == pytest.approx(2e-5) and f(5.0) == pytest.approx(1e-5)
    assert f(7.5) == pytest.approx(1e-5) and f(12.0) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        R.parse_points("0:0,0:1")


def test_drift_report_norm_ratio_and_cosine():
    m = Toy()
    t = R.select_targets(m, "mlp")  # down_proj in the toy
    ref = {k: v.weight.detach().clone() for k, v in t.items()}
    with torch.no_grad():
        for v in t.values():
            v.weight.mul_(0.5)
    rep = R.drift_report(m, ref, ("down_proj",))
    assert rep["down_proj"]["norm_ratio"] == pytest.approx(0.5)
    assert rep["down_proj"]["cosine"] == pytest.approx(1.0)


def test_target_blocks_filter():
    m = Toy(n=3)
    assert R.parse_blocks("all") is None and R.parse_blocks("1-2") == {1, 2} and R.parse_blocks("0,2") == {0, 2}
    got = set(R.select_targets(m, "decoder_linear", blocks=R.parse_blocks("1-2")))
    assert got == {f"model.layers.{i}.{p}" for i in (1, 2) for p in ("q_proj", "down_proj")}


def test_full_data_source_is_example_weighted_mean():
    gf, gr = torch.ones(2, 2), torch.zeros(2, 2)
    assert torch.allclose(R.mix_sources(gf, gr, 1, 3), torch.full((2, 2), 0.25))
    with pytest.raises(ValueError):
        R.mix_sources(gf, gr, 0, 3)


def test_subspace_alignment_matches_classification_formula():
    torch.manual_seed(6)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    src = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, _ = R.warm_start(t, gr, 0.2, src, 0.98, alignment_scope="restriction_subspace")
    for k, r in restr.items():
        pw = R.project(before[k], r.basis.u, r.basis.vh)
        s = R.shrink_factors(R.row_alignment(pw, R.project(gr[k], r.basis.u, r.basis.vh)), 0.2)
        expected = before[k] + R.project((s - 1.0)[:, None] * pw, r.basis.u, r.basis.vh)
        assert torch.allclose(t[k].weight, expected, atol=1e-5)


def test_subspace_norm_layer_floors():
    torch.manual_seed(7)
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: v.weight.detach().clone() for k, v in t.items()}  # cos(W, -W) = -1: every row at its floor
    src = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, rep = R.warm_start(t, gr, 0.3, src, 0.98, layer_scaling="subspace_norm")
    rel = {k: rep[k]["subspace_rel_norm"] for k in t}
    top = max(rel.values())
    for k, r in restr.items():
        floor = 1.0 - 0.7 * rel[k] / top
        assert rep[k]["floor"] == pytest.approx(floor)
        expected = before[k] + (floor - 1.0) * R.project(before[k], r.basis.u, r.basis.vh)
        assert torch.allclose(t[k].weight, expected, atol=1e-5)
    assert min(rep[k]["floor"] for k in t) == pytest.approx(0.3)
    with pytest.raises(ValueError):  # scaling needs a restriction
        R.warm_start(R.select_targets(Toy(), "mlp"), {k: torch.zeros(6, 6) for k in R.select_targets(Toy(), "mlp")},
                     0.3, layer_scaling="subspace_norm")


def test_subspace_norm_mean_keeps_average_shrink():
    torch.manual_seed(8)
    m = Toy(n=4)
    t = R.select_targets(m, "decoder_linear")
    gr = {k: v.weight.detach().clone() for k, v in t.items()}  # every row at its floor
    src = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.98, layer_scaling="subspace_norm_mean")
    floors = [rep[k]["floor"] for k in t]
    assert all(0.0 <= f <= 1.0 for f in floors)
    if all(0.0 < f < 1.0 for f in floors):  # no clipping: the mean shrink amount is exactly 1 - s_min
        assert sum(1.0 - f for f in floors) / len(floors) == pytest.approx(0.7)
    rel = {k: rep[k]["subspace_rel_norm"] for k in t}
    mean_r = sum(rel.values()) / len(rel)
    for k in t:
        assert rep[k]["floor"] == pytest.approx(min(1.0, max(0.0, 1.0 - 0.7 * rel[k] / mean_r)))


def test_free_parameters_select_groups_and_skip_adapters():
    torch.manual_seed(9)
    m = Toy()
    t = R.select_targets(m, "mlp_down")  # down_proj gets an adapter, q_proj stays nn.Linear
    gr = {k: torch.randn_like(v.weight) for k, v in t.items()}
    gf = {k: torch.randn(6, 3) @ torch.randn(3, 6) for k in t}
    restr, _ = R.warm_start(t, gr, 0.3, gf, 0.98)
    R.apply_semu_adapters(m, restr)
    assert R.free_parameters(m, "none") == {}
    norms = R.free_parameters(m, "norms")
    assert set(norms) == {"norms"} and set(norms["norms"]) == {"model.norm.weight", "model.norm.bias"}
    assert norms["norms"]["model.norm.weight"] is m.model.norm.weight
    assert set(R.free_parameters(m, "attention")["attention"]) == {f"model.layers.{i}.q_proj.weight" for i in range(2)}
    every = R.free_parameters(m, "all")
    assert set(every) == {"norms", "attention"}  # the only MLP projection carries an adapter
    assert set(R.free_parameters(m, "norms,attention")) == {"norms", "attention"}
    with pytest.raises(ValueError):  # a named group with nothing to train
        R.free_parameters(m, "mlp")
    with pytest.raises(ValueError):
        R.free_parameters(m, "heads")


def test_piecewise_schedule_step():
    pts = R.parse_points("0:0,1:3e-5,5:3e-5,5:1e-5,10:1e-5")
    f = lambda x: R.piecewise_linear(pts, x)  # noqa: E731
    assert f(0.0) == 0.0 and f(3.0) == pytest.approx(3e-5) and f(4.999) == pytest.approx(3e-5)
    assert f(5.0) == pytest.approx(1e-5) and f(7.0) == pytest.approx(1e-5) and f(10.0) == pytest.approx(1e-5)
    with pytest.raises(ValueError):  # an x three times
        R.parse_points("0:0,5:1,5:2,5:3,10:1")
    with pytest.raises(ValueError):  # decreasing x
        R.parse_points("0:0,5:1,4:1")


class SeqToy(nn.Module):
    """Token sequences (batch, time, d) through one block's down_proj."""

    def __init__(self, d=6):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block(d)])

    def forward(self, input_ids, attention_mask=None):
        return self.model.layers[0].down_proj(input_ids)


def test_retain_input_covariance_is_answer_token_masked():
    torch.manual_seed(10)
    m = SeqToy()
    t = R.select_targets(m, "mlp_down")
    x = torch.randn(3, 5, 6)
    labels = torch.full((3, 5), -100)
    labels[:2, 2:] = 1  # answer tokens: positions 2-4 of the first two sequences
    batches = [{"input_ids": x[:2], "attention_mask": torch.ones(2, 5), "labels": labels[:2]},
               {"input_ids": x[2:], "attention_mask": torch.ones(1, 5), "labels": labels[2:]}]
    cov, n = R.collect_input_covariances(m, batches, t, lambda b: b)
    sel = x[labels != -100]
    assert n == sel.shape[0] == 6
    assert torch.allclose(cov["model.layers.0.down_proj"], sel.T @ sel / 6, atol=1e-5)
    assert m.training  # mode restored


def test_ridge_input_filter_weights_eigen_directions():
    lam = torch.tensor([4.0, 1.0, 0.0, 0.0])
    q, _ = torch.linalg.qr(torch.randn(4, 4, generator=torch.Generator().manual_seed(11)))
    cov = q @ torch.diag(lam) @ q.T
    m, mu = R.ridge_input_filter(cov, 1.0)
    assert mu == pytest.approx(5.0 / 4)
    assert torch.allclose(m, q @ torch.diag(mu / (lam + mu)) @ q.T, atol=1e-5)
    big, _ = R.ridge_input_filter(cov, 1e6)  # kappa -> inf: no filter
    assert torch.allclose(big, torch.eye(4), atol=1e-5)
    with pytest.raises(ValueError):
        R.ridge_input_filter(cov, 0.0)


def test_restricted_warm_start_input_filter_right_multiplies_delta():
    torch.manual_seed(12)
    m1, m2 = Toy(), Toy()
    m2.load_state_dict(m1.state_dict())
    t1, t2 = R.select_targets(m1, "decoder_linear"), R.select_targets(m2, "decoder_linear")
    before = {k: v.weight.detach().clone() for k, v in t1.items()}
    gr = {k: torch.randn(6, 6) for k in t1}
    gf = {k: torch.randn(6, 2) @ torch.randn(2, 6) for k in t1}
    filters = {k: R.ridge_input_filter(torch.diag(torch.rand(6) * 3), 1.0)[0] for k in t1}
    R.warm_start(t1, gr, 0.3, gf, 0.98)
    _, rep = R.warm_start(t2, gr, 0.3, gf, 0.98, input_filters=filters)
    for k in t1:
        d1 = t1[k].weight.detach() - before[k]
        d2 = t2[k].weight.detach() - before[k]
        assert torch.allclose(d2, d1 @ filters[k], atol=1e-5)
        assert 0.0 < rep[k]["retain_filter_kept"] <= 1.0 + 1e-6
    with pytest.raises(ValueError):  # a filter needs a restriction
        R.warm_start(R.select_targets(Toy(), "mlp_down"), {k: torch.zeros(6, 6) for k in R.select_targets(Toy(), "mlp_down")},
                     0.3, input_filters={})


def test_whitened_basis_with_isotropic_retain_equals_plain_basis():
    torch.manual_seed(13)
    g = torch.randn(5, 3) @ torch.randn(3, 6)
    plain = R.svd_basis(g, 0.9)
    white = R.whitened_svd_basis(g, torch.eye(6) * 2.0, 0.9, rho=0.5)
    assert white.rank == plain.rank
    assert torch.allclose(white.u @ white.u.T, plain.u @ plain.u.T, atol=1e-5)
    assert torch.allclose(white.vh.T @ white.vh, plain.vh.T @ plain.vh, atol=1e-5)
    assert torch.allclose(white.vh @ white.vh.T, torch.eye(white.rank), atol=1e-5)  # orthonormal covectors


def test_whitened_basis_prefers_directions_retain_rarely_uses():
    # Equal forget-gradient energy on inputs e0 and e1; retain uses e0 a hundred times more.
    u = torch.linalg.qr(torch.randn(3, 2, generator=torch.Generator().manual_seed(14)))[0]
    e = torch.eye(4)
    g = u[:, :1] @ e[:1] + u[:, 1:] @ e[1:2]
    cov = torch.diag(torch.tensor([100.0, 1.0, 1.0, 1.0]))
    assert R.svd_basis(g, 0.9).rank == 2
    white = R.whitened_svd_basis(g, cov, 0.9, rho=0.01)
    assert white.rank == 1
    assert torch.allclose(white.vh.abs(), e[1:2], atol=1e-4)
    t = R.select_targets(Toy(), "mlp_down")
    with pytest.raises(ValueError):  # whitening needs its rho
        R.warm_start(t, {k: torch.zeros(6, 6) for k in t}, 0.3, {k: torch.randn(6, 6) for k in t}, 0.9,
                     whitening={k: torch.eye(6) for k in t})


def test_explicit_rank_budget_matches_plain_and_whitened_bases():
    torch.manual_seed(15)
    g = torch.randn(6, 6)
    plain = R.svd_basis(g, 0.6)
    fixed = R.svd_basis(g, 0.6, rank=plain.rank + 2)
    assert fixed.rank == plain.rank + 2 and fixed.evr > plain.evr
    assert R.svd_basis(g, 0.6, rank=50).rank == 6  # clipped to the total rank
    m = Toy()
    t = R.select_targets(m, "decoder_linear")
    gr = {k: torch.randn(6, 6) for k in t}
    src = {k: torch.randn(6, 6) for k in t}
    cov = {k: torch.diag(torch.rand(6) + 0.1) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.6, rank_multiplier=1.5)
    m2 = Toy()
    t2 = R.select_targets(m2, "decoder_linear")
    _, rep2 = R.warm_start(t2, gr, 0.3, src, 0.6, whitening=cov, whitening_rho=1.0, rank_multiplier=1.5)
    for k in t:
        r0 = R.svd_basis(src[k], 0.6).rank
        assert rep[k]["reference_rank"] == rep2[k]["reference_rank"] == r0
        assert rep[k]["rank"] == rep2[k]["rank"] == min(6, max(1, round(1.5 * r0)))
    with pytest.raises(ValueError):  # a budget needs a restriction
        R.warm_start(R.select_targets(Toy(), "mlp_down"), {k: torch.zeros(6, 6) for k in R.select_targets(Toy(), "mlp_down")},
                     0.3, rank_multiplier=1.5)


def test_one_sided_projection_and_adapter():
    torch.manual_seed(16)
    vh = torch.linalg.qr(torch.randn(6, 2))[0].T
    d = torch.randn(4, 6)
    assert torch.allclose(R.project(d, None, vh), d @ vh.T @ vh, atol=1e-6)
    basis = R.Basis(True, "", None, vh, 2, 4, 1.0)
    w_star = torch.randn(4, 6)
    core, anchor = R.adapter_core_and_anchor(w_star, basis)
    adapter = R.SemuAdapterLinear(6, 4, core, anchor, None, vh)
    assert torch.allclose(adapter.effective_weight(), w_star, atol=1e-5)
    opt = torch.optim.SGD([adapter.weight], lr=0.1)
    adapter(torch.randn(8, 6)).square().sum().backward()
    opt.step()
    moved = adapter.effective_weight().detach() - w_star
    assert moved.norm() > 0 and torch.allclose(moved @ vh.T @ vh, moved, atol=1e-5)  # rows stay in span(vh)


def test_representation_basis_picks_forget_heavy_inputs():
    e = torch.eye(4)
    cov_forget = torch.diag(torch.tensor([1.0, 5.0, 1.0, 0.1]))
    cov_retain = torch.diag(torch.tensor([5.0, 1.0, 1.0, 0.1]))
    g = torch.randn(3, 4, generator=torch.Generator().manual_seed(17))
    w = torch.randn(3, 4, generator=torch.Generator().manual_seed(18))
    for side in ("identity", "gradient", "representation"):
        b = R.representation_basis(g, cov_forget, cov_retain, w, rank=1, rho=0.01, output_side=side)
        assert b.active and b.rank == 1 and torch.allclose(b.vh.abs(), e[1:2], atol=1e-4)
        assert (b.u is None) == (side == "identity")
        if b.u is not None:
            assert torch.allclose(b.u.T @ b.u, torch.eye(b.u.shape[1]), atol=1e-5)
        assert 0.0 < b.evr <= 1.0 + 1e-6
    g1 = R.representation_basis(g, cov_forget, cov_retain, w, rank=1, rho=0.01, output_side="gradient")
    target = g @ e[1]
    assert torch.allclose((g1.u[:, 0] @ target).abs(), target.norm(), atol=1e-4)  # u along G_f v
    with pytest.raises(ValueError):
        R.representation_basis(g, cov_forget, cov_retain, w, rank=1, rho=0.01, output_side="both")


def test_combined_bases_reduce_to_c_and_d():
    gen = torch.Generator().manual_seed(19)
    g = torch.randn(5, 3, generator=gen) @ torch.randn(3, 6, generator=gen)
    x_r, x_f = torch.randn(40, 6, generator=gen), torch.randn(40, 6, generator=gen) * torch.tensor([1, 3, 1, 1, 2, 1.0])
    cov_r, cov_f = x_r.T @ x_r / 40, x_f.T @ x_f / 40
    w = torch.randn(5, 6, generator=gen)
    proj = lambda m: m @ m.T  # noqa: E731
    c = R.whitened_svd_basis(g, cov_r, 0.9, rho=0.5, rank=2)
    # beta 0 = the gradient contrast: the input side of C; with the whitened-gradient output side it is C
    seq0 = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="whitened_gradient", input_mix=0.0)
    assert torch.allclose(proj(seq0.vh.T), proj(c.vh.T), atol=1e-4)
    assert torch.allclose(proj(seq0.u), proj(c.u), atol=1e-4)
    joint0 = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=0.0)
    assert torch.allclose(proj(joint0.u), proj(c.u), atol=1e-4)  # span(G_f V) = span of C's U
    # beta 1 = D; scaling the forget covariance changes nothing
    d = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient")
    d2 = R.representation_basis(g, 7.0 * cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=1.0)
    assert torch.allclose(proj(d.vh.T), proj(d2.vh.T), atol=1e-5)
    mid = R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=0.5)
    assert mid.active and mid.vh.shape == (2, 6) and 0.0 < mid.evr <= 1.0 + 1e-6
    with pytest.raises(ValueError):
        R.representation_basis(g, cov_f, cov_r, w, rank=2, rho=0.5, output_side="gradient", input_mix=1.5)


def test_v_proj_and_head_profiles():
    m = Toy()
    assert set(R.select_targets(m, "head")) == {"lm_head"}
    assert set(R.select_targets(m, "down_up_v")) == {f"model.layers.{i}.down_proj" for i in range(2)}  # no up/v
    assert set(R.select_targets(m, "down_up_o_v")) == {f"model.layers.{i}.down_proj" for i in range(2)}
    assert set(R.select_targets(m, "attn_q")) == {f"model.layers.{i}.q_proj" for i in range(2)}  # one type only
    with pytest.raises(ValueError):
        R.select_targets(m, "attn_k")


def test_basis_prefix_matches_direct_bases():
    gen = torch.Generator().manual_seed(20)
    g = torch.randn(5, 7, generator=gen)
    x_r, x_f = torch.randn(60, 7, generator=gen), torch.randn(60, 7, generator=gen) * torch.linspace(0.5, 3.0, 7)
    cov_r, cov_f = x_r.T @ x_r / 60, x_f.T @ x_f / 60
    w = torch.randn(5, 7, generator=gen)
    proj = lambda m: m @ m.T  # noqa: E731
    families = {
        "plain": (R.svd_basis(g, 0.9, rank=5), lambda r: R.svd_basis(g, 0.9, rank=r), False),
        "C": (R.whitened_svd_basis(g, cov_r, 0.9, rho=0.5, rank=5),
              lambda r: R.whitened_svd_basis(g, cov_r, 0.9, rho=0.5, rank=r), False),
        "D": (R.representation_basis(g, cov_f, cov_r, w, rank=5, rho=0.5, output_side="gradient"),
              lambda r: R.representation_basis(g, cov_f, cov_r, w, rank=r, rho=0.5, output_side="gradient"), True),
    }
    for name, (full, direct_at, gradient_output) in families.items():
        for r in (1, 2, 3):
            pre, direct = R.basis_prefix(full, r, g, gradient_output), direct_at(r)
            assert pre.rank == r, name
            assert torch.allclose(proj(pre.vh.T), proj(direct.vh.T), atol=1e-4), name
            assert torch.allclose(proj(pre.u), proj(direct.u), atol=1e-4), name
            kept = R.project(g, direct.u, direct.vh).square().sum() / g.square().sum()
            assert abs(pre.evr - float(kept)) < 1e-4, name


def test_forget_output_budget_picks_smallest_rank_reaching_target():
    torch.manual_seed(21)
    gen = torch.Generator().manual_seed(21)
    t = R.select_targets(Toy(), "decoder_linear")
    w0 = {k: v.weight.detach().clone() for k, v in t.items()}
    gr = {k: torch.randn(6, 6, generator=gen) for k in t}
    src = {k: torch.randn(6, 6, generator=gen) for k in t}
    xs = {k: torch.randn(50, 6, generator=gen) for k in t}
    cov_f = {k: x.T @ x / 50 for k, x in xs.items()}
    cov_r = {k: torch.diag(torch.rand(6, generator=gen) + 0.1) for k in t}
    _, rep = R.warm_start(t, gr, 0.3, src, 0.9, forget_covariances=cov_f, forget_output_target=0.1)
    for k in t:
        q, r = rep[k]["forget_output_change"], rep[k]["rank"]
        assert abs(R.forget_output_change(t[k].weight.detach() - w0[k], w0[k], cov_f[k]) - q) < 1e-5
        assert q >= 0.1 or r == 6
        assert rep[k]["reference_rank"] == R.svd_basis(src[k], 0.9).rank
        if r > 1:  # one rank less misses the target
            proposal = (R.shrink_factors(R.row_alignment(w0[k], gr[k]), 0.3) - 1.0).unsqueeze(1) * w0[k]
            below = R.svd_basis(src[k], 0.9, rank=r - 1)
            assert R.forget_output_change(R.project(proposal, below.u, below.vh), w0[k], cov_f[k]) < 0.1
    t2 = R.select_targets(Toy(), "decoder_linear")
    _, rep2 = R.warm_start(t2, gr, 0.3, src, 0.9, whitening_rho=0.5, representation_output="gradient",
                           forget_covariances=cov_f, retain_covariances=cov_r, forget_output_target=0.1)
    assert all(rep2[k]["forget_output_change"] >= 0.1 or rep2[k]["rank"] == 6 for k in t2)
    for bad in (dict(rank_multiplier=1.5, forget_covariances=cov_f), dict(), dict(forget_covariances=cov_f,
                                                                              layer_scaling="subspace_norm")):
        with pytest.raises(ValueError):
            R.warm_start(R.select_targets(Toy(), "decoder_linear"), gr, 0.3, src, 0.9, forget_output_target=0.1, **bad)
