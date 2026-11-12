import re
import numpy as np
import pytest
import torch
from hydra import initialize_config_module, compose
from omegaconf import OmegaConf
from wetw import WETW, WESee, SearchIndex
from wetw.data import distance, znorm, cross_distances, Targets, prepare_npz
from wetw.losses import log_ranking


@pytest.fixture
def cfg():
    with initialize_config_module(config_module="wetw.conf", version_base="1.3"):
        c = compose(config_name="config", overrides=["study=smoke"])
    c.dataset.dim_series = 32
    c.encoder_model.num_en_resblock = 2
    c.encoder_model.multi_temporal_head_kernels = [3, 5, 7]
    c.encoder_model.time_fusion_pool_target = 16
    c.system.device = "cpu"
    return c


def test_distance_properties():
    torch.manual_seed(1)
    h = WETW(8)
    with torch.no_grad():
        h.net[-1].weight.normal_(0, 0.1)
    h.eval()
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    torch.testing.assert_close(h(a, b), h(b, a))
    assert torch.all(h(a, b) >= 0) and torch.equal(h(a, a), torch.zeros(5))
    assert (h.weights(a, b) > 0).all()


def test_ranking_scale_invariance():
    a, b = torch.tensor([1.0, 2.0]), torch.tensor([1.2, 1.5])
    torch.testing.assert_close(log_ranking(a, b), log_ranking(20 * a, 20 * b))


@pytest.mark.parametrize("variant", ["full", "sfe"])
def test_encoder_shapes_and_gradients(cfg, variant):
    torch.set_num_threads(1)
    model = WESee(cfg, variant)
    for n in [1, 3]:
        z = model(torch.randn(n, 32))
        assert z.shape == (n, 128) and torch.isfinite(z).all()
    (z * torch.randn_like(z)).sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_exact_fallback(cfg):
    torch.set_num_threads(1)
    rng = np.random.RandomState(7)
    db = znorm(rng.randn(15, 32))
    q = znorm(rng.randn(1, 32))[0]
    index = SearchIndex(WESee(cfg), WETW(128), db, normalized=True)
    result = index.query(q, k=4, rho=1, normalized=True)
    expected = cross_distances(q[None], db)[0]
    np.testing.assert_array_equal(
        result.indices, np.argsort(expected, kind="stable")[:4]
    )
    np.testing.assert_allclose(result.distances, expected[result.indices])


def test_cache_and_zero_window():
    x = znorm(np.random.RandomState(2).randn(5, 16))
    t = Targets(x)
    values = t.pairs([0, 1], [1, 0])
    assert values[0] == values[1] and len(t.cache) == 1
    assert distance(x[0], x[1], 0) == pytest.approx(
        np.linalg.norm(x[0] - x[1]) / 4, rel=1e-6
    )


def test_split_overlap_rejected(tmp_path):
    arrays = {
        s: np.ones((2, 8)) for s in ["train", "valid", "test", "database", "queries"]
    }
    arrays["train_ids"] = np.array([1, 2])
    arrays["valid_ids"] = np.array([2, 3])
    np.savez(tmp_path / "data.npz", **arrays)
    with pytest.raises(ValueError, match="Overlapping"):
        prepare_npz(tmp_path / "data.npz", tmp_path / "out")


def test_paper_presets_compose():
    names = [
        "whalesounds",
        "cornellwhalechallenge",
        "audiomnistds",
        "dreamera",
        "lendb",
        "w01_seismic",
        "mosquitosound",
        "fruitflies",
    ]
    for name in names:
        with initialize_config_module(config_module="wetw.conf", version_base="1.3"):
            c = compose(config_name="config", overrides=[f"dataset={name}"])
        assert c.release.dataset == name and c.encoder_model.dim_emb == 128
        assert c.dataset.refine.steps in [30000, 100000] and not c.smoke
        assert not re.search(r"/(lustre|data/users|home)/", OmegaConf.to_yaml(c))


def test_generator_parameter_count():
    assert sum(p.numel() for p in WETW(128).parameters()) == 65793


def test_scalability_presets_compose():
    from importlib.resources import files

    paths = list(files("wetw.conf").joinpath("dataset").iterdir())
    for path in paths:
        if not path.name.startswith("audiomnist_L"):
            continue
        with initialize_config_module(config_module="wetw.conf", version_base="1.3"):
            c = compose(config_name="config", overrides=[f"dataset={path.name[:-5]}"])
        assert c.reference_layout and c.release.dataset.startswith("audiomnist_L")


def test_lowrank_head_defaults_to_full_rank():
    with initialize_config_module(config_module="wetw.conf", version_base="1.3"):
        c = compose(config_name="config", overrides=["dataset=cornellwhalechallenge", "head=lowrank"])
    head = WETW(128, kind="lowrank", rank=int(c.head_rank or 128))
    assert head.projection.shape == (128, 128)
    assert c.dataset.dim_series == 4000
