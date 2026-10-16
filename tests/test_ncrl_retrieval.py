"""NCRL retrieval: kNN order, action-dim filter, de-duplication across
iterations, the iteration cap, an under-filled DB, and the DB build tool on a
synthetic converted dataset (embedding = encoder(first frame))."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from _ncrl_online_synthetic import EMBODIMENTS, make_pretrain_artifacts

from rl_garden.algorithms.ncrl import (
    RETRIEVAL_DB_FORMAT,
    encoder_fingerprint,
    load_retrieval_db,
    load_rssm_pretrain_checkpoint,
    retrieval_embedding,
    retrieve_nearest_episodes,
    rssm_from_pretrain_checkpoint,
)


def _line_db(n: int, action_dims: list[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry ``i`` sits at ``(i, 0)``: distance from the origin is ``i``."""
    embedding = torch.zeros(n, 2)
    embedding[:, 0] = torch.arange(n, dtype=torch.float32)
    dims = torch.tensor(action_dims if action_dims is not None else [6] * n)
    return embedding, dims


def _fixed_query(point=(0.0, 0.0)):
    calls = []

    def query():
        calls.append(1)
        return torch.tensor(point)

    return query, calls


def test_knn_returns_nearest_first():
    embedding, dims = _line_db(10)
    embedding = embedding[torch.randperm(10)]
    query, _ = _fixed_query((3.2, 0.0))
    selected, iterations = retrieve_nearest_episodes(query, embedding, dims, 6, num_episodes=4)
    distances = (embedding[selected, 0] - 3.2).abs().tolist()
    assert distances == sorted(distances)
    assert sorted(embedding[selected, 0].tolist()) == [2.0, 3.0, 4.0, 5.0]
    assert iterations == 1


def test_action_dim_filter():
    embedding, dims = _line_db(8, [6, 12, 6, 12, 6, 12, 6, 12])
    query, _ = _fixed_query()
    selected, _ = retrieve_nearest_episodes(query, embedding, dims, 12, num_episodes=3)
    assert selected == [1, 3, 5]
    with pytest.raises(ValueError, match="action dim 4"):
        retrieve_nearest_episodes(query, embedding, dims, 4, num_episodes=3)


def test_deduplicated_accumulation_across_iterations():
    embedding, dims = _line_db(20)
    queries = iter([torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([19.0, 0.0])])
    selected, iterations = retrieve_nearest_episodes(
        lambda: next(queries), embedding, dims, 6, num_episodes=9, per_iteration=3
    )
    assert iterations == 3
    assert len(selected) == len(set(selected)) == 9
    # Same query twice: the second pass takes the next-nearest unseen entries.
    assert selected[:6] == [0, 1, 2, 3, 4, 5]
    assert selected[6:] == [19, 18, 17]


def test_iteration_cap_stops_an_unfillable_request():
    embedding, dims = _line_db(100)
    query, calls = _fixed_query()
    selected, iterations = retrieve_nearest_episodes(
        query, embedding, dims, 6, num_episodes=50, max_iterations=10, per_iteration=2
    )
    assert iterations == 10 and len(calls) == 10
    assert len(selected) == len(set(selected)) == 20


def test_too_few_eligible_entries_stops_when_exhausted():
    embedding, dims = _line_db(6, [6, 6, 6, 12, 12, 12])
    query, calls = _fixed_query()
    selected, iterations = retrieve_nearest_episodes(query, embedding, dims, 6, num_episodes=500)
    assert sorted(selected) == [0, 1, 2]
    assert iterations == 1 and len(calls) == 1


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    return make_pretrain_artifacts(tmp_path_factory.mktemp("ncrl_retrieval"))


def test_build_retrieval_db_embeds_first_frames(artifacts):
    db = load_retrieval_db(artifacts["retrieval_db"])
    num_episodes = sum(len(lengths) for _, lengths in EMBODIMENTS.values())
    assert db["metadata"]["format"] == RETRIEVAL_DB_FORMAT
    assert db["embedding"].dtype == torch.float16 and db["embedding"].shape[0] == num_episodes
    assert db["embodiment"] == ["quadruped"] * 2 + ["walker"] * 4  # part directories sorted by name
    assert db["episode_id"].tolist() == [0, 1, 0, 1, 2, 3]
    assert db["action_dim"].tolist() == [12, 12, 6, 6, 6, 6]
    assert db["episode_length"].tolist() == [15, 13, 14, 16, 12, 15]
    assert db["metadata"]["padded_action_dim"] == 12

    world_model = rssm_from_pretrain_checkpoint(load_rssm_pretrain_checkpoint(artifacts["checkpoint"]))
    assert db["metadata"]["encoder_fingerprint"] == encoder_fingerprint(world_model.encoder)
    obs = np.load(artifacts["dataset_dir"] / "walker" / "obs.npy", mmap_mode="r")
    second_walker_first_frame = torch.from_numpy(np.array(obs[14:15]))  # walker episode 1 starts at step 14
    expected = retrieval_embedding(world_model, {"rgb_default": second_walker_first_frame})
    torch.testing.assert_close(db["embedding"][3:4].float(), expected, atol=1e-2, rtol=1e-2)
