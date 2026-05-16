import numpy as np
import pytest
import tensorflow as tf

import sys
sys.path.insert(0, "src")

from models.two_tower import build_two_tower_model


NUM_USERS = 100
NUM_ITEMS = 500
USER_FEAT_DIM = 22   # 4 base features + 18 genre preference features
ITEM_FEAT_DIM = 21   # 3 item features + 18 genre flags
EMBEDDING_DIM = 16
BATCH = 8


@pytest.fixture(scope="module")
def model_and_towers():
    return build_two_tower_model(NUM_USERS, NUM_ITEMS, USER_FEAT_DIM, ITEM_FEAT_DIM, EMBEDDING_DIM)


def test_model_builds(model_and_towers):
    model, user_tower, item_tower = model_and_towers
    assert model is not None
    assert user_tower is not None
    assert item_tower is not None


def test_forward_pass_output_shape(model_and_towers):
    model, _, _ = model_and_towers
    inputs = {
        "user_id": np.random.randint(0, NUM_USERS, (BATCH, 1)),
        "user_features": np.random.rand(BATCH, USER_FEAT_DIM).astype(np.float32),
        "item_id": np.random.randint(0, NUM_ITEMS, (BATCH, 1)),
        "item_features": np.random.rand(BATCH, ITEM_FEAT_DIM).astype(np.float32),
    }
    out = model(inputs, training=False)
    assert out.shape == (BATCH, 1)


def test_user_tower_output_is_unit_norm(model_and_towers):
    _, user_tower, _ = model_and_towers
    inputs = {
        "user_id": np.random.randint(0, NUM_USERS, (BATCH, 1)),
        "user_features": np.random.rand(BATCH, USER_FEAT_DIM).astype(np.float32),
    }
    vecs = user_tower(inputs, training=False).numpy()
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(BATCH), atol=1e-5)


def test_item_tower_output_is_unit_norm(model_and_towers):
    _, _, item_tower = model_and_towers
    inputs = {
        "item_id": np.random.randint(0, NUM_ITEMS, (BATCH, 1)),
        "item_features": np.random.rand(BATCH, ITEM_FEAT_DIM).astype(np.float32),
    }
    vecs = item_tower(inputs, training=False).numpy()
    norms = np.linalg.norm(vecs, axis=1)
    np.testing.assert_allclose(norms, np.ones(BATCH), atol=1e-5)
