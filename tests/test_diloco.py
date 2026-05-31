"""Unit tests for the DiLoCo building blocks.

Run from the repo root with the project venv:
    .venv\\Scripts\\python.exe -m pytest tests -q
"""

import numpy as np

from centurion_worker import model_spec, tensor_codec


def test_init_params_matches_spec():
    state = model_spec.init_params(seed=0)
    model_spec.validate(state)  # should not raise
    for name in model_spec.PARAM_NAMES:
        assert state[name].dtype == model_spec.PARAM_DTYPE
        assert tuple(state[name].shape) == model_spec.PARAM_SHAPES[name]


def test_validate_rejects_bad_shape():
    state = model_spec.init_params(seed=0)
    state["fc1.bias"] = np.zeros((model_spec.HIDDEN_DIM + 1,), dtype=np.float32)
    try:
        model_spec.validate(state)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_codec_round_trip_is_exact():
    state = model_spec.init_params(seed=3)
    blob = tensor_codec.state_to_bytes(state)
    restored = tensor_codec.state_from_bytes(blob)
    assert set(restored) == set(state)
    for name in state:
        assert np.array_equal(state[name], restored[name])


def test_average_states_is_elementwise_mean():
    a = model_spec.init_params(seed=1)
    b = model_spec.init_params(seed=2)
    avg = tensor_codec.average_states([a, b])
    for name in a:
        expected = ((a[name].astype(np.float64) + b[name].astype(np.float64)) / 2.0)
        np.testing.assert_allclose(avg[name], expected.astype(np.float32), rtol=0, atol=1e-6)


def test_average_single_state_is_identity():
    a = model_spec.init_params(seed=7)
    avg = tensor_codec.average_states([a])
    for name in a:
        assert np.array_equal(avg[name], a[name])


def test_average_rejects_empty():
    try:
        tensor_codec.average_states([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_average_rejects_mismatched_keys():
    a = model_spec.init_params(seed=1)
    b = model_spec.init_params(seed=2)
    del b["fc2.bias"]
    try:
        tensor_codec.average_states([a, b])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_coordinator_barrier_advances_after_world_size_submits():
    from centurion_coord.server import Coordinator

    coord = Coordinator(world_size=2, total_rounds=5)
    assert coord.current_round == 0

    # global[0] is the canonical init
    g0 = tensor_codec.state_from_bytes(coord.global_params(0))
    model_spec.validate(g0)

    s_a = tensor_codec.state_to_bytes(model_spec.init_params(seed=11))
    s_b = tensor_codec.state_to_bytes(model_spec.init_params(seed=22))

    r1 = coord.submit("a", 0, s_a)
    assert r1["advanced"] is False
    assert coord.current_round == 0

    r2 = coord.submit("b", 0, s_b)
    assert r2["advanced"] is True
    assert coord.current_round == 1

    # global[1] must equal the average of the two submissions
    g1 = tensor_codec.state_from_bytes(coord.global_params(1))
    expected = tensor_codec.average_states([
        tensor_codec.state_from_bytes(s_a),
        tensor_codec.state_from_bytes(s_b),
    ])
    for name in expected:
        np.testing.assert_array_equal(g1[name], expected[name])
