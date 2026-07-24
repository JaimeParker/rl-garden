from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from robot_infra.teleop.utils.telo_op_control_twist import EETwistTeleOpWrapper


def _pico_sample() -> list[float]:
    data = [0.0] * 13
    data[9] = 1.0  # valid xyzw quaternion w component
    return data


class _FakeAgain(Exception):
    pass


class _FakeSocket:
    def __init__(self, samples):
        self._samples = list(samples)
        self.connected_url = None

    def connect(self, url):
        self.connected_url = url

    def setsockopt_string(self, *args):
        del args

    def recv_json(self, flags=None):
        del flags
        if not self._samples:
            raise _FakeAgain()
        sample = self._samples.pop(0)
        if isinstance(sample, BaseException):
            raise sample
        return sample

    def close(self):
        pass


class _FakeContext:
    def __init__(self, socket):
        self._socket = socket

    def socket(self, kind):
        del kind
        return self._socket

    def term(self):
        pass


def _install_fake_zmq(monkeypatch, samples):
    socket = _FakeSocket(samples)
    module = types.SimpleNamespace(
        SUB=1,
        SUBSCRIBE="",
        NOBLOCK=1,
        Again=_FakeAgain,
        Context=lambda: _FakeContext(socket),
    )
    monkeypatch.setitem(sys.modules, "zmq", module)
    return socket


def test_pico_init_waits_for_first_sample_and_reuses_it(monkeypatch):
    _install_fake_zmq(monkeypatch, [_FakeAgain(), _pico_sample()])

    teleop = EETwistTeleOpWrapper(init_timeout_s=0.1)
    sample = teleop.poll()

    assert teleop.last_received is True
    assert sample.action.shape == (7,)
    np.testing.assert_allclose(sample.twist, np.zeros(6, dtype=np.float32))
    teleop.close()


def test_pico_init_timeout_raises_when_no_sample_arrives(monkeypatch):
    _install_fake_zmq(monkeypatch, [])

    with pytest.raises(TimeoutError, match="Timed out waiting for pico"):
        EETwistTeleOpWrapper(init_timeout_s=0.01)


def test_make_teleop_source_constructs_pico(monkeypatch):
    _install_fake_zmq(monkeypatch, [_pico_sample()])

    from robot_infra.teleop.source import make_teleop_source

    teleop = make_teleop_source(device="pico", init_timeout_s=0.1)

    assert isinstance(teleop, EETwistTeleOpWrapper)
    teleop.close()


def test_make_teleop_source_constructs_spacemouse_without_pico_kwargs(monkeypatch):
    captured = {}

    class _FakeSpaceMouse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import robot_infra.teleop.spacemouse as spacemouse

    monkeypatch.setattr(spacemouse, "SpaceMouseTeleOpWrapper", _FakeSpaceMouse)

    from robot_infra.teleop.source import make_teleop_source

    teleop = make_teleop_source(
        device="spacemouse",
        zmq_url="tcp://unused:7777",
        hand="left",
        pos_scale=99.0,
        rot_scale=99.0,
        twist_limit=99.0,
        spacemouse_index=2,
    )

    assert isinstance(teleop, _FakeSpaceMouse)
    assert captured == {"intervention_threshold": 1e-4, "spacemouse_index": 2}


def test_make_teleop_source_rejects_unknown_device():
    from robot_infra.teleop.source import make_teleop_source

    with pytest.raises(ValueError, match="device must be"):
        make_teleop_source(device="unknown")
