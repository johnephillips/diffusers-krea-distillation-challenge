# This file is autogenerated by the command `make fix-copies`, do not edit.
# flake8: noqa

from ..utils import DummyObject, requires_backends


class FlaxPNDMScheduler(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])


class FlaxUNet2DConditionModel(metaclass=DummyObject):
    _backends = ["flax"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])
