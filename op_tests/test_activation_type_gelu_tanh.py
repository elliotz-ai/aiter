# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Reference-level checks for ActivationType.GeluTanh.

These do not launch any GPU kernel, so they run without an MI accelerator.
They only verify the pieces added on the host/Python side: the enum value
resolves, its torch reference implementation matches PyTorch's tanh-GELU
approximation, and the CLI string-parsing helper resolves the multi-word
enum name correctly.
"""

import torch
import torch.nn.functional as F

from aiter import ActivationType
from aiter.ops.quant import get_torch_act
from aiter.utility.dtypes import str2ActivationType


def test_gelu_tanh_enum_value_exists():
    assert ActivationType.GeluTanh is not None
    assert ActivationType.GeluTanh != ActivationType.Gelu
    assert ActivationType.GeluTanh != ActivationType.Silu
    assert ActivationType.GeluTanh != ActivationType.Swiglu


def test_gelu_tanh_torch_reference_matches_pytorch_tanh_approx():
    x = torch.randn(1024, dtype=torch.float32)
    act = get_torch_act(ActivationType.GeluTanh)
    expected = F.gelu(x, approximate="tanh")
    torch.testing.assert_close(act(x), expected)

    # Confirm it is numerically distinct from erf-exact GELU, otherwise this
    # would silently be aliasing the existing (wrong, for Gemma4) kernel.
    erf_out = F.gelu(x, approximate="none")
    assert not torch.allclose(act(x), erf_out)


def test_str2activation_type_resolves_multiword_names():
    assert str2ActivationType("gelu_tanh") is ActivationType.GeluTanh
    # Existing single-word names must keep working.
    assert str2ActivationType("silu") is ActivationType.Silu
    assert str2ActivationType("gelu") is ActivationType.Gelu
    assert str2ActivationType("swiglu") is ActivationType.Swiglu


if __name__ == "__main__":
    test_gelu_tanh_enum_value_exists()
    test_gelu_tanh_torch_reference_matches_pytorch_tanh_approx()
    test_str2activation_type_resolves_multiword_names()
    print("ok")
