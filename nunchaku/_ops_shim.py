"""
Compatibility shim: maps nunchaku._C.ops.* calls to torch.ops.nunchaku.* .

With the stable ABI, stateless ops are registered via STABLE_TORCH_LIBRARY
and accessed via torch.ops.nunchaku.*. This module provides backward
compatibility for code that does `from nunchaku._C import ops; ops.foo(...)`.

Key change: lora_scales is now passed as a 1D float Tensor instead of a list.
"""

import math
import sys
import types

import torch

# Importing _C triggers loading the .so, which runs the STABLE_TORCH_LIBRARY
# static initializers that register torch.ops.nunchaku.*
import nunchaku._C  # noqa: F401


class _OpsNamespace:
    """Mimics the old _C.ops submodule interface, delegating to torch.ops.nunchaku.*"""

    @staticmethod
    def gemm_w4a4(
        act=None,
        wgt=None,
        out=None,
        qout=None,
        ascales=None,
        wscales=None,
        oscales=None,
        poolout=None,
        lora_act_in=None,
        lora_up=None,
        lora_down=None,
        lora_act_out=None,
        norm_q=None,
        norm_k=None,
        rotary_emb=None,
        bias=None,
        smooth_factor=None,
        out_vk=None,
        out_linearattn=None,
        act_unsigned=False,
        lora_scales=None,
        fuse_silu=False,
        fp4=False,
        alpha=1.0,
        wcscales=None,
        out_q=None,
        out_k=None,
        out_v=None,
        attn_tokens=0,
    ):
        # Convert lora_scales list to tensor
        if lora_scales is None:
            lora_scales_tensor = torch.empty(0, dtype=torch.float32)
        else:
            lora_scales_tensor = torch.tensor(lora_scales, dtype=torch.float32)

        torch.ops.nunchaku.gemm_w4a4(
            act, wgt, out, qout, ascales, wscales, oscales, poolout,
            lora_act_in, lora_up, lora_down, lora_act_out,
            norm_q, norm_k, rotary_emb, bias, smooth_factor,
            out_vk, out_linearattn, act_unsigned, lora_scales_tensor,
            fuse_silu, fp4, float(alpha), wcscales,
            out_q, out_k, out_v, attn_tokens,
        )

    @staticmethod
    def quantize_w4a4_act_fuse_lora(input=None, output=None, oscales=None,
                                     lora_down=None, lora_act_out=None,
                                     smooth=None, fuse_glu=False, fp4=False):
        torch.ops.nunchaku.quantize_w4a4_act_fuse_lora(
            input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4
        )

    @staticmethod
    def attention_fp16(q, k, v, o, scale):
        torch.ops.nunchaku.attention_fp16(q, k, v, o, float(scale))

    @staticmethod
    def gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size):
        return torch.ops.nunchaku.gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size)

    @staticmethod
    def gemm_awq(in_feats, kernel, scaling_factors, zeros):
        return torch.ops.nunchaku.gemm_awq(in_feats, kernel, scaling_factors, zeros)

    @staticmethod
    def test_rmsnorm_rope(input, output, norm_q, norm_k, rotary_emb):
        torch.ops.nunchaku.test_rmsnorm_rope(input, output, norm_q, norm_k, rotary_emb)

    @staticmethod
    def test_pack_qkv(input, out_q, out_k, out_v, numTokens):
        torch.ops.nunchaku.test_pack_qkv(input, out_q, out_k, out_v, numTokens)


# Expose a real module at `nunchaku._C.ops` so both:
#   from nunchaku._C import ops
#   from nunchaku._C.ops import foo
# continue to work.
_ops_namespace = _OpsNamespace()
_ops_module = types.ModuleType("nunchaku._C.ops")
for _name in dir(_ops_namespace):
    if _name.startswith("_"):
        continue
    _attr = getattr(_ops_namespace, _name)
    if callable(_attr):
        setattr(_ops_module, _name, _attr)

sys.modules["nunchaku._C.ops"] = _ops_module
nunchaku._C.ops = _ops_module
