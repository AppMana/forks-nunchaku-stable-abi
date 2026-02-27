#include "ops.h"

#include <torch/csrc/stable/library.h>

using StableTensor = torch::stable::Tensor;

namespace nunchaku::ops {

static auto getT(std::optional<StableTensor> &t) {
    ::Tensor ret = t.has_value() ? from_torch(t.value()) : ::Tensor{};
    if (ret.valid()) {
        spdlog::trace("  {}", ret.shape.str());
    } else {
        spdlog::trace("  <invalid>");
    }
    return ret;
}

void gemm_w4a4(std::optional<StableTensor> act,
               std::optional<StableTensor> wgt,
               std::optional<StableTensor> out,
               std::optional<StableTensor> qout,
               std::optional<StableTensor> ascales,
               std::optional<StableTensor> wscales,
               std::optional<StableTensor> oscales,
               std::optional<StableTensor> poolout,
               std::optional<StableTensor> lora_act_in,
               std::optional<StableTensor> lora_up,
               std::optional<StableTensor> lora_down,
               std::optional<StableTensor> lora_act_out,
               std::optional<StableTensor> norm_q,
               std::optional<StableTensor> norm_k,
               std::optional<StableTensor> rotary_emb,
               std::optional<StableTensor> bias,
               std::optional<StableTensor> smooth_factor,
               std::optional<StableTensor> out_vk,
               std::optional<StableTensor> out_linearattn,
               bool act_unsigned,
               StableTensor lora_scales_tensor,
               bool fuse_silu,
               bool fp4,
               double alpha,
               std::optional<StableTensor> wcscales,
               std::optional<StableTensor> out_q,
               std::optional<StableTensor> out_k,
               std::optional<StableTensor> out_v,
               int64_t attn_tokens) {
    TorchOpContext ctx;
    spdlog::trace("running gemm_w4a4: ");

    // Extract lora_scales from tensor
    std::vector<float> lora_scales;
    int64_t num_scales = lora_scales_tensor.numel();
    if (num_scales > 0) {
        float *data = reinterpret_cast<float *>(lora_scales_tensor.data_ptr());
        lora_scales.assign(data, data + num_scales);
    }

    nunchaku::kernels::gemm_w4a4(getT(act),
                                 getT(wgt),
                                 getT(out),
                                 getT(qout),
                                 getT(ascales),
                                 getT(wscales),
                                 getT(oscales),
                                 getT(poolout),
                                 getT(lora_act_in),
                                 getT(lora_up),
                                 getT(lora_down),
                                 getT(lora_act_out),
                                 getT(norm_q),
                                 getT(norm_k),
                                 getT(rotary_emb),
                                 getT(bias),
                                 getT(smooth_factor),
                                 getT(out_vk),
                                 getT(out_linearattn),
                                 act_unsigned,
                                 lora_scales,
                                 fuse_silu,
                                 fp4,
                                 (float)alpha,
                                 getT(wcscales),
                                 getT(out_q),
                                 getT(out_k),
                                 getT(out_v),
                                 (int)attn_tokens);
}

void quantize_w4a4_act_fuse_lora(std::optional<StableTensor> input,
                                  std::optional<StableTensor> output,
                                  std::optional<StableTensor> oscales,
                                  std::optional<StableTensor> lora_down,
                                  std::optional<StableTensor> lora_act_out,
                                  std::optional<StableTensor> smooth,
                                  bool fuse_glu,
                                  bool fp4) {
    TorchOpContext ctx;
    spdlog::trace("running quantize_w4a4_act_fuse_lora: ");

    nunchaku::kernels::quantize_w4a4_act_fuse_lora(
        getT(input), getT(output), getT(oscales), getT(lora_down), getT(lora_act_out), getT(smooth), fuse_glu, fp4);
}

void attention_fp16(StableTensor q, StableTensor k, StableTensor v, StableTensor o, double scale) {
    TorchOpContext ctx;
    nunchaku::kernels::attention_fp16(
        from_torch(q), from_torch(k), from_torch(v), from_torch(o), (float)scale);
}

StableTensor gemv_awq(StableTensor in_feats, StableTensor kernel, StableTensor scaling_factors, StableTensor zeros, int64_t m, int64_t n, int64_t k, int64_t group_size) {
    TorchOpContext ctx;
    ::Tensor result = ::gemv_awq(from_torch(in_feats),
                                 from_torch(kernel),
                                 from_torch(scaling_factors),
                                 from_torch(zeros),
                                 (int)m,
                                 (int)n,
                                 (int)k,
                                 (int)group_size);

    return to_torch(result);
}

StableTensor gemm_awq(StableTensor in_feats, StableTensor kernel, StableTensor scaling_factors, StableTensor zeros) {
    TorchOpContext ctx;
    ::Tensor result = ::awq_gemm_forward_cuda(from_torch(in_feats),
                                              from_torch(kernel),
                                              from_torch(scaling_factors),
                                              from_torch(zeros));

    return to_torch(result);
}

void test_rmsnorm_rope(StableTensor input, StableTensor output, StableTensor norm_q, StableTensor norm_k, StableTensor rotary_emb) {
    nunchaku::kernels::test_rmsnorm_rope(
        from_torch(input), from_torch(output), from_torch(norm_q), from_torch(norm_k), from_torch(rotary_emb));
}

void test_pack_qkv(StableTensor input, StableTensor out_q, StableTensor out_k, StableTensor out_v, int64_t numTokens) {
    nunchaku::kernels::test_pack_qkv(
        from_torch(input), from_torch(out_q), from_torch(out_k), from_torch(out_v), (int)numTokens);
}

} // namespace nunchaku::ops

// ============================================================================
// STABLE_TORCH_LIBRARY registration for stateless ops
// ============================================================================

// --- Boxed wrapper functions ---

static void boxed_gemm_w4a4(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto act            = to<std::optional<StableTensor>>(stack[0]);
    auto wgt            = to<std::optional<StableTensor>>(stack[1]);
    auto out            = to<std::optional<StableTensor>>(stack[2]);
    auto qout           = to<std::optional<StableTensor>>(stack[3]);
    auto ascales        = to<std::optional<StableTensor>>(stack[4]);
    auto wscales        = to<std::optional<StableTensor>>(stack[5]);
    auto oscales        = to<std::optional<StableTensor>>(stack[6]);
    auto poolout        = to<std::optional<StableTensor>>(stack[7]);
    auto lora_act_in    = to<std::optional<StableTensor>>(stack[8]);
    auto lora_up        = to<std::optional<StableTensor>>(stack[9]);
    auto lora_down      = to<std::optional<StableTensor>>(stack[10]);
    auto lora_act_out   = to<std::optional<StableTensor>>(stack[11]);
    auto norm_q         = to<std::optional<StableTensor>>(stack[12]);
    auto norm_k         = to<std::optional<StableTensor>>(stack[13]);
    auto rotary_emb     = to<std::optional<StableTensor>>(stack[14]);
    auto bias           = to<std::optional<StableTensor>>(stack[15]);
    auto smooth_factor  = to<std::optional<StableTensor>>(stack[16]);
    auto out_vk         = to<std::optional<StableTensor>>(stack[17]);
    auto out_linearattn = to<std::optional<StableTensor>>(stack[18]);
    auto act_unsigned  = to<bool>(stack[19]);
    auto lora_scales_tensor = to<StableTensor>(stack[20]);
    auto fuse_silu     = to<bool>(stack[21]);
    auto fp4           = to<bool>(stack[22]);
    auto alpha         = to<double>(stack[23]);
    auto wcscales      = to<std::optional<StableTensor>>(stack[24]);
    auto out_q         = to<std::optional<StableTensor>>(stack[25]);
    auto out_k         = to<std::optional<StableTensor>>(stack[26]);
    auto out_v         = to<std::optional<StableTensor>>(stack[27]);
    auto attn_tokens   = to<int64_t>(stack[28]);

    nunchaku::ops::gemm_w4a4(
        act, wgt, out, qout, ascales, wscales, oscales, poolout,
        lora_act_in, lora_up, lora_down, lora_act_out,
        norm_q, norm_k, rotary_emb, bias, smooth_factor,
        out_vk, out_linearattn, act_unsigned, lora_scales_tensor,
        fuse_silu, fp4, alpha, wcscales, out_q, out_k, out_v, attn_tokens);
}

static void boxed_quantize_w4a4_act_fuse_lora(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto input        = to<std::optional<StableTensor>>(stack[0]);
    auto output       = to<std::optional<StableTensor>>(stack[1]);
    auto oscales      = to<std::optional<StableTensor>>(stack[2]);
    auto lora_down    = to<std::optional<StableTensor>>(stack[3]);
    auto lora_act_out = to<std::optional<StableTensor>>(stack[4]);
    auto smooth       = to<std::optional<StableTensor>>(stack[5]);
    auto fuse_glu  = to<bool>(stack[6]);
    auto fp4       = to<bool>(stack[7]);

    nunchaku::ops::quantize_w4a4_act_fuse_lora(input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4);
}

static void boxed_attention_fp16(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto q     = to<StableTensor>(stack[0]);
    auto k     = to<StableTensor>(stack[1]);
    auto v     = to<StableTensor>(stack[2]);
    auto o     = to<StableTensor>(stack[3]);
    auto scale = to<double>(stack[4]);

    nunchaku::ops::attention_fp16(q, k, v, o, scale);
}

static void boxed_gemv_awq(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto in_feats        = to<StableTensor>(stack[0]);
    auto kernel          = to<StableTensor>(stack[1]);
    auto scaling_factors = to<StableTensor>(stack[2]);
    auto zeros           = to<StableTensor>(stack[3]);
    auto m               = to<int64_t>(stack[4]);
    auto n               = to<int64_t>(stack[5]);
    auto k               = to<int64_t>(stack[6]);
    auto group_size      = to<int64_t>(stack[7]);

    auto result = nunchaku::ops::gemv_awq(in_feats, kernel, scaling_factors, zeros, m, n, k, group_size);
    stack[0] = from(result);
}

static void boxed_gemm_awq(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto in_feats        = to<StableTensor>(stack[0]);
    auto kernel          = to<StableTensor>(stack[1]);
    auto scaling_factors = to<StableTensor>(stack[2]);
    auto zeros           = to<StableTensor>(stack[3]);

    auto result = nunchaku::ops::gemm_awq(in_feats, kernel, scaling_factors, zeros);
    stack[0] = from(result);
}

static void boxed_test_rmsnorm_rope(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto input      = to<StableTensor>(stack[0]);
    auto output     = to<StableTensor>(stack[1]);
    auto norm_q     = to<StableTensor>(stack[2]);
    auto norm_k     = to<StableTensor>(stack[3]);
    auto rotary_emb = to<StableTensor>(stack[4]);

    nunchaku::ops::test_rmsnorm_rope(input, output, norm_q, norm_k, rotary_emb);
}

static void boxed_test_pack_qkv(StableIValue *stack, uint64_t num_args, uint64_t num_outputs) {
    auto input     = to<StableTensor>(stack[0]);
    auto out_q     = to<StableTensor>(stack[1]);
    auto out_k     = to<StableTensor>(stack[2]);
    auto out_v     = to<StableTensor>(stack[3]);
    auto numTokens = to<int64_t>(stack[4]);

    nunchaku::ops::test_pack_qkv(input, out_q, out_k, out_v, numTokens);
}

// --- Schema definitions ---

STABLE_TORCH_LIBRARY(nunchaku, m) {
    m.def("gemm_w4a4("
          "Tensor? act, "
          "Tensor? wgt, "
          "Tensor? out, "
          "Tensor? qout, "
          "Tensor? ascales, "
          "Tensor? wscales, "
          "Tensor? oscales, "
          "Tensor? poolout, "
          "Tensor? lora_act_in, "
          "Tensor? lora_up, "
          "Tensor? lora_down, "
          "Tensor? lora_act_out, "
          "Tensor? norm_q, "
          "Tensor? norm_k, "
          "Tensor? rotary_emb, "
          "Tensor? bias, "
          "Tensor? smooth_factor, "
          "Tensor? out_vk, "
          "Tensor? out_linearattn, "
          "bool act_unsigned, "
          "Tensor lora_scales, "
          "bool fuse_silu, "
          "bool fp4, "
          "float alpha, "
          "Tensor? wcscales, "
          "Tensor? out_q, "
          "Tensor? out_k, "
          "Tensor? out_v, "
          "int attn_tokens"
          ") -> ()");

    m.def("quantize_w4a4_act_fuse_lora("
          "Tensor? input, "
          "Tensor? output, "
          "Tensor? oscales, "
          "Tensor? lora_down, "
          "Tensor? lora_act_out, "
          "Tensor? smooth, "
          "bool fuse_glu, "
          "bool fp4"
          ") -> ()");

    m.def("attention_fp16("
          "Tensor q, "
          "Tensor k, "
          "Tensor v, "
          "Tensor o, "
          "float scale"
          ") -> ()");

    m.def("gemv_awq("
          "Tensor in_feats, "
          "Tensor kernel, "
          "Tensor scaling_factors, "
          "Tensor zeros, "
          "int m, "
          "int n, "
          "int k, "
          "int group_size"
          ") -> Tensor");

    m.def("gemm_awq("
          "Tensor in_feats, "
          "Tensor kernel, "
          "Tensor scaling_factors, "
          "Tensor zeros"
          ") -> Tensor");

    m.def("test_rmsnorm_rope("
          "Tensor input, "
          "Tensor output, "
          "Tensor norm_q, "
          "Tensor norm_k, "
          "Tensor rotary_emb"
          ") -> ()");

    m.def("test_pack_qkv("
          "Tensor input, "
          "Tensor out_q, "
          "Tensor out_k, "
          "Tensor out_v, "
          "int numTokens"
          ") -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(nunchaku, CUDA, m) {
    m.impl("gemm_w4a4", &boxed_gemm_w4a4);
    m.impl("quantize_w4a4_act_fuse_lora", &boxed_quantize_w4a4_act_fuse_lora);
    m.impl("attention_fp16", &boxed_attention_fp16);
    m.impl("gemv_awq", &boxed_gemv_awq);
    m.impl("gemm_awq", &boxed_gemm_awq);
    m.impl("test_rmsnorm_rope", &boxed_test_rmsnorm_rope);
    m.impl("test_pack_qkv", &boxed_test_pack_qkv);
}
