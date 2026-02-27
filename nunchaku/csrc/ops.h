#pragma once

#include "interop/torch.h"
#include "kernels/zgemm/zgemm.h"
#include "kernels/awq/gemv_awq.h"
#include "kernels/awq/gemm_awq.h"

namespace nunchaku::ops {

// gemm_w4a4 using stable::Tensor; lora_scales passed as a float Tensor instead of vector<float>
void gemm_w4a4(std::optional<torch::stable::Tensor> act,
               std::optional<torch::stable::Tensor> wgt,
               std::optional<torch::stable::Tensor> out,
               std::optional<torch::stable::Tensor> qout,
               std::optional<torch::stable::Tensor> ascales,
               std::optional<torch::stable::Tensor> wscales,
               std::optional<torch::stable::Tensor> oscales,
               std::optional<torch::stable::Tensor> poolout,
               std::optional<torch::stable::Tensor> lora_act_in,
               std::optional<torch::stable::Tensor> lora_up,
               std::optional<torch::stable::Tensor> lora_down,
               std::optional<torch::stable::Tensor> lora_act_out,
               std::optional<torch::stable::Tensor> norm_q,
               std::optional<torch::stable::Tensor> norm_k,
               std::optional<torch::stable::Tensor> rotary_emb,
               std::optional<torch::stable::Tensor> bias,
               std::optional<torch::stable::Tensor> smooth_factor,
               std::optional<torch::stable::Tensor> out_vk,
               std::optional<torch::stable::Tensor> out_linearattn,
               bool act_unsigned,
               torch::stable::Tensor lora_scales_tensor, // 1D float tensor instead of vector<float>
               bool fuse_silu,
               bool fp4,
               double alpha,
               std::optional<torch::stable::Tensor> wcscales,
               std::optional<torch::stable::Tensor> out_q,
               std::optional<torch::stable::Tensor> out_k,
               std::optional<torch::stable::Tensor> out_v,
               int64_t attn_tokens);

void quantize_w4a4_act_fuse_lora(std::optional<torch::stable::Tensor> input,
                                  std::optional<torch::stable::Tensor> output,
                                  std::optional<torch::stable::Tensor> oscales,
                                  std::optional<torch::stable::Tensor> lora_down,
                                  std::optional<torch::stable::Tensor> lora_act_out,
                                  std::optional<torch::stable::Tensor> smooth,
                                  bool fuse_glu,
                                  bool fp4);

void attention_fp16(torch::stable::Tensor q,
                    torch::stable::Tensor k,
                    torch::stable::Tensor v,
                    torch::stable::Tensor o,
                    double scale);

torch::stable::Tensor gemv_awq(torch::stable::Tensor in_feats,
                                torch::stable::Tensor kernel,
                                torch::stable::Tensor scaling_factors,
                                torch::stable::Tensor zeros,
                                int64_t m,
                                int64_t n,
                                int64_t k,
                                int64_t group_size);

torch::stable::Tensor gemm_awq(torch::stable::Tensor in_feats,
                                 torch::stable::Tensor kernel,
                                 torch::stable::Tensor scaling_factors,
                                 torch::stable::Tensor zeros);

void test_rmsnorm_rope(torch::stable::Tensor input,
                       torch::stable::Tensor output,
                       torch::stable::Tensor norm_q,
                       torch::stable::Tensor norm_k,
                       torch::stable::Tensor rotary_emb);

void test_pack_qkv(torch::stable::Tensor input,
                   torch::stable::Tensor out_q,
                   torch::stable::Tensor out_k,
                   torch::stable::Tensor out_v,
                   int64_t numTokens);

} // namespace nunchaku::ops
