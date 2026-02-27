#pragma once

#include "interop/torch.h"
#include "FluxModel.h"
#include "Serialization.h"
#include "debug.h"
#include "Linear.h"
#include "module.h"

class QuantizedFluxModel : public ModuleWrapper<FluxModel> {
public:
    void init(bool use_fp4, bool offload, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedFluxModel on device {}", deviceId);
        if (!bf16) {
            spdlog::info("Use FP16 model");
        }
        if (offload) {
            spdlog::info("Layer offloading enabled");
        }
        ModuleWrapper::init(deviceId);

        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<FluxModel>(
            use_fp4, offload, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    bool isBF16() {
        checkModel();
        return net->dtype == Tensor::BF16;
    }

    // Callback storage: raw PyObject* with manual refcounting
    PyObject *residual_callback_ = nullptr;
    PyObject *attention_callback_ = nullptr;

    void set_residual_callback(PyObject *callback) {
        PyGILState_STATE gstate = PyGILState_Ensure();

        // Release old callback
        Py_XDECREF(residual_callback_);
        residual_callback_ = nullptr;

        if (!callback || callback == Py_None) {
            if (net) {
                net->set_residual_callback(nullptr);
            }
            PyGILState_Release(gstate);
            return;
        }

        Py_INCREF(callback);
        residual_callback_ = callback;

        if (net) {
            PyObject *cb = residual_callback_;
            net->set_residual_callback([cb](const Tensor &x) -> Tensor {
                PyGILState_STATE gs = PyGILState_Ensure();
                torch::stable::Tensor st_x = to_torch(x, true);
                PyObject *py_x             = tensor_to_py(st_x);
                PyObject *result           = PyObject_CallFunctionObjArgs(cb, py_x, NULL);
                Py_DECREF(py_x);
                if (!result) {
                    PyErr_Print();
                    PyGILState_Release(gs);
                    throw std::runtime_error("Residual callback failed");
                }
                // Keep Python tensor alive during conversion
                Py_INCREF(result);
                torch::stable::Tensor st_y = py_to_tensor(result);
                Tensor y                   = from_torch(st_y);
                Py_DECREF(result);
                PyGILState_Release(gs);
                return y;
            });
        }

        PyGILState_Release(gstate);
    }

    torch::stable::Tensor forward(torch::stable::Tensor hidden_states,
                                   torch::stable::Tensor encoder_hidden_states,
                                   torch::stable::Tensor temb,
                                   torch::stable::Tensor rotary_emb_img,
                                   torch::stable::Tensor rotary_emb_context,
                                   torch::stable::Tensor rotary_emb_single,
                                   std::optional<torch::stable::Tensor> controlnet_block_samples        = std::nullopt,
                                   std::optional<torch::stable::Tensor> controlnet_single_block_samples = std::nullopt,
                                   bool skip_first_layer                                                = false) {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward");

        Tensor result = net->forward(
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            from_torch(rotary_emb_single),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value()) : Tensor{},
            controlnet_single_block_samples.has_value() ? from_torch(controlnet_single_block_samples.value())
                                                        : Tensor{},
            skip_first_layer);

        torch::stable::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    std::pair<torch::stable::Tensor, torch::stable::Tensor>
    forward_layer(int64_t idx,
                  torch::stable::Tensor hidden_states,
                  torch::stable::Tensor encoder_hidden_states,
                  torch::stable::Tensor temb,
                  torch::stable::Tensor rotary_emb_img,
                  torch::stable::Tensor rotary_emb_context,
                  std::optional<torch::stable::Tensor> controlnet_block_samples        = std::nullopt,
                  std::optional<torch::stable::Tensor> controlnet_single_block_samples = std::nullopt) {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_layer {}", idx);

        auto &&[hidden_states_, encoder_hidden_states_] = net->forward_layer(
            idx,
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value()) : Tensor{},
            controlnet_single_block_samples.has_value() ? from_torch(controlnet_single_block_samples.value())
                                                        : Tensor{});

        auto out_hs  = to_torch(hidden_states_);
        auto out_ehs = to_torch(encoder_hidden_states_);
        Tensor::synchronizeDevice();

        return {out_hs, out_ehs};
    }

    torch::stable::Tensor forward_single_layer(int64_t idx,
                                                torch::stable::Tensor hidden_states,
                                                torch::stable::Tensor temb,
                                                torch::stable::Tensor rotary_emb_single) {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_single_layer {}", idx);

        if (net->isOffloadEnabled()) {
            net->single_transformer_blocks.at(idx)->loadLazyParams();
        }

        Tensor result = net->single_transformer_blocks.at(idx)->forward(
            from_torch(hidden_states), from_torch(temb), from_torch(rotary_emb_single));

        if (net->isOffloadEnabled()) {
            net->single_transformer_blocks.at(idx)->releaseLazyParams();
        }

        auto output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    struct NormOneResult {
        torch::stable::Tensor x, gate_msa, shift_mlp, scale_mlp, gate_mlp;
    };

    NormOneResult norm_one_forward(int64_t idx, torch::stable::Tensor hidden_states, torch::stable::Tensor temb) {
        AdaLayerNormZero::Output result =
            net->transformer_blocks.at(idx)->norm1.forward(from_torch(hidden_states), from_torch(temb));
        return {to_torch(result.x),
                to_torch(result.gate_msa),
                to_torch(result.shift_mlp),
                to_torch(result.scale_mlp),
                to_torch(result.gate_mlp)};
    }

    void setLoraScale(int skipRanks, float scale) {
        if (skipRanks % 16 != 0) {
            throw std::invalid_argument("skipRanks must be multiples of 16");
        }

        CUDADeviceContext ctx(deviceId);

        spdlog::info("Set lora scale to {} (skip {} ranks)", scale, skipRanks);

        net->traverse([&](Module *module) {
            if (auto *m = dynamic_cast<GEMV_AWQ *>(module)) {
                m->lora_scale = scale;
            } else if (auto *m = dynamic_cast<GEMM_W4A4 *>(module)) {
                for (int i = 0; i < skipRanks / 16; i++) {
                    m->lora_scales[i] = 1.0f;
                }
                for (int i = skipRanks / 16; i < (int)m->lora_scales.size(); i++) {
                    m->lora_scales[i] = scale;
                }
            }
        });
    }

    void setAttentionImpl(std::string name, PyObject *attn_func) {
        if (name.empty() || name == "default") {
            name = "flashattn2";
        }

        spdlog::info("Set attention implementation to {}", name);

        if (name == "flashattn2") {
            net->setAttentionImpl(AttentionImpl::FlashAttention2, nullptr);
        } else if (name == "nunchaku-fp16") {
            net->setAttentionImpl(AttentionImpl::NunchakuFP16, nullptr);
        } else if (name == "custom") {
            // Release old callback
            Py_XDECREF(attention_callback_);
            Py_INCREF(attn_func);
            attention_callback_ = attn_func;

            PyObject *f = attention_callback_;
            net->setAttentionImpl(AttentionImpl::Custom, [f](Tensor qkv) -> Tensor {
                PyGILState_STATE gs = PyGILState_Ensure();
                torch::stable::Tensor st_qkv = to_torch(qkv, true);
                PyObject *py_qkv             = tensor_to_py(st_qkv);
                PyObject *result             = PyObject_CallFunctionObjArgs(f, py_qkv, NULL);
                Py_DECREF(py_qkv);
                if (!result) {
                    PyErr_Print();
                    PyGILState_Release(gs);
                    throw std::runtime_error("Attention callback failed");
                }
                Py_INCREF(result);
                torch::stable::Tensor st_out = py_to_tensor(result);
                Tensor output                = from_torch(st_out);
                Py_DECREF(result);
                PyGILState_Release(gs);
                return output;
            });
        } else {
            throw std::invalid_argument(spdlog::fmt_lib::format("Invalid attention implementation {}", name));
        }
    }

    struct ForwardLayerIPAdapterResult {
        torch::stable::Tensor hidden_states, encoder_hidden_states, ip_query;
    };

    ForwardLayerIPAdapterResult
    forward_layer_ip_adapter(int64_t idx,
                             torch::stable::Tensor hidden_states,
                             torch::stable::Tensor encoder_hidden_states,
                             torch::stable::Tensor temb,
                             torch::stable::Tensor rotary_emb_img,
                             torch::stable::Tensor rotary_emb_context,
                             std::optional<torch::stable::Tensor> controlnet_block_samples        = std::nullopt,
                             std::optional<torch::stable::Tensor> controlnet_single_block_samples = std::nullopt) {
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedFluxModel forward_layer {}", idx);

        auto &&[hidden_states_, encoder_hidden_states_, ip_query_] = net->forward_ip_adapter(
            idx,
            from_torch(hidden_states),
            from_torch(encoder_hidden_states),
            from_torch(temb),
            from_torch(rotary_emb_img),
            from_torch(rotary_emb_context),
            controlnet_block_samples.has_value() ? from_torch(controlnet_block_samples.value()) : Tensor{},
            controlnet_single_block_samples.has_value() ? from_torch(controlnet_single_block_samples.value())
                                                        : Tensor{});

        auto out_hs  = to_torch(hidden_states_);
        auto out_ehs = to_torch(encoder_hidden_states_);
        auto out_ipq = to_torch(ip_query_);
        Tensor::synchronizeDevice();

        return {out_hs, out_ehs, out_ipq};
    }

    ~QuantizedFluxModel() {
        Py_XDECREF(residual_callback_);
        Py_XDECREF(attention_callback_);
    }
};
