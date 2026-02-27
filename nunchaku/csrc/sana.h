#pragma once

#include "interop/torch.h"
#include "SanaModel.h"
#include "Serialization.h"
#include "debug.h"
#include "module.h"

class QuantizedSanaModel : public ModuleWrapper<SanaModel> {
public:
    // init with a Python dict config and a list of pag_layers
    void init(PyObject *config, std::vector<int> pag_layers, bool use_fp4, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedSanaModel on device {}", deviceId);

        // Parse config dict using Python C API
        auto getInt = [config](const char *key) -> int {
            PyObject *val = PyDict_GetItemString(config, key);
            if (!val) {
                throw std::runtime_error(spdlog::fmt_lib::format("Missing config key: {}", key));
            }
            return (int)PyLong_AsLong(val);
        };
        auto getDouble = [config](const char *key) -> double {
            PyObject *val = PyDict_GetItemString(config, key);
            if (!val) {
                throw std::runtime_error(spdlog::fmt_lib::format("Missing config key: {}", key));
            }
            return PyFloat_AsDouble(val);
        };

        SanaConfig cfg{
            .num_layers                = getInt("num_layers"),
            .num_attention_heads       = getInt("num_attention_heads"),
            .attention_head_dim        = getInt("attention_head_dim"),
            .num_cross_attention_heads = getInt("num_cross_attention_heads"),
            .expand_ratio              = getDouble("mlp_ratio"),
            .pag_layers                = pag_layers,
            .use_fp4                   = use_fp4,
        };

        ModuleWrapper::init(deviceId);
        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<SanaModel>(cfg, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    torch::stable::Tensor forward(torch::stable::Tensor hidden_states,
                                   torch::stable::Tensor encoder_hidden_states,
                                   torch::stable::Tensor timestep,
                                   torch::stable::Tensor cu_seqlens_img,
                                   torch::stable::Tensor cu_seqlens_txt,
                                   int H,
                                   int W,
                                   bool pag,
                                   bool cfg,
                                   bool skip_first_layer = false) {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedSanaModel forward");

        Tensor result = net->forward(from_torch(hidden_states),
                                     from_torch(encoder_hidden_states),
                                     from_torch(timestep),
                                     from_torch(cu_seqlens_img),
                                     from_torch(cu_seqlens_txt),
                                     H,
                                     W,
                                     pag,
                                     cfg,
                                     skip_first_layer);

        torch::stable::Tensor output = to_torch(result);

        return output;
    }

    torch::stable::Tensor forward_layer(int64_t idx,
                                         torch::stable::Tensor hidden_states,
                                         torch::stable::Tensor encoder_hidden_states,
                                         torch::stable::Tensor timestep,
                                         torch::stable::Tensor cu_seqlens_img,
                                         torch::stable::Tensor cu_seqlens_txt,
                                         int H,
                                         int W,
                                         bool pag,
                                         bool cfg) {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        spdlog::debug("QuantizedSanaModel forward_layer {}", idx);

        Tensor result = net->transformer_blocks.at(idx)->forward(from_torch(hidden_states),
                                                                 from_torch(encoder_hidden_states),
                                                                 from_torch(timestep),
                                                                 from_torch(cu_seqlens_img),
                                                                 from_torch(cu_seqlens_txt),
                                                                 H,
                                                                 W,
                                                                 pag,
                                                                 cfg);

        torch::stable::Tensor output = to_torch(result);

        return output;
    }
};
