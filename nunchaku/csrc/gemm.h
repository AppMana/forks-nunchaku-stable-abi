#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Linear.h"
#include "debug.h"
#include "module.h"

class QuantizedGEMM : public ModuleWrapper<GEMM_W4A4> {
public:
    void init(int64_t in_features, int64_t out_features, bool bias, bool use_fp4, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedGEMM");

        size_t val = 0;
        checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
        spdlog::debug("Stack={}", val);

        net = std::make_unique<GEMM_W4A4>((int)in_features,
                                          (int)out_features,
                                          bias,
                                          use_fp4,
                                          bf16 ? Tensor::BF16 : Tensor::FP16,
                                          Device::cuda((int)deviceId));
    }

    torch::stable::Tensor forward(torch::stable::Tensor x) {
        checkModel();

        Tensor result = net->forward(from_torch(x));

        torch::stable::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    void quantize(torch::stable::Tensor x, bool fuse_glu) {
        checkModel();

        spdlog::debug("QuantizedGEMM quantize");

        auto qout = net->quantize(from_torch(x), fuse_glu);

        Tensor act      = qout.act.copy(Device::cpu());
        Tensor ascales  = qout.ascales.copy(Device::cpu());
        Tensor lora_act = qout.lora_act.copy(Device::cpu());

        Tensor::synchronizeDevice();
    }
};
