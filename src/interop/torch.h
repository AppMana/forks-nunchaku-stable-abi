#pragma once

#define Py_LIMITED_API 0x03090000
#include <Python.h>

#include <torch/csrc/stable/tensor_struct.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include "common.h"
#include "Tensor.h"

// --- Buffer wrapper that keeps a stable::Tensor alive ---
class BufferTorchTensor : public Buffer {
public:
    BufferTorchTensor(torch::stable::Tensor tensor);
    virtual bool isAsyncBuffer() override {
        return this->device.type == Device::CUDA;
    }

private:
    torch::stable::Tensor tensor_;
};

// --- CUDA stream context using stable ABI ---
class TorchOpContext {
public:
    TorchOpContext();
    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&)      = delete;
    ~TorchOpContext();
};

// --- Tensor conversion: stable::Tensor <-> internal Tensor ---
Tensor from_torch(torch::stable::Tensor input);
torch::stable::Tensor to_torch(Tensor input, bool inplace = false);

// --- TensorsProvider using stable::Tensor dict ---
class TensorsProviderTorch : public TensorsProvider {
public:
    TensorsProviderTorch(std::map<std::string, torch::stable::Tensor> dict)
        : storage(std::move(dict)) {}

    virtual bool contains(const std::string &key) const override {
        return storage.contains(key);
    }
    virtual Tensor getTensor(const std::string &key) override {
        if (!storage.contains(key)) {
            return Tensor{};
        }
        return from_torch(storage.at(key));
    }

private:
    std::map<std::string, torch::stable::Tensor> storage;
};

// --- Python <-> stable::Tensor conversion helpers ---
// Converts a Python torch.Tensor (PyObject*) to torch::stable::Tensor
// by extracting metadata via Python C API and using aoti_torch_create_tensor_from_blob
torch::stable::Tensor py_to_tensor(PyObject *obj);

// Converts a torch::stable::Tensor to a Python torch.Tensor (PyObject*)
// Returns a new reference. Uses torch.frombuffer or empty_strided + copy approach.
PyObject *tensor_to_py(const torch::stable::Tensor &tensor);

// Converts a Python torch.Tensor to an optional stable::Tensor.
// Returns empty optional if obj is Py_None or nullptr.
std::optional<torch::stable::Tensor> py_to_optional_tensor(PyObject *obj);

// Helper to get dtype int32 from a Python dtype object
int32_t py_dtype_to_aoti_dtype(PyObject *dtype_obj);

// Helper to get device info from a Python device object
void py_device_to_aoti(PyObject *device_obj, int32_t &device_type, int32_t &device_index);
