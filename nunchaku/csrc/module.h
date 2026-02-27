#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Module.h"
#include "debug.h"
#include "utils.h"
#include <vector>

template<typename M>
class ModuleWrapper {
public:
    void init(int deviceId) {
        this->deviceId = deviceId;
    }
    void reset() {
        CUDADeviceContext ctx(this->deviceId);

        debugContext.reset();
        net.reset();
        Tensor::synchronizeDevice();

        nunchaku::utils::trim_memory();
        Tensor::synchronizeDevice();
    }

    void load(std::string path, bool partial = false) {
        checkModel();
        CUDADeviceContext ctx(this->deviceId);

        spdlog::info("{} weights from {}", partial ? "Loading partial" : "Loading", path);

        std::shared_ptr<SafeTensors> provider = std::make_shared<SafeTensors>(path);
        net->loadParams(*provider, partial);
        Tensor::synchronizeDevice();

        spdlog::info("Done.");
    }

    void loadDict(std::map<std::string, torch::stable::Tensor> dict, bool partial = false) {
        checkModel();
        CUDADeviceContext ctx(this->deviceId);

        spdlog::info("{} weights from pytorch", partial ? "Loading partial" : "Loading");

        std::shared_ptr<TensorsProviderTorch> provider =
            std::make_shared<TensorsProviderTorch>(std::move(dict));
        net->loadParams(*provider, partial);
        Tensor::synchronizeDevice();

        spdlog::info("Done.");
    }

    // Python dict-based loadDict: accepts PyObject* dict, iterates with PyDict_Next
    void loadDictPy(PyObject *dict, bool partial = false) {
        checkModel();
        CUDADeviceContext ctx(this->deviceId);

        spdlog::info("{} weights from pytorch", partial ? "Loading partial" : "Loading");

        std::map<std::string, torch::stable::Tensor> tensor_dict;
        std::vector<PyObject *> held_values;
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(dict, &pos, &key, &value)) {
            PyObject *key_utf8 = PyUnicode_AsUTF8String(key);
            if (!key_utf8) {
                throw std::runtime_error("Dictionary keys must be convertible to UTF-8 strings");
            }

            char *key_str = PyBytes_AsString(key_utf8);
            if (!key_str) {
                Py_DECREF(key_utf8);
                throw std::runtime_error("Failed to extract UTF-8 key bytes");
            }

            // Keep reference to the Python tensor alive during loading.
            Py_INCREF(value);
            held_values.push_back(value);
            tensor_dict.emplace(key_str, py_to_tensor(value));
            Py_DECREF(key_utf8);
        }

        std::shared_ptr<TensorsProviderTorch> provider =
            std::make_shared<TensorsProviderTorch>(std::move(tensor_dict));
        net->loadParams(*provider, partial);
        Tensor::synchronizeDevice();

        // Release Python tensor references.
        for (PyObject *v : held_values) {
            Py_DECREF(v);
        }

        spdlog::info("Done.");
    }

    void startDebug() {
        debugContext = std::make_unique<DebugContext>();
    }
    void stopDebug() {
        debugContext.reset();
    }

    // Returns a Python dict of {str: torch.Tensor}
    PyObject *getDebugResultsPy() {
        CUDADeviceContext ctx(this->deviceId);

        PyObject *result = PyDict_New();

        if (debugContext) {
            for (auto &&[key, value] : debugContext->tensors) {
                torch::stable::Tensor st = to_torch(value);
                PyObject *py_tensor      = tensor_to_py(st);
                PyDict_SetItemString(result, key.c_str(), py_tensor);
                Py_DECREF(py_tensor);
            }
        }

        return result;
    }

protected:
    void checkModel() {
        if (!net) {
            throw std::runtime_error("Model not initialized");
        }
    }

protected:
    std::unique_ptr<M> net;
    std::unique_ptr<DebugContext> debugContext;

    int deviceId = -1;
};
