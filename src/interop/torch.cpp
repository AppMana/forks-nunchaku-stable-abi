#include "torch.h"

#include <torch/csrc/stable/tensor_inl.h>

using spdlog::fmt_lib::format;

template<typename To, typename Ti>
static To int_cast(Ti x) {
    if (x < std::numeric_limits<To>::min() || x > std::numeric_limits<To>::max()) {
        throw std::runtime_error("integer overflow");
    }
    return static_cast<To>(x);
}

// --- Scalar type mapping: torch headeronly ScalarType <-> internal Tensor::ScalarType ---

static const std::map<torch::headeronly::ScalarType, ::Tensor::ScalarType> stableToInternal = {
    {torch::headeronly::ScalarType::Char, ::Tensor::INT8},
    {torch::headeronly::ScalarType::Byte, ::Tensor::INT8},
    {torch::headeronly::ScalarType::Int, ::Tensor::INT32},
    {torch::headeronly::ScalarType::Long, ::Tensor::INT64},
    {torch::headeronly::ScalarType::Float, ::Tensor::FP32},
    {torch::headeronly::ScalarType::Half, ::Tensor::FP16},
    {torch::headeronly::ScalarType::BFloat16, ::Tensor::BF16},
    {torch::headeronly::ScalarType::Short, ::Tensor::INT16},
    {torch::headeronly::ScalarType::Float8_e4m3fn, ::Tensor::FP8_E4M3},
    {torch::headeronly::ScalarType::Float8_e5m2, ::Tensor::FP8_E5M2},
};

static const std::map<::Tensor::ScalarType, torch::headeronly::ScalarType> internalToStable = {
    {::Tensor::INT8, torch::headeronly::ScalarType::Byte},
    {::Tensor::INT32, torch::headeronly::ScalarType::Int},
    {::Tensor::INT64, torch::headeronly::ScalarType::Long},
    {::Tensor::FP32, torch::headeronly::ScalarType::Float},
    {::Tensor::FP16, torch::headeronly::ScalarType::Half},
    {::Tensor::BF16, torch::headeronly::ScalarType::BFloat16},
    {::Tensor::INT16, torch::headeronly::ScalarType::Short},
    {::Tensor::FP8_E4M3, torch::headeronly::ScalarType::Float8_e4m3fn},
    {::Tensor::FP8_E5M2, torch::headeronly::ScalarType::Float8_e5m2},
};

// Map internal scalar type to aoti dtype int
static int32_t internal_dtype_to_aoti(::Tensor::ScalarType st) {
    switch (st) {
    case ::Tensor::INT8:
        return aoti_torch_dtype_uint8();
    case ::Tensor::INT16:
        return aoti_torch_dtype_int16();
    case ::Tensor::INT32:
        return aoti_torch_dtype_int32();
    case ::Tensor::INT64:
        return aoti_torch_dtype_int64();
    case ::Tensor::FP16:
        return aoti_torch_dtype_float16();
    case ::Tensor::FP32:
        return aoti_torch_dtype_float32();
    case ::Tensor::BF16:
        return aoti_torch_dtype_bfloat16();
    case ::Tensor::FP8_E4M3:
        return aoti_torch_dtype_float8_e4m3fn();
    case ::Tensor::FP8_E5M2:
        return aoti_torch_dtype_float8_e5m2();
    default:
        throw std::runtime_error("Unsupported scalar type");
    }
}

// --- BufferTorchTensor ---

BufferTorchTensor::BufferTorchTensor(torch::stable::Tensor tensor) : tensor_(std::move(tensor)) {
    this->ptr  = tensor_.data_ptr();
    int64_t numel = tensor_.numel();
    // Compute item size from scalar type
    auto stype = tensor_.scalar_type();
    // Use the internal scalar size map after converting
    auto it = stableToInternal.find(stype);
    if (it != stableToInternal.end()) {
        this->size = numel * ::Tensor::scalarSize.at(it->second);
    } else {
        // Fallback: try to get storage size
        int64_t storage_size;
        aoti_torch_get_storage_size(tensor_.get(), &storage_size);
        this->size = storage_size;
    }

    if (tensor_.is_cuda()) {
        this->device.type = Device::CUDA;
        this->device.idx  = tensor_.get_device();
    } else {
        this->device.type = Device::CPU;
        this->device.idx  = 0;
    }
}

// --- TorchOpContext ---

TorchOpContext::TorchOpContext() {
    // Get current CUDA device index
    int32_t device_index = torch::stable::accelerator::getCurrentDeviceIndex();
    void *stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
    stackCUDAStreams.push(reinterpret_cast<cudaStream_t>(stream_ptr));
}

TorchOpContext::~TorchOpContext() {
    stackCUDAStreams.pop();
}

// --- from_torch ---

::Tensor from_torch(torch::stable::Tensor input) {
    ::Tensor result;

    const int64_t ndims = input.dim();
    for (int64_t i = 0; i < ndims; i++) {
        result.shape.dataExtent.push_back(int_cast<int>(input.size(i)));
        result.shape.dataStride.push_back(int_cast<int>(input.stride(i)));
    }

    auto stype = input.scalar_type();
    auto it    = stableToInternal.find(stype);
    if (it == stableToInternal.end()) {
        throw std::runtime_error("Unsupported scalar type in from_torch");
    }
    result.scalarType = it->second;
    result.buffer     = std::make_shared<BufferTorchTensor>(std::move(input));

    ::Tensor::lockBuffer(result.buffer, getCurrentCUDAStream());

    return result;
}

// --- to_torch ---

torch::stable::Tensor to_torch(::Tensor input, bool inplace) {
    assert(input.is_contiguous());

    int64_t ndims = input.ndims();
    std::vector<int64_t> sizes(ndims);
    std::vector<int64_t> strides(ndims);

    // Compute contiguous strides
    int64_t stride = 1;
    for (int64_t i = ndims - 1; i >= 0; i--) {
        sizes[i]   = input.size(i);
        strides[i] = stride;
        stride *= sizes[i];
    }

    int32_t dtype = internal_dtype_to_aoti(input.scalar_type());

    int32_t device_type;
    int32_t device_index;
    if (input.device().type == Device::CPU) {
        device_type  = aoti_torch_device_type_cpu();
        device_index = 0;
    } else {
        device_type  = aoti_torch_device_type_cuda();
        device_index = input.device().idx;
    }

    if (inplace) {
        // Create a view of the existing data
        AtenTensorHandle handle;
        TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
            input.data_ptr(), ndims, sizes.data(), strides.data(), 0, dtype, device_type, device_index, &handle));
        return torch::stable::Tensor(handle);
    } else {
        // Allocate new tensor and copy
        AtenTensorHandle handle;
        TORCH_ERROR_CODE_CHECK(
            aoti_torch_empty_strided(ndims, sizes.data(), strides.data(), dtype, device_type, device_index, &handle));
        torch::stable::Tensor result(handle);

        // Copy data from internal tensor to the new stable tensor
        ::Tensor dst = from_torch(result);
        dst.copy_(input);

        return result;
    }
}

// --- Python <-> stable::Tensor conversion ---

// Helper to call a Python method that returns an int
static int64_t py_call_int_method(PyObject *obj, const char *method) {
    PyObject *result = PyObject_CallMethod(obj, method, NULL);
    if (!result) {
        PyErr_Print();
        throw std::runtime_error(format("Failed to call {}", method));
    }
    int64_t val = PyLong_AsLongLong(result);
    Py_DECREF(result);
    return val;
}

static std::string pyobject_to_utf8_string(PyObject *obj, const char *context) {
    PyObject *utf8 = PyUnicode_AsUTF8String(obj);
    if (!utf8) {
        PyErr_Print();
        throw std::runtime_error(format("Failed to convert {} to UTF-8", context));
    }

    char *data      = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(utf8, &data, &size) < 0) {
        Py_DECREF(utf8);
        PyErr_Print();
        throw std::runtime_error(format("Failed to read UTF-8 bytes for {}", context));
    }

    std::string result(data, static_cast<size_t>(size));
    Py_DECREF(utf8);
    return result;
}

// Map Python dtype name to aoti dtype int
static int32_t dtype_name_to_aoti(const char *name) {
    if (strcmp(name, "torch.uint8") == 0)
        return aoti_torch_dtype_uint8();
    if (strcmp(name, "torch.int8") == 0)
        return aoti_torch_dtype_int8();
    if (strcmp(name, "torch.int16") == 0)
        return aoti_torch_dtype_int16();
    if (strcmp(name, "torch.int32") == 0)
        return aoti_torch_dtype_int32();
    if (strcmp(name, "torch.int64") == 0)
        return aoti_torch_dtype_int64();
    if (strcmp(name, "torch.float16") == 0)
        return aoti_torch_dtype_float16();
    if (strcmp(name, "torch.float32") == 0)
        return aoti_torch_dtype_float32();
    if (strcmp(name, "torch.float64") == 0)
        return aoti_torch_dtype_float64();
    if (strcmp(name, "torch.bfloat16") == 0)
        return aoti_torch_dtype_bfloat16();
    if (strcmp(name, "torch.float8_e4m3fn") == 0)
        return aoti_torch_dtype_float8_e4m3fn();
    if (strcmp(name, "torch.float8_e5m2") == 0)
        return aoti_torch_dtype_float8_e5m2();
    throw std::runtime_error(format("Unsupported dtype: {}", name));
}

int32_t py_dtype_to_aoti_dtype(PyObject *dtype_obj) {
    PyObject *str = PyObject_Str(dtype_obj);
    if (!str) {
        PyErr_Print();
        throw std::runtime_error("Failed to stringify dtype object");
    }
    std::string name = pyobject_to_utf8_string(str, "dtype");
    int32_t result   = dtype_name_to_aoti(name.c_str());
    Py_DECREF(str);
    return result;
}

void py_device_to_aoti(PyObject *device_obj, int32_t &device_type, int32_t &device_index) {
    PyObject *type_attr = PyObject_GetAttrString(device_obj, "type");
    if (!type_attr) {
        PyErr_Print();
        throw std::runtime_error("Failed to get device.type");
    }
    std::string type_str = pyobject_to_utf8_string(type_attr, "device.type");
    if (type_str == "cuda") {
        device_type = aoti_torch_device_type_cuda();
    } else {
        device_type = aoti_torch_device_type_cpu();
    }
    Py_DECREF(type_attr);

    PyObject *index_attr = PyObject_GetAttrString(device_obj, "index");
    if (index_attr && index_attr != Py_None) {
        device_index = (int32_t)PyLong_AsLong(index_attr);
    } else {
        device_index = 0;
    }
    Py_XDECREF(index_attr);
}

torch::stable::Tensor py_to_tensor(PyObject *obj) {
    if (!obj || obj == Py_None) {
        throw std::runtime_error("Cannot convert None to stable::Tensor");
    }

    // Get data_ptr
    intptr_t data = (intptr_t)py_call_int_method(obj, "data_ptr");

    // Get shape
    PyObject *shape = PyObject_GetAttrString(obj, "shape");
    Py_ssize_t ndim = PyTuple_Size(shape);
    std::vector<int64_t> sizes(ndim);
    for (Py_ssize_t i = 0; i < ndim; i++) {
        sizes[i] = PyLong_AsLongLong(PyTuple_GetItem(shape, i));
    }
    Py_DECREF(shape);

    // Get strides
    PyObject *strides_obj = PyObject_CallMethod(obj, "stride", NULL);
    std::vector<int64_t> strides(ndim);
    for (Py_ssize_t i = 0; i < ndim; i++) {
        strides[i] = PyLong_AsLongLong(PyTuple_GetItem(strides_obj, i));
    }
    Py_DECREF(strides_obj);

    // Get dtype
    PyObject *dtype_obj = PyObject_GetAttrString(obj, "dtype");
    int32_t dtype = py_dtype_to_aoti_dtype(dtype_obj);
    Py_DECREF(dtype_obj);

    // Get device
    PyObject *device_obj = PyObject_GetAttrString(obj, "device");
    int32_t device_type, device_index;
    py_device_to_aoti(device_obj, device_type, device_index);
    Py_DECREF(device_obj);

    // Get storage offset
    PyObject *offset_obj = PyObject_CallMethod(obj, "storage_offset", NULL);
    int64_t storage_offset = PyLong_AsLongLong(offset_obj);
    Py_DECREF(offset_obj);

    // Create AtenTensorHandle from blob
    // Note: create_tensor_from_blob creates a non-owning view, so the Python tensor
    // must be kept alive by the caller. We use Py_INCREF on the source object
    // in higher-level code when needed.
    AtenTensorHandle handle;
    TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
        (void *)data, ndim, sizes.data(), strides.data(), storage_offset, dtype, device_type, device_index, &handle));

    return torch::stable::Tensor(handle);
}

PyObject *tensor_to_py(const torch::stable::Tensor &tensor) {
    // Strategy: call torch.empty_strided() from Python, then copy data using
    // the internal tensor copy mechanism. We create an internal Tensor from
    // both source and destination and use the internal copy_ method.

    int64_t ndim = tensor.dim();
    std::vector<int64_t> sizes(ndim);
    std::vector<int64_t> strides_vec(ndim);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; i--) {
        sizes[i]       = tensor.size(i);
        strides_vec[i] = stride;
        stride *= sizes[i];
    }

    // Import torch and call torch.empty_strided
    PyObject *torch_mod = PyImport_ImportModule("torch");
    if (!torch_mod) {
        PyErr_Print();
        throw std::runtime_error("Failed to import torch");
    }

    // Build sizes tuple
    PyObject *py_sizes = PyTuple_New(ndim);
    for (int64_t i = 0; i < ndim; i++) {
        PyObject *item = PyLong_FromLongLong(sizes[i]);
        if (!item || PyTuple_SetItem(py_sizes, i, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(py_sizes);
            Py_DECREF(torch_mod);
            PyErr_Print();
            throw std::runtime_error("Failed to build sizes tuple");
        }
    }

    // Build strides tuple
    PyObject *py_strides = PyTuple_New(ndim);
    for (int64_t i = 0; i < ndim; i++) {
        PyObject *item = PyLong_FromLongLong(strides_vec[i]);
        if (!item || PyTuple_SetItem(py_strides, i, item) < 0) {
            Py_XDECREF(item);
            Py_DECREF(py_strides);
            Py_DECREF(py_sizes);
            Py_DECREF(torch_mod);
            PyErr_Print();
            throw std::runtime_error("Failed to build strides tuple");
        }
    }

    // Build dtype - get from tensor and convert to Python torch.dtype
    int32_t dtype_int;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(tensor.get(), &dtype_int));

    // Map aoti dtype int to Python dtype attribute name
    const char *dtype_attr = nullptr;
    if (dtype_int == aoti_torch_dtype_uint8())
        dtype_attr = "uint8";
    else if (dtype_int == aoti_torch_dtype_int8())
        dtype_attr = "int8";
    else if (dtype_int == aoti_torch_dtype_int16())
        dtype_attr = "int16";
    else if (dtype_int == aoti_torch_dtype_int32())
        dtype_attr = "int32";
    else if (dtype_int == aoti_torch_dtype_int64())
        dtype_attr = "int64";
    else if (dtype_int == aoti_torch_dtype_float16())
        dtype_attr = "float16";
    else if (dtype_int == aoti_torch_dtype_float32())
        dtype_attr = "float32";
    else if (dtype_int == aoti_torch_dtype_float64())
        dtype_attr = "float64";
    else if (dtype_int == aoti_torch_dtype_bfloat16())
        dtype_attr = "bfloat16";
    else if (dtype_int == aoti_torch_dtype_float8_e4m3fn())
        dtype_attr = "float8_e4m3fn";
    else if (dtype_int == aoti_torch_dtype_float8_e5m2())
        dtype_attr = "float8_e5m2";
    else
        throw std::runtime_error("Unsupported dtype in tensor_to_py");

    PyObject *py_dtype = PyObject_GetAttrString(torch_mod, dtype_attr);

    // Build device string
    int32_t device_type_int, device_idx;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(tensor.get(), &device_type_int));
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_index(tensor.get(), &device_idx));

    PyObject *py_device;
    if (device_type_int == aoti_torch_device_type_cuda()) {
        py_device = PyUnicode_FromFormat("cuda:%d", device_idx);
    } else {
        py_device = PyUnicode_FromString("cpu");
    }

    // Call torch.empty_strided(sizes, strides, dtype=dtype, device=device)
    PyObject *empty_strided = PyObject_GetAttrString(torch_mod, "empty_strided");
    PyObject *args          = PyTuple_Pack(2, py_sizes, py_strides);
    PyObject *kwargs        = PyDict_New();
    PyDict_SetItemString(kwargs, "dtype", py_dtype);
    PyDict_SetItemString(kwargs, "device", py_device);

    PyObject *result_tensor = PyObject_Call(empty_strided, args, kwargs);

    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(empty_strided);
    Py_DECREF(py_device);
    Py_DECREF(py_dtype);
    Py_DECREF(py_strides);
    Py_DECREF(py_sizes);
    Py_DECREF(torch_mod);

    if (!result_tensor) {
        PyErr_Print();
        throw std::runtime_error("Failed to create output tensor via torch.empty_strided");
    }

    // Copy data: create stable tensor from the Python result, then use internal copy
    torch::stable::Tensor dst_stable = py_to_tensor(result_tensor);
    ::Tensor dst                     = from_torch(dst_stable);
    ::Tensor src                     = from_torch(tensor);
    dst.copy_(src);

    return result_tensor; // new reference
}

std::optional<torch::stable::Tensor> py_to_optional_tensor(PyObject *obj) {
    if (!obj || obj == Py_None) {
        return std::nullopt;
    }
    return py_to_tensor(obj);
}
