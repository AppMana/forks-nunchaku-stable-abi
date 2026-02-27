// Main module definition for nunchaku._C using Python stable ABI (Py_LIMITED_API)
// and Torch stable ABI (TORCH_STABLE_ONLY).
//
// This replaces pybind.cpp with raw CPython type definitions using PyType_Spec.
// Tensor arguments are converted via Python C API -> aoti_torch_create_tensor_from_blob.

#define Py_LIMITED_API 0x03090000
#include <Python.h>

#include <torch/csrc/stable/library.h>

#include "flux.h"
#include "sana.h"
#include "gemm.h"
#include "gemm88.h"
#include "ops.h"
#include "utils.h"

// ============================================================================
// Helper macros for CPython type methods
// ============================================================================

// Helper to convert a Python tensor arg to stable::Tensor, handling contiguous
static torch::stable::Tensor py_tensor_arg(PyObject *obj) {
    // Call .contiguous() on the tensor
    PyObject *contig = PyObject_CallMethod(obj, "contiguous", NULL);
    if (!contig) {
        PyErr_Print();
        throw std::runtime_error("Failed to call .contiguous()");
    }
    // We need to keep contiguous alive, so incref; caller's Python scope manages it
    torch::stable::Tensor result = py_to_tensor(contig);
    Py_DECREF(contig);
    return result;
}

static std::optional<torch::stable::Tensor> py_optional_tensor_arg(PyObject *obj) {
    if (!obj || obj == Py_None) {
        return std::nullopt;
    }
    return py_tensor_arg(obj);
}

// ============================================================================
// PyQuantizedFluxModel
// ============================================================================

struct PyQuantizedFluxModel {
    PyObject_HEAD
    QuantizedFluxModel *model;
};

static void PyQuantizedFluxModel_dealloc(PyQuantizedFluxModel *self) {
    delete self->model;
    PyObject_Free(self);
}

static PyObject *PyQuantizedFluxModel_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyQuantizedFluxModel *self = (PyQuantizedFluxModel *)PyType_GenericAlloc(type, 0);
    if (self) {
        self->model = new QuantizedFluxModel();
    }
    return (PyObject *)self;
}

static PyObject *PyQuantizedFluxModel_init_method(PyQuantizedFluxModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"use_fp4", "offload", "bf16", "deviceId", NULL};
    int use_fp4, offload, bf16;
    int8_t deviceId;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "pppb", (char **)kwlist, &use_fp4, &offload, &bf16, &deviceId))
        return NULL;
    try {
        self->model->init(use_fp4, offload, bf16, deviceId);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_set_residual_callback(PyQuantizedFluxModel *self, PyObject *args) {
    PyObject *callback;
    if (!PyArg_ParseTuple(args, "O", &callback))
        return NULL;
    try {
        self->model->set_residual_callback(callback);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_reset(PyQuantizedFluxModel *self, PyObject *) {
    try {
        self->model->reset();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_load(PyQuantizedFluxModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"path", "partial", NULL};
    const char *path;
    int partial = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", (char **)kwlist, &path, &partial))
        return NULL;
    try {
        self->model->load(path, partial);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_loadDict(PyQuantizedFluxModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"dict", "partial", NULL};
    PyObject *dict;
    int partial = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|p", (char **)kwlist, &PyDict_Type, &dict, &partial))
        return NULL;
    try {
        self->model->loadDictPy(dict, partial);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_forward(PyQuantizedFluxModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"hidden_states",
                                   "encoder_hidden_states",
                                   "temb",
                                   "rotary_emb_img",
                                   "rotary_emb_context",
                                   "rotary_emb_single",
                                   "controlnet_block_samples",
                                   "controlnet_single_block_samples",
                                   "skip_first_layer",
                                   NULL};
    PyObject *py_hs, *py_ehs, *py_temb, *py_rei, *py_rec, *py_res;
    PyObject *py_cbs = Py_None, *py_csbs = Py_None;
    int skip_first = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOO|OOp", (char **)kwlist,
                                     &py_hs, &py_ehs, &py_temb, &py_rei, &py_rec, &py_res,
                                     &py_cbs, &py_csbs, &skip_first))
        return NULL;
    try {
        auto result = self->model->forward(
            py_tensor_arg(py_hs), py_tensor_arg(py_ehs), py_tensor_arg(py_temb),
            py_tensor_arg(py_rei), py_tensor_arg(py_rec), py_tensor_arg(py_res),
            py_optional_tensor_arg(py_cbs), py_optional_tensor_arg(py_csbs), skip_first);
        return tensor_to_py(result);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedFluxModel_forward_layer(PyQuantizedFluxModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"idx",
                                   "hidden_states",
                                   "encoder_hidden_states",
                                   "temb",
                                   "rotary_emb_img",
                                   "rotary_emb_context",
                                   "controlnet_block_samples",
                                   "controlnet_single_block_samples",
                                   NULL};
    int64_t idx;
    PyObject *py_hs, *py_ehs, *py_temb, *py_rei, *py_rec;
    PyObject *py_cbs = Py_None, *py_csbs = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "LOOOOO|OO", (char **)kwlist,
                                     &idx, &py_hs, &py_ehs, &py_temb, &py_rei, &py_rec,
                                     &py_cbs, &py_csbs))
        return NULL;
    try {
        auto [hs, ehs] = self->model->forward_layer(
            idx, py_tensor_arg(py_hs), py_tensor_arg(py_ehs), py_tensor_arg(py_temb),
            py_tensor_arg(py_rei), py_tensor_arg(py_rec),
            py_optional_tensor_arg(py_cbs), py_optional_tensor_arg(py_csbs));
        PyObject *py_hs_out  = tensor_to_py(hs);
        PyObject *py_ehs_out = tensor_to_py(ehs);
        return PyTuple_Pack(2, py_hs_out, py_ehs_out);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedFluxModel_forward_layer_ip_adapter(PyQuantizedFluxModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"idx",
                                   "hidden_states",
                                   "encoder_hidden_states",
                                   "temb",
                                   "rotary_emb_img",
                                   "rotary_emb_context",
                                   "controlnet_block_samples",
                                   "controlnet_single_block_samples",
                                   NULL};
    int64_t idx;
    PyObject *py_hs, *py_ehs, *py_temb, *py_rei, *py_rec;
    PyObject *py_cbs = Py_None, *py_csbs = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "LOOOOO|OO", (char **)kwlist,
                                     &idx, &py_hs, &py_ehs, &py_temb, &py_rei, &py_rec,
                                     &py_cbs, &py_csbs))
        return NULL;
    try {
        auto result = self->model->forward_layer_ip_adapter(
            idx, py_tensor_arg(py_hs), py_tensor_arg(py_ehs), py_tensor_arg(py_temb),
            py_tensor_arg(py_rei), py_tensor_arg(py_rec),
            py_optional_tensor_arg(py_cbs), py_optional_tensor_arg(py_csbs));
        PyObject *py_hs_out  = tensor_to_py(result.hidden_states);
        PyObject *py_ehs_out = tensor_to_py(result.encoder_hidden_states);
        PyObject *py_ipq_out = tensor_to_py(result.ip_query);
        return PyTuple_Pack(3, py_hs_out, py_ehs_out, py_ipq_out);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedFluxModel_forward_single_layer(PyQuantizedFluxModel *self, PyObject *args) {
    int64_t idx;
    PyObject *py_hs, *py_temb, *py_res;
    if (!PyArg_ParseTuple(args, "LOOO", &idx, &py_hs, &py_temb, &py_res))
        return NULL;
    try {
        auto result = self->model->forward_single_layer(idx, py_tensor_arg(py_hs), py_tensor_arg(py_temb), py_tensor_arg(py_res));
        return tensor_to_py(result);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedFluxModel_norm_one_forward(PyQuantizedFluxModel *self, PyObject *args) {
    int64_t idx;
    PyObject *py_hs, *py_temb;
    if (!PyArg_ParseTuple(args, "LOO", &idx, &py_hs, &py_temb))
        return NULL;
    try {
        auto result = self->model->norm_one_forward(idx, py_tensor_arg(py_hs), py_tensor_arg(py_temb));
        PyObject *t0 = tensor_to_py(result.x);
        PyObject *t1 = tensor_to_py(result.gate_msa);
        PyObject *t2 = tensor_to_py(result.shift_mlp);
        PyObject *t3 = tensor_to_py(result.scale_mlp);
        PyObject *t4 = tensor_to_py(result.gate_mlp);
        return PyTuple_Pack(5, t0, t1, t2, t3, t4);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedFluxModel_startDebug(PyQuantizedFluxModel *self, PyObject *) {
    self->model->startDebug();
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_stopDebug(PyQuantizedFluxModel *self, PyObject *) {
    self->model->stopDebug();
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_getDebugResults(PyQuantizedFluxModel *self, PyObject *) {
    try {
        return self->model->getDebugResultsPy();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedFluxModel_setLoraScale(PyQuantizedFluxModel *self, PyObject *args) {
    int skipRanks;
    float scale;
    if (!PyArg_ParseTuple(args, "if", &skipRanks, &scale))
        return NULL;
    try {
        self->model->setLoraScale(skipRanks, scale);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_setAttentionImpl(PyQuantizedFluxModel *self, PyObject *args) {
    const char *name;
    PyObject *attn_func;
    if (!PyArg_ParseTuple(args, "sO", &name, &attn_func))
        return NULL;
    try {
        self->model->setAttentionImpl(name, attn_func);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedFluxModel_isBF16(PyQuantizedFluxModel *self, PyObject *) {
    try {
        return PyBool_FromLong(self->model->isBF16());
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyMethodDef PyQuantizedFluxModel_methods[] = {
    {"init", (PyCFunction)PyQuantizedFluxModel_init_method, METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_residual_callback", (PyCFunction)PyQuantizedFluxModel_set_residual_callback, METH_VARARGS, NULL},
    {"reset", (PyCFunction)PyQuantizedFluxModel_reset, METH_NOARGS, NULL},
    {"load", (PyCFunction)PyQuantizedFluxModel_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"loadDict", (PyCFunction)PyQuantizedFluxModel_loadDict, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward", (PyCFunction)PyQuantizedFluxModel_forward, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward_layer", (PyCFunction)PyQuantizedFluxModel_forward_layer, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward_layer_ip_adapter", (PyCFunction)PyQuantizedFluxModel_forward_layer_ip_adapter, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward_single_layer", (PyCFunction)PyQuantizedFluxModel_forward_single_layer, METH_VARARGS, NULL},
    {"norm_one_forward", (PyCFunction)PyQuantizedFluxModel_norm_one_forward, METH_VARARGS, NULL},
    {"startDebug", (PyCFunction)PyQuantizedFluxModel_startDebug, METH_NOARGS, NULL},
    {"stopDebug", (PyCFunction)PyQuantizedFluxModel_stopDebug, METH_NOARGS, NULL},
    {"getDebugResults", (PyCFunction)PyQuantizedFluxModel_getDebugResults, METH_NOARGS, NULL},
    {"setLoraScale", (PyCFunction)PyQuantizedFluxModel_setLoraScale, METH_VARARGS, NULL},
    {"setAttentionImpl", (PyCFunction)PyQuantizedFluxModel_setAttentionImpl, METH_VARARGS, NULL},
    {"isBF16", (PyCFunction)PyQuantizedFluxModel_isBF16, METH_NOARGS, NULL},
    {NULL}};

static PyType_Slot PyQuantizedFluxModel_slots[] = {
    {Py_tp_dealloc, (void *)PyQuantizedFluxModel_dealloc},
    {Py_tp_new, (void *)PyQuantizedFluxModel_new},
    {Py_tp_methods, PyQuantizedFluxModel_methods},
    {0, NULL}};

static PyType_Spec PyQuantizedFluxModel_spec = {
    "nunchaku._C.QuantizedFluxModel",
    sizeof(PyQuantizedFluxModel),
    0,
    Py_TPFLAGS_DEFAULT,
    PyQuantizedFluxModel_slots};

// ============================================================================
// PyQuantizedSanaModel
// ============================================================================

struct PyQuantizedSanaModel {
    PyObject_HEAD
    QuantizedSanaModel *model;
};

static void PyQuantizedSanaModel_dealloc(PyQuantizedSanaModel *self) {
    delete self->model;
    PyObject_Free(self);
}

static PyObject *PyQuantizedSanaModel_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyQuantizedSanaModel *self = (PyQuantizedSanaModel *)PyType_GenericAlloc(type, 0);
    if (self) {
        self->model = new QuantizedSanaModel();
    }
    return (PyObject *)self;
}

static PyObject *PyQuantizedSanaModel_init_method(PyQuantizedSanaModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"config", "pag_layers", "use_fp4", "bf16", "deviceId", NULL};
    PyObject *config;
    PyObject *pag_layers_obj;
    int use_fp4, bf16;
    int8_t deviceId;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Oppb", (char **)kwlist,
                                     &PyDict_Type, &config, &pag_layers_obj, &use_fp4, &bf16, &deviceId))
        return NULL;

    // Parse pag_layers list
    std::vector<int> pag_layers;
    if (pag_layers_obj && pag_layers_obj != Py_None) {
        Py_ssize_t n = PyList_Size(pag_layers_obj);
        for (Py_ssize_t i = 0; i < n; i++) {
            pag_layers.push_back((int)PyLong_AsLong(PyList_GetItem(pag_layers_obj, i)));
        }
    }

    try {
        self->model->init(config, pag_layers, use_fp4, bf16, deviceId);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedSanaModel_reset(PyQuantizedSanaModel *self, PyObject *) {
    try { self->model->reset(); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedSanaModel_load(PyQuantizedSanaModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"path", "partial", NULL};
    const char *path; int partial = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", (char **)kwlist, &path, &partial)) return NULL;
    try { self->model->load(path, partial); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedSanaModel_loadDict(PyQuantizedSanaModel *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"dict", "partial", NULL};
    PyObject *dict; int partial = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!|p", (char **)kwlist, &PyDict_Type, &dict, &partial)) return NULL;
    try { self->model->loadDictPy(dict, partial); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedSanaModel_forward(PyQuantizedSanaModel *self, PyObject *args) {
    PyObject *py_hs, *py_ehs, *py_ts, *py_csi, *py_cst;
    int H, W, pag, cfg, skip_first = 0;
    if (!PyArg_ParseTuple(args, "OOOOOiippI|p", &py_hs, &py_ehs, &py_ts, &py_csi, &py_cst, &H, &W, &pag, &cfg, &skip_first))
        return NULL;
    try {
        auto result = self->model->forward(
            py_tensor_arg(py_hs), py_tensor_arg(py_ehs), py_tensor_arg(py_ts),
            py_tensor_arg(py_csi), py_tensor_arg(py_cst), H, W, pag, cfg, skip_first);
        return tensor_to_py(result);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedSanaModel_forward_layer(PyQuantizedSanaModel *self, PyObject *args) {
    int64_t idx;
    PyObject *py_hs, *py_ehs, *py_ts, *py_csi, *py_cst;
    int H, W, pag, cfg;
    if (!PyArg_ParseTuple(args, "LOOOOOiipp", &idx, &py_hs, &py_ehs, &py_ts, &py_csi, &py_cst, &H, &W, &pag, &cfg))
        return NULL;
    try {
        auto result = self->model->forward_layer(
            idx, py_tensor_arg(py_hs), py_tensor_arg(py_ehs), py_tensor_arg(py_ts),
            py_tensor_arg(py_csi), py_tensor_arg(py_cst), H, W, pag, cfg);
        return tensor_to_py(result);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

static PyObject *PyQuantizedSanaModel_startDebug(PyQuantizedSanaModel *self, PyObject *) { self->model->startDebug(); Py_RETURN_NONE; }
static PyObject *PyQuantizedSanaModel_stopDebug(PyQuantizedSanaModel *self, PyObject *) { self->model->stopDebug(); Py_RETURN_NONE; }
static PyObject *PyQuantizedSanaModel_getDebugResults(PyQuantizedSanaModel *self, PyObject *) {
    try { return self->model->getDebugResultsPy(); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
}

static PyMethodDef PyQuantizedSanaModel_methods[] = {
    {"init", (PyCFunction)PyQuantizedSanaModel_init_method, METH_VARARGS | METH_KEYWORDS, NULL},
    {"reset", (PyCFunction)PyQuantizedSanaModel_reset, METH_NOARGS, NULL},
    {"load", (PyCFunction)PyQuantizedSanaModel_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"loadDict", (PyCFunction)PyQuantizedSanaModel_loadDict, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward", (PyCFunction)PyQuantizedSanaModel_forward, METH_VARARGS, NULL},
    {"forward_layer", (PyCFunction)PyQuantizedSanaModel_forward_layer, METH_VARARGS, NULL},
    {"startDebug", (PyCFunction)PyQuantizedSanaModel_startDebug, METH_NOARGS, NULL},
    {"stopDebug", (PyCFunction)PyQuantizedSanaModel_stopDebug, METH_NOARGS, NULL},
    {"getDebugResults", (PyCFunction)PyQuantizedSanaModel_getDebugResults, METH_NOARGS, NULL},
    {NULL}};

static PyType_Slot PyQuantizedSanaModel_slots[] = {
    {Py_tp_dealloc, (void *)PyQuantizedSanaModel_dealloc},
    {Py_tp_new, (void *)PyQuantizedSanaModel_new},
    {Py_tp_methods, PyQuantizedSanaModel_methods},
    {0, NULL}};

static PyType_Spec PyQuantizedSanaModel_spec = {
    "nunchaku._C.QuantizedSanaModel",
    sizeof(PyQuantizedSanaModel),
    0,
    Py_TPFLAGS_DEFAULT,
    PyQuantizedSanaModel_slots};

// ============================================================================
// PyQuantizedGEMM
// ============================================================================

struct PyQuantizedGEMM {
    PyObject_HEAD
    QuantizedGEMM *model;
};

static void PyQuantizedGEMM_dealloc(PyQuantizedGEMM *self) { delete self->model; PyObject_Free(self); }

static PyObject *PyQuantizedGEMM_new(PyTypeObject *type, PyObject *, PyObject *) {
    PyQuantizedGEMM *self = (PyQuantizedGEMM *)PyType_GenericAlloc(type, 0);
    if (self) self->model = new QuantizedGEMM();
    return (PyObject *)self;
}

static PyObject *PyQuantizedGEMM_init_method(PyQuantizedGEMM *self, PyObject *args) {
    int64_t in_f, out_f; int bias, use_fp4, bf16; int8_t did;
    if (!PyArg_ParseTuple(args, "LLpppb", &in_f, &out_f, &bias, &use_fp4, &bf16, &did)) return NULL;
    try { self->model->init(in_f, out_f, bias, use_fp4, bf16, did); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM_reset(PyQuantizedGEMM *self, PyObject *) {
    try { self->model->reset(); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM_load(PyQuantizedGEMM *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"path", "partial", NULL};
    const char *path; int partial = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", (char **)kwlist, &path, &partial)) return NULL;
    try { self->model->load(path, partial); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM_forward(PyQuantizedGEMM *self, PyObject *args) {
    PyObject *py_x;
    if (!PyArg_ParseTuple(args, "O", &py_x)) return NULL;
    try {
        auto result = self->model->forward(py_tensor_arg(py_x));
        return tensor_to_py(result);
    } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
}

static PyObject *PyQuantizedGEMM_quantize(PyQuantizedGEMM *self, PyObject *args) {
    PyObject *py_x; int fuse_glu;
    if (!PyArg_ParseTuple(args, "Op", &py_x, &fuse_glu)) return NULL;
    try { self->model->quantize(py_tensor_arg(py_x), fuse_glu); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM_startDebug(PyQuantizedGEMM *self, PyObject *) { self->model->startDebug(); Py_RETURN_NONE; }
static PyObject *PyQuantizedGEMM_stopDebug(PyQuantizedGEMM *self, PyObject *) { self->model->stopDebug(); Py_RETURN_NONE; }
static PyObject *PyQuantizedGEMM_getDebugResults(PyQuantizedGEMM *self, PyObject *) {
    try { return self->model->getDebugResultsPy(); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
}

static PyMethodDef PyQuantizedGEMM_methods[] = {
    {"init", (PyCFunction)PyQuantizedGEMM_init_method, METH_VARARGS, NULL},
    {"reset", (PyCFunction)PyQuantizedGEMM_reset, METH_NOARGS, NULL},
    {"load", (PyCFunction)PyQuantizedGEMM_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward", (PyCFunction)PyQuantizedGEMM_forward, METH_VARARGS, NULL},
    {"quantize", (PyCFunction)PyQuantizedGEMM_quantize, METH_VARARGS, NULL},
    {"startDebug", (PyCFunction)PyQuantizedGEMM_startDebug, METH_NOARGS, NULL},
    {"stopDebug", (PyCFunction)PyQuantizedGEMM_stopDebug, METH_NOARGS, NULL},
    {"getDebugResults", (PyCFunction)PyQuantizedGEMM_getDebugResults, METH_NOARGS, NULL},
    {NULL}};

static PyType_Slot PyQuantizedGEMM_slots[] = {
    {Py_tp_dealloc, (void *)PyQuantizedGEMM_dealloc},
    {Py_tp_new, (void *)PyQuantizedGEMM_new},
    {Py_tp_methods, PyQuantizedGEMM_methods},
    {0, NULL}};

static PyType_Spec PyQuantizedGEMM_spec = {
    "nunchaku._C.QuantizedGEMM", sizeof(PyQuantizedGEMM), 0, Py_TPFLAGS_DEFAULT, PyQuantizedGEMM_slots};

// ============================================================================
// PyQuantizedGEMM88
// ============================================================================

struct PyQuantizedGEMM88 {
    PyObject_HEAD
    QuantizedGEMM88 *model;
};

static void PyQuantizedGEMM88_dealloc(PyQuantizedGEMM88 *self) { delete self->model; PyObject_Free(self); }

static PyObject *PyQuantizedGEMM88_new(PyTypeObject *type, PyObject *, PyObject *) {
    PyQuantizedGEMM88 *self = (PyQuantizedGEMM88 *)PyType_GenericAlloc(type, 0);
    if (self) self->model = new QuantizedGEMM88();
    return (PyObject *)self;
}

static PyObject *PyQuantizedGEMM88_init_method(PyQuantizedGEMM88 *self, PyObject *args) {
    int64_t in_f, out_f; int bias, bf16; int8_t did;
    if (!PyArg_ParseTuple(args, "LLppb", &in_f, &out_f, &bias, &bf16, &did)) return NULL;
    try { self->model->init(in_f, out_f, bias, bf16, did); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM88_reset(PyQuantizedGEMM88 *self, PyObject *) {
    try { self->model->reset(); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM88_load(PyQuantizedGEMM88 *self, PyObject *args, PyObject *kwds) {
    static const char *kwlist[] = {"path", "partial", NULL};
    const char *path; int partial = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", (char **)kwlist, &path, &partial)) return NULL;
    try { self->model->load(path, partial); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
    Py_RETURN_NONE;
}

static PyObject *PyQuantizedGEMM88_forward(PyQuantizedGEMM88 *self, PyObject *args) {
    PyObject *py_x;
    if (!PyArg_ParseTuple(args, "O", &py_x)) return NULL;
    try {
        auto result = self->model->forward(py_tensor_arg(py_x));
        return tensor_to_py(result);
    } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
}

static PyObject *PyQuantizedGEMM88_startDebug(PyQuantizedGEMM88 *self, PyObject *) { self->model->startDebug(); Py_RETURN_NONE; }
static PyObject *PyQuantizedGEMM88_stopDebug(PyQuantizedGEMM88 *self, PyObject *) { self->model->stopDebug(); Py_RETURN_NONE; }
static PyObject *PyQuantizedGEMM88_getDebugResults(PyQuantizedGEMM88 *self, PyObject *) {
    try { return self->model->getDebugResultsPy(); } catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return NULL; }
}

static PyMethodDef PyQuantizedGEMM88_methods[] = {
    {"init", (PyCFunction)PyQuantizedGEMM88_init_method, METH_VARARGS, NULL},
    {"reset", (PyCFunction)PyQuantizedGEMM88_reset, METH_NOARGS, NULL},
    {"load", (PyCFunction)PyQuantizedGEMM88_load, METH_VARARGS | METH_KEYWORDS, NULL},
    {"forward", (PyCFunction)PyQuantizedGEMM88_forward, METH_VARARGS, NULL},
    {"startDebug", (PyCFunction)PyQuantizedGEMM88_startDebug, METH_NOARGS, NULL},
    {"stopDebug", (PyCFunction)PyQuantizedGEMM88_stopDebug, METH_NOARGS, NULL},
    {"getDebugResults", (PyCFunction)PyQuantizedGEMM88_getDebugResults, METH_NOARGS, NULL},
    {NULL}};

static PyType_Slot PyQuantizedGEMM88_slots[] = {
    {Py_tp_dealloc, (void *)PyQuantizedGEMM88_dealloc},
    {Py_tp_new, (void *)PyQuantizedGEMM88_new},
    {Py_tp_methods, PyQuantizedGEMM88_methods},
    {0, NULL}};

static PyType_Spec PyQuantizedGEMM88_spec = {
    "nunchaku._C.QuantizedGEMM88", sizeof(PyQuantizedGEMM88), 0, Py_TPFLAGS_DEFAULT, PyQuantizedGEMM88_slots};

// ============================================================================
// Utils submodule methods
// ============================================================================

static PyObject *utils_set_log_level(PyObject *self, PyObject *args) {
    const char *level;
    if (!PyArg_ParseTuple(args, "s", &level))
        return NULL;
    spdlog::set_level(spdlog::level::from_str(level));
    Py_RETURN_NONE;
}

static PyObject *utils_set_cuda_stack_limit(PyObject *self, PyObject *args) {
    int64_t newval;
    if (!PyArg_ParseTuple(args, "L", &newval))
        return NULL;
    try {
        nunchaku::utils::set_cuda_stack_limit(newval);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *utils_disable_memory_auto_release(PyObject *self, PyObject *) {
    try {
        nunchaku::utils::disable_memory_auto_release();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *utils_trim_memory(PyObject *self, PyObject *) {
    try {
        nunchaku::utils::trim_memory();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *utils_set_faster_i2f_mode(PyObject *self, PyObject *args) {
    const char *mode;
    if (!PyArg_ParseTuple(args, "s", &mode))
        return NULL;
    try {
        nunchaku::utils::set_faster_i2f_mode(mode);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyMethodDef utils_methods[] = {
    {"set_log_level", utils_set_log_level, METH_VARARGS, NULL},
    {"set_cuda_stack_limit", utils_set_cuda_stack_limit, METH_VARARGS, NULL},
    {"disable_memory_auto_release", utils_disable_memory_auto_release, METH_NOARGS, NULL},
    {"trim_memory", utils_trim_memory, METH_NOARGS, NULL},
    {"set_faster_i2f_mode", utils_set_faster_i2f_mode, METH_VARARGS, NULL},
    {NULL}};

static struct PyModuleDef utils_module_def = {
    PyModuleDef_HEAD_INIT,
    "utils",
    NULL,
    -1,
    utils_methods};

// ============================================================================
// Module init
// ============================================================================

PyMODINIT_FUNC PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",
        "Nunchaku C extension with stable ABI",
        -1,
        NULL};

    PyObject *m = PyModule_Create(&module_def);
    if (!m)
        return NULL;

    // Add types
    PyObject *flux_type = PyType_FromSpec(&PyQuantizedFluxModel_spec);
    if (!flux_type) return NULL;
    PyModule_AddObject(m, "QuantizedFluxModel", flux_type);

    PyObject *sana_type = PyType_FromSpec(&PyQuantizedSanaModel_spec);
    if (!sana_type) return NULL;
    PyModule_AddObject(m, "QuantizedSanaModel", sana_type);

    PyObject *gemm_type = PyType_FromSpec(&PyQuantizedGEMM_spec);
    if (!gemm_type) return NULL;
    PyModule_AddObject(m, "QuantizedGEMM", gemm_type);

    PyObject *gemm88_type = PyType_FromSpec(&PyQuantizedGEMM88_spec);
    if (!gemm88_type) return NULL;
    PyModule_AddObject(m, "QuantizedGEMM88", gemm88_type);

    // Add utils submodule
    PyObject *utils_mod = PyModule_Create(&utils_module_def);
    if (!utils_mod) return NULL;
    PyModule_AddObject(m, "utils", utils_mod);

    return m;
}
