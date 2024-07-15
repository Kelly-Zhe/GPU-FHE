#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kernels.h"
namespace py = pybind11;
PYBIND11_MODULE(fhe_gpu, m) {
    m.def("moddown_core_cuda", &moddown_core_cuda);
    m.def("modup_core_cuda", &modup_core_cuda);
}