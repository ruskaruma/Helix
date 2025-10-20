#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include"helix/common/types.hpp"
#include"helix/index/index_flat.hpp"

namespace py = pybind11;

static helix::SearchResults searchFlat(helix::IndexFlat& idx,py::array_t<float,py::array::c_style|py::array::forcecast> query,int k)
{
    auto buf=query.request();
    if(buf.ndim!=1) throw std::runtime_error("query must be 1D");
    if(buf.size!=idx.dimension()) throw std::runtime_error("dim mismatch");
    return idx.search(static_cast<float*>(buf.ptr),k);
}

static void addFlat(helix::IndexFlat& idx,py::array_t<float,py::array::c_style|py::array::forcecast> vectors)
{
    auto buf=vectors.request();
    if(buf.ndim!=2) throw std::runtime_error("vectors must be 2D");
    if(buf.shape[1]!=idx.dimension()) throw std::runtime_error("dim mismatch");
    idx.add(static_cast<float*>(buf.ptr),(helix::idx_t)buf.shape[0]);
}

static void trainFlat(helix::IndexFlat& idx,py::array_t<float,py::array::c_style|py::array::forcecast> vectors)
{
    auto buf=vectors.request();
    if(buf.ndim!=2) throw std::runtime_error("vectors must be 2D");
    if(buf.shape[1]!=idx.dimension()) throw std::runtime_error("dim mismatch");
    idx.train(static_cast<float*>(buf.ptr),(helix::idx_t)buf.shape[0]);
}

PYBIND11_MODULE(helix_py,m)
{
    py::class_<helix::IndexConfig>(m,"IndexConfig")
        .def(py::init<helix::dim_t,helix::MetricType,helix::IndexType>())
        .def_readonly("dimension",&helix::IndexConfig::dimension)
        .def_readonly("metric",&helix::IndexConfig::metric)
        .def_readonly("type",&helix::IndexConfig::type);

    py::enum_<helix::MetricType>(m,"MetricType")
        .value("L2",helix::MetricType::L2)
        .value("InnerProduct",helix::MetricType::InnerProduct)
        .value("Cosine",helix::MetricType::Cosine);

    py::enum_<helix::IndexType>(m,"IndexType")
        .value("Flat",helix::IndexType::Flat)
        .value("IVF",helix::IndexType::IVF)
        .value("PQ",helix::IndexType::PQ)
        .value("IVFPQ",helix::IndexType::IVFPQ)
        .value("HNSW",helix::IndexType::HNSW)
        .value("Hybrid",helix::IndexType::Hybrid);

    py::class_<helix::IndexFlat>(m,"IndexFlat")
        .def(py::init<const helix::IndexConfig&>())
        .def("train",&trainFlat)
        .def("add",&addFlat)
        .def("search",&searchFlat)
        .def("ntotal",&helix::IndexFlat::ntotal)
        .def("dimension",&helix::IndexFlat::dimension);
}

