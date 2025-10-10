#pragma once

#include"helix/common/types.hpp"
#include"helix/core/vector_utils.hpp"
#include<string>
#include<vector>

namespace helix {

class Dataset {
public:
    Dataset()=default;
    
    void loadFvecs(const std::string& path);
    void loadIvecs(const std::string& path);
    
    void generateRandom(idx_t n,dim_t dim,float min=0.0f,float max=1.0f,int seed=42);
    void generateGaussian(idx_t n,dim_t dim,float mean=0.0f,float stddev=1.0f,int seed=42);
    
    const float* data() const { return vectors_.data(); }
    float* data() { return vectors_.data(); }
    
    const int* labels() const { return labels_.data(); }
    int* labels() { return labels_.data(); }
    
    idx_t size() const { return nrows_; }
    dim_t dimension() const { return dim_; }
    
    void normalize();
    
    void setLabels(idx_t n,dim_t k,std::vector<int>&& labels) {
        nrows_=n;
        dim_=k;
        labels_=std::move(labels);
    }
    
private:
    std::vector<float> vectors_;
    std::vector<int> labels_;
    idx_t nrows_=0;
    dim_t dim_=0;
};

struct BenchmarkDataset {
    Dataset train;
    Dataset database;
    Dataset queries;
    Dataset groundtruth;
    
    void loadSIFT1M(const std::string& basePath);
    void generateSynthetic(idx_t ntrain,idx_t nbase,idx_t nquery,dim_t dim);
};

}

