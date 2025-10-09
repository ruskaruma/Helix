#pragma once

#include"helix/common/types.hpp"
#include<vector>
#include<cstring>

namespace helix {

template<typename T>
class Vector2D {
public:
    Vector2D()=default;
    Vector2D(idx_t rows,dim_t cols) : rows_(rows),cols_(cols)
    {
        data_.resize(rows*cols);
    }
    
    void resize(idx_t rows,dim_t cols)
    {
        rows_=rows;
        cols_=cols;
        data_.resize(rows*cols);
    }
    
    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
    
    T* operator[](idx_t i) { return data_.data()+i*cols_; }
    const T* operator[](idx_t i) const { return data_.data()+i*cols_; }
    
    idx_t rows() const { return rows_; }
    dim_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }
    
private:
    std::vector<T> data_;
    idx_t rows_=0;
    dim_t cols_=0;
};

void normalizeVectors(float* vectors,idx_t n,dim_t dim);
void normalizeVector(float* vector,dim_t dim);

float vectorNorm(const float* vector,dim_t dim);

void copyVectors(float* dst,const float* src,idx_t n,dim_t dim);

}

