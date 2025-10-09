#include"helix/core/vector_utils.hpp"
#include<cmath>
#include<cstring>

namespace helix {

void normalizeVector(float* vector,dim_t dim)
{
    float norm=vectorNorm(vector,dim);
    if(norm<1e-9f)
    {
        return;
    }
    
    for(dim_t i=0;i<dim;++i)
    {
        vector[i]/=norm;
    }
}

void normalizeVectors(float* vectors,idx_t n,dim_t dim)
{
    for(idx_t i=0;i<n;++i)
    {
        normalizeVector(vectors+i*dim,dim);
    }
}

float vectorNorm(const float* vector,dim_t dim)
{
    float sum=0.0f;
    for(dim_t i=0;i<dim;++i)
    {
        sum+=vector[i]*vector[i];
    }
    return std::sqrt(sum);
}

void copyVectors(float* dst,const float* src,idx_t n,dim_t dim)
{
    std::memcpy(dst,src,n*dim*sizeof(float));
}

}

