#include"helix/core/distance.hpp"
#include"helix/common/utils.hpp"
#include<algorithm>
#include<cstring>

namespace helix {

float L2Distance::compute(const float* a,const float* b,dim_t dim) const
{
    if(dim>=8)
    {
        return computeSimd(a,b,dim);
    }
    
    float sum=0.0f;
    for(dim_t i=0;i<dim;++i)
    {
        float diff=a[i]-b[i];
        sum+=diff*diff;
    }
    return sum;
}

float L2Distance::computeSimd(const float* a,const float* b,dim_t dim) const
{
#ifdef __AVX2__
    __m256 sum_vec=_mm256_setzero_ps();
    dim_t i=0;
    
    for(;i+7<dim;i+=8)
    {
        __m256 va=_mm256_loadu_ps(a+i);
        __m256 vb=_mm256_loadu_ps(b+i);
        __m256 diff=_mm256_sub_ps(va,vb);
        sum_vec=_mm256_fmadd_ps(diff,diff,sum_vec);
    }
    
    float result[8];
    _mm256_storeu_ps(result,sum_vec);
    float sum=result[0]+result[1]+result[2]+result[3]+result[4]+result[5]+result[6]+result[7];
    
    for(;i<dim;++i)
    {
        float diff=a[i]-b[i];
        sum+=diff*diff;
    }
    
    return sum;
#else
    return compute(a,b,dim);
#endif
}

void L2Distance::computeBatch(const float* queries,const float* database,idx_t nq,idx_t nb,dim_t dim,float* distances) const
{
    for(idx_t i=0;i<nq;++i)
    {
        for(idx_t j=0;j<nb;++j)
        {
            distances[i*nb+j]=compute(queries+i*dim,database+j*dim,dim);
        }
    }
}

float InnerProductDistance::compute(const float* a,const float* b,dim_t dim) const
{
    if(dim>=8)
    {
        return computeSimd(a,b,dim);
    }
    
    float sum=0.0f;
    for(dim_t i=0;i<dim;++i)
    {
        sum+=a[i]*b[i];
    }
    return -sum;
}

float InnerProductDistance::computeSimd(const float* a,const float* b,dim_t dim) const
{
#ifdef __AVX2__
    __m256 sum_vec=_mm256_setzero_ps();
    dim_t i=0;
    
    for(;i+7<dim;i+=8)
    {
        __m256 va=_mm256_loadu_ps(a+i);
        __m256 vb=_mm256_loadu_ps(b+i);
        sum_vec=_mm256_fmadd_ps(va,vb,sum_vec);
    }
    
    float result[8];
    _mm256_storeu_ps(result,sum_vec);
    float sum=result[0]+result[1]+result[2]+result[3]+result[4]+result[5]+result[6]+result[7];
    
    for(;i<dim;++i)
    {
        sum+=a[i]*b[i];
    }
    
    return -sum;
#else
    return compute(a,b,dim);
#endif
}

void InnerProductDistance::computeBatch(const float* queries,const float* database,idx_t nq,idx_t nb,dim_t dim,float* distances) const
{
    for(idx_t i=0;i<nq;++i)
    {
        for(idx_t j=0;j<nb;++j)
        {
            distances[i*nb+j]=compute(queries+i*dim,database+j*dim,dim);
        }
    }
}

float CosineDistance::compute(const float* a,const float* b,dim_t dim) const
{
    float dot=0.0f,norm_a=0.0f,norm_b=0.0f;
    
    for(dim_t i=0;i<dim;++i)
    {
        dot+=a[i]*b[i];
        norm_a+=a[i]*a[i];
        norm_b+=b[i]*b[i];
    }
    
    float denom=std::sqrt(norm_a)*std::sqrt(norm_b);
    if(denom<1e-9f)
    {
        return 1.0f;
    }
    
    return 1.0f-(dot/denom);
}

void CosineDistance::computeBatch(const float* queries,const float* database,idx_t nq,idx_t nb,dim_t dim,float* distances) const
{
    for(idx_t i=0;i<nq;++i)
    {
        for(idx_t j=0;j<nb;++j)
        {
            distances[i*nb+j]=compute(queries+i*dim,database+j*dim,dim);
        }
    }
}

std::unique_ptr<DistanceComputer> createDistanceComputer(MetricType metric)
{
    switch(metric)
    {
        case MetricType::L2:
            return std::make_unique<L2Distance>();
        case MetricType::InnerProduct:
            return std::make_unique<InnerProductDistance>();
        case MetricType::Cosine:
            return std::make_unique<CosineDistance>();
        default:
            throw HelixException("unsupported metric type");
    }
}

}

