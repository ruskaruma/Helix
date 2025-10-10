#include"helix/quantization/product_quantizer.hpp"
#include"helix/common/utils.hpp"
#include"helix/core/threading.hpp"
#include<cstring>
#include<random>
#include<limits>
#include<cmath>

namespace helix {

ProductQuantizer::ProductQuantizer(dim_t dimension,int nsub,int nbits)
    : dim_(dimension),nsub_(nsub),nbits_(nbits),isTrained_(false)
{
    HELIX_CHECK(dim_%nsub_==0,"dimension must be divisible by nsub");
    HELIX_CHECK(nbits_>=1 && nbits_<=8,"nbits must be between 1 and 8");
    
    ksub_=1<<nbits_;
    dsub_=dim_/nsub_;
    
    centroids_.resize(nsub_*ksub_*dsub_);
}

void ProductQuantizer::train(const float* vectors,idx_t n)
{
    HELIX_CHECK(n>=ksub_,"need at least ksub vectors for training");
    
    for(int subIdx=0;subIdx<nsub_;++subIdx)
    {
        trainSubspace(vectors,n,subIdx);
    }
    
    isTrained_=true;
}

void ProductQuantizer::trainSubspace(const float* vectors,idx_t n,int subIdx)
{
    kmeansSubspace(vectors,n,subIdx);
}

void ProductQuantizer::kmeansSubspace(const float* vectors,idx_t n,int subIdx)
{
    std::vector<float> subVectors(n*dsub_);
    
    for(idx_t i=0;i<n;++i)
    {
        std::memcpy(subVectors.data()+i*dsub_,vectors+i*dim_+subIdx*dsub_,dsub_*sizeof(float));
    }
    
    std::mt19937 rng(42+subIdx);
    std::uniform_int_distribution<idx_t> dist(0,n-1);
    
    float* centroids=centroids_.data()+subIdx*ksub_*dsub_;
    
    for(int k=0;k<ksub_;++k)
    {
        idx_t idx=dist(rng);
        std::memcpy(centroids+k*dsub_,subVectors.data()+idx*dsub_,dsub_*sizeof(float));
    }
    
    std::vector<int> assignments(n);
    const int maxIter=25;
    
    for(int iter=0;iter<maxIter;++iter)
    {
        bool changed=false;
        
        for(idx_t i=0;i<n;++i)
        {
            float minDist=std::numeric_limits<float>::max();
            int bestK=-1;
            
            for(int k=0;k<ksub_;++k)
            {
                float dist=0.0f;
                for(int d=0;d<dsub_;++d)
                {
                    float diff=subVectors[i*dsub_+d]-centroids[k*dsub_+d];
                    dist+=diff*diff;
                }
                
                if(dist<minDist)
                {
                    minDist=dist;
                    bestK=k;
                }
            }
            
            if(assignments[i]!=bestK)
            {
                assignments[i]=bestK;
                changed=true;
            }
        }
        
        if(!changed)
        {
            break;
        }
        
        std::vector<float> newCentroids(ksub_*dsub_,0.0f);
        std::vector<int> counts(ksub_,0);
        
        for(idx_t i=0;i<n;++i)
        {
            int k=assignments[i];
            counts[k]++;
            for(int d=0;d<dsub_;++d)
            {
                newCentroids[k*dsub_+d]+=subVectors[i*dsub_+d];
            }
        }
        
        for(int k=0;k<ksub_;++k)
        {
            if(counts[k]>0)
            {
                for(int d=0;d<dsub_;++d)
                {
                    newCentroids[k*dsub_+d]/=counts[k];
                }
            }
        }
        
        std::memcpy(centroids,newCentroids.data(),ksub_*dsub_*sizeof(float));
    }
}

void ProductQuantizer::encode(const float* vectors,idx_t n,uint8_t* codes) const
{
    HELIX_CHECK(isTrained_,"quantizer not trained");
    
    for(idx_t i=0;i<n;++i)
    {
        for(int subIdx=0;subIdx<nsub_;++subIdx)
        {
            const float* subVec=vectors+i*dim_+subIdx*dsub_;
            const float* centroids=centroids_.data()+subIdx*ksub_*dsub_;
            
            float minDist=std::numeric_limits<float>::max();
            uint8_t bestK=0;
            
            for(int k=0;k<ksub_;++k)
            {
                float dist=0.0f;
                for(int d=0;d<dsub_;++d)
                {
                    float diff=subVec[d]-centroids[k*dsub_+d];
                    dist+=diff*diff;
                }
                
                if(dist<minDist)
                {
                    minDist=dist;
                    bestK=k;
                }
            }
            
            codes[i*nsub_+subIdx]=bestK;
        }
    }
}

void ProductQuantizer::decode(const uint8_t* codes,idx_t n,float* vectors) const
{
    HELIX_CHECK(isTrained_,"quantizer not trained");
    
    for(idx_t i=0;i<n;++i)
    {
        for(int subIdx=0;subIdx<nsub_;++subIdx)
        {
            uint8_t k=codes[i*nsub_+subIdx];
            const float* centroid=centroids_.data()+subIdx*ksub_*dsub_+k*dsub_;
            std::memcpy(vectors+i*dim_+subIdx*dsub_,centroid,dsub_*sizeof(float));
        }
    }
}

void ProductQuantizer::computeDistanceTable(const float* query,float* distTable) const
{
    HELIX_CHECK(isTrained_,"quantizer not trained");
    
    for(int subIdx=0;subIdx<nsub_;++subIdx)
    {
        const float* subQuery=query+subIdx*dsub_;
        const float* centroids=centroids_.data()+subIdx*ksub_*dsub_;
        
        for(int k=0;k<ksub_;++k)
        {
            float dist=0.0f;
            for(int d=0;d<dsub_;++d)
            {
                float diff=subQuery[d]-centroids[k*dsub_+d];
                dist+=diff*diff;
            }
            distTable[subIdx*ksub_+k]=dist;
        }
    }
}

float ProductQuantizer::computeAsymmetricDistance(const float* query,const uint8_t* code) const
{
    HELIX_CHECK(isTrained_,"quantizer not trained");
    
    float dist=0.0f;
    
    for(int subIdx=0;subIdx<nsub_;++subIdx)
    {
        const float* subQuery=query+subIdx*dsub_;
        uint8_t k=code[subIdx];
        const float* centroid=centroids_.data()+subIdx*ksub_*dsub_+k*dsub_;
        
        for(int d=0;d<dsub_;++d)
        {
            float diff=subQuery[d]-centroid[d];
            dist+=diff*diff;
        }
    }
    
    return dist;
}

void ProductQuantizer::loadCentroids(const float* centroids,size_t size)
{
    HELIX_CHECK_EQ(size,centroids_.size(),"centroid size mismatch");
    std::memcpy(centroids_.data(),centroids,size*sizeof(float));
    isTrained_=true;
}

}

