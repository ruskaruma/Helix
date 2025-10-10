#include"helix/benchmark/dataset.hpp"
#include"helix/core/io.hpp"
#include"helix/core/distance.hpp"
#include<random>
#include<fstream>
#include<algorithm>
#include<queue>

namespace helix {

void Dataset::loadFvecs(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    HELIX_CHECK(file.is_open(),"failed to open file: "+path);
    
    file.seekg(0,std::ios::end);
    size_t fileSize=file.tellg();
    file.seekg(0,std::ios::beg);
    
    int d;
    file.read(reinterpret_cast<char*>(&d),sizeof(int));
    HELIX_CHECK(d>0,"invalid dimension");
    
    dim_=d;
    size_t recordSize=4+dim_*4;
    nrows_=fileSize/recordSize;
    
    vectors_.resize(nrows_*dim_);
    file.seekg(0,std::ios::beg);
    
    for(idx_t i=0;i<nrows_;++i)
    {
        file.read(reinterpret_cast<char*>(&d),sizeof(int));
        file.read(reinterpret_cast<char*>(vectors_.data()+i*dim_),dim_*sizeof(float));
    }
    
    file.close();
}

void Dataset::loadIvecs(const std::string& path)
{
    std::ifstream file(path,std::ios::binary);
    HELIX_CHECK(file.is_open(),"failed to open file: "+path);
    
    file.seekg(0,std::ios::end);
    size_t fileSize=file.tellg();
    file.seekg(0,std::ios::beg);
    
    int k;
    file.read(reinterpret_cast<char*>(&k),sizeof(int));
    HELIX_CHECK(k>0,"invalid k");
    
    size_t recordSize=4+k*4;
    nrows_=fileSize/recordSize;
    dim_=k;
    
    labels_.resize(nrows_*k);
    file.seekg(0,std::ios::beg);
    
    for(idx_t i=0;i<nrows_;++i)
    {
        file.read(reinterpret_cast<char*>(&k),sizeof(int));
        file.read(reinterpret_cast<char*>(labels_.data()+i*k),k*sizeof(int));
    }
    
    file.close();
}

void Dataset::generateRandom(idx_t n,dim_t dim,float min,float max,int seed)
{
    nrows_=n;
    dim_=dim;
    vectors_.resize(n*dim);
    
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min,max);
    
    for(auto& v : vectors_)
    {
        v=dist(rng);
    }
}

void Dataset::generateGaussian(idx_t n,dim_t dim,float mean,float stddev,int seed)
{
    nrows_=n;
    dim_=dim;
    vectors_.resize(n*dim);
    
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(mean,stddev);
    
    for(auto& v : vectors_)
    {
        v=dist(rng);
    }
}

void Dataset::normalize()
{
    normalizeVectors(vectors_.data(),nrows_,dim_);
}

void BenchmarkDataset::loadSIFT1M(const std::string& basePath)
{
    train.loadFvecs(basePath+"/sift_learn.fvecs");
    database.loadFvecs(basePath+"/sift_base.fvecs");
    queries.loadFvecs(basePath+"/sift_query.fvecs");
    groundtruth.loadIvecs(basePath+"/sift_groundtruth.ivecs");
}

void BenchmarkDataset::generateSynthetic(idx_t ntrain,idx_t nbase,idx_t nquery,dim_t dim)
{
    train.generateRandom(ntrain,dim,0.0f,1.0f,42);
    database.generateRandom(nbase,dim,0.0f,1.0f,43);
    queries.generateRandom(nquery,dim,0.0f,1.0f,44);
    
    L2Distance dist;
    std::vector<int> gt_labels(nquery*100);
    
    for(idx_t q=0;q<nquery;++q)
    {
        using HeapElement=std::pair<float,int>;
        auto cmp=[](const HeapElement& a,const HeapElement& b) {
            return a.first<b.first;
        };
        std::priority_queue<HeapElement,std::vector<HeapElement>,decltype(cmp)> heap(cmp);
        
        for(idx_t i=0;i<nbase;++i)
        {
            float d=dist.compute(queries.data()+q*dim,database.data()+i*dim,dim);
            
            if(heap.size()<100)
            {
                heap.push({d,static_cast<int>(i)});
            }
            else if(d<heap.top().first)
            {
                heap.pop();
                heap.push({d,static_cast<int>(i)});
            }
        }
        
        std::vector<int> results(100);
        for(int i=99;i>=0;--i)
        {
            results[i]=heap.top().second;
            heap.pop();
        }
        
        std::memcpy(gt_labels.data()+q*100,results.data(),100*sizeof(int));
    }
    
    groundtruth.setLabels(nquery,100,std::move(gt_labels));
}

}

