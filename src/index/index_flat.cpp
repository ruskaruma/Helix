#include"helix/index/index_flat.hpp"
#include"helix/core/io.hpp"
#include"helix/core/threading.hpp"
#include<queue>
#include<cstring>

namespace helix {

IndexFlat::IndexFlat(const IndexConfig& config)
    : IndexBase(config)
{
    distance_=createDistanceComputer(config.metric);
}

void IndexFlat::train(const float*,idx_t)
{
    isTrained_=true;
}

void IndexFlat::add(const float* vectors,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    
    size_t oldSize=vectors_.size();
    vectors_.resize(oldSize+n*config_.dimension);
    std::memcpy(vectors_.data()+oldSize,vectors,n*config_.dimension*sizeof(float));
    
    for(idx_t i=0;i<n;++i)
    {
        ids_.push_back(nextId_++);
    }
    
    ntotal_+=n;
}

void IndexFlat::addWithIds(const float* vectors,const idx_t* ids,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    
    size_t oldSize=vectors_.size();
    vectors_.resize(oldSize+n*config_.dimension);
    std::memcpy(vectors_.data()+oldSize,vectors,n*config_.dimension*sizeof(float));
    
    for(idx_t i=0;i<n;++i)
    {
        ids_.push_back(ids[i]);
        nextId_=std::max(nextId_,ids[i]+1);
    }
    
    ntotal_+=n;
}

SearchResults IndexFlat::search(const float* query,int k) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(k>0,"k must be positive");
    
    k=std::min(k,static_cast<int>(ntotal_));
    
    using HeapElement=std::pair<float,idx_t>;
    auto cmp=[](const HeapElement& a,const HeapElement& b) {
        return a.first<b.first;
    };
    std::priority_queue<HeapElement,std::vector<HeapElement>,decltype(cmp)> heap(cmp);
    
    for(idx_t i=0;i<ntotal_;++i)
    {
        float dist=distance_->compute(query,vectors_.data()+i*config_.dimension,config_.dimension);
        
        if(heap.size()<static_cast<size_t>(k))
        {
            heap.push({dist,ids_[i]});
        }
        else if(dist<heap.top().first)
        {
            heap.pop();
            heap.push({dist,ids_[i]});
        }
    }
    
    SearchResults results(k);
    while(!heap.empty())
    {
        auto elem=heap.top();
        heap.pop();
        results.results.push_back({elem.second,elem.first});
    }
    
    std::reverse(results.results.begin(),results.results.end());
    return results;
}

void IndexFlat::searchBatch(const float* queries,idx_t nq,int k,std::vector<SearchResults>& results) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(k>0,"k must be positive");
    
    results.resize(nq);
    
    parallelFor(0,nq,[&](idx_t i) {
        results[i]=search(queries+i*config_.dimension,k);
    });
}

void IndexFlat::save(const std::string& path) const
{
    FileWriter writer(path);
    
    writer.write(&config_.dimension,1);
    int metricType=static_cast<int>(config_.metric);
    writer.write(&metricType,1);
    writer.write(&ntotal_,1);
    writer.write(&nextId_,1);
    
    if(ntotal_>0)
    {
        writer.write(vectors_.data(),vectors_.size());
        writer.write(ids_.data(),ids_.size());
    }
    
    writer.close();
}

void IndexFlat::load(const std::string& path)
{
    FileReader reader(path);
    
    dim_t dim;
    reader.read(&dim,1);
    HELIX_CHECK_EQ(dim,config_.dimension,"dimension mismatch");
    
    int metricType;
    reader.read(&metricType,1);
    HELIX_CHECK_EQ(metricType,static_cast<int>(config_.metric),"metric type mismatch");
    
    reader.read(&ntotal_,1);
    reader.read(&nextId_,1);
    
    if(ntotal_>0)
    {
        vectors_.resize(ntotal_*config_.dimension);
        reader.read(vectors_.data(),vectors_.size());
        
        ids_.resize(ntotal_);
        reader.read(ids_.data(),ids_.size());
    }
    
    isTrained_=true;
    reader.close();
}

}

