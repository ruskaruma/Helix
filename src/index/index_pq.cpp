#include"helix/index/index_pq.hpp"
#include"helix/core/io.hpp"
#include"helix/core/threading.hpp"
#include<queue>
#include<cstring>
#include<algorithm>

namespace helix {

IndexPQ::IndexPQ(const IndexConfig& config,int nsub,int nbits)
    : IndexBase(config),nextId_(0)
{
    pq_=std::make_unique<ProductQuantizer>(config.dimension,nsub,nbits);
}

void IndexPQ::train(const float* vectors,idx_t n)
{
    pq_->train(vectors,n);
    isTrained_=true;
}

void IndexPQ::add(const float* vectors,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    
    size_t codeSize=pq_->nsub();
    size_t oldSize=codes_.size();
    codes_.resize(oldSize+n*codeSize);
    
    pq_->encode(vectors,n,codes_.data()+oldSize);
    
    for(idx_t i=0;i<n;++i)
    {
        ids_.push_back(nextId_++);
    }
    
    ntotal_+=n;
}

void IndexPQ::addWithIds(const float* vectors,const idx_t* ids,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    
    size_t codeSize=pq_->nsub();
    size_t oldSize=codes_.size();
    codes_.resize(oldSize+n*codeSize);
    
    pq_->encode(vectors,n,codes_.data()+oldSize);
    
    for(idx_t i=0;i<n;++i)
    {
        ids_.push_back(ids[i]);
        nextId_=std::max(nextId_,ids[i]+1);
    }
    
    ntotal_+=n;
}

SearchResults IndexPQ::search(const float* query,int k) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(k>0,"k must be positive");
    
    k=std::min(k,static_cast<int>(ntotal_));
    
    using HeapElement=std::pair<float,idx_t>;
    auto cmp=[](const HeapElement& a,const HeapElement& b) {
        return a.first<b.first;
    };
    std::priority_queue<HeapElement,std::vector<HeapElement>,decltype(cmp)> heap(cmp);
    
    int codeSize=pq_->nsub();
    
    for(idx_t i=0;i<ntotal_;++i)
    {
        float dist=pq_->computeAsymmetricDistance(query,codes_.data()+i*codeSize);
        
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

void IndexPQ::searchBatch(const float* queries,idx_t nq,int k,std::vector<SearchResults>& results) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(k>0,"k must be positive");
    
    results.resize(nq);
    
    parallelFor(0,nq,[&](idx_t i) {
        results[i]=search(queries+i*config_.dimension,k);
    });
}

void IndexPQ::save(const std::string& path) const
{
    FileWriter writer(path);
    
    writer.write(&config_.dimension,1);
    int metricType=static_cast<int>(config_.metric);
    writer.write(&metricType,1);
    
    int nsub=pq_->nsub();
    int nbits=pq_->nbits();
    writer.write(&nsub,1);
    writer.write(&nbits,1);
    
    writer.write(&ntotal_,1);
    writer.write(&nextId_,1);
    
    if(isTrained_)
    {
        size_t centroidSize=nsub*pq_->ksub()*pq_->dsub();
        writer.write(pq_->centroids(),centroidSize);
    }
    
    if(ntotal_>0)
    {
        writer.write(codes_.data(),codes_.size());
        writer.write(ids_.data(),ids_.size());
    }
    
    writer.close();
}

void IndexPQ::load(const std::string& path)
{
    FileReader reader(path);
    
    dim_t dim;
    reader.read(&dim,1);
    HELIX_CHECK_EQ(dim,config_.dimension,"dimension mismatch");
    
    int metricType;
    reader.read(&metricType,1);
    HELIX_CHECK_EQ(metricType,static_cast<int>(config_.metric),"metric type mismatch");
    
    int nsub,nbits;
    reader.read(&nsub,1);
    reader.read(&nbits,1);
    
    pq_=std::make_unique<ProductQuantizer>(dim,nsub,nbits);
    
    reader.read(&ntotal_,1);
    reader.read(&nextId_,1);
    
    size_t centroidSize=nsub*pq_->ksub()*pq_->dsub();
    std::vector<float> centroids(centroidSize);
    reader.read(centroids.data(),centroidSize);
    pq_->loadCentroids(centroids.data(),centroidSize);
    
    if(ntotal_>0)
    {
        codes_.resize(ntotal_*nsub);
        reader.read(codes_.data(),codes_.size());
        
        ids_.resize(ntotal_);
        reader.read(ids_.data(),ids_.size());
    }
    
    isTrained_=true;
    reader.close();
}

}

