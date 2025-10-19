#include"helix/index/index_ivf.hpp"
#include"helix/core/io.hpp"
#include"helix/core/threading.hpp"
#include"helix/common/utils.hpp"
#include<queue>
#include<algorithm>
#include<cstring>

namespace helix {

IndexIVF::IndexIVF(const IndexConfig& config,int nlist,int nprobe)
    : IndexBase(config),nlist_(nlist),nprobe_(nprobe)
{
    HELIX_CHECK(nlist_>0,"nlist must be positive");
    HELIX_CHECK(nprobe_>0,"nprobe must be positive");
    HELIX_CHECK(nprobe_<=nlist_,"nprobe must be <= nlist");

    quantizer_=std::make_unique<KMeans>(nlist_);
    invlists_.resize(nlist_);
    vectors_.resize(nlist_);
    ids_.resize(nlist_);
    distance_=createDistanceComputer(config.metric);
}

void IndexIVF::train(const float* vectors,idx_t n)
{
    HELIX_CHECK(vectors!=nullptr,"vectors cannot be null");
    HELIX_CHECK(n>0,"n must be positive");

    quantizer_->train(vectors,n,config_.dimension);
    isTrained_=true;
}

void IndexIVF::add(const float* vectors,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(vectors!=nullptr,"vectors cannot be null");
    HELIX_CHECK(n>0,"n must be positive");

    for(idx_t i=0;i<n;++i)
    {
        idx_t cluster=quantizer_->assign(vectors+i*config_.dimension,config_.dimension);
        addToInvlist(cluster,vectors+i*config_.dimension,nextId_++);
    }

    ntotal_+=n;
}

void IndexIVF::addWithIds(const float* vectors,const idx_t* ids,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(vectors!=nullptr,"vectors cannot be null");
    HELIX_CHECK(ids!=nullptr,"ids cannot be null");
    HELIX_CHECK(n>0,"n must be positive");

    for(idx_t i=0;i<n;++i)
    {
        idx_t cluster=quantizer_->assign(vectors+i*config_.dimension,config_.dimension);
        addToInvlist(cluster,vectors+i*config_.dimension,ids[i]);
        nextId_=std::max(nextId_,ids[i]+1);
    }

    ntotal_+=n;
}

SearchResults IndexIVF::search(const float* query,int k) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(query!=nullptr,"query cannot be null");
    HELIX_CHECK(k>0,"k must be positive");

    k=std::min(k,static_cast<int>(ntotal_));

    std::vector<int> clusters=selectClusters(query);

    using HeapElement=std::pair<float,idx_t>;
    auto cmp=[](const HeapElement& a,const HeapElement& b) {
        return a.first<b.first;
    };
    std::priority_queue<HeapElement,std::vector<HeapElement>,decltype(cmp)> heap(cmp);

    for(int clusterId : clusters)
    {
        const auto& invlist=invlists_[clusterId];
        const auto& clusterVectors=vectors_[clusterId];
        const auto& clusterIds=ids_[clusterId];

        for(size_t i=0;i<invlist.size();++i)
        {
            float dist=distance_->compute(query,clusterVectors.data()+i*config_.dimension,config_.dimension);

            if(heap.size()<static_cast<size_t>(k))
            {
                heap.push({dist,clusterIds[i]});
            }
            else if(dist<heap.top().first)
            {
                heap.pop();
                heap.push({dist,clusterIds[i]});
            }
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

void IndexIVF::searchBatch(const float* queries,idx_t nq,int k,std::vector<SearchResults>& results) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(queries!=nullptr,"queries cannot be null");
    HELIX_CHECK(k>0,"k must be positive");

    results.resize(nq);

    parallelFor(0,nq,[&](idx_t i) {
        results[i]=search(queries+i*config_.dimension,k);
    });
}

void IndexIVF::save(const std::string& path) const
{
    FileWriter writer(path);

    writer.write(&config_.dimension,1);
    int metricType=static_cast<int>(config_.metric);
    writer.write(&metricType,1);
    writer.write(&ntotal_,1);
    writer.write(&nextId_,1);
    writer.write(&nlist_,1);
    writer.write(&nprobe_,1);

    //save quantizer centroids
    if(quantizer_->isTrained())
    {
        const auto& centroids=quantizer_->getCentroids();
        writer.write(centroids.data(),centroids.size());
    }

    //save inverted lists
    for(int i=0;i<nlist_;++i)
    {
        idx_t listSize=invlists_[i].size();
        writer.write(&listSize,1);

        if(listSize>0)
        {
            writer.write(invlists_[i].data(),listSize);
            writer.write(vectors_[i].data(),vectors_[i].size());
            writer.write(ids_[i].data(),ids_[i].size());
        }
    }

    writer.close();
}

void IndexIVF::load(const std::string& path)
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
    reader.read(&nlist_,1);
    reader.read(&nprobe_,1);

    //load quantizer centroids
    std::vector<float> centroids(nlist_*config_.dimension);
    reader.read(centroids.data(),centroids.size());

    //reconstruct quantizer
    quantizer_=std::make_unique<KMeans>(nlist_);
    quantizer_->setCentroids(centroids);

    //load inverted lists
    invlists_.resize(nlist_);
    vectors_.resize(nlist_);
    ids_.resize(nlist_);

    for(int i=0;i<nlist_;++i)
    {
        idx_t listSize;
        reader.read(&listSize,1);

        if(listSize>0)
        {
            invlists_[i].resize(listSize);
            vectors_[i].resize(listSize*config_.dimension);
            ids_[i].resize(listSize);

            reader.read(invlists_[i].data(),listSize);
            reader.read(vectors_[i].data(),vectors_[i].size());
            reader.read(ids_[i].data(),ids_[i].size());
        }
    }

    isTrained_=true;
    reader.close();
}

void IndexIVF::addToInvlist(int listId,const float* vector,idx_t id)
{
    invlists_[listId].push_back(id);
    vectors_[listId].insert(vectors_[listId].end(),vector,vector+config_.dimension);
    ids_[listId].push_back(id);
}

std::vector<int> IndexIVF::selectClusters(const float* query) const
{
    std::vector<std::pair<float,int>> distances;
    distances.reserve(nlist_);

    const auto& centroids=quantizer_->getCentroids();
    for(int i=0;i<nlist_;++i)
    {
        float dist=distance_->compute(query,centroids.data()+i*config_.dimension,config_.dimension);
        distances.emplace_back(dist,i);
    }

    std::sort(distances.begin(),distances.end());

    std::vector<int> selected;
    selected.reserve(nprobe_);
    for(int i=0;i<nprobe_ && i<nlist_;++i)
    {
        selected.push_back(distances[i].second);
    }

    return selected;
}

} // namespace helix
