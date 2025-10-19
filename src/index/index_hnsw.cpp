#include"helix/index/index_hnsw.hpp"
#include"helix/core/io.hpp"
#include"helix/core/threading.hpp"
#include"helix/common/utils.hpp"
#include<algorithm>
#include<cstring>

namespace helix {

IndexHNSW::IndexHNSW(const IndexConfig& config,int m,int efConstruction,int efSearch)
    : IndexBase(config),m_(m),efConstruction_(efConstruction),efSearch_(efSearch),
      maxLevel_(-1),entryPoint_(-1),rng_(std::random_device{}())
{
    HELIX_CHECK(m_>0,"m must be positive");
    HELIX_CHECK(efConstruction_>0,"efConstruction must be positive");
    HELIX_CHECK(efSearch_>0,"efSearch must be positive");
    
    distance_=createDistanceComputer(config.metric);
}

void IndexHNSW::train(const float*,idx_t)
{
    isTrained_=true;
}

void IndexHNSW::add(const float* vectors,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(vectors!=nullptr,"vectors cannot be null");
    HELIX_CHECK(n>0,"n must be positive");

    for(idx_t i=0;i<n;++i)
    {
        insertNode(vectors+i*config_.dimension,nextId_++);
    }

    ntotal_+=n;
}

void IndexHNSW::addWithIds(const float* vectors,const idx_t* ids,idx_t n)
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(vectors!=nullptr,"vectors cannot be null");
    HELIX_CHECK(ids!=nullptr,"ids cannot be null");
    HELIX_CHECK(n>0,"n must be positive");

    for(idx_t i=0;i<n;++i)
    {
        insertNode(vectors+i*config_.dimension,ids[i]);
        nextId_=std::max(nextId_,ids[i]+1);
    }

    ntotal_+=n;
}

SearchResults IndexHNSW::search(const float* query,int k) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(query!=nullptr,"query cannot be null");
    HELIX_CHECK(k>0,"k must be positive");

    k=std::min(k,static_cast<int>(ntotal_));

    if(ntotal_==0)
    {
        return SearchResults(k);
    }

    std::vector<idx_t> candidates=greedySearch(query,efSearch_);
    
    //if greedySearch fails, fall back to searching all nodes
    if(candidates.empty())
    {
        for(idx_t i=0;i<static_cast<idx_t>(nodes_.size());++i)
        {
            if(nodes_[i])
            {
                candidates.push_back(i);
            }
        }
    }

    using HeapElement=std::pair<float,idx_t>;
    auto cmp=[](const HeapElement& a,const HeapElement& b) {
        return a.first<b.first;
    };
    std::priority_queue<HeapElement,std::vector<HeapElement>,decltype(cmp)> heap(cmp);

    for(idx_t candidate : candidates)
    {
        if(candidate<static_cast<idx_t>(nodes_.size()) && nodes_[candidate])
        {
            float dist=computeDistance(query,nodes_[candidate]->data.data());
            heap.push({dist,nodes_[candidate]->id});
        }
    }

    SearchResults results(k);
    while(!heap.empty() && results.results.size()<static_cast<size_t>(k))
    {
        auto elem=heap.top();
        heap.pop();
        results.results.push_back({elem.second,elem.first});
    }

    return results;
}

void IndexHNSW::searchBatch(const float* queries,idx_t nq,int k,std::vector<SearchResults>& results) const
{
    HELIX_CHECK(isTrained_,"index not trained");
    HELIX_CHECK(queries!=nullptr,"queries cannot be null");
    HELIX_CHECK(k>0,"k must be positive");

    results.resize(nq);

    parallelFor(0,nq,[&](idx_t i) {
        results[i]=search(queries+i*config_.dimension,k);
    });
}

void IndexHNSW::save(const std::string& path) const
{
    FileWriter writer(path);

    writer.write(&config_.dimension,1);
    int metricType=static_cast<int>(config_.metric);
    writer.write(&metricType,1);
    writer.write(&ntotal_,1);
    writer.write(&nextId_,1);
    writer.write(&m_,1);
    writer.write(&efConstruction_,1);
    writer.write(&efSearch_,1);
    writer.write(&maxLevel_,1);
    writer.write(&entryPoint_,1);

    //save nodes
    for(idx_t i=0;i<static_cast<idx_t>(nodes_.size());++i)
    {
        if(nodes_[i])
        {
            writer.write(&i,1);
            writer.write(&nodes_[i]->id,1);
            writer.write(&nodes_[i]->level,1);
            writer.write(nodes_[i]->data.data(),config_.dimension);
            
            for(int l=0;l<=nodes_[i]->level;++l)
            {
                idx_t neighborCount=nodes_[i]->neighbors[l].size();
                writer.write(&neighborCount,1);
                if(neighborCount>0)
                {
                    writer.write(nodes_[i]->neighbors[l].data(),neighborCount);
                }
            }
        }
    }

    writer.close();
}

void IndexHNSW::load(const std::string& path)
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
    reader.read(&m_,1);
    reader.read(&efConstruction_,1);
    reader.read(&efSearch_,1);
    reader.read(&maxLevel_,1);
    reader.read(&entryPoint_,1);

    //load nodes
    nodes_.clear();
    nodes_.resize(ntotal_);

    for(idx_t i=0;i<ntotal_;++i)
    {
        idx_t nodeId,id;
        int level;
        reader.read(&nodeId,1);
        reader.read(&id,1);
        reader.read(&level,1);

        nodes_[nodeId]=std::make_unique<HNSWNode>(id,level,config_.dimension);
        reader.read(nodes_[nodeId]->data.data(),config_.dimension);

        for(int l=0;l<=level;++l)
        {
            idx_t neighborCount;
            reader.read(&neighborCount,1);
            nodes_[nodeId]->neighbors[l].resize(neighborCount);
            if(neighborCount>0)
            {
                reader.read(nodes_[nodeId]->neighbors[l].data(),neighborCount);
            }
        }
    }

    isTrained_=true;
    reader.close();
}

int IndexHNSW::selectLevel()
{
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    float r=dist(rng_);
    int level=0;
    while(r<0.5f && level<maxLevel_)
    {
        r=dist(rng_);
        level++;
    }
    return level;
}

void IndexHNSW::insertNode(const float* vector,idx_t id)
{
    int level=selectLevel();
    maxLevel_=std::max(maxLevel_,level);

    auto node=std::make_unique<HNSWNode>(id,level,config_.dimension);
    std::memcpy(node->data.data(),vector,config_.dimension*sizeof(float));

    if(ntotal_==1)
    {
        entryPoint_=nodes_.size();
        nodes_.push_back(std::move(node));
        return;
    }

    //search for entry point
    std::vector<idx_t> candidates;
    if(entryPoint_>=0 && entryPoint_<static_cast<idx_t>(nodes_.size()))
    {
        candidates=greedySearch(vector,efConstruction_);
    }

    //insert node
    idx_t nodeIndex=nodes_.size();
    nodes_.push_back(std::move(node));

    //connect neighbors at each level
    for(int l=0;l<=level;++l)
    {
        connectNeighbors(nodes_[nodeIndex].get(),l);
    }
}

void IndexHNSW::connectNeighbors(HNSWNode* node,int level)
{
    std::vector<idx_t> candidates;
    if(entryPoint_>=0 && entryPoint_<static_cast<idx_t>(nodes_.size()))
    {
        candidates=greedySearch(node->data.data(),efConstruction_);
    }

    //select m neighbors
    std::vector<std::pair<float,idx_t>> distances;
    for(idx_t candidate : candidates)
    {
        if(candidate<static_cast<idx_t>(nodes_.size()) && nodes_[candidate] && 
           nodes_[candidate]->level>=level)
        {
            float dist=computeDistance(node->data.data(),nodes_[candidate]->data.data());
            distances.emplace_back(dist,candidate);
        }
    }

    std::sort(distances.begin(),distances.end());
    
    int neighborsToAdd=std::min(m_,static_cast<int>(distances.size()));
    for(int i=0;i<neighborsToAdd;++i)
    {
        node->neighbors[level].push_back(distances[i].second);
        //bidirectional connection
        if(nodes_[distances[i].second])
        {
            nodes_[distances[i].second]->neighbors[level].push_back(node->id);
        }
    }
}

std::vector<idx_t> IndexHNSW::searchLayer(const float* query,const std::vector<idx_t>& candidates,int ef,int level) const
{
    std::unordered_set<idx_t> visited;
    std::vector<std::pair<float,idx_t>> candidatesList;
    
    for(idx_t candidate : candidates)
    {
        if(candidate<static_cast<idx_t>(nodes_.size()) && nodes_[candidate] && 
           nodes_[candidate]->level>=level)
        {
            float dist=computeDistance(query,nodes_[candidate]->data.data());
            candidatesList.emplace_back(dist,candidate);
            visited.insert(candidate);
        }
    }

    if(candidatesList.empty())
    {
        return candidates;
    }

    std::sort(candidatesList.begin(),candidatesList.end());

    std::vector<idx_t> result;
    for(int i=0;i<std::min(ef,static_cast<int>(candidatesList.size()));++i)
    {
        result.push_back(candidatesList[i].second);
    }

    return result;
}

std::vector<idx_t> IndexHNSW::greedySearch(const float* query,int ef) const
{
    if(entryPoint_<0 || entryPoint_>=static_cast<idx_t>(nodes_.size()) || !nodes_[entryPoint_])
    {
        return {};
    }

    std::vector<idx_t> currentCandidates={entryPoint_};
    
    for(int level=maxLevel_;level>=0;--level)
    {
        currentCandidates=searchLayer(query,currentCandidates,ef,level);
        if(currentCandidates.empty())
        {
            break;
        }
    }

    return currentCandidates;
}

float IndexHNSW::computeDistance(const float* a,const float* b) const
{
    return distance_->compute(a,b,config_.dimension);
}

} // namespace helix
