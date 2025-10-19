#pragma once

#include"helix/core/distance.hpp"
#include"helix/index/index_base.hpp"
#include<vector>
#include<unordered_set>
#include<queue>
#include<random>

namespace helix {

struct HNSWNode
{
    std::vector<float> data;
    std::vector<std::vector<idx_t>> neighbors;
    int level;
    idx_t id;

    HNSWNode(idx_t id_,int level_,int dim) : level(level_),id(id_)
    {
        data.resize(dim);
        neighbors.resize(level_+1);
    }
};

class IndexHNSW : public IndexBase
{
  public:
    IndexHNSW(const IndexConfig& config,int m=16,int efConstruction=200,int efSearch=50);
    ~IndexHNSW() override=default;
    void train(const float* vectors,idx_t n) override;
    void add(const float* vectors,idx_t n) override;
    void addWithIds(const float* vectors,const idx_t* ids,idx_t n) override;
    SearchResults search(const float* query,int k) const override;
    void searchBatch(const float* queries,idx_t nq,int k,
                     std::vector<SearchResults>& results) const override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    void setEfSearch(int ef) { efSearch_=ef; }
    int getEfSearch() const { return efSearch_; }
    int getM() const { return m_; }
    int getEfConstruction() const { return efConstruction_; }

  private:
    int m_;
    int efConstruction_;
    int efSearch_;
    int maxLevel_;
    idx_t entryPoint_;
    std::vector<std::unique_ptr<HNSWNode>> nodes_;
    std::unique_ptr<DistanceComputer> distance_;
    mutable std::mt19937 rng_;

    int selectLevel();
    void insertNode(const float* vector,idx_t id);
    void connectNeighbors(HNSWNode* node,int level);
    std::vector<idx_t> searchLayer(const float* query,const std::vector<idx_t>& candidates,int ef,int level) const;
    std::vector<idx_t> greedySearch(const float* query,int ef) const;
    float computeDistance(const float* a,const float* b) const;
};

} // namespace helix
