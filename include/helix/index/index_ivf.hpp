#pragma once

#include"helix/core/distance.hpp"
#include"helix/core/kmeans.hpp"
#include"helix/index/index_base.hpp"
#include<vector>
#include<unordered_map>

namespace helix {

class IndexIVF : public IndexBase
{
  public:
    IndexIVF(const IndexConfig& config,int nlist,int nprobe=1);
    ~IndexIVF() override=default;
    void train(const float* vectors,idx_t n) override;
    void add(const float* vectors,idx_t n) override;
    void addWithIds(const float* vectors,const idx_t* ids,idx_t n) override;
    SearchResults search(const float* query,int k) const override;
    void searchBatch(const float* queries,idx_t nq,int k,
                     std::vector<SearchResults>& results) const override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;

    void setNprobe(int nprobe) { nprobe_=nprobe; }
    int getNprobe() const { return nprobe_; }
    int getNlist() const { return nlist_; }

  private:
    int nlist_;
    int nprobe_;
    std::unique_ptr<KMeans> quantizer_;
    std::vector<std::vector<idx_t>> invlists_;
    std::vector<std::vector<float>> vectors_;
    std::vector<std::vector<idx_t>> ids_;
    std::unique_ptr<DistanceComputer> distance_;

    void addToInvlist(int listId,const float* vector,idx_t id);
    std::vector<int> selectClusters(const float* query) const;
};

} // namespace helix
