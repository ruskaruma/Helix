#pragma once

#include"helix/index/index_base.hpp"
#include"helix/quantization/product_quantizer.hpp"
#include<vector>
#include<memory>

namespace helix {

class IndexPQ : public IndexBase {
public:
    IndexPQ(const IndexConfig& config,int nsub,int nbits);
    ~IndexPQ() override=default;
    
    void train(const float* vectors,idx_t n) override;
    void add(const float* vectors,idx_t n) override;
    void addWithIds(const float* vectors,const idx_t* ids,idx_t n) override;
    SearchResults search(const float* query,int k) const override;
    void searchBatch(const float* queries,idx_t nq,int k,std::vector<SearchResults>& results) const override;
    
    void save(const std::string& path) const override;
    void load(const std::string& path) override;
    
private:
    std::unique_ptr<ProductQuantizer> pq_;
    std::vector<uint8_t> codes_;
    std::vector<idx_t> ids_;
    idx_t nextId_;
};

}

