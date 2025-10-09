#include<gtest/gtest.h>
#include"helix/index/index_flat.hpp"
#include<vector>
#include<random>
#include<algorithm>

class IndexFlatTest : public ::testing::Test {
protected:
    IndexFlatTest() : dim(128),config(dim,helix::MetricType::L2,helix::IndexType::Flat) {}
    
    helix::dim_t dim;
    helix::IndexConfig config;
};

TEST_F(IndexFlatTest,Construction) {
    helix::IndexFlat index(config);
    EXPECT_EQ(index.dimension(),dim);
    EXPECT_EQ(index.ntotal(),0);
    EXPECT_FALSE(index.isTrained());
}

TEST_F(IndexFlatTest,TrainAndAdd) {
    helix::IndexFlat index(config);
    
    std::vector<float> vectors(10*dim);
    for(size_t i=0;i<vectors.size();++i)
    {
        vectors[i]=static_cast<float>(i);
    }
    
    index.train(vectors.data(),10);
    EXPECT_TRUE(index.isTrained());
    
    index.add(vectors.data(),10);
    EXPECT_EQ(index.ntotal(),10);
}

TEST_F(IndexFlatTest,AddWithIds) {
    helix::IndexFlat index(config);
    
    std::vector<float> vectors(5*dim,1.0f);
    std::vector<helix::idx_t> ids={100,200,300,400,500};
    
    index.train(vectors.data(),5);
    index.addWithIds(vectors.data(),ids.data(),5);
    
    EXPECT_EQ(index.ntotal(),5);
    
    const helix::idx_t* storedIds=index.getIds();
    for(int i=0;i<5;++i)
    {
        EXPECT_EQ(storedIds[i],ids[i]);
    }
}

TEST_F(IndexFlatTest,SearchBasic) {
    helix::IndexFlat index(config);
    
    std::vector<float> vectors(10*dim,0.0f);
    for(int i=0;i<10;++i)
    {
        vectors[i*dim]=static_cast<float>(i);
    }
    
    index.train(vectors.data(),10);
    index.add(vectors.data(),10);
    
    std::vector<float> query(dim,0.0f);
    query[0]=0.5f;
    
    auto results=index.search(query.data(),3);
    
    EXPECT_EQ(results.results.size(),3);
    EXPECT_EQ(results.results[0].id,0);
}

TEST_F(IndexFlatTest,SearchBatch) {
    helix::IndexFlat index(config);
    
    std::vector<float> vectors(100*dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    index.train(vectors.data(),100);
    index.add(vectors.data(),100);
    
    std::vector<float> queries(5*dim);
    for(auto& q : queries)
    {
        q=dist(rng);
    }
    
    std::vector<helix::SearchResults> results;
    index.searchBatch(queries.data(),5,10,results);
    
    EXPECT_EQ(results.size(),5);
    for(const auto& res : results)
    {
        EXPECT_EQ(res.results.size(),10);
    }
}

TEST_F(IndexFlatTest,SaveAndLoad) {
    helix::IndexFlat index1(config);
    
    std::vector<float> vectors(20*dim);
    for(size_t i=0;i<vectors.size();++i)
    {
        vectors[i]=static_cast<float>(i%100);
    }
    
    index1.train(vectors.data(),20);
    index1.add(vectors.data(),20);
    
    std::string path="/tmp/helix_test_index.bin";
    index1.save(path);
    
    helix::IndexFlat index2(config);
    index2.load(path);
    
    EXPECT_EQ(index2.ntotal(),20);
    EXPECT_TRUE(index2.isTrained());
    
    std::vector<float> query(dim,0.0f);
    auto results1=index1.search(query.data(),5);
    auto results2=index2.search(query.data(),5);
    
    EXPECT_EQ(results1.results.size(),results2.results.size());
    for(size_t i=0;i<results1.results.size();++i)
    {
        EXPECT_EQ(results1.results[i].id,results2.results[i].id);
        EXPECT_NEAR(results1.results[i].distance,results2.results[i].distance,1e-5f);
    }
}

TEST_F(IndexFlatTest,DifferentMetrics) {
    std::vector<float> vectors(10*dim,1.0f);
    std::vector<float> query(dim,1.0f);
    
    helix::IndexConfig configL2(dim,helix::MetricType::L2,helix::IndexType::Flat);
    helix::IndexFlat indexL2(configL2);
    indexL2.train(vectors.data(),10);
    indexL2.add(vectors.data(),10);
    auto resultsL2=indexL2.search(query.data(),5);
    EXPECT_EQ(resultsL2.results.size(),5);
    
    helix::IndexConfig configIP(dim,helix::MetricType::InnerProduct,helix::IndexType::Flat);
    helix::IndexFlat indexIP(configIP);
    indexIP.train(vectors.data(),10);
    indexIP.add(vectors.data(),10);
    auto resultsIP=indexIP.search(query.data(),5);
    EXPECT_EQ(resultsIP.results.size(),5);
}

