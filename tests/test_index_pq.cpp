#include<gtest/gtest.h>
#include"helix/index/index_pq.hpp"
#include<vector>
#include<random>

class IndexPQTest : public ::testing::Test {
protected:
    IndexPQTest() : dim(128),config(dim,helix::MetricType::L2,helix::IndexType::PQ) {}
    
    helix::dim_t dim;
    helix::IndexConfig config;
};

TEST_F(IndexPQTest,Construction) {
    helix::IndexPQ index(config,8,8);
    EXPECT_EQ(index.dimension(),dim);
    EXPECT_EQ(index.ntotal(),0);
    EXPECT_FALSE(index.isTrained());
}

TEST_F(IndexPQTest,TrainAndAdd) {
    helix::IndexPQ index(config,8,6);
    
    std::vector<float> vectors(1000*dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    index.train(vectors.data(),1000);
    EXPECT_TRUE(index.isTrained());
    
    index.add(vectors.data(),100);
    EXPECT_EQ(index.ntotal(),100);
}

TEST_F(IndexPQTest,AddWithIds) {
    helix::IndexPQ index(config,8,6);
    
    std::vector<float> vectors(500*dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    index.train(vectors.data(),500);
    
    std::vector<helix::idx_t> ids={100,200,300,400,500};
    index.addWithIds(vectors.data(),ids.data(),5);
    
    EXPECT_EQ(index.ntotal(),5);
}

TEST_F(IndexPQTest,SearchBasic) {
    helix::IndexPQ index(config,8,6);
    
    std::vector<float> vectors(500*dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    index.train(vectors.data(),500);
    index.add(vectors.data(),500);
    
    std::vector<float> query(dim);
    for(auto& q : query)
    {
        q=dist(rng);
    }
    
    auto results=index.search(query.data(),10);
    
    EXPECT_EQ(results.results.size(),10);
    
    for(int i=0;i<9;++i)
    {
        EXPECT_LE(results.results[i].distance,results.results[i+1].distance);
    }
}

TEST_F(IndexPQTest,SearchBatch) {
    helix::IndexPQ index(config,8,6);
    
    std::vector<float> vectors(500*dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    index.train(vectors.data(),500);
    index.add(vectors.data(),500);
    
    std::vector<float> queries(10*dim);
    for(auto& q : queries)
    {
        q=dist(rng);
    }
    
    std::vector<helix::SearchResults> results;
    index.searchBatch(queries.data(),10,5,results);
    
    EXPECT_EQ(results.size(),10);
    for(const auto& res : results)
    {
        EXPECT_EQ(res.results.size(),5);
    }
}

TEST_F(IndexPQTest,SaveAndLoad) {
    helix::IndexPQ index1(config,4,6);
    
    std::vector<float> vectors(300*dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f,1.0f);
    
    for(auto& v : vectors)
    {
        v=dist(rng);
    }
    
    index1.train(vectors.data(),300);
    index1.add(vectors.data(),200);
    
    std::string path="/tmp/helix_test_pq.bin";
    index1.save(path);
    
    helix::IndexPQ index2(config,4,6);
    index2.load(path);
    
    EXPECT_EQ(index2.ntotal(),200);
    EXPECT_TRUE(index2.isTrained());
    
    std::vector<float> query(dim);
    for(auto& q : query)
    {
        q=dist(rng);
    }
    
    auto results1=index1.search(query.data(),5);
    auto results2=index2.search(query.data(),5);
    
    EXPECT_EQ(results1.results.size(),results2.results.size());
}

TEST_F(IndexPQTest,SmallIndex) {
    helix::dim_t small_dim=16;
    helix::IndexConfig small_config(small_dim,helix::MetricType::L2,helix::IndexType::PQ);
    helix::IndexPQ index(small_config,2,3);
    
    std::vector<float> vectors(50*small_dim,1.0f);
    index.train(vectors.data(),50);
    index.add(vectors.data(),50);
    
    std::vector<float> query(small_dim,1.0f);
    auto results=index.search(query.data(),3);
    
    EXPECT_EQ(results.results.size(),3);
}

