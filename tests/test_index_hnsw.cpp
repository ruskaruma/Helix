#include<gtest/gtest.h>
#include"helix/index/index_hnsw.hpp"
#include"helix/index/index_flat.hpp"
#include"helix/common/utils.hpp"
#include<random>

class IndexHNSWTest : public ::testing::Test
{
  protected:
    IndexHNSWTest() : dim(128),n(1000),k(10),m(16),efConstruction(200),efSearch(50),
                      config(dim,helix::MetricType::L2,helix::IndexType::HNSW) {}
    
    void SetUp() override
    {
        trainData.resize(n*dim);
        queryData.resize(dim);
        
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f,1.0f);
        for(int i=0;i<n*dim;++i)
        {
            trainData[i]=dist(rng);
        }
        for(int i=0;i<dim;++i)
        {
            queryData[i]=dist(rng);
        }
    }

    int dim;
    int n;
    int k;
    int m;
    int efConstruction;
    int efSearch;
    helix::IndexConfig config;
    std::vector<float> trainData;
    std::vector<float> queryData;
};

TEST_F(IndexHNSWTest,BasicConstruction)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    EXPECT_EQ(hnsw.getM(),m);
    EXPECT_EQ(hnsw.getEfConstruction(),efConstruction);
    EXPECT_EQ(hnsw.getEfSearch(),efSearch);
}

TEST_F(IndexHNSWTest,Training)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    EXPECT_TRUE(hnsw.isTrained());
}

TEST_F(IndexHNSWTest,AddAndSearch)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    hnsw.add(trainData.data(),n);
    
    auto results=hnsw.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),k);
    
    for(const auto& result : results.results)
    {
        EXPECT_GE(result.id,0);
        EXPECT_LT(result.id,n);
        EXPECT_GE(result.distance,0.0f);
    }
}

TEST_F(IndexHNSWTest,AddWithIds)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    
    std::vector<helix::idx_t> ids(n);
    for(int i=0;i<n;++i)
    {
        ids[i]=i*2;
    }
    
    hnsw.addWithIds(trainData.data(),ids.data(),n);
    
    auto results=hnsw.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),k);
    
    for(const auto& result : results.results)
    {
        EXPECT_TRUE(std::find(ids.begin(),ids.end(),result.id)!=ids.end());
    }
}

TEST_F(IndexHNSWTest,BatchSearch)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    hnsw.add(trainData.data(),n);
    
    int nq=5;
    std::vector<float> queries(nq*dim);
    std::vector<helix::SearchResults> results;
    
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f,1.0f);
    for(int i=0;i<nq*dim;++i)
    {
        queries[i]=dist(rng);
    }
    
    hnsw.searchBatch(queries.data(),nq,k,results);
    EXPECT_EQ(results.size(),nq);
    
    for(const auto& result : results)
    {
        EXPECT_EQ(result.results.size(),k);
    }
}

TEST_F(IndexHNSWTest,SaveAndLoad)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    hnsw.add(trainData.data(),n);
    
    std::string path="/tmp/test_hnsw.bin";
    hnsw.save(path);
    
    helix::IndexHNSW loadedHnsw(config,m,efConstruction,efSearch);
    loadedHnsw.load(path);
    
    EXPECT_TRUE(loadedHnsw.isTrained());
    EXPECT_EQ(loadedHnsw.getM(),m);
    EXPECT_EQ(loadedHnsw.getEfConstruction(),efConstruction);
    EXPECT_EQ(loadedHnsw.getEfSearch(),efSearch);
    
    auto originalResults=hnsw.search(queryData.data(),k);
    auto loadedResults=loadedHnsw.search(queryData.data(),k);
    
    EXPECT_EQ(originalResults.results.size(),loadedResults.results.size());
    for(size_t i=0;i<originalResults.results.size();++i)
    {
        EXPECT_EQ(originalResults.results[i].id,loadedResults.results[i].id);
        EXPECT_FLOAT_EQ(originalResults.results[i].distance,loadedResults.results[i].distance);
    }
}

TEST_F(IndexHNSWTest,SetEfSearch)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    hnsw.add(trainData.data(),n);
    
    int newEfSearch=100;
    hnsw.setEfSearch(newEfSearch);
    EXPECT_EQ(hnsw.getEfSearch(),newEfSearch);
    
    auto results=hnsw.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),k);
}

TEST_F(IndexHNSWTest,InvalidInputs)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    
    EXPECT_THROW(hnsw.add(trainData.data(),n),helix::HelixException);
    EXPECT_THROW(hnsw.search(queryData.data(),k),helix::HelixException);
    
    hnsw.train(trainData.data(),n);
    EXPECT_THROW(hnsw.add(nullptr,n),helix::HelixException);
    EXPECT_THROW(hnsw.add(trainData.data(),0),helix::HelixException);
    EXPECT_THROW(hnsw.search(nullptr,k),helix::HelixException);
    EXPECT_THROW(hnsw.search(queryData.data(),0),helix::HelixException);
}

TEST_F(IndexHNSWTest,RecallComparison)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    helix::IndexFlat flat(config);
    
    hnsw.train(trainData.data(),n);
    flat.train(trainData.data(),n);
    
    hnsw.add(trainData.data(),n);
    flat.add(trainData.data(),n);
    
    auto hnswResults=hnsw.search(queryData.data(),k);
    auto flatResults=flat.search(queryData.data(),k);
    
    EXPECT_EQ(hnswResults.results.size(),flatResults.results.size());
    
    //check that distances are reasonable
    for(size_t i=0;i<hnswResults.results.size();++i)
    {
        EXPECT_GE(hnswResults.results[i].distance,0.0f);
        EXPECT_GE(flatResults.results[i].distance,0.0f);
    }
}

TEST_F(IndexHNSWTest,SmallIndex)
{
    int smallN=10;
    std::vector<float> smallData(smallN*dim);
    std::mt19937 rng(456);
    std::normal_distribution<float> dist(0.0f,1.0f);
    for(int i=0;i<smallN*dim;++i)
    {
        smallData[i]=dist(rng);
    }
    
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(smallData.data(),smallN);
    hnsw.add(smallData.data(),smallN);
    
    auto results=hnsw.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),std::min(k,smallN));
}

TEST_F(IndexHNSWTest,EmptyIndex)
{
    helix::IndexHNSW hnsw(config,m,efConstruction,efSearch);
    hnsw.train(trainData.data(),n);
    
    auto results=hnsw.search(queryData.data(),k);
    EXPECT_TRUE(results.results.empty());
}
