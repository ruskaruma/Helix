#include<gtest/gtest.h>
#include"helix/index/index_ivf.hpp"
#include"helix/index/index_flat.hpp"
#include"helix/common/utils.hpp"
#include<random>

class IndexIVFTest : public ::testing::Test
{
  protected:
    IndexIVFTest() : dim(128),n(1000),k(10),nlist(64),nprobe(8),
                     config(dim,helix::MetricType::L2,helix::IndexType::IVF) {}
    
    void SetUp() override
    {
        dim=128;
        n=1000;
        k=10;
        nlist=64;
        nprobe=8;
        
        config=helix::IndexConfig(dim,helix::MetricType::L2,helix::IndexType::IVF);
        
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
    int nlist;
    int nprobe;
    helix::IndexConfig config;
    std::vector<float> trainData;
    std::vector<float> queryData;
};

TEST_F(IndexIVFTest,BasicConstruction)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    EXPECT_EQ(ivf.getNlist(),nlist);
    EXPECT_EQ(ivf.getNprobe(),nprobe);
}

TEST_F(IndexIVFTest,Training)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    ivf.train(trainData.data(),n);
    EXPECT_TRUE(ivf.isTrained());
}

TEST_F(IndexIVFTest,AddAndSearch)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    ivf.train(trainData.data(),n);
    ivf.add(trainData.data(),n);
    
    auto results=ivf.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),k);
    
    for(const auto& result : results.results)
    {
        EXPECT_GE(result.id,0);
        EXPECT_LT(result.id,n);
        EXPECT_GE(result.distance,0.0f);
    }
}

TEST_F(IndexIVFTest,AddWithIds)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    ivf.train(trainData.data(),n);
    
    std::vector<helix::idx_t> ids(n);
    for(int i=0;i<n;++i)
    {
        ids[i]=i*2;
    }
    
    ivf.addWithIds(trainData.data(),ids.data(),n);
    
    auto results=ivf.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),k);
    
    for(const auto& result : results.results)
    {
        EXPECT_TRUE(std::find(ids.begin(),ids.end(),result.id)!=ids.end());
    }
}

TEST_F(IndexIVFTest,BatchSearch)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    ivf.train(trainData.data(),n);
    ivf.add(trainData.data(),n);
    
    int nq=5;
    std::vector<float> queries(nq*dim);
    std::vector<helix::SearchResults> results;
    
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f,1.0f);
    for(int i=0;i<nq*dim;++i)
    {
        queries[i]=dist(rng);
    }
    
    ivf.searchBatch(queries.data(),nq,k,results);
    EXPECT_EQ(results.size(),nq);
    
    for(const auto& result : results)
    {
        EXPECT_EQ(result.results.size(),k);
    }
}

TEST_F(IndexIVFTest,SaveAndLoad)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    ivf.train(trainData.data(),n);
    ivf.add(trainData.data(),n);
    
    std::string path="/tmp/test_ivf.bin";
    ivf.save(path);
    
    helix::IndexIVF loadedIvf(config,nlist,nprobe);
    loadedIvf.load(path);
    
    EXPECT_TRUE(loadedIvf.isTrained());
    EXPECT_EQ(loadedIvf.getNlist(),nlist);
    EXPECT_EQ(loadedIvf.getNprobe(),nprobe);
    
    auto originalResults=ivf.search(queryData.data(),k);
    auto loadedResults=loadedIvf.search(queryData.data(),k);
    
    EXPECT_EQ(originalResults.results.size(),loadedResults.results.size());
    for(size_t i=0;i<originalResults.results.size();++i)
    {
        EXPECT_EQ(originalResults.results[i].id,loadedResults.results[i].id);
        EXPECT_FLOAT_EQ(originalResults.results[i].distance,loadedResults.results[i].distance);
    }
}

TEST_F(IndexIVFTest,SetNprobe)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    ivf.train(trainData.data(),n);
    ivf.add(trainData.data(),n);
    
    int newNprobe=16;
    ivf.setNprobe(newNprobe);
    EXPECT_EQ(ivf.getNprobe(),newNprobe);
    
    auto results=ivf.search(queryData.data(),k);
    EXPECT_EQ(results.results.size(),k);
}

TEST_F(IndexIVFTest,InvalidInputs)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    
    EXPECT_THROW(ivf.add(trainData.data(),n),helix::HelixException);
    EXPECT_THROW(ivf.search(queryData.data(),k),helix::HelixException);
    
    ivf.train(trainData.data(),n);
    EXPECT_THROW(ivf.add(nullptr,n),helix::HelixException);
    EXPECT_THROW(ivf.add(trainData.data(),0),helix::HelixException);
    EXPECT_THROW(ivf.search(nullptr,k),helix::HelixException);
    EXPECT_THROW(ivf.search(queryData.data(),0),helix::HelixException);
}

TEST_F(IndexIVFTest,RecallComparison)
{
    helix::IndexIVF ivf(config,nlist,nprobe);
    helix::IndexFlat flat(config);
    
    ivf.train(trainData.data(),n);
    flat.train(trainData.data(),n);
    
    ivf.add(trainData.data(),n);
    flat.add(trainData.data(),n);
    
    auto ivfResults=ivf.search(queryData.data(),k);
    auto flatResults=flat.search(queryData.data(),k);
    
    EXPECT_EQ(ivfResults.results.size(),flatResults.results.size());
    
    //check that distances are reasonable
    for(size_t i=0;i<ivfResults.results.size();++i)
    {
        EXPECT_GE(ivfResults.results[i].distance,0.0f);
        EXPECT_GE(flatResults.results[i].distance,0.0f);
    }
}
