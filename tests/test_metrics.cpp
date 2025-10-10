#include<gtest/gtest.h>
#include"helix/benchmark/metrics.hpp"
#include<thread>
#include<chrono>
#include<fstream>

TEST(MetricsTest,RecallAtK) {
    helix::SearchResults result;
    result.results.push_back({0,1.0f});
    result.results.push_back({1,2.0f});
    result.results.push_back({2,3.0f});
    result.results.push_back({5,4.0f});
    result.results.push_back({7,5.0f});
    
    int groundtruth[]={0,1,2,3,4,5,6,7,8,9};
    
    float recall1=helix::computeRecallAtK(result,groundtruth,10,1);
    EXPECT_FLOAT_EQ(recall1,1.0f);
    
    float recall3=helix::computeRecallAtK(result,groundtruth,10,3);
    EXPECT_FLOAT_EQ(recall3,1.0f);
    
    float recall5=helix::computeRecallAtK(result,groundtruth,10,5);
    EXPECT_NEAR(recall5,0.6f,0.01f);
}

TEST(MetricsTest,RecallAtKNoMatches) {
    helix::SearchResults result;
    result.results.push_back({100,1.0f});
    result.results.push_back({101,2.0f});
    
    int groundtruth[]={0,1,2,3,4,5,6,7,8,9};
    
    float recall=helix::computeRecallAtK(result,groundtruth,10,2);
    EXPECT_FLOAT_EQ(recall,0.0f);
}

TEST(MetricsTest,MetricsCollector) {
    helix::MetricsCollector collector;
    
    collector.startTimer();
    for(int i=0;i<10;++i)
    {
        collector.recordQuery();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        collector.stopTimer();
    }
    
    auto latency=collector.computeLatency();
    EXPECT_GT(latency.mean_ms,0.0);
    EXPECT_GT(latency.p50_ms,0.0);
    EXPECT_GT(latency.p95_ms,0.0);
    
    auto throughput=collector.computeThroughput();
    EXPECT_EQ(throughput.num_queries,10);
    EXPECT_GT(throughput.qps,0.0);
}

TEST(MetricsTest,ComputeRecall) {
    helix::MetricsCollector collector;
    
    std::vector<helix::SearchResults> results(3);
    results[0].results={{0,1.0f},{1,2.0f},{2,3.0f}};
    results[1].results={{5,1.0f},{6,2.0f},{7,3.0f}};
    results[2].results={{10,1.0f},{11,2.0f},{12,3.0f}};
    
    int groundtruth[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};
    
    auto recall=collector.computeRecall(results,groundtruth,3,10);
    EXPECT_GT(recall.recall_at_1,0.0f);
}

TEST(MetricsTest,BenchmarkResultsPrint) {
    helix::BenchmarkResults results;
    results.recall.recall_at_1=0.95f;
    results.recall.recall_at_10=0.98f;
    results.recall.recall_at_100=0.99f;
    results.latency.mean_ms=1.5;
    results.latency.p50_ms=1.2;
    results.latency.p95_ms=2.5;
    results.latency.p99_ms=3.0;
    results.throughput.qps=666.67;
    results.throughput.total_time_ms=150.0;
    results.throughput.num_queries=100;
    results.memory_bytes=1024*1024*10;
    
    results.print();
    
    std::string path="/tmp/helix_test_results.json";
    results.saveJson(path);
    
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open());
    file.close();
}

TEST(MetricsTest,EmptyMetrics) {
    helix::MetricsCollector collector;
    
    auto latency=collector.computeLatency();
    EXPECT_EQ(latency.mean_ms,0.0);
    
    auto throughput=collector.computeThroughput();
    EXPECT_EQ(throughput.num_queries,0);
}

TEST(MetricsTest,Reset) {
    helix::MetricsCollector collector;
    
    collector.startTimer();
    for(int i=0;i<5;++i)
    {
        collector.recordQuery();
        collector.stopTimer();
    }
    
    auto throughput1=collector.computeThroughput();
    EXPECT_EQ(throughput1.num_queries,5);
    
    collector.reset();
    
    auto throughput2=collector.computeThroughput();
    EXPECT_EQ(throughput2.num_queries,0);
}

