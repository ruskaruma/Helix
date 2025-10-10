#pragma once

#include"helix/common/types.hpp"
#include<vector>
#include<chrono>
#include<string>
#include<unordered_set>

namespace helix {

struct RecallMetrics {
    float recall_at_1;
    float recall_at_10;
    float recall_at_100;
    
    RecallMetrics() : recall_at_1(0.0f),recall_at_10(0.0f),recall_at_100(0.0f) {}
};

struct LatencyMetrics {
    double mean_ms;
    double p50_ms;
    double p95_ms;
    double p99_ms;
    double min_ms;
    double max_ms;
    
    LatencyMetrics() : mean_ms(0.0),p50_ms(0.0),p95_ms(0.0),p99_ms(0.0),min_ms(0.0),max_ms(0.0) {}
};

struct ThroughputMetrics {
    double qps;
    double total_time_ms;
    idx_t num_queries;
    
    ThroughputMetrics() : qps(0.0),total_time_ms(0.0),num_queries(0) {}
};

struct BenchmarkResults {
    RecallMetrics recall;
    LatencyMetrics latency;
    ThroughputMetrics throughput;
    size_t memory_bytes;
    
    void print() const;
    void saveJson(const std::string& path) const;
};

class MetricsCollector {
public:
    MetricsCollector()=default;
    
    void startTimer();
    void stopTimer();
    void recordQuery();
    
    RecallMetrics computeRecall(const std::vector<SearchResults>& results,const int* groundtruth,idx_t nq,int gt_k) const;
    LatencyMetrics computeLatency() const;
    ThroughputMetrics computeThroughput() const;
    
    void reset();
    
private:
    std::vector<double> queryTimes_;
    std::chrono::high_resolution_clock::time_point startTime_;
    std::chrono::high_resolution_clock::time_point lastQueryStart_;
    bool timerRunning_=false;
};

float computeRecallAtK(const SearchResults& result,const int* groundtruth,int gt_k,int k);

}

