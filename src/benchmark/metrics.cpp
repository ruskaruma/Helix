#include"helix/benchmark/metrics.hpp"
#include<algorithm>
#include<numeric>
#include<cmath>
#include<fstream>
#include<iostream>
#include<iomanip>

namespace helix {

void MetricsCollector::startTimer()
{
    startTime_=std::chrono::high_resolution_clock::now();
    timerRunning_=true;
}

void MetricsCollector::stopTimer()
{
    auto endTime=std::chrono::high_resolution_clock::now();
    double elapsed=std::chrono::duration<double,std::milli>(endTime-lastQueryStart_).count();
    queryTimes_.push_back(elapsed);
}

void MetricsCollector::recordQuery()
{
    lastQueryStart_=std::chrono::high_resolution_clock::now();
}

RecallMetrics MetricsCollector::computeRecall(const std::vector<SearchResults>& results,const int* groundtruth,idx_t nq,int gt_k) const
{
    RecallMetrics metrics;
    
    float sum_at_1=0.0f;
    float sum_at_10=0.0f;
    float sum_at_100=0.0f;
    
    for(idx_t i=0;i<nq;++i)
    {
        sum_at_1+=computeRecallAtK(results[i],groundtruth+i*gt_k,gt_k,1);
        sum_at_10+=computeRecallAtK(results[i],groundtruth+i*gt_k,gt_k,10);
        sum_at_100+=computeRecallAtK(results[i],groundtruth+i*gt_k,gt_k,100);
    }
    
    metrics.recall_at_1=sum_at_1/nq;
    metrics.recall_at_10=sum_at_10/nq;
    metrics.recall_at_100=sum_at_100/nq;
    
    return metrics;
}

LatencyMetrics MetricsCollector::computeLatency() const
{
    LatencyMetrics metrics;
    
    if(queryTimes_.empty())
    {
        return metrics;
    }
    
    std::vector<double> sorted=queryTimes_;
    std::sort(sorted.begin(),sorted.end());
    
    metrics.mean_ms=std::accumulate(sorted.begin(),sorted.end(),0.0)/sorted.size();
    metrics.min_ms=sorted.front();
    metrics.max_ms=sorted.back();
    metrics.p50_ms=sorted[sorted.size()*50/100];
    metrics.p95_ms=sorted[sorted.size()*95/100];
    metrics.p99_ms=sorted[sorted.size()*99/100];
    
    return metrics;
}

ThroughputMetrics MetricsCollector::computeThroughput() const
{
    ThroughputMetrics metrics;
    
    if(queryTimes_.empty())
    {
        return metrics;
    }
    
    metrics.num_queries=queryTimes_.size();
    metrics.total_time_ms=std::accumulate(queryTimes_.begin(),queryTimes_.end(),0.0);
    metrics.qps=(metrics.num_queries*1000.0)/metrics.total_time_ms;
    
    return metrics;
}

void MetricsCollector::reset()
{
    queryTimes_.clear();
    timerRunning_=false;
}

float computeRecallAtK(const SearchResults& result,const int* groundtruth,int gt_k,int k)
{
    k=std::min(k,static_cast<int>(result.results.size()));
    k=std::min(k,gt_k);
    
    std::unordered_set<idx_t> gt_set;
    for(int i=0;i<k;++i)
    {
        gt_set.insert(groundtruth[i]);
    }
    
    int hits=0;
    for(int i=0;i<k;++i)
    {
        if(gt_set.count(result.results[i].id))
        {
            hits++;
        }
    }
    
    return static_cast<float>(hits)/k;
}

void BenchmarkResults::print() const
{
    std::cout<<std::fixed<<std::setprecision(4);
    std::cout<<"Recall Metrics:"<<std::endl;
    std::cout<<"  Recall@1:   "<<recall.recall_at_1<<std::endl;
    std::cout<<"  Recall@10:  "<<recall.recall_at_10<<std::endl;
    std::cout<<"  Recall@100: "<<recall.recall_at_100<<std::endl;
    
    std::cout<<"\nLatency Metrics:"<<std::endl;
    std::cout<<"  Mean:  "<<latency.mean_ms<<" ms"<<std::endl;
    std::cout<<"  P50:   "<<latency.p50_ms<<" ms"<<std::endl;
    std::cout<<"  P95:   "<<latency.p95_ms<<" ms"<<std::endl;
    std::cout<<"  P99:   "<<latency.p99_ms<<" ms"<<std::endl;
    std::cout<<"  Min:   "<<latency.min_ms<<" ms"<<std::endl;
    std::cout<<"  Max:   "<<latency.max_ms<<" ms"<<std::endl;
    
    std::cout<<"\nThroughput Metrics:"<<std::endl;
    std::cout<<"  QPS:         "<<throughput.qps<<std::endl;
    std::cout<<"  Total time:  "<<throughput.total_time_ms<<" ms"<<std::endl;
    std::cout<<"  Queries:     "<<throughput.num_queries<<std::endl;
    
    std::cout<<"\nMemory: "<<(memory_bytes/1024.0/1024.0)<<" MB"<<std::endl;
}

void BenchmarkResults::saveJson(const std::string& path) const
{
    std::ofstream file(path);
    file<<std::fixed<<std::setprecision(4);
    file<<"{\n";
    file<<"  \"recall\": {\n";
    file<<"    \"recall_at_1\": "<<recall.recall_at_1<<",\n";
    file<<"    \"recall_at_10\": "<<recall.recall_at_10<<",\n";
    file<<"    \"recall_at_100\": "<<recall.recall_at_100<<"\n";
    file<<"  },\n";
    file<<"  \"latency\": {\n";
    file<<"    \"mean_ms\": "<<latency.mean_ms<<",\n";
    file<<"    \"p50_ms\": "<<latency.p50_ms<<",\n";
    file<<"    \"p95_ms\": "<<latency.p95_ms<<",\n";
    file<<"    \"p99_ms\": "<<latency.p99_ms<<",\n";
    file<<"    \"min_ms\": "<<latency.min_ms<<",\n";
    file<<"    \"max_ms\": "<<latency.max_ms<<"\n";
    file<<"  },\n";
    file<<"  \"throughput\": {\n";
    file<<"    \"qps\": "<<throughput.qps<<",\n";
    file<<"    \"total_time_ms\": "<<throughput.total_time_ms<<",\n";
    file<<"    \"num_queries\": "<<throughput.num_queries<<"\n";
    file<<"  },\n";
    file<<"  \"memory_bytes\": "<<memory_bytes<<"\n";
    file<<"}\n";
    file.close();
}

}

