
#include <random>
#include <chrono>

#include <fmt/core.h>
#include <omp.h>

#include "benchmark.h"
#include "deglib.h"

/**
 * Convert the queue into a vector with ascending distance order
 **/
static auto topListAscending(deglib::search::ResultSet& queue) {
    const auto size = (int32_t) queue.size();
    auto topList = std::vector<deglib::search::ObjectDistance>(size);
    for(int32_t i = size - 1; i >= 0; i--) {
        topList[i] = std::move(const_cast<deglib::search::ObjectDistance&>(queue.top()));
        queue.pop();
    }
    return topList;
}

/**
 * Write ivecs file
 **/
void ivecs_write(const char *fname, uint32_t d, size_t n, const uint32_t* v) {
    auto out = std::ofstream(fname, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
        fmt::print(stderr, "Error in open file {}\n", fname);
        perror("");
        abort();
    }
    for (uint32_t i = 0; i < n; i++) {
        const auto ptr = v + i * d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));    
        out.write(reinterpret_cast<const char*>(ptr), sizeof(uint32_t) * d);    
    }

    out.close();
}

/**
 * Write fvecs file
 **/
void fvecs_write(const char *fname, uint32_t d, size_t n, const float* v) {
    auto out = std::ofstream(fname, std::ios::out | std::ios::binary);

    // check open file for write
    if (!out.is_open()) {
        fmt::print(stderr, "Error in open file {}\n", fname);
        perror("");
        abort();
    }
    for (uint32_t i = 0; i < n; i++) {
        const auto ptr = v + i * d;
        out.write(reinterpret_cast<const char*>(&d), sizeof(d));    
        out.write(reinterpret_cast<const char*>(ptr), sizeof(float) * d);    
    }

    out.close();
}


/**
 * Compute the gt data
 */
static std::vector<uint32_t> compute_gt(const deglib::StaticFeatureRepository& base_repo, const deglib::StaticFeatureRepository& query_repo, const deglib::Metric metric, const uint32_t k_target) {
    const auto start = std::chrono::steady_clock::now();

    const auto base_size = (uint32_t)base_repo.size();
    const auto query_size = (uint32_t)query_repo.size();
    const auto dims = base_repo.dims();

    const auto feature_space = deglib::FloatSpace(dims, metric);
    const auto dist_func = feature_space.get_dist_func();
    const auto dist_func_param = feature_space.get_dist_func_param();

    auto count = 0;
    auto topLists = std::vector<uint32_t>(k_target*query_size);
    #pragma omp parallel for
    for (int q = 0; q < (int)query_size; q++) {
        const auto query = query_repo.getFeature(q);

        auto worst_distance = std::numeric_limits<float>::max();
        auto results = deglib::search::ResultSet(); 
        for (uint32_t b = 0; b < base_size; b++) {
            const auto distance = dist_func(query, base_repo.getFeature(b), dist_func_param);
            if(distance < worst_distance) {
                results.emplace(b, distance);
                if (results.size() > k_target) {
                    results.pop();
                    worst_distance = results.top().getDistance();
                }
            }
        }

        if (results.size() != k_target) {
            fmt::print(stderr, "For query {} only {} base elements have been found, {} are required.\n", q, results.size(), k_target);
            perror("");
            abort();
        }

        auto topList = topLists.data() + (k_target*q);
        for(int32_t i = k_target - 1; i >= 0; i--) {
            topList[i] = results.top().getInternalIndex();
            results.pop();
        }

        #pragma omp critical
        {
            count++;
            if(count % 100 == 0) {
                const auto duration_ms = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
                fmt::print("Computed {} ground truth lists after {}ms\n", count, duration_ms);
            }
        }
    }

    return topLists;
}

int main() {

    #if defined(USE_AVX)
        fmt::print("use AVX2  ...\n");
    #elif defined(USE_SSE)
        fmt::print("use SSE  ...\n");
    #else
        fmt::print("use arch  ...\n");
    #endif
    fmt::print("Actual memory usage: {} Mb, Max memory usage: {} Mb \n", getCurrentRSS() / 1000000, getPeakRSS() / 1000000);

    omp_set_num_threads(1);
    std::cout << "_OPENMP " << omp_get_num_threads() << " threads" << std::endl;

    const auto data_path = std::filesystem::path(DATA_PATH);

    // ------------------------------------------------------- laion -----------------------------------------------------------------
    const auto repository_file          = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K_512byteFloat.fvecs").string();
    const auto query_file               = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k_512byteFloat.fvecs").string();
    const auto gt_file                  = (data_path / "laion2B" / "gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.ivecs").string();

    const auto repository_file_uint8    = (data_path / "laion2B" / "laion2B-en-clip768v2-n=300K_512byte.u8vecs").string();
    const auto query_file_uint8         = (data_path / "laion2B" / "public-queries-2024-laion2B-en-clip768v2-n=10k_512byte.u8vecs").string();

    // compute ground truth using uint8 files 
    std::vector<uint32_t> topListsUint8;
    {
        const uint32_t k_target = 100;
        const auto base_repository = deglib::load_static_repository(repository_file_uint8.c_str());
        const auto query_repository = deglib::load_static_repository(query_file_uint8.c_str());

        const auto start = std::chrono::system_clock::now();
        const auto topLists = compute_gt(base_repository, query_repository, deglib::Metric::L2_Uint8, k_target);
        auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
        fmt::print("Computing {:5} top lists of a {:8} base took {:5}s \n", query_repository.size(), base_repository.size(), duration);
        topListsUint8 = topLists;

        // store ground truth
        // ivecs_write(gt_file.c_str(), k_target, query_repository.size(), topListsUint8.data());
    }

    // compute ground truth using float files 
    std::vector<uint32_t> topListsFloat;
    {
        const uint32_t k_target = 100;
        const auto base_repository = deglib::load_static_repository(repository_file.c_str());
        const auto query_repository = deglib::load_static_repository(query_file.c_str());

        const auto start = std::chrono::system_clock::now();
        const auto topLists = compute_gt(base_repository, query_repository, deglib::Metric::L2, k_target);
        auto duration = uint32_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - start).count());
        fmt::print("Computing {:5} top lists of a {:8} base took {:5}s \n", query_repository.size(), base_repository.size(), duration);
        topListsFloat = topLists;

        // store ground truth
        // ivecs_write(gt_file.c_str(), k_target, query_repository.size(), topListsFloat.data());
    }

    // compare ground truth
    for (size_t g = 0; g < topListsFloat.size(); g++) {
        if(topListsFloat[g] != topListsUint8[g]) {
            fmt::print(stderr, "Found different gt information at index {}, expected {} got {}.\n", g, topListsFloat[g], topListsUint8[g]);
            perror("");
            abort();
        }         
    }

    // compare uint8 and uint8-in-float files
    {
        size_t dims_f;
        size_t count_f;
        auto y1 = deglib::fvecs_read(query_file.c_str(), dims_f, count_f);
        auto y_f = reinterpret_cast<float*>(y1.get());
        std::cout << "dims_f=" << dims_f << ", count_f=" << count_f << std::endl;
        std::cout << "x[0]=" << uint32_t(y_f[0]) << ", x[511]=" << uint32_t(y_f[511]) << ", x[512]=" << uint32_t(y_f[512]) << ", x[513]=" << uint32_t(y_f[513]) << std::endl;

        size_t dims_u8;
        size_t count_u8;
        auto y = deglib::u8vecs_read(query_file_uint8.c_str(), dims_u8, count_u8);
        std::cout << "dims_u8=" << dims_u8 << ", count_u8=" << count_u8 << std::endl;
        std::cout << "x[0]=" << uint32_t(y[0]) << ", x[511]=" << uint32_t(y[511]) << ", x[512]=" << uint32_t(y[512]) << ", x[513]=" << uint32_t(y[513]) << std::endl;
    }
}