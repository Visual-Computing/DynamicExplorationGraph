#include "test_regression.h"

TEST(DeglibDatasetDeterminism, ClusteredDatasetBitExactness)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<float> base_data;
    std::vector<float> query_data;
    generate_synthetic_clustered_dataset(base_count, dim, base_data, query_data, query_count, num_clusters);

    const uint64_t base_hash = float_vector_checksum(base_data);
    const uint64_t query_hash = float_vector_checksum(query_data);

    auto gt_l2 = compute_groundtruth_l2(base_data, base_count, query_data, query_count, dim, 10);
    const uint64_t gt_l2_hash = groundtruth_checksum(gt_l2);

    auto gt_ip = compute_groundtruth_innerproduct(base_data, base_count, query_data, query_count, dim, 10);
    const uint64_t gt_ip_hash = groundtruth_checksum(gt_ip);

    std::cout << "[DatasetDeterminism] base_hash = 0x" << std::hex << base_hash << std::dec << std::endl;
    std::cout << "[DatasetDeterminism] query_hash = 0x" << std::hex << query_hash << std::dec << std::endl;
    std::cout << "[DatasetDeterminism] gt_l2_hash = 0x" << std::hex << gt_l2_hash << std::dec << std::endl;
    std::cout << "[DatasetDeterminism] gt_ip_hash = 0x" << std::hex << gt_ip_hash << std::dec << std::endl;

    // Hardcoded expected hashes computed on Windows
    // Checksum verification ensures 100% bit-exact dataset generation and groundtruth across OS/compilers.
    EXPECT_EQ(base_hash, 0xef1c7686c6a529f4ULL) << "base_data checksum mismatch across platforms!";
    EXPECT_EQ(query_hash, 0xea1c88f507eeb188ULL) << "query_data checksum mismatch across platforms!";
    EXPECT_EQ(gt_l2_hash, 0xe04ea5bc6428cc3dULL) << "gt_l2 checksum mismatch across platforms!";
    EXPECT_EQ(gt_ip_hash, 0x10c26c33619a5928ULL) << "gt_ip checksum mismatch across platforms!";
}

TEST(DeglibDatasetDeterminism, Uint8ClusteredDatasetBitExactness)
{
    const size_t dim = 128;
    const size_t base_count = 100000;
    const size_t query_count = 100;
    const size_t num_clusters = 1000;

    std::vector<uint8_t> base_data;
    std::vector<uint8_t> query_data;
    generate_synthetic_clustered_dataset_uint8(base_count, dim, base_data, query_data, query_count, num_clusters);

    const uint64_t base_hash = fnv1a_64(base_data.data(), base_data.size());
    const uint64_t query_hash = fnv1a_64(query_data.data(), query_data.size());

    auto gt_u8 = compute_groundtruth_l2_uint8(base_data, base_count, query_data, query_count, dim, 10);
    const uint64_t gt_u8_hash = groundtruth_checksum(gt_u8);

    std::cout << "[DatasetDeterminism] uint8_base_hash = 0x" << std::hex << base_hash << std::dec << std::endl;
    std::cout << "[DatasetDeterminism] uint8_query_hash = 0x" << std::hex << query_hash << std::dec << std::endl;
    std::cout << "[DatasetDeterminism] gt_u8_hash = 0x" << std::hex << gt_u8_hash << std::dec << std::endl;

    EXPECT_EQ(base_hash, 0x4d94d32f77a245daULL) << "uint8_base_data checksum mismatch across platforms!";
    EXPECT_EQ(query_hash, 0x5493b1b61d1b8a94ULL) << "uint8_query_data checksum mismatch across platforms!";
    EXPECT_EQ(gt_u8_hash, 0xde241e831af209fcULL) << "gt_u8 checksum mismatch across platforms!";
}
