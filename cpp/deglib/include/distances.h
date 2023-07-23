#pragma once

#include <config.h>

namespace deglib {
    
    namespace distances {

        class L2Float {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                float diff0, diff1, diff2, diff3;
                const float* last = a + size;
                const float* unroll_group = last - 3;

                // Process 4 items with each loop for efficiency. 
                float result = 0;
                while (a < unroll_group) {
                    diff0 = a[0] - b[0];
                    diff1 = a[1] - b[1];
                    diff2 = a[2] - b[2];
                    diff3 = a[3] - b[3];
                    result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
                    a += 4;
                    b += 4;
                }
                // Process last 0-3 pixels.  Not needed for standard vector lengths. 
                while (a < last) {
                    diff0 = *a++ - *b++;
                    result += diff0 * diff0;
                }

                return result;
            }
        };

        class L2Float16Ext {
        public:
            // AVX instructions don't require their memory operands to be aligned, but SSE does
            // https://stackoverflow.com/questions/52147378/choice-between-aligned-vs-unaligned-x86-simd-instructions
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;
            #if defined(USE_AVX512)
                __m512 sum512 = _mm512_setzero_ps();
                while (a < last) {
                    __m512 v = _mm512_sub_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b));
                    sum512 = _mm512_fmadd_ps(v, v, sum512);
                    a += 16;
                    b += 16;
                }

                __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_AVX)
                
                // TODO two sum and v's to increase throughput
                // newer CPUs have reciprocal throughput less than its latency -> performance can be improved if multiple instructions are executed in parallel
                // https://stackoverflow.com/questions/65818232/improving-performance-of-floating-point-dot-product-of-an-array-with-simd/65827668#65827668
                __m256 sum256 = _mm256_setzero_ps();
                __m256 v;
                while (a < last) {
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;        
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;
                }

                // TODO cast faster then extract?
                //__m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
                //sum128 = _mm_add_ps(sum128, _mm_unpackhi_ps(sum128, sum128));

                // TODO horizontal add faster?
                // https://doc.rust-lang.org/core/arch/x86/fn._mm256_hadd_ps.html
                // https://stackoverflow.com/questions/51274287/computing-8-horizontal-sums-of-eight-avx-single-precision-floating-point-vectors/51275249#51275249
                // _mm256_hadd_ps(sum256)

                // TODO down to a single number 
                // https://www.py4u.net/discuss/73145

                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE)
                __m128 sum128 = _mm_setzero_ps();
                __m128 v;
                while (a < last) {
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };
        
        class L2Float8Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
            #if defined(USE_AVX)
                __m256 sum256 = _mm256_setzero_ps();
                __m256 v;
                while (a < last) {
                    v = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
                    sum256 = _mm256_fmadd_ps(v, v, sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE)
                __m128 sum128 = _mm_setzero_ps();
                __m128 v;
                while (a < last) {
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        class L2Float4Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
                __m128 sum128 = _mm_setzero_ps();
                __m128 v;
                while (a < last) {
                    v = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
                    sum128 = _mm_fmadd_ps(v, v, sum128);
                    a += 4;
                    b += 4;
                }

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        class L2Float16ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = deglib::distances::L2Float16Ext::compare(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16 ;
                float res_tail = deglib::distances::L2Float::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };

        class L2Float4ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = deglib::distances::L2Float4Ext::compare(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = deglib::distances::L2Float::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };



        class InnerProductFloat {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                float dot0, dot1, dot2, dot3;
                const float* last = a + size;
                const float* unroll_group = last - 3;

                // Process 4 items with each loop for efficiency. 
                float result = 0;
                while (a < unroll_group) {
                    dot0 = a[0] * b[0];
                    dot1 = a[1] * b[1];
                    dot2 = a[2] * b[2];
                    dot3 = a[3] * b[3];
                    result += dot0 + dot1 + dot2 + dot3;
                    a += 4;
                    b += 4;
                }

                // Process last 0-3 pixels.  Not needed for standard vector lengths. 
                while (a < last) {
                    result += *a++ * *b++;
                }

                return result;
            }
        };

        class InnerProductFloat16Ext {
        public:
            // AVX instructions don't require their memory operands to be aligned, but SSE does
            // https://stackoverflow.com/questions/52147378/choice-between-aligned-vs-unaligned-x86-simd-instructions
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) 
            {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);

                const float *last = a + size;
            #if defined(USE_AVX512)
                __m512 sum512 = _mm512_setzero_ps();
                while (a < last) {
                    sum512 = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(b), sum512);
                    a += 16;
                    b += 16;
                }

                __m256 sum256 = _mm256_add_ps(_mm512_extractf32x8_ps(sum512, 0), _mm512_extractf32x8_ps(sum512, 1));
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_AVX)
                __m256 sum256 = _mm256_setzero_ps();
                while (a < last) {
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;        
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE)
                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };
        
        class InnerProductFloat8Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
            #if defined(USE_AVX)
                __m256 sum256 = _mm256_setzero_ps();
                while (a < last) {
                    sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b), sum256);
                    a += 8;
                    b += 8;
                }
                __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            #elif defined(USE_SSE)
                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                }
            #endif 

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return f[0] + f[1] + f[2] + f[3];
            }
        };

        class InnerProductFloat4Ext {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                float *a = (float *) pVect1v;
                float *b = (float *) pVect2v;
                size_t size = *((size_t *) qty_ptr);
                
                const float *last = a + size;
                __m128 sum128 = _mm_setzero_ps();
                while (a < last) {
                    sum128 = _mm_fmadd_ps(_mm_loadu_ps(a), _mm_loadu_ps(b), sum128);
                    a += 4;
                    b += 4;
                }

                alignas(32) float f[4];
                _mm_store_ps(f, sum128);
                return abs(f[0] + f[1] + f[2] + f[3]);
            }
        };

        class InnerProductFloat16ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty16 = qty >> 4 << 4;
                float res = deglib::distances::InnerProductFloat16Ext::compare(pVect1v, pVect2v, &qty16);
                float *pVect1 = (float *) pVect1v + qty16;
                float *pVect2 = (float *) pVect2v + qty16;

                size_t qty_left = qty - qty16;
                float res_tail = deglib::distances::InnerProductFloat::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };

        class InnerProductFloat4ExtResiduals {
        public:
            inline static float compare(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
                size_t qty = *((size_t *) qty_ptr);

                size_t qty4 = qty >> 2 << 2;
                float res = deglib::distances::InnerProductFloat4Ext::compare(pVect1v, pVect2v, &qty4);
                float *pVect1 = (float *) pVect1v + qty4;
                float *pVect2 = (float *) pVect2v + qty4;

                size_t qty_left = qty - qty4;
                float res_tail = deglib::distances::InnerProductFloat::compare(pVect1, pVect2, &qty_left);
                return (res + res_tail);
            }
        };

    } // end namespace Distances


    enum class Metric {
        L2 = 1,
        InnerProduct = 2
    };

    template <typename MTYPE>
    using DISTFUNC = MTYPE (*)(const void*, const void*, const void*);

    template <typename MTYPE>
    class SpaceInterface
    {
    public:
        virtual const size_t dim() const = 0;
        virtual const deglib::Metric metric() const = 0;
        virtual const size_t get_data_size() const = 0;
        virtual const DISTFUNC<MTYPE> get_dist_func() const = 0;
        virtual const void* get_dist_func_param() const = 0;
    };


    class FloatSpace : public SpaceInterface<float> {

        static DISTFUNC<float> select_dist_func(const size_t dim, const deglib::Metric metric) {
            DISTFUNC<float> distfunc = deglib::distances::L2Float::compare;
            if(metric == deglib::Metric::L2) {
                #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                    if (dim % 16 == 0)
                        distfunc = deglib::distances::L2Float16Ext::compare;
                    else if (dim % 8 == 0)
                        distfunc = deglib::distances::L2Float8Ext::compare;
                    else if (dim % 4 == 0)
                        distfunc = deglib::distances::L2Float4Ext::compare;
                    else if (dim > 16)
                        distfunc = deglib::distances::L2Float16ExtResiduals::compare;
                    else if (dim > 4)
                        distfunc = deglib::distances::L2Float4ExtResiduals::compare;
                #endif
            }
            else 
            {
                #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
                    if (dim % 16 == 0)
                        distfunc = deglib::distances::InnerProductFloat16Ext::compare;
                    else if (dim % 8 == 0)
                        distfunc = deglib::distances::InnerProductFloat8Ext::compare;
                    else if (dim % 4 == 0)
                        distfunc = deglib::distances::InnerProductFloat4Ext::compare;
                    else if (dim > 16)
                        distfunc = deglib::distances::InnerProductFloat16ExtResiduals::compare;
                    else if (dim > 4)
                        distfunc = deglib::distances::InnerProductFloat4ExtResiduals::compare;
                #elif
                    distfunc = deglib::distances::InnerProductFloat::compare;
                #endif
            }

            // TODO add cosine but convert to a distance = 2 - (cosine + 1)
            // https://www.kaggle.com/cdabakoglu/word-vectors-cosine-similarity
            // https://github.com/yahoojapan/NGT/blob/master/lib/NGT/PrimitiveComparator.h#L431

            return distfunc;
        }

        const DISTFUNC<float> fstdistfunc_;
        const size_t data_size_;
        const size_t dim_;
        const deglib::Metric metric_;

    public:
        FloatSpace(const size_t dim, const deglib::Metric metric) 
            : metric_(metric), dim_(dim), data_size_(dim * sizeof(float)), fstdistfunc_(select_dist_func(dim, metric)) {
        }

        const size_t dim() const {
            return dim_;
        }

        const deglib::Metric metric() const {
            return metric_;
        }


        const size_t get_data_size() const {
            return data_size_;
        }

        const DISTFUNC<float> get_dist_func() const {
            return fstdistfunc_;
        }

        const void *get_dist_func_param() const {
            return &dim_;
        }

        ~FloatSpace() {}
    };

}  // end namespace deglib