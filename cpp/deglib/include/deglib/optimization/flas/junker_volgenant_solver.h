// Apache 2.0 License — Visual Computing Group, HTW Berlin
#ifndef EVP_FLAS_JUNKER_VOLGENANT_SOLVER_H
#define EVP_FLAS_JUNKER_VOLGENANT_SOLVER_H

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <vector>

// ---------------------------------------------------------------------------
// Linear assignment (Junker-Volgenant / shortest-augmenting-path) solver.
//
// `matrix` is a dim x dim row-major cost matrix (e.g. the quantized distance
// LUT). Returns the column assignment `row -> col` (i.e. permutation[row] =
// col). The algorithm body is allocation-free — it operates entirely on the
// buffers owned by JVScratch.
//
// Usage:
//   JVScratch scratch(dim);
//   compute_assignment(matrix, dim, scratch);
//   scratch.perm()  // holds the result
//
// For a self-contained result:
//   auto perm = compute_assignment(matrix, dim);  // returns std::vector<int>
//
// On the hot MT path, construct one JVScratch per thread and reuse it across
// all calls — scratch.reset() re-zeroes in-place without re-allocating.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// JVScratch — RAII scratch space for the Junker-Volgenant solver.
//
// Holds 8*dim ints (1 perm + 7 scratch: in_col, v, free_, collist, matches,
// pred, d). All zero-initialised on construction.
// ---------------------------------------------------------------------------
class JVScratch {
  public:
    // Number of dim-sized regions in the buffer: 1 output permutation +
    // 7 scratch arrays (in_col, v, free_, collist, matches, pred, d).
    static constexpr int kNumRegions = 8;

    // Construct with a given dimension, allocating kNumRegions*dim ints (zero-filled).
    explicit JVScratch(int dim) : dim_(dim), capacity_(dim), buffer_(static_cast<size_t>(kNumRegions) * static_cast<size_t>(dim), 0) {}

    // Default-constructible for lazy init via init().
    JVScratch() = default;

    // Re-initialise to a new dimension. If dim <= capacity, reuses the
    // existing buffer (zeroing only the used portion). Otherwise re-allocates.
    void init(int dim) {
        if (dim <= capacity_) {
            dim_ = dim;
            std::fill(buffer_.begin(), buffer_.begin() + static_cast<size_t>(kNumRegions) * static_cast<size_t>(dim), 0);
        } else {
            capacity_ = dim;
            dim_ = dim;
            buffer_.assign(static_cast<size_t>(kNumRegions) * static_cast<size_t>(dim), 0);
        }
    }

    // Zero the existing buffer in-place — cheaper than re-allocating.
    // The algorithm guarantees it only reads what it has written, so a full
    // zero is sufficient.
    void reset() noexcept { std::fill(buffer_.begin(), buffer_.begin() + static_cast<size_t>(kNumRegions) * static_cast<size_t>(dim_), 0); }

    // Accessors — return raw pointers into the internal buffer.
    int* perm() noexcept { return buffer_.data(); }
    int* in_col() noexcept { return buffer_.data() + static_cast<size_t>(dim_); }
    int* v() noexcept { return buffer_.data() + static_cast<size_t>(dim_) * 2; }
    int* free() noexcept { return buffer_.data() + static_cast<size_t>(dim_) * 3; }
    int* collist() noexcept { return buffer_.data() + static_cast<size_t>(dim_) * 4; }
    int* matches() noexcept { return buffer_.data() + static_cast<size_t>(dim_) * 5; }
    int* pred() noexcept { return buffer_.data() + static_cast<size_t>(dim_) * 6; }
    int* d() noexcept { return buffer_.data() + static_cast<size_t>(dim_) * 7; }

    int dim() const noexcept { return dim_; }

  private:
    int dim_ = 0;
    int capacity_ = 0;
    std::vector<int> buffer_;
};

// ---------------------------------------------------------------------------
// Core algorithm: writes the dim-element column assignment into scratch.perm()
// and uses scratch's internal buffer for all working arrays. The scratch is
// reset (zeroed) at entry so results are identical to the original
// No allocation or memset happens inside the algorithm body — all buffer
// management is handled by JVScratch.
// ---------------------------------------------------------------------------
inline void compute_assignment(const int* matrix, int dim, JVScratch& scratch) {
    if (scratch.dim() != dim) {
        scratch.init(dim);
    } else {
        scratch.reset();
    }

    // Wire up raw pointers into the scratch's buffer for the hot loop.
    // out_perm serves double-duty: it is both the row->col assignment output
    // and the row->col lookup (in_col is the inverse: col->row).
    int* out_perm = scratch.perm();
    int* in_col = scratch.in_col();
    int* v = scratch.v();
    int* free_ = scratch.free();
    int* collist = scratch.collist();
    int* matches = scratch.matches();
    int* pred = scratch.pred();
    int* d = scratch.d();

    int i, imin, i0, freerow;
    int j, j1, j2 = 0, endofpath = 0, last = 0, min = 0;

    for (j = dim - 1; j >= 0; j--) {
        min = matrix[0 * dim + j];
        imin = 0;
        for (i = 1; i < dim; i++) {
            if (matrix[i * dim + j] < min) {
                min = matrix[i * dim + j];
                imin = i;
            }
        }

        v[j] = min;
        matches[imin]++;
        if (matches[imin] == 1) {
            out_perm[imin] = j;
            in_col[j] = imin;
        } else {
            in_col[j] = -1;
        }
    }

    int num_free = 0;
    for (i = 0; i < dim; i++) {
        if (matches[i] == 0) {
            free_[num_free] = i;
            num_free++;
        } else if (matches[i] == 1) {
            j1 = out_perm[i];
            min = INT_MAX;
            for (j = 0; j < dim; j++) {
                if (j != j1 && matrix[i * dim + j] - v[j] < min) {
                    min = matrix[i * dim + j] - v[j];
                }
            }
            v[j1] -= min;
        }
    }

    for (int loop_cmt = 0; loop_cmt < 2; loop_cmt++) {
        int k = 0;
        int prv_num_free = num_free;
        num_free = 0;
        while (k < prv_num_free) {
            i = free_[k];
            k++;
            int umin = matrix[i * dim + 0] - v[0];
            j1 = 0;
            int usubmin = INT_MAX;

            for (j = 1; j < dim; j++) {
                int h = matrix[i * dim + j] - v[j];

                if (h < usubmin) {
                    if (h >= umin) {
                        usubmin = h;
                        j2 = j;
                    } else {
                        usubmin = umin;
                        umin = h;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }

            i0 = in_col[j1];
            if (umin < usubmin) {
                v[j1] = v[j1] - (usubmin - umin);
            } else if (i0 >= 0) {
                j1 = j2;
                i0 = in_col[j2];
            }

            out_perm[i] = j1;
            in_col[j1] = i;
            if (i0 >= 0) {
                if (umin < usubmin) {
                    k--;
                    free_[k] = i0;
                } else {
                    free_[num_free] = i0;
                    num_free++;
                }
            }
        }
    }

    for (int f = 0; f < num_free; f++) {
        freerow = free_[f];
        for (j = 0; j < dim; j++) {
            d[j] = matrix[freerow * dim + j] - v[j];
            pred[j] = freerow;
            collist[j] = j;
        }

        int low = 0;
        int up = 0;
        int unassigned_found = 0;
        int max_iter = dim * dim + 10;

        while (!unassigned_found && max_iter > 0) {
            max_iter--;
            if (up == low) {
                last = low - 1;
                min = d[collist[up]];
                up++;

                for (int k = up; k < dim; k++) {
                    j = collist[k];
                    int h = d[j];
                    if (h <= min) {
                        if (h < min) {
                            up = low;
                            min = h;
                        }
                        collist[k] = collist[up];
                        collist[up] = j;
                        up++;
                    }
                }

                for (int k = low; k < up; k++) {
                    if (in_col[collist[k]] < 0) {
                        endofpath = collist[k];
                        unassigned_found = 1;
                        break;
                    }
                }
            }

            if (!unassigned_found) {
                j1 = collist[low];
                low++;
                i = in_col[j1];
                int h = matrix[i * dim + j1] - v[j1] - min;

                for (int k = up; k < dim; k++) {
                    j = collist[k];
                    int v2 = matrix[i * dim + j] - v[j] - h;

                    if (v2 < d[j]) {
                        pred[j] = i;

                        if (v2 == min) {
                            if (in_col[j] < 0) {
                                endofpath = j;
                                unassigned_found = 1;
                                break;
                            } else {
                                collist[k] = collist[up];
                                collist[up] = j;
                                up++;
                            }
                        }

                        d[j] = v2;
                    }
                }
            }
        }

        for (int k = 0; k <= last; k++) {
            j1 = collist[k];
            v[j1] += d[j1] - min;
        }

        i = freerow + 1;
        while (i != freerow) {
            i = pred[endofpath];
            in_col[endofpath] = i;
            j1 = endofpath;
            endofpath = out_perm[i];
            out_perm[i] = j1;
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience overload: allocates a JVScratch, runs the core algorithm, and
// returns the permutation as a std::vector (owning, RAII).
// ---------------------------------------------------------------------------
inline std::vector<int> compute_assignment(const int* matrix, int dim) {
    JVScratch scratch(dim);
    compute_assignment(matrix, dim, scratch);
    return std::vector<int>(scratch.perm(), scratch.perm() + dim);
}

#endif  // EVP_FLAS_JUNKER_VOLGENANT_SOLVER_H
