#pragma once
#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <cassert>
#include <cstring>
#include <cmath>
#include <random>
#include <sstream>
#include <iomanip>
#include <unordered_set>

// ── CUDA backend (opt-in) ─────────────────────────────────────────────────────
// Define STAKML_CUDA at compile time to swap the three matmul implementations
// from the OpenMP blocked loop to cuBLAS calls.
// Everything else (autograd, layers, loss, optimizer) is completely untouched.
#ifdef STAKML_CUDA
#include "cuda_matmul.cuh"
#endif

namespace stakml {

// ─────────────────────────────────────────────────────────────────────────────
// CONCEPT: What IS a Tensor?
//
// A Tensor is just a multi-dimensional array with some extra bookkeeping.
// A scalar is a 0-D tensor. A vector is 1-D. A matrix is 2-D. An image
// batch is 4-D (batch, channels, height, width). Same idea throughout.
//
// The raw data lives in ONE flat 1-D array in memory, always. The shape
// and strides tell us how to interpret that flat array as N dimensions.
//
// Example: a 2×3 matrix
//   shape  = [2, 3]          → 2 rows, 3 cols
//   strides= [3, 1]          → to move one row, jump 3 elements;
//                               to move one col, jump 1 element
//   data   = [a,b,c,d,e,f]  → flat in memory (row-major)
//
//   element [i][j] = data[ i*strides[0] + j*strides[1] ]
//                  = data[ i*3 + j ]
//
// WHY STRIDES?
//   Reshape, transpose, and slicing become ZERO-COPY — you just change
//   the strides/shape metadata, not the underlying data. PyTorch does
//   exactly this. Your FPGA work has the same idea: you lay data out flat
//   in BRAM and use address arithmetic to index it.
// ─────────────────────────────────────────────────────────────────────────────

class Tensor {
public:
    // ── Data storage ─────────────────────────────────────────────────────────
    //
    // shared_ptr<vector<float>>: multiple Tensors can SHARE the same buffer.
    // When you do a reshape or transpose, the new Tensor points at the same
    // memory — no copy. The last Tensor to die frees the buffer.
    // This is exactly how PyTorch's storage system works.
    //
    std::string op_name_;
    std::vector<std::shared_ptr<Tensor>> inputs_;
    std::shared_ptr<std::vector<float>> data_;
    std::vector<size_t> shape_;    // e.g. {2, 3} for a 2×3 matrix
    std::vector<size_t> strides_;  // e.g. {3, 1} for row-major 2×3
    size_t offset_;                // where in data_ this tensor starts (for views)

    // ── For autograd (Week 3) ─────────────────────────────────────────────────
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;           // accumulated gradient
    std::function<void()> backward_fn_;     // how to propagate gradient back

    // ─────────────────────────────────────────────────────────────────────────
    // Constructors
    // ─────────────────────────────────────────────────────────────────────────

    // Default: empty tensor
    Tensor() : offset_(0) {}

    // Construct from shape — allocates memory, fills with zeros
    explicit Tensor(const std::vector<size_t>& shape)
        : shape_(shape), offset_(0)
    {
        strides_ = compute_strides(shape);
        size_t total = num_elements();
        data_ = std::make_shared<std::vector<float>>(total, 0.0f);
    }

    // Construct from shape + fill value
    Tensor(const std::vector<size_t>& shape, float fill_value)
        : shape_(shape), offset_(0)
    {
        strides_ = compute_strides(shape);
        size_t total = num_elements();
        data_ = std::make_shared<std::vector<float>>(total, fill_value);
    }

    // Construct from raw data (copies in)
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data)
        : shape_(shape), offset_(0)
    {
        strides_ = compute_strides(shape);
        if (data.size() != num_elements())
            throw std::runtime_error("Data size doesn't match shape");
        data_ = std::make_shared<std::vector<float>>(data);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Shape & Size utilities
    // ─────────────────────────────────────────────────────────────────────────

    size_t ndim() const { return shape_.size(); }

    size_t num_elements() const {
        if (shape_.empty()) return 1; // scalar
        return std::accumulate(shape_.begin(), shape_.end(),
                               size_t(1), std::multiplies<size_t>());
    }

    // Returns size along a given dimension
    size_t size(size_t dim) const {
        if (dim >= shape_.size())
            throw std::out_of_range("Dimension out of range");
        return shape_[dim];
    }

    bool is_scalar() const { return shape_.empty(); }

    // ─────────────────────────────────────────────────────────────────────────
    // Element access
    //
    // CONCEPT: element(i, j, k, ...) = data_[ offset_
    //                                        + i * strides_[0]
    //                                        + j * strides_[1]
    //                                        + k * strides_[2] + ... ]
    //
    // The offset_ handles the case where this Tensor is a VIEW into
    // a larger buffer (e.g. a row-slice of a matrix).
    // ─────────────────────────────────────────────────────────────────────────

    float& at(const std::vector<size_t>& idx) {
        return (*data_)[flat_index(idx)];
    }

    const float& at(const std::vector<size_t>& idx) const {
        return (*data_)[flat_index(idx)];
    }

    // Raw flat access (useful for loops over all elements)
    float* raw_ptr() { return data_->data() + offset_; }
    const float* raw_ptr() const { return data_->data() + offset_; }

    // ─────────────────────────────────────────────────────────────────────────
    // Reshape  — ZERO COPY
    //
    // CONCEPT: reshape just reinterprets the same flat data with a new shape.
    // Works only if the tensor is contiguous (strides are standard row-major).
    // Exactly like NumPy's reshape or PyTorch's view().
    // ─────────────────────────────────────────────────────────────────────────

    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(),
                                           size_t(1), std::multiplies<size_t>());
        if (new_total != num_elements())
            throw std::runtime_error("reshape: element count must not change");

        if (!is_contiguous())
            throw std::runtime_error("reshape: tensor must be contiguous (call contiguous() first)");

        Tensor result;
        result.data_    = data_;          // shared buffer — no copy!
        result.shape_   = new_shape;
        result.strides_ = compute_strides(new_shape);
        result.offset_  = offset_;
        result.requires_grad_ = requires_grad_;
        return result;
    }

    // Flatten to 1-D
    Tensor flatten() const {
        return reshape({num_elements()});
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Transpose (2-D only for now)
    //
    // CONCEPT: swap shape and strides. The data doesn't move.
    //   Original 2×3: shape={2,3}, strides={3,1}
    //   Transposed 3×2: shape={3,2}, strides={1,3}
    //   element [i][j] of transposed = element [j][i] of original ✓
    // ─────────────────────────────────────────────────────────────────────────

    Tensor transpose() const {
        if (ndim() != 2)
            throw std::runtime_error("transpose: only 2-D tensors supported");

        Tensor result;
        result.data_    = data_;
        result.shape_   = {shape_[1], shape_[0]};
        result.strides_ = {strides_[1], strides_[0]};
        result.offset_  = offset_;
        result.requires_grad_ = requires_grad_;
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Contiguous check & copy
    //
    // A tensor is contiguous if its strides match what compute_strides()
    // would give for its shape — i.e., row-major with no gaps.
    // After a transpose, the tensor is NOT contiguous.
    // contiguous() returns a fresh tensor where data IS laid out row-major.
    // ─────────────────────────────────────────────────────────────────────────

    bool is_contiguous() const {
        return strides_ == compute_strides(shape_);
    }

    Tensor contiguous() const {
        if (is_contiguous()) return *this;
        // Copy elements out in logical order
        Tensor result(shape_);
        size_t n = num_elements();
        for (size_t i = 0; i < n; ++i) {
            auto idx = unravel_index(i, shape_);
            result.raw_ptr()[i] = at(idx);
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Element-wise ops (the simple ones)
    // ─────────────────────────────────────────────────────────────────────────

    Tensor operator+(const Tensor& other) const {
        check_same_shape(other, "+");
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = raw_ptr()[i] + other.raw_ptr()[i];
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        check_same_shape(other, "-");
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = raw_ptr()[i] - other.raw_ptr()[i];
        return result;
    }

    Tensor operator*(const Tensor& other) const {  // element-wise multiply
        check_same_shape(other, "*");
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = raw_ptr()[i] * other.raw_ptr()[i];
        return result;
    }

    Tensor operator*(float scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = raw_ptr()[i] * scalar;
        return result;
    }

    Tensor operator/(float scalar) const {
        return (*this) * (1.0f / scalar);
    }

    Tensor operator-() const {
        return (*this) * -1.0f;
    }

    // In-place add (for gradient accumulation later)
    Tensor& operator+=(const Tensor& other) {
        check_same_shape(other, "+=");
        for (size_t i = 0; i < num_elements(); ++i)
            raw_ptr()[i] += other.raw_ptr()[i];
        return *this;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Matrix multiply  (2-D only for now)
    //
    // CONCEPT: C[i][j] = sum_k( A[i][k] * B[k][j] )
    // Shape rule: (M×K) @ (K×N) → (M×N)
    //
    // Two backends selected at compile time via #ifdef STAKML_CUDA:
    //   CPU: blocked OpenMP loop (your existing code, untouched)
    //   GPU: cuBLAS SGEMM via cuda_matmul.cuh
    // ─────────────────────────────────────────────────────────────────────────

    Tensor matmul(const Tensor& other) const {
        if (ndim() != 2 || other.ndim() != 2)
            throw std::runtime_error("matmul: both tensors must be 2-D");
        if (shape_[1] != other.shape_[0])
            throw std::runtime_error("matmul: inner dimensions must match");

        size_t M = shape_[0], K = shape_[1], N = other.shape_[1];
        Tensor result({M, N}, 0.0f);

#ifdef STAKML_CUDA
        cublas_matmul(raw_ptr(), other.raw_ptr(), result.raw_ptr(), M, K, N);
#else
        const float* A = raw_ptr();
        const float* B = other.raw_ptr();
        float* C = result.raw_ptr();

        const size_t BLOCK = 64;

        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t i = 0; i < M; i += BLOCK) {
            for (size_t j = 0; j < N; j += BLOCK) {
                for (size_t k = 0; k < K; k += BLOCK) {
                    size_t i_end = std::min(i + BLOCK, M);
                    size_t j_end = std::min(j + BLOCK, N);
                    size_t k_end = std::min(k + BLOCK, K);
                    for (size_t ii = i; ii < i_end; ++ii) {
                        for (size_t kk = k; kk < k_end; ++kk) {
                            float a_ik = A[ii*K + kk];
                            #pragma omp simd
                            for (size_t jj = j; jj < j_end; ++jj) {
                                C[ii*N + jj] += a_ik * B[kk*N + jj];
                            }
                        }
                    }
                }
            }
        }
#endif
        return result;
    }

    // ── Transposed matmul helpers (used in autograd) ──────────────────────────
    //
    // matmul_A_BT: this @ B.T
    //   this: {M,K},  B: {N,K}  →  result: {M,N}
    //
    // Used in backward: dA = dC @ W.T   where dC:{batch,out}, W:{in,out}
    //
    Tensor matmul_A_BT(const Tensor& B) const {
        if (ndim() != 2 || B.ndim() != 2)
            throw std::runtime_error("matmul_A_BT: both tensors must be 2-D");
        if (shape_[1] != B.shape_[1])
            throw std::runtime_error("matmul_A_BT: inner dims must match (K)");

        size_t M = shape_[0], K = shape_[1], N = B.shape_[0];
        Tensor result({M, N}, 0.0f);

#ifdef STAKML_CUDA
        cublas_matmul_A_BT(raw_ptr(), B.raw_ptr(), result.raw_ptr(), M, K, N);
#else
        const float* a = raw_ptr();
        const float* b = B.raw_ptr();
        float* c = result.raw_ptr();

        const size_t BLOCK = 64;

        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t i = 0; i < M; i += BLOCK) {
            for (size_t j = 0; j < N; j += BLOCK) {
                for (size_t k = 0; k < K; k += BLOCK) {
                    size_t i_end = std::min(i + BLOCK, M);
                    size_t j_end = std::min(j + BLOCK, N);
                    size_t k_end = std::min(k + BLOCK, K);
                    for (size_t ii = i; ii < i_end; ++ii) {
                        for (size_t kk = k; kk < k_end; ++kk) {
                            float a_ik = a[ii*K + kk];
                            #pragma omp simd
                            for (size_t jj = j; jj < j_end; ++jj) {
                                c[ii*N + jj] += a_ik * b[jj*K + kk];
                            }
                        }
                    }
                }
            }
        }
#endif
        return result;
    }

    // matmul_AT_B: this.T @ B
    //   this: {M,K},  B: {M,N}  →  result: {K,N}
    //
    // Used in backward: dW = X.T @ dC   where X:{batch,in}, dC:{batch,out}
    //
    Tensor matmul_AT_B(const Tensor& B) const {
        if (ndim() != 2 || B.ndim() != 2)
            throw std::runtime_error("matmul_AT_B: both tensors must be 2-D");
        if (shape_[0] != B.shape_[0])
            throw std::runtime_error("matmul_AT_B: outer dimensions must match (M)");

        size_t M = shape_[0], K = shape_[1], N = B.shape_[1];
        Tensor result({K, N}, 0.0f);

#ifdef STAKML_CUDA
        cublas_matmul_AT_B(raw_ptr(), B.raw_ptr(), result.raw_ptr(), M, K, N);
#else
        const float* a = raw_ptr();
        const float* b = B.raw_ptr();
        float*       c = result.raw_ptr();

        const size_t BLOCK = 64;

        #pragma omp parallel for schedule(static) collapse(2)
        for (size_t k = 0; k < K; k += BLOCK) {
            for (size_t n = 0; n < N; n += BLOCK) {
                for (size_t m = 0; m < M; m += BLOCK) {
                    size_t k_end = std::min(k + BLOCK, K);
                    size_t n_end = std::min(n + BLOCK, N);
                    size_t m_end = std::min(m + BLOCK, M);
                    for (size_t mm = m; mm < m_end; ++mm) {
                        for (size_t kk = k; kk < k_end; ++kk) {
                            float a_mk = a[mm*K + kk];
                            #pragma omp simd
                            for (size_t nn = n; nn < n_end; ++nn) {
                                c[kk*N + nn] += a_mk * b[mm*N + nn];
                            }
                        }
                    }
                }
            }
        }
#endif
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reductions
    // ─────────────────────────────────────────────────────────────────────────

    float sum_all() const {
        float s = 0.0f;
        for (size_t i = 0; i < num_elements(); ++i)
            s += raw_ptr()[i];
        return s;
    }

    float mean_all() const {
        return sum_all() / static_cast<float>(num_elements());
    }

    float max_all() const {
        float m = raw_ptr()[0];
        for (size_t i = 1; i < num_elements(); ++i)
            m = std::max(m, raw_ptr()[i]);
        return m;
    }

    // Sum along a single axis → drops that dimension
    Tensor sum(size_t axis) const {
        if (axis >= ndim())
            throw std::runtime_error("sum: axis out of range");

        // ── Fast path: 2-D contiguous (the only case used in practice) ──────────
        if (ndim() == 2 && is_contiguous()) {
            size_t rows = shape_[0], cols = shape_[1];
            const float* src = raw_ptr();

            if (axis == 0) {
                Tensor result({cols}, 0.0f);
                float* dst = result.raw_ptr();
                for (size_t i = 0; i < rows; ++i)
                    for (size_t j = 0; j < cols; ++j)
                        dst[j] += src[i*cols + j];
                return result;
            } else {
                Tensor result({rows}, 0.0f);
                float* dst = result.raw_ptr();
                for (size_t i = 0; i < rows; ++i) {
                    float acc = 0.0f;
                    for (size_t j = 0; j < cols; ++j)
                        acc += src[i*cols + j];
                    dst[i] = acc;
                }
                return result;
            }
        }

        // ── General path: N-D ────────────────────────────────────────────────────
        std::vector<size_t> out_shape;
        for (size_t i = 0; i < ndim(); ++i)
            if (i != axis) out_shape.push_back(shape_[i]);
        if (out_shape.empty()) out_shape = {1};

        Tensor result(out_shape, 0.0f);
        size_t n = num_elements();
        for (size_t flat = 0; flat < n; ++flat) {
            auto idx = unravel_index(flat, shape_);
            std::vector<size_t> out_idx;
            for (size_t i = 0; i < ndim(); ++i)
                if (i != axis) out_idx.push_back(idx[i]);
            if (out_idx.empty()) out_idx = {0};
            result.at(out_idx) += at(idx);
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Activation functions  (element-wise)
    // ─────────────────────────────────────────────────────────────────────────

    Tensor relu() const {
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i){
            float v = raw_ptr()[i];
            result.raw_ptr()[i] = v > 0.0f ? v : 0.0f;
        }
        return result;
    }

    Tensor sigmoid() const {
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = 1.0f / (1.0f + std::exp(-raw_ptr()[i]));
        return result;
    }

    Tensor tanh_act() const {
        Tensor result(shape_);
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = std::tanh(raw_ptr()[i]);
        return result;
    }

    // Softmax along last dimension
    Tensor softmax() const {
        if (ndim() != 2)
            throw std::runtime_error("softmax: only 2-D supported for now");
        Tensor result(shape_);
        size_t rows = shape_[0], cols = shape_[1];

        const float* src = raw_ptr();
        float*       dst = result.raw_ptr();

        for (size_t i = 0; i < rows; ++i) {
            const float* row_src = src + i*cols;
            float*       row_dst = dst + i*cols;

            float max_val = row_src[0];
            for (size_t j = 1; j < cols; ++j)
                max_val = std::max(max_val, row_src[j]);

            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                row_dst[j] = std::exp(row_src[j] - max_val);
                sum_exp += row_dst[j];
            }

            float inv_sum = 1.0f / sum_exp;
            for (size_t j = 0; j < cols; ++j)
                row_dst[j] *= inv_sum;
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Initializers (static factory methods)
    // ─────────────────────────────────────────────────────────────────────────

    static Tensor zeros(const std::vector<size_t>& shape) {
        return Tensor(shape, 0.0f);
    }

    static Tensor ones(const std::vector<size_t>& shape) {
        return Tensor(shape, 1.0f);
    }

    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float std = 1.0f) {
        Tensor t(shape);
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(mean, std);
        for (size_t i = 0; i < t.num_elements(); ++i)
            t.raw_ptr()[i] = dist(rng);
        return t;
    }

    // Xavier / Glorot init — good default for weights in MLPs
    // std = sqrt(2 / (fan_in + fan_out))
    static Tensor xavier(const std::vector<size_t>& shape) {
        if (shape.size() < 2) throw std::runtime_error("xavier: need at least 2 dims");
        size_t receptive = 1;
        for (size_t i = 2; i < shape.size(); ++i) receptive *= shape[i];
        float fan_in  = static_cast<float>(shape[1] * receptive);
        float fan_out = static_cast<float>(shape[0] * receptive);
        float std = std::sqrt(2.0f / (fan_in + fan_out));
        return randn(shape, 0.0f, std);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Printing
    // ─────────────────────────────────────────────────────────────────────────

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " = ";
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i)
            std::cout << shape_[i] << (i+1 < shape_.size() ? "," : "");
        std::cout << "], data=";

        if (ndim() == 0) {
            std::cout << (*data_)[offset_] << ")\n";
            return;
        }
        if (ndim() == 1) {
            std::cout << "[";
            for (size_t i = 0; i < shape_[0]; ++i)
                std::cout << std::fixed << std::setprecision(4) << at({i})
                          << (i+1 < shape_[0] ? ", " : "");
            std::cout << "])\n";
            return;
        }
        if (ndim() == 2) {
            std::cout << "\n";
            for (size_t i = 0; i < shape_[0]; ++i) {
                std::cout << "  [";
                for (size_t j = 0; j < shape_[1]; ++j)
                    std::cout << std::fixed << std::setprecision(4) << at({i,j})
                              << (j+1 < shape_[1] ? ", " : "");
                std::cout << "]\n";
            }
            std::cout << ")\n";
            return;
        }
        std::cout << "[...] (ndim=" << ndim() << ", elems=" << num_elements() << "))\n";
    }

    std::string shape_str() const {
        std::ostringstream ss;
        ss << "(";
        for (size_t i = 0; i < shape_.size(); ++i)
            ss << shape_[i] << (i+1 < shape_.size() ? ", " : "");
        ss << ")";
        return ss.str();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Gradient helpers
    // ─────────────────────────────────────────────────────────────────────────

    void zero_grad() {
        if (grad_) {
            for (size_t i = 0; i < grad_->num_elements(); ++i)
                grad_->raw_ptr()[i] = 0.0f;
        }
    }

    Tensor& grad() {
        if (!grad_) grad_ = std::make_shared<Tensor>(shape_, 0.0f);
        return *grad_;
    }

    void backward() {
        bool needs_seed = true;
        if (grad_) {
            const float* gp = grad_->raw_ptr();
            size_t n = grad_->num_elements();
            for (size_t i = 0; i < n; ++i) {
                if (gp[i] != 0.0f) { needs_seed = false; break; }
            }
        }
        if (needs_seed)
            grad() = Tensor(shape_, 1.0f);

        std::vector<Tensor*> topo;
        std::unordered_set<Tensor*> visited;

        std::function<void(Tensor*)> build_topo = [&](Tensor* t) {
            if (visited.count(t)) return;
            visited.insert(t);
            for (auto& inp : t->inputs_)
                build_topo(inp.get());
            topo.push_back(t);
        };
        build_topo(this);

        std::reverse(topo.begin(), topo.end());

        for (Tensor* t : topo)
            if (t->backward_fn_) t->backward_fn_();
    }

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        if (shape.empty()) return {};
        std::vector<size_t> strides(shape.size());
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            strides[i] = strides[i+1] * shape[i+1];
        return strides;
    }

    size_t flat_index(const std::vector<size_t>& idx) const {
        if (idx.size() != shape_.size())
            throw std::runtime_error("Index dimensionality mismatch");
        size_t flat = offset_;
        for (size_t i = 0; i < idx.size(); ++i) {
            if (idx[i] >= shape_[i])
                throw std::out_of_range("Index out of bounds");
            flat += idx[i] * strides_[i];
        }
        return flat;
    }

    static std::vector<size_t> unravel_index(size_t flat, const std::vector<size_t>& shape) {
        std::vector<size_t> idx(shape.size());
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            idx[i] = flat % shape[i];
            flat /= shape[i];
        }
        return idx;
    }

    void check_same_shape(const Tensor& other, const std::string& op) const {
        if (shape_ != other.shape_)
            throw std::runtime_error("Shape mismatch in op '" + op + "': "
                + shape_str() + " vs " + other.shape_str());
    }
};

// Free function: scalar * tensor
inline Tensor operator*(float scalar, const Tensor& t) { return t * scalar; }

} // namespace stakml