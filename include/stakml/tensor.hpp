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
    // This naive O(M*K*N) triple loop is intentionally simple.
    // You'll feel the pain on large matrices — that's the lesson.
    // We'll plug in BLAS later if needed.
    // ─────────────────────────────────────────────────────────────────────────

    Tensor matmul(const Tensor& other) const {
        if (ndim() != 2 || other.ndim() != 2)
            throw std::runtime_error("matmul: both tensors must be 2-D");
        if (shape_[1] != other.shape_[0])
            throw std::runtime_error("matmul: inner dimensions must match");

        size_t M = shape_[0], K = shape_[1], N = other.shape_[1];
        Tensor result({M, N}, 0.0f);

        for (size_t i = 0; i < M; ++i)
            for (size_t k = 0; k < K; ++k)       // k-loop outside j: cache-friendly
                for (size_t j = 0; j < N; ++j)
                    result.at({i,j}) += at({i,k}) * other.at({k,j});

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

        std::vector<size_t> out_shape;
        for (size_t i = 0; i < ndim(); ++i)
            if (i != axis) out_shape.push_back(shape_[i]);
        if (out_shape.empty()) out_shape = {1};

        Tensor result(out_shape, 0.0f);
        // iterate over all elements of this tensor
        size_t n = num_elements();
        for (size_t flat = 0; flat < n; ++flat) {
            auto idx = unravel_index(flat, shape_);
            // build output index by dropping the axis dimension
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
        for (size_t i = 0; i < num_elements(); ++i)
            result.raw_ptr()[i] = std::max(0.0f, raw_ptr()[i]);
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
        for (size_t i = 0; i < rows; ++i) {
            // subtract max for numerical stability
            float max_val = at({i, 0});
            for (size_t j = 1; j < cols; ++j)
                max_val = std::max(max_val, at({i, j}));
            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                result.at({i,j}) = std::exp(at({i,j}) - max_val);
                sum_exp += result.at({i,j});
            }
            for (size_t j = 0; j < cols; ++j)
                result.at({i,j}) /= sum_exp;
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
        float fan_in  = static_cast<float>(shape[0]);
        float fan_out = static_cast<float>(shape[1]);
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
        // Higher dims: just show flat
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
    // Gradient helpers (stubs for Week 3)
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

private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────

    // Compute row-major strides for a given shape
    // e.g. shape {2,3,4} → strides {12, 4, 1}
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        if (shape.empty()) return {};
        std::vector<size_t> strides(shape.size());
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
            strides[i] = strides[i+1] * shape[i+1];
        return strides;
    }

    // Convert multi-dim index → flat index in data_
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

    // Convert flat index → multi-dim index (inverse of flat_index for contiguous)
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
