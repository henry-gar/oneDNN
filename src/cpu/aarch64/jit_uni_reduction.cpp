/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_uni_reduction.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

#define GET_OFF(field) offsetof(jit_reduction_args_t, field)

struct jit_uni_reduction_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_reduction_kernel)

    explicit jit_uni_reduction_kernel_t(const jit_reduction_conf_t &conf)
        : conf_(conf) {}

    void operator()(const jit_reduction_args_t *args) const {
        jit_generator_t::operator()(args);
    }

private:
    void load_params();
    void init_acc();
    void reduce_full_vectors();
    void reduce_tail();
    void horizontal_finalize();
    void store_result();
    void generate() override;

    const jit_reduction_conf_t conf_;

    const XReg reg_param_ = abi_param1;
    const XReg reg_src_ = x9;
    const XReg reg_dst_ = x10;
    const XReg reg_work_ = x11;

    const ZReg z_acc_ = z0;
    const ZReg z_tmp_ = z1;
    const PReg p_tail_ = p1;
};

void jit_uni_reduction_kernel_t::load_params() {
    ldr(reg_src_, ptr(reg_param_, static_cast<uint32_t>(GET_OFF(src))));
    ldr(reg_dst_, ptr(reg_param_, static_cast<uint32_t>(GET_OFF(dst))));
    mov_imm(reg_work_, static_cast<uint64_t>(conf_.reduce_size / conf_.simd_w));
}

void jit_uni_reduction_kernel_t::init_acc() {
    eor(z_acc_.d, z_acc_.d, z_acc_.d);
}

void jit_uni_reduction_kernel_t::reduce_full_vectors() {
    Label loop, loop_end;

    L(loop);
    cmp(reg_work_, 0);
    b(EQ, loop_end);

    ld1w(z_tmp_.s, P_ALL_ONE / T_z, ptr(reg_src_));
    fadd(z_acc_.s, z_acc_.s, z_tmp_.s);
    add_imm(reg_src_, reg_src_, conf_.simd_w * sizeof(float), X_TMP_0);
    sub_imm(reg_work_, reg_work_, 1, X_TMP_0);
    b(loop);

    L(loop_end);
}

void jit_uni_reduction_kernel_t::reduce_tail() {
    if (conf_.reduce_tail == 0) return;

    eor(z_tmp_.d, z_tmp_.d, z_tmp_.d);
    ld1w(z_tmp_.s, p_tail_ / T_z, ptr(reg_src_));
    fadd(z_acc_.s, z_acc_.s, z_tmp_.s);
}

void jit_uni_reduction_kernel_t::horizontal_finalize() {
    faddv(SReg(z_acc_.getIdx()), P_ALL_ONE, z_acc_.s);

    if (conf_.alg == alg_kind::reduction_mean) {
        mov_imm(W_TMP_0, float2int(static_cast<float>(conf_.reduce_size)));
        dup(z_tmp_.s, W_TMP_0);
        fdiv(z_acc_.s, P_ALL_ONE, z_tmp_.s);
    }
}

void jit_uni_reduction_kernel_t::store_result() {
    str(SReg(z_acc_.getIdx()), ptr(reg_dst_));
}

void jit_uni_reduction_kernel_t::generate() {
    preamble();

    if (conf_.reduce_tail != 0)
        set_preg(p_tail_.s, conf_.reduce_tail, X_TMP_0, X_TMP_1);

    load_params();
    init_acc();
    reduce_full_vectors();
    reduce_tail();
    horizontal_finalize();
    store_result();

    postamble();
}

status_t jit_uni_reduction_t::pd_t::init(engine_t *engine) {
    using namespace alg_kind;
    using namespace data_type;
    using namespace format_tag;

    conf_ = jit_reduction_conf_t();
    conf_.isa = sve;
    conf_.alg = desc()->alg_kind;
    conf_.src_type = src_md()->data_type;
    conf_.dst_type = dst_md()->data_type;
    conf_.src_dt_size = types::data_type_size(conf_.src_type);
    conf_.dst_dt_size = types::data_type_size(conf_.dst_type);

    VDISPATCH_REDUCTION(mayiuse(sve_128), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_REDUCTION(conf_.src_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REDUCTION(conf_.dst_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REDUCTION(
            set_default_params() == status::success, VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_REDUCTION(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_REDUCTION(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_REDUCTION(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const format_tag_t src_tag = memory_desc_matches_one_of_tag(
            *src_md(), x, nc, ncw, nchw, ncdhw);
    const format_tag_t dst_tag = memory_desc_matches_one_of_tag(
            *dst_md(), x, nc, ncw, nchw, ncdhw);
    VDISPATCH_REDUCTION(src_tag != format_tag::undef && src_tag == dst_tag,
            VERBOSE_UNSUPPORTED_TAG);

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    const int ndims = src_mdw.ndims();
    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();

    int num_reduced_dims = 0;
    conf_.idle_size = dst_mdw.nelems();
    conf_.reduce_size = 1;
    for (int d = ndims - 1; d >= 0; --d) {
        if (src_dims[d] != dst_dims[d]) {
            ++num_reduced_dims;
            conf_.reduce_size *= src_dims[d];
        } else {
            break;
        }
    }

    VDISPATCH_REDUCTION(
            num_reduced_dims != 0, "dimensionality reduction not possible");

    for (int d = 0; d < ndims - num_reduced_dims; ++d) {
        VDISPATCH_REDUCTION(
                src_dims[d] == dst_dims[d], "non-suffix reduction");
    }

    VDISPATCH_REDUCTION(conf_.idle_size > 0, VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_REDUCTION(conf_.reduce_size > 0, VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_REDUCTION(utils::one_of(conf_.alg, reduction_sum, reduction_mean),
            VERBOSE_BAD_ALGORITHM);

    conf_.simd_w = simd_elems(f32, conf_.isa);
    VDISPATCH_REDUCTION(conf_.simd_w > 0 && conf_.simd_w <= 64,
            VERBOSE_UNSUPPORTED_ISA);
    conf_.reduce_tail = conf_.reduce_size % conf_.simd_w;

    return status::success;
}

jit_uni_reduction_t::jit_uni_reduction_t(const pd_t *apd) : primitive_t(apd) {}

jit_uni_reduction_t::~jit_uni_reduction_t() = default;

status_t jit_uni_reduction_t::init(engine_t *engine) {
    kernel_ = utils::make_unique<jit_uni_reduction_kernel_t>(pd()->conf());
    if (!kernel_) return status::out_of_memory;
    return kernel_->create_kernel();
}

status_t jit_uni_reduction_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);

    const auto &conf = pd()->conf();
    const dim_t idle_size = conf.idle_size;
    const dim_t reduce_size = conf.reduce_size;
    const size_t src_dt_size = conf.src_dt_size;
    const size_t dst_dt_size = conf.dst_dt_size;

    parallel_nd(idle_size, [= COMPAT_THIS_CAPTURE](dim_t i) {
        jit_reduction_args_t args;
        args.src = src + i * reduce_size * src_dt_size;
        args.dst = dst + i * dst_dt_size;
        (*kernel_)(&args);
    });

    return status::success;
}

#undef GET_OFF

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
