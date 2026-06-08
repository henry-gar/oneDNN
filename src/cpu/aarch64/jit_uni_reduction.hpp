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

#ifndef CPU_AARCH64_JIT_UNI_REDUCTION_HPP
#define CPU_AARCH64_JIT_UNI_REDUCTION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/cpu_reduction_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

struct jit_reduction_conf_t {
    cpu_isa_t isa = isa_undef;
    alg_kind_t alg = alg_kind::undef;
    data_type_t src_type = data_type::undef;
    data_type_t dst_type = data_type::undef;
    dim_t idle_size = 0;
    dim_t reduce_size = 0;
    size_t src_dt_size = 0;
    size_t dst_dt_size = 0;
    size_t simd_w = 0;
    size_t reduce_tail = 0;
};

struct jit_reduction_args_t {
    const uint8_t *src = nullptr;
    uint8_t *dst = nullptr;
};

struct jit_uni_reduction_kernel_t;

struct jit_uni_reduction_t : public primitive_t {
    struct pd_t : public cpu_reduction_pd_t {
        using cpu_reduction_pd_t::cpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("jit:sve", jit_uni_reduction_t);

        status_t init(engine_t *engine);

        const jit_reduction_conf_t &conf() const { return conf_; }

    private:
        jit_reduction_conf_t conf_;
    };

    jit_uni_reduction_t(const pd_t *apd);
    ~jit_uni_reduction_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_uni_reduction_kernel_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
