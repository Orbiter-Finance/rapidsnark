#ifndef __GROTH16_CUDA_CU__
#define __GROTH16_CUDA_CU__
#include <iostream>
#include <chrono>
#include <fstream>
#include <gmp.h>
#include <memory>
#include <stdexcept>


#include <nlohmann/json.hpp>

#include <alt_bn128.hpp>
#include "binfile_utils.hpp"
#include "zkey_utils.hpp"
#include "wtns_utils.hpp"
#include "groth16.hpp"

using json = nlohmann::json;

#define G2_DEFINED
#define CURVE_ID 1
#include "appUtils/ntt/ntt.cu"
#include "appUtils/ntt/kernel_ntt.cu"
#include "appUtils/msm/msm.cu"
using namespace curve_config;

// using namespace curve_config;
using namespace ntt;
using namespace AltBn128;
using namespace Groth16;

// Operate on scalars
typedef scalar_t ICICLE_S;
typedef scalar_t ICICLE_E;
using FpMilliseconds = std::chrono::duration<float, std::chrono::milliseconds::period>;
#define START_TIMER(timer) auto timer##_start = std::chrono::high_resolution_clock::now();
#define END_TIMER(timer, msg) printf("%s: %.0f ms\n", msg, FpMilliseconds(std::chrono::high_resolution_clock::now() - timer##_start).count());

static void init_icicle_cuda_ntt_ctx(
    device_context::DeviceContext& ntt_ctx,
    const u_int32_t log_ntt_size
){
    const ICICLE_S basic_root = ICICLE_S::omega(log_ntt_size);
    InitDomain(basic_root, ntt_ctx);
}

// need to call init_ntt_ctx once!
static cudaError_t icicle_cuda_ntt(
    Engine::FrElement* inoutput,
    u_int32_t ntt_size,
    NTTConfig<ICICLE_S> &config,
    NTTDir direction, // NTTDir::kForward for ntt, NTTDir::kInverse for intt
    device_context::DeviceContext ntt_ctx // 
){  
    auto rep_inoutput = reinterpret_cast<ICICLE_E *>(inoutput);

    return NTT<ICICLE_S, ICICLE_E>(
        rep_inoutput, 
        ntt_size, 
        direction, 
        config, 
        rep_inoutput
    );
}

static void bn254_icicle_cuda_g1_msm(
    Engine::G1Point &r,
    Engine::G1PointAffine *bases,
    uint8_t *scalars,
    unsigned int msm_size,
    msm::MSMConfig &config
){
    auto rep_bases = reinterpret_cast<affine_t *>(bases);
    auto rep_scalars = reinterpret_cast<scalar_t *>(scalars);
    projective_t temp;
    
    cudaError_t res = msm::MSM<scalar_t, affine_t, projective_t>(rep_scalars, rep_bases, msm_size, config, &temp);
    if(res!=cudaSuccess){
        std::cerr << res << "\n";
        exit(EXIT_FAILURE);
    }
    affine_t affine_temp = projective_t::to_affine(temp);
    affine_t affine_temp_mont = affine_t::ToMontgomery(affine_temp);

    Engine::G1PointAffine affine_temp2;
    std::memcpy((void*)&affine_temp2, (void*)&affine_temp_mont, sizeof(affine_temp_mont));
    G1.copy(r, affine_temp2);
}

static void bn254_icicle_cuda_g2_msm(
    Engine::G2Point &r,
    Engine::G2PointAffine *bases,
    uint8_t *scalars,
    unsigned int msm_size,
    msm::MSMConfig &config
){
    auto rep_bases = reinterpret_cast<g2_affine_t *>(bases);
    auto rep_scalars = reinterpret_cast<scalar_t *>(scalars);
    g2_projective_t temp;
    cudaError_t res = msm::MSM<scalar_t, g2_affine_t, g2_projective_t>(rep_scalars, rep_bases, msm_size, config, &temp);
    if(res!=cudaSuccess){
        std::cerr << res << "\n";
        exit(EXIT_FAILURE);
    }
    g2_affine_t affine_temp = g2_projective_t::to_affine(temp);
    g2_affine_t affine_temp_mont = g2_affine_t::ToMontgomery(affine_temp);
    Engine::G2PointAffine affine_temp2;
    std::memcpy((void*)&affine_temp2, (void*)&affine_temp_mont, sizeof(affine_temp_mont));
    G2.copy(r, affine_temp2);
}

template <typename Engine>
class Cuda_Prover: public Prover<Engine>{
    using Prover<Engine>::E;
    using Prover<Engine>::pointsA;
    using Prover<Engine>::nVars;
    using Prover<Engine>::pointsB1;
    using Prover<Engine>::pointsB2;
    using Prover<Engine>::pointsC;
    using Prover<Engine>::domainSize;
    using Prover<Engine>::nCoefs;
    using Prover<Engine>::coefs;
    using Prover<Engine>::fft;
    using Prover<Engine>::pointsH;
    using Prover<Engine>::vk_alpha1;
    using Prover<Engine>::vk_delta1;
    using Prover<Engine>::vk_beta2;
    using Prover<Engine>::vk_delta2;
    using Prover<Engine>::vk_beta1;
    using Prover<Engine>::nPublic;

public:  
    Cuda_Prover(Engine &_E, 
            u_int32_t _nVars, 
            u_int32_t _nPublic, 
            u_int32_t _domainSize, 
            u_int64_t _nCoefs, 
            typename Engine::G1PointAffine &_vk_alpha1,
            typename Engine::G1PointAffine &_vk_beta1,
            typename Engine::G2PointAffine &_vk_beta2,
            typename Engine::G1PointAffine &_vk_delta1,
            typename Engine::G2PointAffine &_vk_delta2,
            Coef<Engine> *_coefs, 
            typename Engine::G1PointAffine *_pointsA,
            typename Engine::G1PointAffine *_pointsB1,
            typename Engine::G2PointAffine *_pointsB2,
            typename Engine::G1PointAffine *_pointsC,
            typename Engine::G1PointAffine *_pointsH
            ): Prover<Engine>(
                _E,
                _nVars,
                _nPublic,
                _domainSize,
                _nCoefs, 
                _vk_alpha1,
                _vk_beta1,
                _vk_beta2,
                _vk_delta1,
                _vk_delta2,
                _coefs, 
                _pointsA,
                _pointsB1,
                _pointsB2,
                _pointsC,
                _pointsH

            ){

            }
    std::unique_ptr<Proof<Engine>> prove_cuda(typename Engine::FrElement *wtns);
};


template <typename Engine>
std::unique_ptr<Proof<Engine>> Cuda_Prover<Engine>::prove_cuda(typename Engine::FrElement *wtns) {

#ifdef USE_OPENMP
    START_TIMER(get_msm_config_timer);
    // msm::MSMConfig msm_config = msm::DefaultMSMConfig<affine_t>();
    device_context::DeviceContext msm_ctx = device_context::get_default_device_context();
    msm::MSMConfig msm_config = {
        msm_ctx,   // ctx
        0,     // points_size
        1,     // precompute_factor
        0,     // c
        0,     // bitsize
        10,    // large_bucket_factor
        1,     // batch_size
        false, // are_scalars_on_device
        false, // are_scalars_montgomery_form
        false, // are_points_on_device
        true, // are_points_montgomery_form
        false, // are_results_on_device
        false, // is_big_triangle
        false, // is_async
    };
    END_TIMER(get_msm_config_timer, "get MSM config");
    START_TIMER(multiexp_a_timer);
    LOG_TRACE("Start Multiexp A");
    uint32_t sW = sizeof(wtns[0]);
    typename Engine::G1Point pi_a;
    //E.g1.multiMulByScalar(pi_a, pointsA, (uint8_t *)wtns, sW, nVars);
    bn254_icicle_cuda_g1_msm(pi_a, pointsA, (uint8_t *)wtns, nVars, msm_config);
    std::ostringstream ss2;
    ss2 << "pi_a: " << E.g1.toString(pi_a);
    LOG_DEBUG(ss2);
    END_TIMER(multiexp_a_timer, "Multiexp A");

    START_TIMER(multiexp_b1_timer);
    LOG_TRACE("Start Multiexp B1");
    typename Engine::G1Point pib1;
    //E.g1.multiMulByScalar(pib1, pointsB1, (uint8_t *)wtns, sW, nVars);
    bn254_icicle_cuda_g1_msm(pib1, pointsB1,(uint8_t *)wtns, nVars, msm_config);
    std::ostringstream ss3;
    ss3 << "pib1: " << E.g1.toString(pib1);
    LOG_DEBUG(ss3);
    END_TIMER(multiexp_b1_timer, "Multiexp B1");

    START_TIMER(multiexp_b2_timer);
    LOG_TRACE("Start Multiexp B2");
    typename Engine::G2Point pi_b;
    // E.g2.multiMulByScalar(pi_b, pointsB2, (uint8_t *)wtns, sW, nVars);
    bn254_icicle_cuda_g2_msm(pi_b, pointsB2, (uint8_t *)wtns, nVars, msm_config );
    std::ostringstream ss4;
    ss4 << "pi_b: " << E.g2.toString(pi_b);
    LOG_DEBUG(ss4);
    END_TIMER(multiexp_b2_timer, "Multiexp B2");


    START_TIMER(multiexp_c_timer);
    LOG_TRACE("Start Multiexp C");
    typename Engine::G1Point pi_c;
    //E.g1.multiMulByScalar(pi_c, pointsC, (uint8_t *)((uint64_t)wtns + (nPublic +1)*sW), sW, nVars-nPublic-1);
    bn254_icicle_cuda_g1_msm(pi_c, pointsC, (uint8_t *)((uint64_t)wtns + (nPublic +1)*sW), nVars-nPublic-1, msm_config);
    std::ostringstream ss5;
    ss5 << "pi_c: " << E.g1.toString(pi_c);
    LOG_DEBUG(ss5);
    END_TIMER(multiexp_c_timer, "Multiexp C");
#else
    LOG_TRACE("Start Multiexp A");
    uint32_t sW = sizeof(wtns[0]);
    typename Engine::G1Point pi_a;
    auto pA_future = std::async([&]() {
        E.g1.multiMulByScalar(pi_a, pointsA, (uint8_t *)wtns, sW, nVars);
    });

    LOG_TRACE("Start Multiexp B1");
    typename Engine::G1Point pib1;
    auto pB1_future = std::async([&]() {
        E.g1.multiMulByScalar(pib1, pointsB1, (uint8_t *)wtns, sW, nVars);
    });

    LOG_TRACE("Start Multiexp B2");
    typename Engine::G2Point pi_b;
    auto pB2_future = std::async([&]() {
        E.g2.multiMulByScalar(pi_b, pointsB2, (uint8_t *)wtns, sW, nVars);
    });

    LOG_TRACE("Start Multiexp C");
    typename Engine::G1Point pi_c;
    auto pC_future = std::async([&]() {
        E.g1.multiMulByScalar(pi_c, pointsC, (uint8_t *)((uint64_t)wtns + (nPublic +1)*sW), sW, nVars-nPublic-1);
    });
#endif

    START_TIMER(init_a_b_c_A_timer);
    LOG_TRACE("Start Initializing a b c A");
    auto a = new typename Engine::FrElement[domainSize];
    auto b = new typename Engine::FrElement[domainSize];
    auto c = new typename Engine::FrElement[domainSize];

    #pragma omp parallel for
    for (u_int32_t i=0; i<domainSize; i++) {
        E.fr.copy(a[i], E.fr.zero());
        E.fr.copy(b[i], E.fr.zero());
    }
    END_TIMER(init_a_b_c_A_timer, "Initializing a b c A");

    START_TIMER(processing_coefs_timer);

    LOG_TRACE("Processing coefs");
#ifdef _OPENMP
    #define NLOCKS 1024
    omp_lock_t locks[NLOCKS];
    for (int i=0; i<NLOCKS; i++) omp_init_lock(&locks[i]);
    #pragma omp parallel for 
#endif
    for (u_int64_t i=0; i<nCoefs; i++) {
        typename Engine::FrElement *ab = (coefs[i].m == 0) ? a : b;
        typename Engine::FrElement aux;

        E.fr.mul(
            aux,
            wtns[coefs[i].s],
            coefs[i].coef
        );
#ifdef _OPENMP
        omp_set_lock(&locks[coefs[i].c % NLOCKS]);
#endif
        E.fr.add(
            ab[coefs[i].c],
            ab[coefs[i].c],
            aux
        );
#ifdef _OPENMP
        omp_unset_lock(&locks[coefs[i].c % NLOCKS]);
#endif
    }
#ifdef _OPENMP
    for (int i=0; i<NLOCKS; i++) omp_destroy_lock(&locks[i]);
#endif

    END_TIMER(processing_coefs_timer, "Processing coefs");

    START_TIMER(cal_c_timer);
    LOG_TRACE("Calculating c");
    #pragma omp parallel for
    for (u_int32_t i=0; i<domainSize; i++) {
        E.fr.mul(
            c[i],
            a[i],
            b[i]
        );
    }
    END_TIMER(cal_c_timer, "Calculating c");


    START_TIMER(init_fft_timer);
    LOG_TRACE("Initializing fft");
    u_int32_t domainPower = fft->log2(domainSize);
    auto ctx = device_context::get_default_device_context();
    init_icicle_cuda_ntt_ctx(ctx, domainPower);
    NTTConfig<ICICLE_S> config = DefaultNTTConfig<ICICLE_S>();
    config.ntt_algorithm = NttAlgorithm::MixedRadix;
    config.batch_size = 1;
    END_TIMER(init_fft_timer, "Initializing fft");

    START_TIMER(ifft_a_timer);
    LOG_TRACE("Start iFFT A");
    //fft->ifft(a, domainSize);
    icicle_cuda_ntt(a, domainSize,config, NTTDir::kInverse, ctx);
    END_TIMER(ifft_a_timer, "iFFT A");


    START_TIMER(a_after_ifft_timer);
    LOG_TRACE("a After ifft:");
    LOG_DEBUG(E.fr.toString(a[0]).c_str());
    LOG_DEBUG(E.fr.toString(a[1]).c_str());
    END_TIMER(a_after_ifft_timer, "a After ifft");

    START_TIMER(shift_a_timer);
    LOG_TRACE("Start Shift A");
    #pragma omp parallel for
    for (u_int64_t i=0; i<domainSize; i++) {
        E.fr.mul(a[i], a[i], fft->root(domainPower+1, i));
    }
    END_TIMER(shift_a_timer, "Shift A");

    START_TIMER(a_after_shift_timer);
    LOG_TRACE("a After shift:");
    LOG_DEBUG(E.fr.toString(a[0]).c_str());
    LOG_DEBUG(E.fr.toString(a[1]).c_str());
    END_TIMER(a_after_shift_timer, "a After shift");

    START_TIMER(fft_a_timer);
    LOG_TRACE("Start FFT A");
    //fft->fft(a, domainSize);
    icicle_cuda_ntt(a, domainSize,config, NTTDir::kForward, ctx);
    END_TIMER(fft_a_timer, "FFT A");

    START_TIMER(a_after_fft_timer);
    LOG_TRACE("a After fft:");
    LOG_DEBUG(E.fr.toString(a[0]).c_str());
    LOG_DEBUG(E.fr.toString(a[1]).c_str());
    END_TIMER(a_after_fft_timer, "a After fft");

    START_TIMER(ifft_b_timer);
    LOG_TRACE("Start iFFT B");
    //fft->ifft(b, domainSize);
    icicle_cuda_ntt(b, domainSize,config, NTTDir::kInverse, ctx);
    END_TIMER(ifft_b_timer, "iFFT B");

    START_TIMER(b_after_ifft_timer);
    LOG_TRACE("b After ifft:");
    LOG_DEBUG(E.fr.toString(b[0]).c_str());
    LOG_DEBUG(E.fr.toString(b[1]).c_str());
    END_TIMER(b_after_ifft_timer, "b After ifft");

    START_TIMER(shift_b_timer);
    LOG_TRACE("Start Shift B");
    #pragma omp parallel for
    for (u_int64_t i=0; i<domainSize; i++) {
        E.fr.mul(b[i], b[i], fft->root(domainPower+1, i));
    }
    END_TIMER(shift_b_timer, "Shift B");

    START_TIMER(b_after_shift_timer);
    LOG_TRACE("b After shift:");
    LOG_DEBUG(E.fr.toString(b[0]).c_str());
    LOG_DEBUG(E.fr.toString(b[1]).c_str());
    END_TIMER(b_after_shift_timer, "b After shift");

    START_TIMER(fft_b_timer);
    LOG_TRACE("Start FFT B");
    //fft->fft(b, domainSize);
    icicle_cuda_ntt(b, domainSize,config, NTTDir::kForward, ctx);
    END_TIMER(fft_b_timer, "FFT B");

    START_TIMER(b_after_fft_timer);
    LOG_TRACE("b After fft:");
    LOG_DEBUG(E.fr.toString(b[0]).c_str());
    LOG_DEBUG(E.fr.toString(b[1]).c_str());
    END_TIMER(b_after_fft_timer, "b After fft");

    START_TIMER(ifft_c_timer);
    LOG_TRACE("Start iFFT C");
    //fft->ifft(c, domainSize);
    icicle_cuda_ntt(c, domainSize,config, NTTDir::kInverse, ctx);
    END_TIMER(ifft_c_timer, "iFFT C");

    START_TIMER(c_after_ifft_timer);
    LOG_TRACE("c After ifft:");
    LOG_DEBUG(E.fr.toString(c[0]).c_str());
    LOG_DEBUG(E.fr.toString(c[1]).c_str());
    END_TIMER(c_after_ifft_timer, "c After ifft");

    START_TIMER(shift_c_timer);
    LOG_TRACE("Start Shift C");
    #pragma omp parallel for
    for (u_int64_t i=0; i<domainSize; i++) {
        E.fr.mul(c[i], c[i], fft->root(domainPower+1, i));
    }
    END_TIMER(shift_c_timer, "Shift C");

    START_TIMER(c_after_shift_timer);
    LOG_TRACE("c After shift:");
    LOG_DEBUG(E.fr.toString(c[0]).c_str());
    LOG_DEBUG(E.fr.toString(c[1]).c_str());
    END_TIMER(c_after_shift_timer, "c After shift");

    START_TIMER(fft_c_timer);
    LOG_TRACE("Start FFT C");
    //fft->fft(c, domainSize);
    icicle_cuda_ntt(c, domainSize,config, NTTDir::kForward, ctx);
    END_TIMER(fft_c_timer, "FFT C");

    START_TIMER(c_after_fft_timer);
    LOG_TRACE("c After fft:");
    LOG_DEBUG(E.fr.toString(c[0]).c_str());
    LOG_DEBUG(E.fr.toString(c[1]).c_str());
    END_TIMER(c_after_fft_timer, "c After fft");

    START_TIMER(start_abc_timer);
    LOG_TRACE("Start ABC");
    #pragma omp parallel for
    for (u_int64_t i=0; i<domainSize; i++) {
        E.fr.mul(a[i], a[i], b[i]);
        E.fr.sub(a[i], a[i], c[i]);
        E.fr.fromMontgomery(a[i], a[i]);
    }
    END_TIMER(start_abc_timer, "Start ABC");

    START_TIMER(abc_timer);
    LOG_TRACE("abc:");
    LOG_DEBUG(E.fr.toString(a[0]).c_str());
    LOG_DEBUG(E.fr.toString(a[1]).c_str());
    END_TIMER(abc_timer, "abc");


    delete [] b;
    delete [] c;

    START_TIMER(multiexp_h_timer);
    LOG_TRACE("Start Multiexp H");
    typename Engine::G1Point pih;
    //E.g1.multiMulByScalar(pih, pointsH, (uint8_t *)a, sizeof(a[0]), domainSize);
    bn254_icicle_cuda_g1_msm(pih, pointsH,(uint8_t *)a, domainSize, msm_config);
    std::ostringstream ss1;
    ss1 << "pih: " << E.g1.toString(pih);
    LOG_DEBUG(ss1);
    END_TIMER(multiexp_h_timer, "Multiexp H");


    delete [] a;

    typename Engine::FrElement r;
    typename Engine::FrElement s;
    typename Engine::FrElement rs;

    E.fr.copy(r, E.fr.zero());
    E.fr.copy(s, E.fr.zero());

    randombytes_buf((void *)&(r.v[0]), sizeof(r)-1);
    randombytes_buf((void *)&(s.v[0]), sizeof(s)-1);

#ifndef USE_OPENMP
    pA_future.get();
    pB1_future.get();
    pB2_future.get();
    pC_future.get();
#endif

    typename Engine::G1Point p1;
    typename Engine::G2Point p2;

    E.g1.add(pi_a, pi_a, vk_alpha1);
    E.g1.mulByScalar(p1, vk_delta1, (uint8_t *)&r, sizeof(r));
    E.g1.add(pi_a, pi_a, p1);

    E.g2.add(pi_b, pi_b, vk_beta2);
    E.g2.mulByScalar(p2, vk_delta2, (uint8_t *)&s, sizeof(s));
    E.g2.add(pi_b, pi_b, p2);

    E.g1.add(pib1, pib1, vk_beta1);
    E.g1.mulByScalar(p1, vk_delta1, (uint8_t *)&s, sizeof(s));
    E.g1.add(pib1, pib1, p1);

    E.g1.add(pi_c, pi_c, pih);

    E.g1.mulByScalar(p1, pi_a, (uint8_t *)&s, sizeof(s));
    E.g1.add(pi_c, pi_c, p1);

    E.g1.mulByScalar(p1, pib1, (uint8_t *)&r, sizeof(r));
    E.g1.add(pi_c, pi_c, p1);

    E.fr.mul(rs, r, s);
    E.fr.toMontgomery(rs, rs);

    E.g1.mulByScalar(p1, vk_delta1, (uint8_t *)&rs, sizeof(rs));
    E.g1.sub(pi_c, pi_c, p1);

    Proof<Engine> *p = new Proof<Engine>(Engine::engine);
    E.g1.copy(p->A, pi_a);
    E.g2.copy(p->B, pi_b);
    E.g1.copy(p->C, pi_c);

    return std::unique_ptr<Proof<Engine>>(p);    

}


template <typename Engine>
std::unique_ptr<Cuda_Prover<Engine>> makeCuda_Prover(
    u_int32_t nVars, 
    u_int32_t nPublic, 
    u_int32_t domainSize, 
    u_int64_t nCoeffs, 
    void *vk_alpha1,
    void *vk_beta_1,
    void *vk_beta_2,
    void *vk_delta_1,
    void *vk_delta_2,
    void *coefs, 
    void *pointsA, 
    void *pointsB1, 
    void *pointsB2, 
    void *pointsC, 
    void *pointsH
) {
    Cuda_Prover<Engine> *p = new Cuda_Prover<Engine>(
        Engine::engine, 
        nVars, 
        nPublic, 
        domainSize, 
        nCoeffs, 
        *(typename Engine::G1PointAffine *)vk_alpha1,
        *(typename Engine::G1PointAffine *)vk_beta_1,
        *(typename Engine::G2PointAffine *)vk_beta_2,
        *(typename Engine::G1PointAffine *)vk_delta_1,
        *(typename Engine::G2PointAffine *)vk_delta_2,
        (Coef<Engine> *)((uint64_t)coefs + 4), 
        (typename Engine::G1PointAffine *)pointsA,
        (typename Engine::G1PointAffine *)pointsB1,
        (typename Engine::G2PointAffine *)pointsB2,
        (typename Engine::G1PointAffine *)pointsC,
        (typename Engine::G1PointAffine *)pointsH
    );
    return std::unique_ptr< Cuda_Prover<Engine> >(p);
}

#endif