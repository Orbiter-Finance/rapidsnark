#include "groth16_cuda.cu"


int main(int argc, char **argv){
    std::cout << "Hello ICICLE" <<"\n";

    std::cout << "Hello CMAKE" << std::endl;
    u_int32_t log_ntt_size = 2;
    u_int32_t domainSize = 1 << log_ntt_size;
    FFT<AltBn128::Engine::Fr> fft(domainSize);
    auto a = new AltBn128::Engine::FrElement[domainSize];
    #pragma omp parallel for
    for (u_int32_t i = 0; i < domainSize; i++){
        Fr.fromUI(a[i], i + 1);
    }
    // auto input = reinterpret_cast<E *>(a);

    auto ctx = device_context::get_default_device_context();
    init_icicle_cuda_ntt_ctx(ctx, log_ntt_size);

    // const S basic_root = S::omega(log_ntt_size);
    // InitDomain(basic_root, ctx);

    NTTConfig<ICICLE_S> config = DefaultNTTConfig<ICICLE_S>();
    config.ntt_algorithm = NttAlgorithm::MixedRadix;
    // config.ntt_algorithm = NttAlgorithm::Radix2;
    config.batch_size = 1;

    START_TIMER(MixedRadix);
    // cudaError_t err = NTT<S, E>(input, domainSize, NTTDir::kForward, config, input);
    cudaError_t err = icicle_cuda_ntt(
        a,
        domainSize,
        config,
        NTTDir::kForward,
        ctx
    );

    END_TIMER(MixedRadix, "MixedRadix NTT");

    fft.printVector(&a[0], domainSize);
    delete [] a;

    return 0;
}