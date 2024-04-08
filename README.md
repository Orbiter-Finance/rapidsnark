# rapidsnark-gpu

This is a CUDA-accelerated version of [rapidsnark](https://github.com/iden3/rapidsnark.git).

We provide a prover_cuda program that uses Ingonyama's [ICICLE](https://github.com/ingonyama-zk/icicle.git) GPU library for NTT/MSM to build proof, current only supporting [standalone mode](https://github.com/iden3/rapidsnark?tab=readme-ov-file#compile-prover-for-x86_64-host-machine).This accelerated implementation has notably reduced the building proof's time from 41.757 seconds to 8.443 seconds.

# Prerequistes

- CUDA Toolkit version, CMake version and GCC version etc. (please see [ICICLE](https://github.com/ingonyama-zk/icicle?tab=readme-ov-file#prerequisites))
- Ubuntu 22.04
- x86_64 host machine

# Dependencies

```bash
sudo apt-get install build-essential libgmp-dev libsodium-dev nasm curl m4
```

# Compile prover_cuda

```bash
git submodule init
git submodule update
./build_gmp.sh host
mkdir build_prover && cd build_prover
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../package
make -j$(nproc) && make install
```

# Building proof

You can replace rapidsnark command:

```bash
./package/bin/prover <circuit.zkey> <witness.wtns> <proof.json> <public.json>
```

by this one

```bash
./package/bin/prover_cuda <circuit.zkey> <witness.wtns> <proof.json> <public.json>
```

# Results

Test machine:
GPU — NVIDIA GeForce RTX 4090 24GB  
CPU — 2\* AMD EPYC 7763 64-Core Processor  
RAM - 256GB

CPU version:

```
init and set str for altBbn128r: 0 ms
get zkey,zkeyHeader,wtns,wtnsHeader: 0 ms
make prover: 64 ms
get wtnsData: 0 ms
Multiexp A: 1816 ms
Multiexp B1: 2020 ms
Multiexp B2: 2520 ms
Multiexp C: 2775 ms
Initializing a b c A: 59 ms
Processing coefs: 593 ms
Calculating c: 49 ms
Initializing fft: 0 ms
iFFT A: 18158 ms
a After ifft: 0 ms
Shift A: 46 ms
a After shift: 0 ms
FFT A: 3188 ms
a After fft: 0 ms
iFFT B: 1805 ms
b After ifft: 0 ms
Shift B: 45 ms
b After shift: 0 ms
FFT B: 750 ms
b After fft: 0 ms
iFFT C: 971 ms
c After ifft: 0 ms
Shift C: 49 ms
c After shift: 0 ms
FFT C: 779 ms
c After fft: 0 ms
Start ABC: 48 ms
abc: 0 ms
Multiexp H: 4720 ms
generate proof: 40708 ms
write proof to file: 0 ms
write public to file: 0 ms
prover total: 41757 ms
```

GPU version:

```
init and set str for altBbn128r: 0 ms
get zkey,zkeyHeader,wtns,wtnsHeader: 0 ms
make prover: 65 ms
get wtnsData: 0 ms
get MSM config: 0 ms
Multiexp A: 848 ms
Multiexp B1: 679 ms
Multiexp B2: 1080 ms
Multiexp C: 683 ms
Initializing a b c A: 75 ms
Processing coefs: 618 ms
Calculating c: 38 ms
Initializing fft: 148 ms
iFFT A: 291 ms
a After ifft: 0 ms
Shift A: 164 ms
a After shift: 0 ms
FFT A: 294 ms
a After fft: 0 ms
iFFT B: 272 ms
b After ifft: 0 ms
Shift B: 33 ms
b After shift: 0 ms
FFT B: 308 ms
b After fft: 0 ms
iFFT C: 255 ms
c After ifft: 0 ms
Shift C: 28 ms
c After shift: 0 ms
FFT C: 278 ms
c After fft: 0 ms
Start ABC: 46 ms
abc: 0 ms
Multiexp H: 929 ms
generate proof: 7362 ms
write proof to file: 0 ms
write public to file: 0 ms
prover cuda total: 8443 ms
```

# Note

This project has been tested only under the following configurations:

- Ubuntu 22.04
- Kernel version 6.5.0-25-generic
- CUDA 12.3
- GCC/G++ 12.3.0
- GPU — NVIDIA GeForce RTX 4090 24GB
- CPU — 2\* AMD EPYC 7763 64-Core Processor
