#include "groth16_cuda.cu"

__uint128_t g_lehmer64_state = 0xAAAAAAAAAAAAAAAALL;

// Fast random generator
// https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/

uint64_t lehmer64() {
  g_lehmer64_state *= 0xda942042e4dd58b5LL;
  return g_lehmer64_state >> 64;
}


int main(int argc, char **argv){
    std::cout << "--------test icicle's msm and rapidsnark's G1.multiMulByScalar-----------" << "\n";
    START_TIMER(prover_cuda_timer);
    if (argc != 9) {
        std::cerr << "Invalid number of parameters:\n";
        std::cerr << "Usage: icicle_msm_test <circuit.zkey> <witness.wtns>  <config_c> <bit_size> <large_bucket_factor> <are_scalars_montgomery_form> <are_points_montgomery_form> <is_big_triangle>\n";
        return EXIT_FAILURE;
    }

    mpz_t altBbn128r;

    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);

    try {
        std::string zkeyFilename = argv[1];
        std::string wtnsFilename = argv[2];

        int config_c = atoi(argv[3]);
        int bit_size = atoi(argv[4]);
        int large_bucket_factor = atoi(argv[5]);
        bool are_scalars_montgomery_form = atoi(argv[6])!=0?true:false;
        bool are_points_montgomery_form = atoi(argv[7])!=0?true:false;
        bool is_big_triangle = atoi(argv[8])!=0?true:false;
        std::cout << "config_c: " << config_c << "\n";
        std::cout << "bit_size: " << bit_size << "\n";
        std::cout << "large bucket factor: " << large_bucket_factor << "\n";                 
        std::cout << "are_scalars_montgomery_form: " << are_scalars_montgomery_form << "\n";
        std::cout << "are_points_montgomery_form: " << are_points_montgomery_form << "\n";
        std::cout << "is_big_triangle: " << is_big_triangle << "\n";     
       

        auto zkey = BinFileUtils::openExisting(zkeyFilename, "zkey", 1);
        auto zkeyHeader = ZKeyUtils::loadHeader(zkey.get());

        std::string proofStr;
        if (mpz_cmp(zkeyHeader->rPrime, altBbn128r) != 0) {
            throw std::invalid_argument( "zkey curve not supported" );
        }

        auto wtns = BinFileUtils::openExisting(wtnsFilename, "wtns", 2);
        auto wtnsHeader = WtnsUtils::loadHeader(wtns.get());

        if (mpz_cmp(wtnsHeader->prime, altBbn128r) != 0) {
            throw std::invalid_argument( "different wtns curve" );
        }

        AltBn128::FrElement *wtnsData = (AltBn128::FrElement *)wtns->getSectionData(2);
        auto nVars = zkeyHeader->nVars;
        // pointsA = (Engine::G1PointAffine *)zkey->getSectionData(5);
        // pointsB2 = (Engine::G1PointAffine *)zkey->getSectionData(7); //G2 group
        // pointsC = (Engine::G1PointAffine *)zkey->getSectionData(8);
        // pointsH = (Engine::G1PointAffine *)zkey->getSectionData(9);
        auto pointsB1 = (Engine::G1PointAffine *)zkey->getSectionData(6);


        device_context::DeviceContext ctx = device_context::get_default_device_context();
        msm::MSMConfig config = {
            ctx,   // ctx
            0,     // points_size
            1,     // precompute_factor
            config_c,     // c
            0,     // bitsize
            large_bucket_factor,    // large_bucket_factor
            1,     // batch_size
            false, // are_scalars_on_device
            are_scalars_montgomery_form, // are_scalars_montgomery_form
            false, // are_points_on_device
            are_points_montgomery_form, // are_points_montgomery_form
            false, // are_results_on_device
            is_big_triangle, // is_big_triangle
            false, // is_async
        };
        Engine::G1Point rapidsnark_res;
        Engine::G1Point icicle_res;

        START_TIMER(icicle_g1_msm_timer);
        bn254_icicle_cuda_g1_msm(icicle_res, pointsB1, (uint8_t*)wtnsData, nVars, config);
        std::cout <<"nVars: " << nVars << "\n";
        END_TIMER(icicle_g1_msm_timer, "icicle g1 msm");
        START_TIMER(rapidsnark_g1_msm_timer);
        auto sW = sizeof(wtnsData[0]);
        G1.multiMulByScalar(rapidsnark_res, pointsB1, (uint8_t*)wtnsData, sW, nVars);
        END_TIMER(rapidsnark_g1_msm_timer, "rapidsnark g1 msm");
        std::cout << "icicle res:" << G1.toString(icicle_res, 10) << "\n";
        std::cout << "rapidsnark res:" << G1.toString(rapidsnark_res, 10) << "\n";

        //assert(G1.eq(rapidsnark_res, icicle_res));
        std::cout << "compare res: " << G1.eq(rapidsnark_res, icicle_res) << "\n";

        std::cout << "----------------------random scalars and pointsB1-------------------------------------------" << "\n";
        uint8_t *scalars = new uint8_t[nVars *32];
        // random scalars
        for (int i=0; i < nVars * 4; i++){
            uint64_t res  = lehmer64();
            
            if(i % 4 == 3) {
                res &= 0x1fffffffffffffff;
            }
            *((uint64_t *)(scalars +i*8)) = res;
        }

        Engine::G1Point rapidsnark_res1;
        Engine::G1Point icicle_res1;
        START_TIMER(icicle_g1_msm_timer1);
        bn254_icicle_cuda_g1_msm(icicle_res1, pointsB1, (uint8_t*)scalars, nVars, config);
        END_TIMER(icicle_g1_msm_timer1, "icicle g1 msm");
        START_TIMER(rapidsnark_g1_msm_timer1);
        G1.multiMulByScalar(rapidsnark_res1, pointsB1, (uint8_t*)scalars, sW, nVars);
        END_TIMER(rapidsnark_g1_msm_timer1, "rapidsnark g1 msm");
        std::cout << "icicle res:" << G1.toString(icicle_res1, 10) << "\n";
        std::cout << "rapidsnark res:" << G1.toString(rapidsnark_res1, 10) << "\n";

        //assert(G1.eq(rapidsnark_res, icicle_res));
        std::cout << "compare res: " << G1.eq(rapidsnark_res1, icicle_res1) << "\n";


        std::cout << "------------------wtnsData and generate bases-----------------------------" << "\n";

        Engine::G1PointAffine *bases = new Engine::G1PointAffine[nVars];
        G1.copy(bases[0], G1.one());
        G1.copy(bases[1], G1.one());

        for (int i=2; i<nVars; i++){
            G1.add(bases[i], bases[i-1], bases[i-2]);
        }

        Engine::G1Point rapidsnark_res2;
        Engine::G1Point icicle_res2;
        START_TIMER(icicle_g1_msm_timer2);
        bn254_icicle_cuda_g1_msm(icicle_res2, bases, (uint8_t*)wtnsData, nVars, config);
        END_TIMER(icicle_g1_msm_timer2, "icicle g1 msm");
        START_TIMER(rapidsnark_g1_msm_timer2);
        G1.multiMulByScalar(rapidsnark_res2, bases, (uint8_t*)wtnsData, sW, nVars);
        END_TIMER(rapidsnark_g1_msm_timer2, "rapidsnark g1 msm");
        std::cout << "icicle res:" << G1.toString(icicle_res2, 10) << "\n";
        std::cout << "rapidsnark res:" << G1.toString(rapidsnark_res2, 10) << "\n";

        //assert(G1.eq(rapidsnark_res, icicle_res));
        std::cout << "compare res: " << G1.eq(rapidsnark_res2, icicle_res2) << "\n";

        std::cout << "------------------random scalars and generate bases-----------------------------------" << "\n";
        Engine::G1Point rapidsnark_res3;
        Engine::G1Point icicle_res3;
        START_TIMER(icicle_g1_msm_timer3);
        bn254_icicle_cuda_g1_msm(icicle_res3, bases, (uint8_t*)scalars, nVars, config);
        END_TIMER(icicle_g1_msm_timer3, "icicle g1 msm");
        START_TIMER(rapidsnark_g1_msm_timer3);
        G1.multiMulByScalar(rapidsnark_res3, bases, (uint8_t*)scalars, sW, nVars);
        END_TIMER(rapidsnark_g1_msm_timer3, "rapidsnark g1 msm");
        std::cout << "icicle res:" << G1.toString(icicle_res3, 10) << "\n";
        std::cout << "rapidsnark res:" << G1.toString(rapidsnark_res3, 10) << "\n";

        //assert(G1.eq(rapidsnark_res, icicle_res));
        std::cout << "compare res: " << G1.eq(rapidsnark_res3, icicle_res3) << "\n";    



        std::cout << "DONE!" <<"\n";
        delete [] scalars; 
        delete [] bases;
    } catch (std::exception* e) {
        mpz_clear(altBbn128r);
        std::cerr << e->what() << '\n';
        return EXIT_FAILURE;
    } catch (std::exception& e) {
        mpz_clear(altBbn128r);
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    mpz_clear(altBbn128r);
    END_TIMER(prover_cuda_timer, "prover cuda total");

    exit(EXIT_SUCCESS);
}