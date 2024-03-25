#include "groth16_cuda.cu"

int main(int argc, char **argv)
{
    START_TIMER(prover_cuda_timer);
    if (argc != 5) {
        std::cerr << "Invalid number of parameters:\n";
        std::cerr << "Usage: prover_cuda <circuit.zkey> <witness.wtns> <proof.json> <public.json>\n";
        return EXIT_FAILURE;
    }

    START_TIMER(altBbn128r_timer);
    mpz_t altBbn128r;

    mpz_init(altBbn128r);
    mpz_set_str(altBbn128r, "21888242871839275222246405745257275088548364400416034343698204186575808495617", 10);
    END_TIMER(altBbn128r_timer, "init and set str for altBbn128r");

    try {
        START_TIMER(get_zkey_wtns_timer);
        std::string zkeyFilename = argv[1];
        std::string wtnsFilename = argv[2];
        std::string proofFilename = argv[3];
        std::string publicFilename = argv[4];

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

        END_TIMER(get_zkey_wtns_timer, "get zkey,zkeyHeader,wtns,wtnsHeader");

        START_TIMER(make_prover_timer);
        auto prover = makeCuda_Prover<AltBn128::Engine>(
            zkeyHeader->nVars,
            zkeyHeader->nPublic,
            zkeyHeader->domainSize,
            zkeyHeader->nCoefs,
            zkeyHeader->vk_alpha1,
            zkeyHeader->vk_beta1,
            zkeyHeader->vk_beta2,
            zkeyHeader->vk_delta1,
            zkeyHeader->vk_delta2,
            zkey->getSectionData(4),    // Coefs
            zkey->getSectionData(5),    // pointsA
            zkey->getSectionData(6),    // pointsB1
            zkey->getSectionData(7),    // pointsB2
            zkey->getSectionData(8),    // pointsC
            zkey->getSectionData(9)     // pointsH1
        );
        END_TIMER(make_prover_timer, "make prover");

        START_TIMER(wtnsData_timer);
        AltBn128::FrElement *wtnsData = (AltBn128::FrElement *)wtns->getSectionData(2);
        END_TIMER(wtnsData_timer, "get wtnsData");

        START_TIMER(get_proof_timer);
        auto proof = prover->prove_cuda(wtnsData);
        END_TIMER(get_proof_timer, "generate proof");


        START_TIMER(write_proof_to_file_timer);
        std::ofstream proofFile;
        proofFile.open (proofFilename);
        proofFile << proof->toJson();
        proofFile.close();
        END_TIMER(write_proof_to_file_timer, "write proof to file");

        START_TIMER(write_public_to_file_timer);
        std::ofstream publicFile;
        publicFile.open (publicFilename);

        json jsonPublic;
        AltBn128::FrElement aux;
        for (int i=1; i<=zkeyHeader->nPublic; i++) {
            AltBn128::Fr.toMontgomery(aux, wtnsData[i]);
            jsonPublic.push_back(AltBn128::Fr.toString(aux));
        }

        publicFile << jsonPublic;
        publicFile.close();
        END_TIMER(write_public_to_file_timer, "write public to file");

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
