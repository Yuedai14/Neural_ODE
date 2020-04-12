#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "resnet50_paramsXX.h"
#include "imagesXX.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t start, end;
    uint64_t im2col_cycles = 0, matmul_cycles = 0, pool_cycles = 0, conv_dw_cycles = 0, res_add_cycles = 0, other_cycles = 0;

    // conv_1
    start = read_cycles();
    im2col(conv_1_params.batch_size, conv_1_params.in_channels, conv_1_params.in_dim,
        conv_1_params.I, conv_1_params.K,
        images, conv_1_in, &conv_1_params);
    
    end = read_cycles();
    im2col_cycles += end - start;
    

    start = read_cycles();
    tiled_matmul_nn_auto(conv_1_params.I, conv_1_params.J, conv_1_params.K,
        conv_1_in, conv_1_w, conv_1_b, conv_1_out,
        RELU, conv_1_params.output_scale, true,
        tiled_matmul_type, check, "conv_1");

    end = read_cycles();
    matmul_cycles += end - start;


    uint64_t total_cycles = im2col_cycles + matmul_cycles + pool_cycles + conv_dw_cycles + res_add_cycles + other_cycles;

    printf("\nTotal cycles: %llu\n", total_cycles);
    printf("Matmul cycles: %llu\n", matmul_cycles);
    printf("Im2col cycles: %llu\n", im2col_cycles);
    printf("Pooling cycles: %llu\n", pool_cycles);
    printf("Depthwise convolution cycles: %llu\n", conv_dw_cycles);
    printf("Other cycles: %llu\n", other_cycles);

    printf("PASS\n");

    exit(0);
}

