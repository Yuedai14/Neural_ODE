#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "Neural_ODE/neuralODE_1layer_params.h"
#include "Neural_ODE/neuralODE_1layer_images.h"

uint64_t run_config(size_t conv_k_len, size_t conv_conv_len, size_t conv_in_size, size_t conv_kernel_size){

	uint64_t start;
	start = read_cycles();
	gemmini_config_conv(conv_k_len, conv_conv_len, conv_in_size, conv_kernel_size, 1);
	
	return start;
}

uint64_t run_compute(int K_sp_addr_start, int B_sp_addr_start){
	
	uint64_t start;
	start = read_cycles();

	gemmini_extended_compute_conv(K_sp_addr_start, B_sp_addr_start, DIM, DIM, DIM, DIM);

	return start;
}

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
    uint64_t config_cycles = 0, conv_cycles = 0;
    	
	int K_sp_addr_start = 0;
	int B_sp_addr_start = 2048;

	load_conv(&conv_w[0][0], &images[0][0][0][0], conv_k_len, conv_conv_len);

	// 2D Conv
	start = run_config(conv_k_len, conv_conv_len, conv_in_size, conv_kernel_size);
	end = read_cycles();
	printf("\nconfig_start_cycles: %llu\n", start);
	printf("\nconfig_end_cycles: %llu\n", end);
	config_cycles = end - start;	

	start= run_compute(K_sp_addr_start, B_sp_addr_start);
	end = read_cycles();
	printf("\ncompute_start_cycles: %llu\n", start);
	printf("\ncompute_end_cycles: %llu\n", end);
	conv_cycles = end - start;

	uint64_t total_cycles = config_cycles + conv_cycles;

	
    printf("\nTotal cycles: %llu\n", total_cycles);
    printf("config cycles: %llu\n", config_cycles);
    printf("conv cycles: %llu\n", conv_cycles);

	

	printf("PASS\n");
	exit(0);
}
