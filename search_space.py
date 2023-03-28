

ss_1 = [[3],               #FILTER_HEIGHT
        [3],               #FILTER_WIDTH,
        [24, 36, 48, 64],  #NUMBER_FILTERS
        [1]]               #NUMBER_STRIDES

nats_bench_tss = ["none",
                  "skip_connect",
                  "nor_conv_1x1",
                  "nor_conv_3x3",
                  "avg_pool_3x3"]

nats_bench_sss = [8,16,24,32,40,48,56,64]
