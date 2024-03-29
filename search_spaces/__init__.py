from search_spaces import nasbig, nasmedium, naslittle, nasbench

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

nasmedium_arg = [5,            #N
                 3,            #in_channels
                 [16, 32, 64], #cell_channels
                 10]           #num_classes


def ss_selector(ss_string, dataset):
    ss = None
    if ss_string == 'nasbench':
        ss = nasbench.NasBench(dataset)

    elif ss_string == 'nasmedium':
        ss = nasmedium.NASMedium(N=nasmedium_arg[0],
                                 in_channels=nasmedium_arg[1],
                                 cell_channels=nasmedium_arg[2],
                                 num_classes=nasmedium_arg[3],
                                 dataset=dataset)

    elif ss_string == 'naslittle':
        ss = naslittle.NASLittle(N=nasmedium_arg[0],
                                 in_channels=nasmedium_arg[1],
                                 cell_channels=nasmedium_arg[2],
                                 num_classes=nasmedium_arg[3],
                                 dataset=dataset)
    elif ss_string == 'nasbig':
        ss = nasbig.NASBig(N=nasmedium_arg[0],
                           in_channels=nasmedium_arg[1],
                           cell_channels=nasmedium_arg[2],
                           num_classes=nasmedium_arg[3],
                           dataset=dataset)

    return ss
