from Instances_Generator import Nk_generator

for N in [32, 64, 128]:
    for K in [1, 2, 4, 8]:
        path = "./benchmark/N_" + str(N) + "_K_" + str(K) + "/test/"
        Nk_generator(N, K, 100, path)