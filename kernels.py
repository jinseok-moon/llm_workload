# This page is future work for the project. It will contain scenarios for more precise cycle calculation.


#def allocate_dram_to_sram(input):
#    cycle = input.size() / dram_parallel_bw  # Parallel unit
#    return  cycle

# def gemm_calculate_cycle(input, weight, output):
#     compute_cycle = 0
#     memory_cycle = 0
    
#     memory_cycle += allocate_dram_to_sram(input)
#     memory_cycle += allocate_dram_to_sram(weight)
    
#     tile_size:List[x,y]
#     for tilex, tiley in tile_size:
#         memory_cycle += allocate_dram_to_sram(input[tilex, tiley])
#         memory_cycle += allocate_dram_to_sram(weight[tilex, tiley])
#         compute_cycle += gemm_cycle(input[tilex, tiley], weight[tilex, tiley], sram_output[tilex, tiley])
#     memory_cycle += store_sram_to_dram(output)
    
#     if async_copy:
#         kernel_total_cycle = max(compute_cycle, memory_cycle)
#     else:
#         kernel_total_cycle = compute_cycle + memory_cycle
    