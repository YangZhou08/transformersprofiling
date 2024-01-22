cd /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topkna 

for t in 0 1 2 3 4 5 6 7 
    do 
    for i in 0 1 2 3 4 5 6 
        do 
        echo "tensor_dir_kernel_7_$(t)_$(i) has "
        ls /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topk20/tensor_dir_kernel_7_$(t)_$(i) | wc -l 
        # cd /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topk20 
        echo "c4synthesized_file1_kernel7_$(t)_$(i).json has " 
        wc -l c4synthesized_file1_kernel7_$(t)_$(i).json 
        # wc -l *.json 
        done 
    done 
wc -l *.json 