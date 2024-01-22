for i in 0 1 2 3 4 5 6 
do 
echo "tensor_dir_kernel_7_0_$i has "
ls /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topk20/tensor_dir_kernel_7_0_$i | wc -l 
done 
cd /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topk20 
wc -l *.json 