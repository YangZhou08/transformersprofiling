cd /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topkna 
for t in 0 1 2 3 4 5 6 7 
do 
    python jsonaggregate.py --task_id $t --kernel_size 7 --model_name tinyllama 
done 
