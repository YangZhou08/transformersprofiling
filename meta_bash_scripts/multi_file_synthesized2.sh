mkdir /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topkna 
echo "tinyllma_topkna directory created!" 

sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid0.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid1.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid2.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid3.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid4.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid5.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid6.sh 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/stupid_scripts/largegpusparallel_taskid7.sh 
echo "All jobs submitted!" 
