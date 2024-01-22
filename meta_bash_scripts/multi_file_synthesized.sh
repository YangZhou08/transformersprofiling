# for i in 1 2 3 4 5 6 7 
# do 
#     sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/single_file_synthesized.sh $i 
# done 
rm -rf /fsx-storygen/beidic/yang/c4llm_synthesized/tinyllama_topkna 
echo "all previous tinyllama_topkna files removed!" 
sbatch /fsx-storygen/beidic/yang/transformersprofiling/meta_bash_scripts/largegpusparalleldatasetgeneration.sh 
echo "All jobs submitted!" 
