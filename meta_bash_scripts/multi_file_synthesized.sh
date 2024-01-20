for i in 1 2 3 4 5 6 7 
do 
    sbatch single_file_synthesized.sh $i 
done 
echo "All jobs submitted!" 
