for B in 1 5 10 15
do
    echo "Runing $1 for B=$B"
    sbatch submission-im.bash $1 $B $2
done
