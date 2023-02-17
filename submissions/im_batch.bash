for B in 2 6 10 14 18
do
    echo "Runing $1 for B=$B"
    sbatch submission-im.bash $1 $B $2
done
