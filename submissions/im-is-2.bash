for B in 2 4 6 8 10
do
    echo "Runing $1 for B=$B"
    sbatch submission-im-is.bash $1 $B $2
done
