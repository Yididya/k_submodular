for B in 1 3 5 7 9
do
    echo "Runing $1 for B=$B"
    sbatch submission-im-is.bash $1 $B $2
done
