for B in 100
do
    echo "Runing $1 for B=$B"
    sbatch submission-im-master.bash $1 $B $2
done
