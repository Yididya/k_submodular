for B in 4 8 12 16 20
do
    echo "Runing $1 for B=$B"
    sbatch submission-im.bash $1 $B $2
done
