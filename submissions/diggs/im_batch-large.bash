for B in 20 25 30 35 40
do
    echo "Runing $1 for B=$B"
    sbatch submission-im.bash $1 $B $2
done
