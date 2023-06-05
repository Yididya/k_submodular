for K in 10 9 8 7 6 5 4 3 2 1
do
  sbatch submission-im-is-varying-k.bash $1 $2 $K
done
