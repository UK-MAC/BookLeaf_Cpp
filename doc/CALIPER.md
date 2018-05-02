# Collecting performance statistics with LLNL Caliper

Caliper is a framework for collecting application runtime data (such as
performance measurements). Two Caliper profiles are included in
`caliper.config`, one each for serial and MPI execution. After building with
Caliper support, measurements can be collected as in the following MPI example:

```
mkdir -p caliper
num_ranks=4
CALI_CONFIG_PROFILE=bookleaf-mpi \
    mpirun -n ${num_ranks} <mpi args> \
    build/bookleaf <bookleaf args>

# e.g. to display percentage total time spent in each function for each rank
for rank in $(seq 0 $((num_ranks-1))); do
    echo "rank ${rank}"
    cali-query -t \
        -q "select annotation, loop, function, mpi.function, percent_total(sum#time.duration)" \
        caliper/caliper-${rank}.cali
    echo
done
```
