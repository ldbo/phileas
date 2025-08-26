dir := "joss-paper"

paper: figures
    rm -rf {{dir}}/jats
    docker run \
        --rm \
        --volume "${PWD}/{{dir}}:/data" \
        --user `id -u`:`id -g` \
        --env JOURNAL=joss \
        openjournals/inara

figures:
    typst compile {{dir}}/figures/experiment_bench.typ \
                  {{dir}}/figures/experiment_bench.svg
    typst compile {{dir}}/figures/acquisition_process.typ \
                  {{dir}}/figures/acquisition_process.svg
