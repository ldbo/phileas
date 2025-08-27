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
