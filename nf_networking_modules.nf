params.TOOL_FOLDER = "$moduleDir/bin/networking"
params.publishdir = "./nf_output"
params.maxforks = 16

maxforks_int = params.maxforks.toInteger()

process calculatePairs_index {
    publishDir "$params.publishdir/temp_pairs", mode: 'copy'
    
    conda "$params.TOOL_FOLDER/conda_env.yml"
    tag "Chunk ${chunk_id}/${parallelism}"

    input:
    file spectrum_file
    each chunk_id
    val ms2_tolerance
    val min_cosine
    val parallelism
    val alignment_strategy
    val enable_peak_filtering

    output:
    file "${chunk_id}.params_aligns.tsv"

    """
    python $params.TOOL_FOLDER/gnps_index.py \
        -t ${spectrum_file} \
        --chunk_id ${chunk_id} \
        --total_chunks $parallelism \
        --tolerance $ms2_tolerance \
        --threshold $min_cosine \
        --alignment_strategy "$alignment_strategy" \
        --enable_peak_filtering $enable_peak_filtering
    """
}


process prepGNPSParams {
    publishDir "$params.publishdir", mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    file spectrum_file
    val parallelism
    val min_matched_peaks
    val ms2_tolerance
    val pm_tolerance
    val min_cosine
    val max_shift

    output:
    file "params/*"

    """
    mkdir params
    python $params.TOOL_FOLDER/prep_molecular_networking_parameters.py \
        "$spectrum_file" \
        "params" \
        --parallelism "$parallelism" \
        --min_matched_peaks "$min_matched_peaks" \
        --ms2_tolerance "$ms2_tolerance" \
        --pm_tolerance "$pm_tolerance" \
        --min_cosine "$min_cosine" \
        --max_shift "$max_shift"
    """
}

process calculateGNPSPairs {
    publishDir "$params.publishdir/temp_pairs", mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    file spectrum_file
    each file(params_file)

    maxForks maxforks_int

    output:
    file "*_aligns.tsv" optional true

    """
    $params.TOOL_FOLDER/main_execmodule \
        ExecMolecularParallelPairs \
        "$params_file" \
        -ccms_INPUT_SPECTRA_MS2 $spectrum_file \
        -ccms_output_aligns ${params_file}_aligns.tsv
    """
}


process calculatePairsEntropy {
    publishDir "$params.publishdir/temp_pairs", mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env_entropy.yml"

    maxForks maxforks_int

    input:
    file spectrum_file
    each i
    val parallelism
    val min_matched_peaks
    val ms2_tolerance
    val min_cosine

    output:
    file "*.tsv" optional true

    """
    python $params.TOOL_FOLDER/run_SpectralEntropy.py \
        $spectrum_file \
        ${i}_pairs.tsv \
        --nodenumber ${i} \
        --nodetotal ${parallelism} \
        --min_matched_peaks ${min_matched_peaks} \
        --ms2_tolerance ${ms2_tolerance} \
        --min_cosine ${min_cosine}
    """
}