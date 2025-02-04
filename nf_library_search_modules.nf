params.TOOL_FOLDER = "$moduleDir/bin/library_search"
params.publishDir = "./nf_output"

process searchDataGNPS {
    //publishDir "./nf_output", mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    cache 'lenient'

    input:
    tuple file(input_library), file(input_spectrum), val(input_path), val(full_path)
    val pm_tolerance
    val fragment_tolerance
    val topk
    val library_min_cosine
    val library_min_matched_peaks
    val analog_search

    output:
    file 'search_results/*' optional true

    """
    mkdir -p search_results

    python $params.TOOL_FOLDER/library_search_wrapper.py \
        "$input_spectrum" \
        "$input_library" \
        search_results \
        $params.TOOL_FOLDER/convert \
        $params.TOOL_FOLDER/main_execmodule.allcandidates \
        --pm_tolerance "$pm_tolerance" \
        --fragment_tolerance "$fragment_tolerance" \
        --topk $topk \
        --library_min_cosine $library_min_cosine \
        --library_min_matched_peaks $library_min_matched_peaks \
        --analog_search "$analog_search" \
        --full_relative_query_path "$full_path"
    """
}

process searchDataGNPSNew{

    publishDir params.publishDir, mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env_gnps_new.yml"

    cache 'lenient'

    input:
    tuple file(input_library), file(input_spectrum)
    val search_algorithm
    val analog_search
    val analog_max_shift
    val pm_tolerance
    val fragment_tolerance
    val library_min_similarity
    val library_min_matched_peaks
    val peak_transformation
    val unmatched_penalty_factor

    output:
    file 'search_results/*' optional true

    """
    mkdir -p search_results

    python $params.TOOL_FOLDER/gnps_new/main_search.py \
        --gnps_lib_mgf "$input_library" \
        --qry_file "$input_spectrum" \
        --algorithm $search_algorithm \
        --analog_search $analog_search \
        --analog_max_shift $analog_max_shift \
        --pm_tol $pm_tolerance \
        --frag_tol $fragment_tolerance \
        --min_score $library_min_similarity \
        --min_matched_peak $library_min_matched_peaks \
        --peak_transformation $peak_transformation \
        --unmatched_penalty_factor $unmatched_penalty_factor
    """
}

process searchDataBlink {
    //publishDir "./nf_output", mode: 'copy'

    conda "$params.TOOL_FOLDER/blink/environment.yml"

    input:
    each file(input_library)
    each file(input_spectrum)
    val blink_ionization
    val blink_minpredict
    val fragment_tolerance

    output:
    file 'search_results/*.csv' optional true

    script:
    def randomFilename = UUID.randomUUID().toString()
    def input_spectrum_abs = input_spectrum.toRealPath()
    def input_library_abs = input_library.toRealPath()
    """
    mkdir -p search_results
    echo $workDir
    previous_cwd=\$(pwd)
    echo \$previous_cwd

    cd $params.TOOL_FOLDER/blink && python -m blink.blink_cli \
    $input_spectrum_abs \
    $input_library_abs \
    \$previous_cwd/search_results/${randomFilename}.csv \
    $params.TOOL_FOLDER/blink/models/positive_random_forest.pickle \
    $params.TOOL_FOLDER/blink/models/negative_random_forest.pickle \
    $blink_ionization \
    --min_predict $blink_minpredict \
    --mass_diffs 0 14.0157 12.000 15.9949 2.01565 27.9949 26.0157 18.0106 30.0106 42.0106 1.9792 17.00284 24.000 13.97925 1.00794 40.0313 \
    --tolerance $fragment_tolerance
    """
}

process formatBlinkResults {
    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    path input_file

    output:
    path '*.tsv'

    """
    python $params.TOOL_FOLDER/format_blink.py \
    $input_file \
    ${input_file}.tsv
    """
}

process chunkResults {
    conda "$params.TOOL_FOLDER/conda_env.yml"

    cache 'lenient'

    input:
    path to_merge, stageAs: './results/*' // To avoid naming collisions
    val topk

    output:
    path "batched_results.tsv" optional true

    """

    python $params.TOOL_FOLDER/tsv_merger.py \
    results \
    batched_results.tsv \
    --topk $topk
    """
}

// Use a separate process to merge all the batched results
process mergeResults {
    publishDir params.publishDir, mode: 'copy'
    
    conda "$params.TOOL_FOLDER/conda_env.yml"

    cache 'lenient'

    input:
    path 'batched_results.tsv', stageAs: './results/batched_results*.tsv' // Will automatically number inputs to avoid name collisions
    val topk

    output:
    path 'merged_results.tsv'

    """
    python $params.TOOL_FOLDER/tsv_merger.py \
    results \
    merged_results.tsv \
    --topk $topk
    """
}

process librarygetGNPSAnnotations {
    publishDir params.publishDir, mode: 'copy'

    cache 'lenient'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    path "merged_results.tsv"
    path "library_summary.tsv"
    val topk
    val filtertostructures

    output:
    path 'merged_results_with_gnps.tsv'

    """
    python $params.TOOL_FOLDER/getGNPS_library_annotations.py \
    merged_results.tsv \
    merged_results_with_gnps.tsv \
    --librarysummary library_summary.tsv \
    --topk $topk \
    --filtertostructures $filtertostructures
    """
}

process filtertop1Annotations {
    publishDir params.publishDir, mode: 'copy'

    cache 'lenient'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    path "merged_results_with_gnps.tsv"

    output:
    path 'merged_results_with_gnps_top1.tsv'

    """
    python $params.TOOL_FOLDER/filter_top1_hits.py \
    merged_results_with_gnps.tsv \
    merged_results_with_gnps_top1.tsv
    """
}

process summaryLibrary {
    publishDir params.publishDir, mode: 'copy'

    cache 'lenient'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    path library_file

    output:
    path '*.tsv'

    """
    python $params.TOOL_FOLDER/library_summary.py \
    $library_file \
    ${library_file}.tsv
    """
}