params.TOOL_FOLDER = "$moduleDir/bin/feature_augment"
params.publishdir = "./nf_output"


process processXIC {
    publishDir "$params.publishdir/perfilecalculation", mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    maxForks 12

    input:
    file input_feature
    each file(input_spectra_file)
    val mz_da_tolernace
    val rt_min_tolernace
    
    output:
    file '*_quant.csv'

    """
    python $params.TOOL_FOLDER/feature_augment_xic.py \
    $input_spectra_file \
    $input_feature \
    ${input_spectra_file}_quant.csv \
    --mz_da_tolernace "${mz_da_tolernace}" \
    --rt_min_tolernace "${rt_min_tolernace}"
    """
}

process reformatQuant {
    publishDir "$params.publishdir/feature_augment_quant", mode: 'copy'

    conda "$params.TOOL_FOLDER/conda_env.yml"

    input:
    file merged_quant_file

    output:
    file 'feature_table.csv'

    """
    python $params.TOOL_FOLDER/reformat_quant.py \
    $merged_quant_file \
    feature_table.csv
    """

}