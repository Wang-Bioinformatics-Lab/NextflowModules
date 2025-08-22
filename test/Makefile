test_masst_usi_file:
	python ../bin/masst/masst_client.py ./data/test_usi.csv test_usi_results.tsv \
	--database gnpslibrary

test_masst_usi_slow_file:
	python ../bin/masst/masst_client.py ./data/test_usi_slow.csv test_results.tsv

test_masst_mgf_sirius:
	python ../bin/masst/masst_client.py ./data/carcinoma_sirius.mgf test_results_mgfsirius.tsv

test_masst_mgf_fbmn:
	python ../bin/masst/masst_client.py ./data/spectra_reformatted.mgf test_results_fbmn.tsv

test_masst_mgf_mzmine:
	python ../bin/masst/masst_client.py ./data/spectra_mzmine.mgf test_results_mzmine.tsv

test_masst_mzml:
	python ../bin/masst/masst_client.py ./data/S_N1.mzML test_results_mzml.tsv

test_masst_usi_file_batch:
	python ../bin/masst/masst_client.py ./data/test_usi.csv test_usi_results_batch.tsv \
	--querymode batch \
	--database gnpslibrary

test_masst_usi_file_big_batch:
	python ../bin/masst/masst_client.py ./data/test_usi_big.csv test_usi_results_batch.tsv \
	--querymode batch \
	--database metabolomicspanrepo_index_nightly