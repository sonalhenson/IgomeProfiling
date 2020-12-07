#!/usr/bin/env bash

# Make sure we are in the working directory because IgomeProfiler uses lots of
# relative paths, but we still use paths relative to the workdir for the real
# script invocation because we don't know where some of the jobs submitted by
# wrapper scripts run from...

WORKDIR=/home/shenson/soft/IgomeProfiling/sonal/IgomeProfiling
cd $WORKDIR

rm -rf output
mkdir output && mkdir output/analysis && mkdir output/logs

# Activate the (previously set up) virtualenv
source .venv/bin/activate

# Run the pipeline on the SLURM batch partition
# Make sure is_run_on_cluster = True in global_params.py

python3 IgOmeProfiling_pipeline.py \
    -q batch \
	--rank_method shuffles \
	--concurrent_cutoffs \
	--gz \
	mock_data/exp12_10M_rows.fastq.gz \
    mock_data/barcode2samplename.txt \
    mock_data/samplename2biologicalcondition.txt \
    output/analysis \
    output/logs
