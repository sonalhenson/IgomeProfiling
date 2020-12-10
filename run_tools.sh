#!/usr/bin/env bash

# Activate the (previously set up) virtualenv
source .venv/bin/activate
python3 tools/distinctive_motifs.py 2>&1 | tee -a distinctive_motifs.log

