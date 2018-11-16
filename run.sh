#!/usr/bin/env bash
python ./src/w2v_feature.py
python ./src/stacking_model.py
python ./src/w2v_feature.py
python ./src/model.py