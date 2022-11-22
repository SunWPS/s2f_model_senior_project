#!/bin/bash

# download library
python install -r requirements.txt

cd ./model/model_saved
chmod 755 load_pre_saved_model.sh
./load_pre_saved_model

cd ../..