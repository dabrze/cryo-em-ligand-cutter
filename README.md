# Cryo-EM ligand cutter
Code for extracting ligand fragments from cryo-EM difference maps.

## Install

To use the scripts, create a python environment, clone the repository, and install all the requirements.

```bash
# set up the virtual environment
# you can use venv as shown below or create a conda environment (conda create -n cryo python)
python3 -m venv cryo
# activate the virtual environment (for conda: conda activate cryo)
source cryo/bin/activate
# clone the repository
git clone https://github.com/dabrze/cryo-em-ligand-cutter.git
# install requirements
cd cryo-em-ligand-cutter
python -m pip install -r requirements.txt
```

## Running

Put the cif files (`<pdb_id>.cif`) and cryo-em difference maps (`<pdb_id>_map_model_difference_1.ccp4`) in the data folder in the root of the repo and run `cut_ligands.py`. 

```bash
python cut_ligands.py --pdb_ids_file sample_input.txt  --input_dir data --output_dir blobs --n_jobs -1 --log_file blob_processing.log
```
    
The numpy arrays with the blob densities will be output into the `blobs` folder. The script will also create a log file called `blob_processing.log`.
