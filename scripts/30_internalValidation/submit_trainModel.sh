#!/bin/bash
#-----------------------------Other information------------------------
#SBATCH --comment=773320000
#SBATCH --job-name=full_full
#-----------------------------Required resources-----------------------
#SBATCH --time=2000
#SBATCH --mem=2048
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#-----------------------------Output files-----------------------------
#SBATCH --output=output_%j.txt
#SBATCH --error=error_output_%j.txt
#-----------------------------Mail address-----------------------------
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sonja.katz@wur.nl

# ACTIVATE ANACONDAi
eval "$(conda shell.bash hook)"
source activate env_myaReg
echo $CONDA_DEFAULT_ENV

HOME="/home/WUR/katz001/PROJECTS/myaReg-genderDifferences"
cd $HOME
N_ITER=300
DATASET="natural"   #"natural"  #"fullRegistry" #histologie_subgroup" 
BORUTAPERC=100


### Bootstrapping
## python scripts/30_internalValidation/00_trainModels_bootstrapping_inclSHAP.py $N_ITER $DATASET $BORUTAPERC



#### Train one classifier on whole dataset  (for external validation)
python scripts/30_internalValidation/10_trainModels_wholeDataset.py $DATASET $BORUTAPERC

