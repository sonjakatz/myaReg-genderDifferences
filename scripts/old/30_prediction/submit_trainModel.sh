#!/bin/bash
#-----------------------------Other information------------------------
#SBATCH --comment=773320000
#SBATCH --job-name=manual_boot
#-----------------------------Required resources-----------------------
#SBATCH --time=1200
#SBATCH --mem=4048
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

HOME="/home/WUR/katz001/PROJECTS/myaReg-genderDifferences/scripts"
cd $HOME

#################################################################################################

### Boruta run
## Individual patient predictions
#python 30_prediction/trainModel_featureSelection.py 

## prediction with confidence intervals
#python 30_prediction/trainModel_featureSelection_bootstrapped.py 




### Manual variable selection run
## Individual patient predictions
#python 30_prediction/trainModel_manualVarSelection.py

## prediction with confidence intervals
#python 30_prediction/trainModel_manualVarSelection_bootstrapped.py
