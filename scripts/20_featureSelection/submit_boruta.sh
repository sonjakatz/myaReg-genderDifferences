#!/bin/bash
#-----------------------------Other information------------------------
#SBATCH --comment=773320000
#SBATCH --job-name=natural
#-----------------------------Required resources-----------------------
#SBATCH --time=1500
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
source activate env_borutaPy
echo $CONDA_DEFAULT_ENV

HOME="/home/WUR/katz001/PROJECTS/myaReg-genderDifferences/scripts"
cd $HOME
python 20_featureSelection/submit_boruta_bootstrapCV.py 
