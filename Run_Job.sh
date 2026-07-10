#!/usr/bin/env bash
#SBATCH --job-name=test_run_memory_efficiency_check
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=../output/test_run_memory_efficiency_check.out
#SBATCH --error=../output/test_run_memory_efficiency_check.err
#SBATCH --mail-type=END
#SBATCH --mail-user=abdulaziz_sobirov@college.harvard.edu

module load python
module load mamba
mamba activate myenv

cd /n/home06/asobirov/ForageWorld-MA
CONFIG=${1:-seperate_ippo_rnn.yaml}
python baselines/seperate_ippo_rnn.py --config_file $CONFIG