#!/usr/bin/env bash
#SBATCH --job-name=forageworld_ma_warrior_only_trading_allowed_07_07_2026
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=../output/forageworld_ma_warrior_only_trading_allowed_07_07_26.out
#SBATCH --error=../output/forageworld_ma_warrior_only_trading_allowed_07_07_26.err
#SBATCH --mail-type=END
#SBATCH --mail-user=abdulaziz_sobirov@college.harvard.edu

module load python
module load mamba
mamba activate myenv

cd /n/home06/asobirov/ForageWorld-MA
CONFIG=${1:-seperate_ippo_rnn_warrior_only_trading_allowed.yaml}
python baselines/seperate_ippo_rnn.py --config_file $CONFIG