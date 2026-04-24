#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
exec "${PYTHON:-python}" baselines/seperate_ippo_rnn.py --config_file seperate_ippo_rnn.yaml
