#!/usr/bin/env bash
set -euo pipefail

push=''

while getopts 'p' OPTION; do
  case "$OPTION" in
    p) push='true' ;;
    ?) echo "Usage: $0 [-p] <submission_folder_path> <dataset_folder_path>"
      exit 1 ;;
  esac
done
shift "$((OPTIND - 1))"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <submission_folder_path> [<dataset_folder_path>]"
  exit 1
fi

REPO_ROOT="/workspaces/Kaggle-tools"
SUBMISSION_DIR="$(cd "$1" && pwd)"
KERNEL_METADATA="${SUBMISSION_DIR}/kernel-metadata.json"

if [[ ! -f "${KERNEL_METADATA}" ]]; then
  echo "kernel-metadata.json not found in ${SUBMISSION_DIR}"
  exit 1
fi

# echo "Building wheel with uv..."
# cd "${REPO_ROOT}"
# uv build

# WHEEL_PATH="$(ls -t "${REPO_ROOT}"/dist/*.whl | head -n 1)"
# if [[ -z "${WHEEL_PATH}" || ! -f "${WHEEL_PATH}" ]]; then
#   echo "No wheel found in ${REPO_ROOT}/dist"
#   exit 1
# fi

# cp "${WHEEL_PATH}" "${DATASET_DIR}/"
# echo "Copied wheel to dataset: ${WHEEL_PATH}"

if [ "${push}" == "true" ]; then
  DATASET_DIR="$(cd "$2" && pwd)"
  cp -r "${REPO_ROOT}/ktools" "${DATASET_DIR}/ktools"
  kaggle datasets version -p "${DATASET_DIR}" -m "Update ktools" -q -r zip
  echo "Waiting 30 seconds for dataset to be processed..."
  sleep 30
fi

# uv pip freeze > "${REPO_ROOT}/${DATASET_DIR}/requirements.txt"
# grep -v -i -E "^(pip|setuptools|wheel|ipykernel|jupyter|jupyter-client|jupyter-core|notebook|traitlets|pywin32|pywinpty|wincertstore|appnope|certifi|urllib3|idna|charset-normalizer)(==|>|<|~|$)" "${REPO_ROOT}/${DATASET_DIR}/requirements.txt" > "${REPO_ROOT}/${DATASET_DIR}/safe_reqs.txt"

kaggle kernels push -p "${SUBMISSION_DIR}"
