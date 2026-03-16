#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/corridorkey"
SRC_DIR="/opt/corridorkey-src"
VENV_DIR="${APP_DIR}/.venv"
DEPS_SIG_FILE="${APP_DIR}/.deps_signature"
SRC_SIG_FILE="${APP_DIR}/.source_signature"
INSTALL_ENV_SIG_FILE="${APP_DIR}/.install_env_signature"

mkdir -p "${APP_DIR}" "${APP_DIR}/ClipsForInference"

export CORRIDORKEY_RESOLUTION="${CORRIDORKEY_RESOLUTION:-1920x1080x24}"

compute_source_signature() {
  (
    cd "${SRC_DIR}"
    find . -type f \
      -not -path './.git/*' \
      -not -path './.venv/*' \
      -not -name '.deps_signature' \
      -not -name '.source_signature' \
      -not -name '.install_env_signature' \
      -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | awk '{print $1}'
  )
}

sync_app_source_if_needed() {
  local src_sig current_sig=""
  src_sig="$(compute_source_signature)"
  if [ -f "${SRC_SIG_FILE}" ]; then
    current_sig="$(cat "${SRC_SIG_FILE}")"
  fi

  if [ "${src_sig}" = "${current_sig}" ]; then
    return
  fi

  (
    cd "${SRC_DIR}"
    tar \
      --exclude='.git' \
      --exclude='.venv' \
      --exclude='.deps_signature' \
      --exclude='.source_signature' \
      --exclude='.install_env_signature' \
      -cf - .
  ) | (
    cd "${APP_DIR}"
    tar -xf -
  )

  echo "${src_sig}" > "${SRC_SIG_FILE}"
}

compute_deps_signature() {
  (
    cd "${APP_DIR}"
    sha256sum pyproject.toml requirements.txt 1-install.sh 2>/dev/null | sha256sum | awk '{print $1}'
  )
}

compute_install_env_signature() {
  {
    echo "CORRIDORKEY_PYTHON_VERSION=${CORRIDORKEY_PYTHON_VERSION:-3.11}"
    echo "CORRIDORKEY_INSTALL_SAM2=${CORRIDORKEY_INSTALL_SAM2:-n}"
    echo "CORRIDORKEY_PREDOWNLOAD_SAM2=${CORRIDORKEY_PREDOWNLOAD_SAM2:-n}"
    echo "CORRIDORKEY_INSTALL_GVM=${CORRIDORKEY_INSTALL_GVM:-n}"
    echo "CORRIDORKEY_INSTALL_VIDEOMAMA=${CORRIDORKEY_INSTALL_VIDEOMAMA:-n}"
  } | sha256sum | awk '{print $1}'
}

install_tracker_if_needed() {
  if [ "${CORRIDORKEY_INSTALL_SAM2:-n}" != "y" ]; then
    return
  fi

  if "${VENV_DIR}/bin/python" -c "import sam2" >/dev/null 2>&1; then
    return
  fi

  echo "[env] CORRIDORKEY_INSTALL_SAM2=y and tracker package missing; installing tracker extras..."
  if command -v uv >/dev/null 2>&1; then
    if uv pip install --python "${VENV_DIR}/bin/python" --torch-backend=auto -e ".[tracker]"; then
      return
    fi
  fi
  "${VENV_DIR}/bin/python" -m pip install -e ".[tracker]" || \
    echo "[env][WARN] Could not install SAM2 tracker extras automatically."
}

apply_optional_model_flags() {
  if [ "${CORRIDORKEY_INSTALL_GVM:-n}" = "y" ]; then
    echo "[env] Ensuring GVM model is installed..."
    "${VENV_DIR}/bin/python" scripts/setup_models.py --gvm || \
      echo "[env][WARN] Failed to ensure GVM model."
  fi

  if [ "${CORRIDORKEY_INSTALL_VIDEOMAMA:-n}" = "y" ]; then
    echo "[env] Ensuring VideoMaMa model is installed..."
    "${VENV_DIR}/bin/python" scripts/setup_models.py --videomama || \
      echo "[env][WARN] Failed to ensure VideoMaMa model."
  fi

  if [ "${CORRIDORKEY_INSTALL_SAM2:-n}" = "y" ] && [ "${CORRIDORKEY_PREDOWNLOAD_SAM2:-n}" != "n" ]; then
    echo "[env] Ensuring SAM2 checkpoint cache is installed..."
    "${VENV_DIR}/bin/python" scripts/setup_models.py --sam2 || \
      echo "[env][WARN] Failed to ensure SAM2 checkpoint cache."
  fi
}

run_installer_if_needed() {
  local current_sig saved_sig=""
  local current_env_sig saved_env_sig=""

  current_sig="$(compute_deps_signature)"
  current_env_sig="$(compute_install_env_signature)"

  if [ -f "${DEPS_SIG_FILE}" ]; then
    saved_sig="$(cat "${DEPS_SIG_FILE}")"
  fi

  if [ -f "${INSTALL_ENV_SIG_FILE}" ]; then
    saved_env_sig="$(cat "${INSTALL_ENV_SIG_FILE}")"
  fi

  if [ -x "${VENV_DIR}/bin/python" ] && [ "${saved_sig}" = "${current_sig}" ]; then
    if [ "${saved_env_sig}" != "${current_env_sig}" ]; then
      echo "[env] Install/model env vars changed; applying updates..."
      (
        cd "${APP_DIR}"
        install_tracker_if_needed
        apply_optional_model_flags
      )
      echo "${current_env_sig}" > "${INSTALL_ENV_SIG_FILE}"
      echo "[env] Environment-driven updates applied."
      return
    fi

    echo "[install] Reusing existing .venv (deps + env signatures unchanged)."
    return
  fi

  echo "[install] Initializing/updating runtime environment..."
  (
    cd "${APP_DIR}"
    CORRIDORKEY_PYTHON_VERSION="${CORRIDORKEY_PYTHON_VERSION:-3.11}" \
    CORRIDORKEY_INSTALL_SAM2="${CORRIDORKEY_INSTALL_SAM2:-n}" \
    CORRIDORKEY_PREDOWNLOAD_SAM2="${CORRIDORKEY_PREDOWNLOAD_SAM2:-n}" \
    CORRIDORKEY_INSTALL_GVM="${CORRIDORKEY_INSTALL_GVM:-n}" \
    CORRIDORKEY_INSTALL_VIDEOMAMA="${CORRIDORKEY_INSTALL_VIDEOMAMA:-n}" \
    CORRIDORKEY_CREATE_SHORTCUT="n" \
    UV_PYTHON_INSTALL_DIR="/opt/uv-python" \
    UV_LINK_MODE=copy \
    bash ./1-install.sh
  )

  echo "${current_sig}" > "${DEPS_SIG_FILE}"
  echo "${current_env_sig}" > "${INSTALL_ENV_SIG_FILE}"
}

echo "noVNC: http://localhost:6080"
echo "Upload: http://localhost:6081"
echo "VNC:   localhost:5900 (no password)"
echo ""

sync_app_source_if_needed
run_installer_if_needed

exec /usr/bin/supervisord -c /etc/supervisor/conf.d/corridorkey-vnc.conf
