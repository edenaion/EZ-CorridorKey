#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: cloud_gpu_installer_smoke.sh [options]

Runs cost-conscious Azure GPU installer smoke tests on Windows and Linux VMs.
Designed for Azure Cloud Shell or GitHub Actions after `azure/login`.

Options:
  --resource-group NAME        Resource group to use/create
  --prefix NAME                Resource name prefix
  --lane-name NAME             Human-readable lane identifier for summaries
  --repo OWNER/REPO            Public GitHub repo to test (default: ezoosk/EZ-CorridorKey)
  --ref REF                    Git ref/branch/tag/sha to download (default: main)
  --windows-size SIZE          Azure VM size for Windows GPU lane
  --linux-size SIZE            Azure VM size for Linux GPU lane
  --windows-regions CSV        Preferred Windows regions, in order
  --linux-regions CSV          Preferred Linux regions, in order
  --windows-python VERSION     Windows Python version to install (default: 3.11.9)
  --linux-python VERSION       Linux Python version to install (default: 3.10)
  --max-budget-usd AMOUNT      Abort if estimated max compute spend exceeds this
                               budget (default: 150)
  --windows-max-minutes N      Max billed minutes budgeted for Windows lane
                               (default: 75)
  --linux-max-minutes N        Max billed minutes budgeted for Linux lane
                               (default: 60)
  --estimate-only              Print cost estimate and exit without provisioning
  --keep-resources             Keep VMs/resources after the run
  --skip-windows               Skip the Windows GPU lane
  --skip-linux                 Skip the Linux GPU lane
  --spot                       Use Spot priority VMs
  --help                       Show this help
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT_DEFAULT="${ROOT_DIR}/logs/azure-gpu-installer-smoke"

RESOURCE_GROUP_BASE="${AZURE_RESOURCE_GROUP_BASE:-${AZURE_RESOURCE_GROUP:-corridorkey-ci-rg}}"
RESOURCE_GROUP="${RESOURCE_GROUP_BASE}"
PREFIX="ckgpu$(date +%m%d%H%M%S)"
LANE_NAME=""
REPO_SLUG="ezoosk/EZ-CorridorKey"
REPO_REF="${GITHUB_REF_NAME:-main}"
WINDOWS_SIZE="Standard_NC4as_T4_v3"
LINUX_SIZE="Standard_NC4as_T4_v3"
WINDOWS_REGIONS="eastus,eastus2,westus2,westus"
LINUX_REGIONS="eastus,eastus2,westus2,westus"
WINDOWS_PYTHON="3.11.9"
LINUX_PYTHON="3.10"
MAX_BUDGET_USD="${AZURE_MAX_BUDGET_USD:-150}"
WINDOWS_MAX_MINUTES=75
LINUX_MAX_MINUTES=60
KEEP_RESOURCES=0
RUN_WINDOWS=1
RUN_LINUX=1
USE_SPOT=0
ESTIMATE_ONLY=0
RESOURCE_GROUP_CREATED=0
WINDOWS_RATE_USD=""
LINUX_RATE_USD=""
WINDOWS_ESTIMATED_USD=""
LINUX_ESTIMATED_USD=""
TOTAL_ESTIMATED_USD=""
FINAL_STATUS="failed"
FAILURE_CONTEXT=""
START_EPOCH="$(date +%s)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --resource-group) RESOURCE_GROUP_BASE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --lane-name) LANE_NAME="$2"; shift 2 ;;
    --repo) REPO_SLUG="$2"; shift 2 ;;
    --ref) REPO_REF="$2"; shift 2 ;;
    --windows-size) WINDOWS_SIZE="$2"; shift 2 ;;
    --linux-size) LINUX_SIZE="$2"; shift 2 ;;
    --windows-regions) WINDOWS_REGIONS="$2"; shift 2 ;;
    --linux-regions) LINUX_REGIONS="$2"; shift 2 ;;
    --windows-python) WINDOWS_PYTHON="$2"; shift 2 ;;
    --linux-python) LINUX_PYTHON="$2"; shift 2 ;;
    --max-budget-usd) MAX_BUDGET_USD="$2"; shift 2 ;;
    --windows-max-minutes) WINDOWS_MAX_MINUTES="$2"; shift 2 ;;
    --linux-max-minutes) LINUX_MAX_MINUTES="$2"; shift 2 ;;
    --estimate-only) ESTIMATE_ONLY=1; shift ;;
    --keep-resources) KEEP_RESOURCES=1; shift ;;
    --skip-windows) RUN_WINDOWS=0; shift ;;
    --skip-linux) RUN_LINUX=0; shift ;;
    --spot) USE_SPOT=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$RUN_WINDOWS" -eq 0 && "$RUN_LINUX" -eq 0 ]]; then
  echo "Nothing to run: both Windows and Linux lanes are disabled." >&2
  exit 1
fi

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) is required. Run this from Azure Cloud Shell or a machine with az installed." >&2
  exit 1
fi

RUN_ID="${PREFIX}-$(date +%Y%m%d-%H%M%S)"
RESOURCE_GROUP="${RESOURCE_GROUP_BASE}-${PREFIX}"
LOG_DIR="${LOG_ROOT_DEFAULT}/${RUN_ID}"
mkdir -p "${LOG_DIR}"
if [[ -z "${LANE_NAME}" ]]; then
  LANE_NAME="${PREFIX}"
fi

WINDOWS_VM="${PREFIX}-win"
LINUX_VM="${PREFIX}-linux"
WINDOWS_REGION=""
LINUX_REGION=""
WINDOWS_CREATED=0
LINUX_CREATED=0

cleanup() {
  local exit_code=$?
  local end_epoch result_json=""
  end_epoch="$(date +%s)"
  if [[ -z "${FAILURE_CONTEXT}" && "${exit_code}" -ne 0 ]]; then
    FAILURE_CONTEXT="process_exit_${exit_code}"
  fi
  if [[ -f "${LOG_DIR}/windows-installer-smoke.log" ]]; then
    result_json="$(extract_result_json "${LOG_DIR}/windows-installer-smoke.log" || true)"
  fi
  if [[ -z "${result_json}" && -f "${LOG_DIR}/linux-installer-smoke.log" ]]; then
    result_json="$(extract_result_json "${LOG_DIR}/linux-installer-smoke.log" || true)"
  fi
  SUMMARY_PATH="${LOG_DIR}/summary.json" \
  LANE_NAME="${LANE_NAME}" \
  RUN_ID="${RUN_ID}" \
  RESOURCE_GROUP="${RESOURCE_GROUP}" \
  REPO_SLUG="${REPO_SLUG}" \
  REPO_REF="${REPO_REF}" \
  WINDOWS_REGION="${WINDOWS_REGION}" \
  WINDOWS_SIZE="${WINDOWS_SIZE}" \
  WINDOWS_PYTHON="${WINDOWS_PYTHON}" \
  WINDOWS_RATE_USD="${WINDOWS_RATE_USD}" \
  WINDOWS_ESTIMATED_USD="${WINDOWS_ESTIMATED_USD}" \
  WINDOWS_MAX_MINUTES="${WINDOWS_MAX_MINUTES}" \
  LINUX_REGION="${LINUX_REGION}" \
  LINUX_SIZE="${LINUX_SIZE}" \
  LINUX_PYTHON="${LINUX_PYTHON}" \
  LINUX_RATE_USD="${LINUX_RATE_USD}" \
  LINUX_ESTIMATED_USD="${LINUX_ESTIMATED_USD}" \
  LINUX_MAX_MINUTES="${LINUX_MAX_MINUTES}" \
  TOTAL_ESTIMATED_USD="${TOTAL_ESTIMATED_USD}" \
  MAX_BUDGET_USD="${MAX_BUDGET_USD}" \
  KEEP_RESOURCES="${KEEP_RESOURCES}" \
  USE_SPOT="${USE_SPOT}" \
  FINAL_STATUS="${FINAL_STATUS}" \
  FAILURE_CONTEXT="${FAILURE_CONTEXT}" \
  START_EPOCH="${START_EPOCH}" \
  END_EPOCH="${end_epoch}" \
  RESULT_JSON="${result_json}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

result_json = os.environ.get("RESULT_JSON", "").strip()
result = json.loads(result_json) if result_json else {}
payload = {
    "lane_name": os.environ["LANE_NAME"],
    "run_id": os.environ["RUN_ID"],
    "resource_group": os.environ["RESOURCE_GROUP"],
    "repo": os.environ["REPO_SLUG"],
    "ref": os.environ["REPO_REF"],
    "status": os.environ["FINAL_STATUS"],
    "failure_context": os.environ.get("FAILURE_CONTEXT", ""),
    "start_epoch": int(float(os.environ["START_EPOCH"])),
    "end_epoch": int(float(os.environ["END_EPOCH"])),
    "duration_seconds": int(float(os.environ["END_EPOCH"])) - int(float(os.environ["START_EPOCH"])),
    "windows_region": os.environ.get("WINDOWS_REGION", ""),
    "windows_size": os.environ.get("WINDOWS_SIZE", ""),
    "windows_python": os.environ.get("WINDOWS_PYTHON", ""),
    "windows_hourly_rate_usd": os.environ.get("WINDOWS_RATE_USD", ""),
    "windows_estimated_max_usd": os.environ.get("WINDOWS_ESTIMATED_USD", ""),
    "windows_max_minutes": os.environ.get("WINDOWS_MAX_MINUTES", ""),
    "linux_region": os.environ.get("LINUX_REGION", ""),
    "linux_size": os.environ.get("LINUX_SIZE", ""),
    "linux_python": os.environ.get("LINUX_PYTHON", ""),
    "linux_hourly_rate_usd": os.environ.get("LINUX_RATE_USD", ""),
    "linux_estimated_max_usd": os.environ.get("LINUX_ESTIMATED_USD", ""),
    "linux_max_minutes": os.environ.get("LINUX_MAX_MINUTES", ""),
    "estimated_total_max_usd": os.environ.get("TOTAL_ESTIMATED_USD", ""),
    "max_budget_usd": os.environ.get("MAX_BUDGET_USD", ""),
    "keep_resources": os.environ.get("KEEP_RESOURCES", ""),
    "use_spot": os.environ.get("USE_SPOT", ""),
    "result": result,
}
Path(os.environ["SUMMARY_PATH"]).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
  if [[ "${KEEP_RESOURCES}" -eq 1 ]]; then
    echo "Keeping resources because --keep-resources was requested."
    echo "Resource group: ${RESOURCE_GROUP}"
    exit "${exit_code}"
  fi

  if [[ "${RESOURCE_GROUP_CREATED}" -eq 1 ]]; then
    az group delete -n "${RESOURCE_GROUP}" --yes --no-wait >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

log() {
  printf '%s %s\n' "[$(date +%H:%M:%S)]" "$*"
}

require_number() {
  local name="$1"
  local value="$2"
  python3 - "$name" "$value" <<'PY'
import sys
name, value = sys.argv[1:]
try:
    float(value)
except ValueError:
    raise SystemExit(f"{name} must be numeric, got {value!r}")
PY
}

estimate_cost() {
  local hourly_rate="$1"
  local max_minutes="$2"
  python3 - "$hourly_rate" "$max_minutes" <<'PY'
import sys
rate = float(sys.argv[1])
minutes = float(sys.argv[2])
buffered = rate * (minutes / 60.0) * 1.15
print(f"{buffered:.4f}")
PY
}

extract_result_json() {
  local logfile="$1"
  python3 - "$logfile" <<'PY'
from pathlib import Path
import sys

log_path = Path(sys.argv[1])
for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
    marker = "CK_RESULT_JSON="
    if marker in line:
        print(line.split(marker, 1)[1].strip())
        raise SystemExit(0)
raise SystemExit(1)
PY
}

rand_password() {
  python3 - <<'PY'
import secrets
alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%^&*"
while True:
    value = ''.join(secrets.choice(alphabet) for _ in range(24))
    if (any(c.islower() for c in value) and any(c.isupper() for c in value)
            and any(c.isdigit() for c in value) and any(c in "!@#$%^&*" for c in value)):
        print(value)
        break
PY
}

sku_available_in_region() {
  local region="$1"
  local size="$2"
  local json
  json="$(az vm list-skus --location "${region}" --resource-type virtualMachines --query "[?name=='${size}']" -o json)"
  python3 - "${size}" <<'PY' <<<"${json}"
import json, sys
size = sys.argv[1]
items = json.load(sys.stdin)
if not items:
    raise SystemExit(1)
item = items[0]
restrictions = item.get("restrictions") or []
raise SystemExit(0 if not restrictions else 1)
PY
}

pick_region() {
  local csv_regions="$1"
  local size="$2"
  IFS=',' read -r -a regions <<<"${csv_regions}"
  for region in "${regions[@]}"; do
    region="${region// /}"
    [[ -z "${region}" ]] && continue
    if sku_available_in_region "${region}" "${size}"; then
      printf '%s\n' "${region}"
      return 0
    fi
  done
  return 1
}

ensure_resource_group() {
  local location="$1"
  az group create -n "${RESOURCE_GROUP}" -l "${location}" --tags purpose=corridorkey-installer-smoke run_id="${RUN_ID}" >/dev/null
  RESOURCE_GROUP_CREATED=1
}

render_template() {
  local template="$1"
  TEMPLATE_PATH="${template}" \
  REPO_SLUG="${REPO_SLUG}" \
  REPO_REF="${REPO_REF}" \
  WINDOWS_PYTHON="${WINDOWS_PYTHON}" \
  LINUX_PYTHON="${LINUX_PYTHON}" \
  python3 - <<'PY'
from pathlib import Path
import os

text = Path(os.environ["TEMPLATE_PATH"]).read_text(encoding="utf-8")
replacements = {
    "__REPO_SLUG__": os.environ["REPO_SLUG"],
    "__REPO_REF__": os.environ["REPO_REF"],
    "__WINDOWS_PYTHON__": os.environ["WINDOWS_PYTHON"],
    "__WINDOWS_PY_SHORT__": os.environ["WINDOWS_PYTHON"].replace(".", "")[:3],
    "__LINUX_PYTHON__": os.environ["LINUX_PYTHON"],
}
for old, new in replacements.items():
    text = text.replace(old, new)
print(text)
PY
}

run_windows_command() {
  local script_content="$1"
  local out_file="$2"
  timeout "${WINDOWS_MAX_MINUTES}m" az vm run-command invoke \
    -g "${RESOURCE_GROUP}" \
    -n "${WINDOWS_VM}" \
    --command-id RunPowerShellScript \
    --scripts "${script_content}" \
    --query "value[0].message" -o tsv | tee "${out_file}"
}

run_linux_command() {
  local script_content="$1"
  local out_file="$2"
  timeout "${LINUX_MAX_MINUTES}m" az vm run-command invoke \
    -g "${RESOURCE_GROUP}" \
    -n "${LINUX_VM}" \
    --command-id RunShellScript \
    --scripts "${script_content}" \
    --query "value[0].message" -o tsv | tee "${out_file}"
}

lookup_hourly_price() {
  local os_name="$1"
  local size="$2"
  local region="$3"
  local use_spot="$4"
  python3 - "$os_name" "$size" "$region" "$use_spot" <<'PY'
import json
import sys
import urllib.parse
import urllib.request

os_name, size, region, use_spot = sys.argv[1:]
flt = (
    "serviceName eq 'Virtual Machines' and "
    f"armSkuName eq '{size}' and "
    f"armRegionName eq '{region}' and "
    "priceType eq 'Consumption'"
)
base = "https://prices.azure.com/api/retail/prices"
url = base + "?" + urllib.parse.urlencode(
    {"api-version": "2023-01-01-preview", "$filter": flt}
)

items = []
while url:
    with urllib.request.urlopen(url) as response:
        payload = json.load(response)
    items.extend(payload.get("Items") or [])
    url = payload.get("NextPageLink")

def is_windows(item):
    return "windows" in (item.get("productName") or "").lower()

def is_spot(item):
    text = " ".join(
        filter(None, [item.get("skuName"), item.get("meterName"), item.get("productName")])
    ).lower()
    return "spot" in text or "low priority" in text

candidates = []
for item in items:
    if os_name == "windows" and not is_windows(item):
        continue
    if os_name == "linux" and is_windows(item):
        continue
    if use_spot == "1" and not is_spot(item):
        continue
    if use_spot != "1" and is_spot(item):
        continue
    candidates.append(item)

if not candidates:
    raise SystemExit(
        f"No retail price found for os={os_name}, size={size}, region={region}, spot={use_spot}"
    )

candidates.sort(key=lambda item: float(item.get("retailPrice") or 0))
print(f"{float(candidates[0]['retailPrice']):.6f}")
PY
}

wait_for_windows_gpu() {
  local attempt
  local probe='
$ErrorActionPreference = "Stop"
$smi = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
if (-not $smi) {
  $candidate = "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
  if (Test-Path $candidate) { $smi = $candidate }
}
if (-not $smi) { throw "nvidia-smi not found" }
& $smi
'
  for attempt in $(seq 1 20); do
    if az vm run-command invoke -g "${RESOURCE_GROUP}" -n "${WINDOWS_VM}" --command-id RunPowerShellScript --scripts "${probe}" --query "value[0].message" -o tsv >"${LOG_DIR}/windows-gpu-probe-${attempt}.log" 2>&1; then
      return 0
    fi
    sleep 30
  done
  return 1
}

wait_for_linux_gpu() {
  local attempt
  for attempt in $(seq 1 20); do
    if az vm run-command invoke -g "${RESOURCE_GROUP}" -n "${LINUX_VM}" --command-id RunShellScript --scripts "set -e; command -v nvidia-smi >/dev/null; nvidia-smi" --query "value[0].message" -o tsv >"${LOG_DIR}/linux-gpu-probe-${attempt}.log" 2>&1; then
      return 0
    fi
    sleep 20
  done
  return 1
}

create_windows_vm() {
  local admin_user="ckadmin"
  local admin_password
  admin_password="$(rand_password)"
  local -a extra_args=()
  if [[ "${USE_SPOT}" -eq 1 ]]; then
    extra_args+=(--priority Spot --max-price -1 --eviction-policy Delete)
  fi
  az vm create \
    -g "${RESOURCE_GROUP}" \
    -n "${WINDOWS_VM}" \
    --location "${WINDOWS_REGION}" \
    --image MicrosoftWindowsServer:WindowsServer:2022-datacenter-azure-edition:latest \
    --size "${WINDOWS_SIZE}" \
    --admin-username "${admin_user}" \
    --admin-password "${admin_password}" \
    --public-ip-address "" \
    --storage-sku StandardSSD_LRS \
    --tags purpose=corridorkey-installer-smoke lane=windows run_id="${RUN_ID}" \
    "${extra_args[@]}" >/dev/null
  WINDOWS_CREATED=1
}

create_linux_vm() {
  local admin_user="ckuser"
  local -a extra_args=()
  if [[ "${USE_SPOT}" -eq 1 ]]; then
    extra_args+=(--priority Spot --max-price -1 --eviction-policy Delete)
  fi
  az vm create \
    -g "${RESOURCE_GROUP}" \
    -n "${LINUX_VM}" \
    --location "${LINUX_REGION}" \
    --image Canonical:0001-com-ubuntu-server-focal:20_04-lts-gen2:latest \
    --size "${LINUX_SIZE}" \
    --admin-username "${admin_user}" \
    --generate-ssh-keys \
    --public-ip-address "" \
    --storage-sku StandardSSD_LRS \
    --tags purpose=corridorkey-installer-smoke lane=linux run_id="${RUN_ID}" \
    "${extra_args[@]}" >/dev/null
  LINUX_CREATED=1
}

install_windows_driver() {
  az vm extension set \
    -g "${RESOURCE_GROUP}" \
    --vm-name "${WINDOWS_VM}" \
    --publisher Microsoft.HpcCompute \
    --name NvidiaGpuDriverWindows >/dev/null
}

install_linux_driver() {
  az vm extension set \
    -g "${RESOURCE_GROUP}" \
    --vm-name "${LINUX_VM}" \
    --publisher Microsoft.HpcCompute \
    --name NvidiaGpuDriverLinux >/dev/null
}

require_number "max budget" "${MAX_BUDGET_USD}"
require_number "windows max minutes" "${WINDOWS_MAX_MINUTES}"
require_number "linux max minutes" "${LINUX_MAX_MINUTES}"

log "Logs will be written to ${LOG_DIR}"
log "Selecting regions"

if [[ "${RUN_WINDOWS}" -eq 1 ]]; then
  WINDOWS_REGION="$(pick_region "${WINDOWS_REGIONS}" "${WINDOWS_SIZE}")" || {
    echo "No unrestricted region found for Windows size ${WINDOWS_SIZE} in [${WINDOWS_REGIONS}]" >&2
    exit 1
  }
  log "Windows lane: ${WINDOWS_SIZE} in ${WINDOWS_REGION}"
  WINDOWS_RATE_USD="$(lookup_hourly_price windows "${WINDOWS_SIZE}" "${WINDOWS_REGION}" "${USE_SPOT}")"
  WINDOWS_ESTIMATED_USD="$(estimate_cost "${WINDOWS_RATE_USD}" "${WINDOWS_MAX_MINUTES}")"
fi

if [[ "${RUN_LINUX}" -eq 1 ]]; then
  LINUX_REGION="$(pick_region "${LINUX_REGIONS}" "${LINUX_SIZE}")" || {
    echo "No unrestricted region found for Linux size ${LINUX_SIZE} in [${LINUX_REGIONS}]" >&2
    exit 1
  }
  log "Linux lane: ${LINUX_SIZE} in ${LINUX_REGION}"
  LINUX_RATE_USD="$(lookup_hourly_price linux "${LINUX_SIZE}" "${LINUX_REGION}" "${USE_SPOT}")"
  LINUX_ESTIMATED_USD="$(estimate_cost "${LINUX_RATE_USD}" "${LINUX_MAX_MINUTES}")"
fi

TOTAL_ESTIMATED_USD="$(python3 - "${WINDOWS_ESTIMATED_USD:-0}" "${LINUX_ESTIMATED_USD:-0}" <<'PY'
import sys
print(f"{sum(float(arg or 0) for arg in sys.argv[1:]):.4f}")
PY
)"

log "Estimated max spend:"
if [[ "${RUN_WINDOWS}" -eq 1 ]]; then
  log "  Windows lane: ~$${WINDOWS_ESTIMATED_USD} max (${WINDOWS_RATE_USD}/hr x ${WINDOWS_MAX_MINUTES} min, incl. 15% buffer)"
fi
if [[ "${RUN_LINUX}" -eq 1 ]]; then
  log "  Linux lane:   ~$${LINUX_ESTIMATED_USD} max (${LINUX_RATE_USD}/hr x ${LINUX_MAX_MINUTES} min, incl. 15% buffer)"
fi
log "  Total:        ~$${TOTAL_ESTIMATED_USD} max (budget cap $${MAX_BUDGET_USD})"

python3 - "${TOTAL_ESTIMATED_USD}" "${MAX_BUDGET_USD}" <<'PY'
import sys
estimate = float(sys.argv[1])
budget = float(sys.argv[2])
if estimate > budget:
    raise SystemExit(f"Estimated max spend ${estimate:.2f} exceeds budget cap ${budget:.2f}")
PY

if [[ "${ESTIMATE_ONLY}" -eq 1 ]]; then
  FINAL_STATUS="estimate_only"
  log "Estimate-only mode requested; exiting before provisioning."
  exit 0
fi

ensure_resource_group "${WINDOWS_REGION:-${LINUX_REGION}}"

if [[ "${RUN_WINDOWS}" -eq 1 ]]; then
  log "Creating Windows GPU VM"
  FAILURE_CONTEXT="create_windows_vm"
  create_windows_vm
  log "Installing Windows NVIDIA driver extension"
  FAILURE_CONTEXT="install_windows_driver"
  install_windows_driver
  log "Waiting for Windows GPU to become ready"
  FAILURE_CONTEXT="wait_for_windows_gpu"
  wait_for_windows_gpu
  log "Running Windows installer smoke"
  FAILURE_CONTEXT="windows_installer_smoke"
  render_template "${ROOT_DIR}/scripts/azure/windows-installer-smoke.ps1.in" >"${LOG_DIR}/windows-installer-smoke.rendered.ps1"
  run_windows_command "$(cat "${LOG_DIR}/windows-installer-smoke.rendered.ps1")" "${LOG_DIR}/windows-installer-smoke.log"
fi

if [[ "${RUN_LINUX}" -eq 1 ]]; then
  log "Creating Linux GPU VM"
  FAILURE_CONTEXT="create_linux_vm"
  create_linux_vm
  log "Installing Linux NVIDIA driver extension"
  FAILURE_CONTEXT="install_linux_driver"
  install_linux_driver
  log "Waiting for Linux GPU to become ready"
  FAILURE_CONTEXT="wait_for_linux_gpu"
  wait_for_linux_gpu
  log "Running Linux installer smoke"
  FAILURE_CONTEXT="linux_installer_smoke"
  render_template "${ROOT_DIR}/scripts/azure/linux-installer-smoke.sh.in" >"${LOG_DIR}/linux-installer-smoke.rendered.sh"
  run_linux_command "$(cat "${LOG_DIR}/linux-installer-smoke.rendered.sh")" "${LOG_DIR}/linux-installer-smoke.log"
fi

cat >"${LOG_DIR}/summary.txt" <<EOF
run_id=${RUN_ID}
resource_group=${RESOURCE_GROUP}
repo=${REPO_SLUG}
ref=${REPO_REF}
windows_region=${WINDOWS_REGION}
windows_size=${WINDOWS_SIZE}
windows_python=${WINDOWS_PYTHON}
windows_hourly_rate_usd=${WINDOWS_RATE_USD}
windows_estimated_max_usd=${WINDOWS_ESTIMATED_USD}
windows_max_minutes=${WINDOWS_MAX_MINUTES}
linux_region=${LINUX_REGION}
linux_size=${LINUX_SIZE}
linux_python=${LINUX_PYTHON}
linux_hourly_rate_usd=${LINUX_RATE_USD}
linux_estimated_max_usd=${LINUX_ESTIMATED_USD}
linux_max_minutes=${LINUX_MAX_MINUTES}
estimated_total_max_usd=${TOTAL_ESTIMATED_USD}
max_budget_usd=${MAX_BUDGET_USD}
keep_resources=${KEEP_RESOURCES}
use_spot=${USE_SPOT}
EOF

log "Azure GPU installer smoke completed successfully"
FINAL_STATUS="success"
FAILURE_CONTEXT=""
