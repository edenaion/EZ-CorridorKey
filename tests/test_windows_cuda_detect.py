from scripts.detect_windows_torch_index import (
    CPU_URL,
    CU126_URL,
    CU128_URL,
    CU130_URL,
    choose_index_url,
    detect,
    find_nvidia_smi,
    parse_cuda_line,
    parse_cuda_version,
    parse_driver_version,
)


def test_windows_installers_use_shared_cuda_helper():
    text = open("1-install.bat", encoding="utf-8", errors="replace").read()
    assert ".venv\\Scripts\\python.exe scripts\\detect_windows_torch_index.py --format env" in text
    assert "findstr" not in text.lower()


def test_windows_installers_do_not_echo_cuda_line_unsafely():
    text = open("1-install.bat", encoding="utf-8", errors="replace").read()
    assert "echo   !CUDA_LINE!" not in text
    assert "echo(  !CUDA_LINE!" in text


def test_windows_installer_uses_managed_python_and_runtime_verification():
    text = open("1-install.bat", encoding="utf-8", errors="replace").read()
    assert "uv python install --managed-python !PYTHON_SPEC!" in text
    assert "uv python find --managed-python !PYTHON_SPEC!" in text
    assert "scripts\\verify_torch_runtime.py" in text


def test_parse_versions_from_standard_nvidia_smi_header():
    text = "| NVIDIA-SMI 581.57     Driver Version: 581.57     CUDA Version: 13.0     |"
    assert parse_driver_version(text) == "581.57"
    assert parse_cuda_line(text) == text
    assert parse_cuda_version(text) == "13.0"


def test_parse_cuda_version_from_non_english_cuda_line():
    text = "| NVIDIA-SMI 581.57     Version del controlador: 581.57     Version de CUDA: 12.3     |"
    assert parse_cuda_version(text) == "12.3"


def test_choose_index_url_maps_supported_cuda_ranges():
    assert choose_index_url("13.0")[0] == CU130_URL
    assert choose_index_url("12.8")[0] == CU128_URL
    assert choose_index_url("12.1")[0] == CU126_URL
    assert choose_index_url("12.0")[0] == CU126_URL


def test_choose_index_url_falls_back_for_unsupported_cuda():
    index_url, wheel_label, note = choose_index_url("11.8")
    assert index_url == CPU_URL
    assert wheel_label == "CPU-only PyTorch"
    assert "11.8" in note


def test_find_nvidia_smi_falls_back_to_standard_install_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("CORRIDORKEY_MOCK_NVIDIA_SMI_FILE", raising=False)
    monkeypatch.delenv("CORRIDORKEY_NVIDIA_SMI_PATH", raising=False)
    monkeypatch.setattr("scripts.detect_windows_torch_index.shutil.which", lambda _: None)
    program_files = tmp_path / "Program Files"
    smi = program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
    smi.parent.mkdir(parents=True)
    smi.write_text("", encoding="utf-8")
    monkeypatch.setenv("ProgramW6432", str(program_files))
    monkeypatch.setenv("ProgramFiles", str(program_files))
    assert find_nvidia_smi() == str(smi)


def test_detect_uses_mock_output_file(monkeypatch, tmp_path):
    mock = tmp_path / "nvidia_smi.txt"
    mock.write_text(
        "| NVIDIA-SMI 581.57     Driver Version: 581.57     CUDA Version: 12.8     |\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CORRIDORKEY_MOCK_NVIDIA_SMI_FILE", str(mock))
    monkeypatch.delenv("CORRIDORKEY_NVIDIA_SMI_PATH", raising=False)
    result = detect()
    assert result["INDEX_URL"] == CU128_URL
    assert result["DRIVER"] == "581.57"
    assert result["CUDA_VERSION"] == "12.8"


def test_detect_without_nvidia_smi_returns_cpu(monkeypatch):
    monkeypatch.delenv("CORRIDORKEY_MOCK_NVIDIA_SMI_FILE", raising=False)
    monkeypatch.delenv("CORRIDORKEY_NVIDIA_SMI_PATH", raising=False)
    monkeypatch.setattr("scripts.detect_windows_torch_index.shutil.which", lambda _: None)
    monkeypatch.setattr("scripts.detect_windows_torch_index._standard_nvidia_smi_paths", lambda: [])
    result = detect()
    assert result["INDEX_URL"] == CPU_URL
    assert result["CUDA_DETECT_MODE"] == "cpu"
