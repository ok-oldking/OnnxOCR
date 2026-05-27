from types import SimpleNamespace

from onnxocr.check_npu import check_npu_driver_valid


class Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


def run_windows_check(monkeypatch, stdout):
    logger = Logger()

    def fake_run(cmd, **kwargs):
        assert cmd[:3] == ["powershell", "-NoProfile", "-Command"]
        assert "Manufacturer -imatch 'Intel'" in cmd[3]
        assert "DeviceName -imatch" in cmd[3]
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr("onnxocr.check_npu.platform.system", lambda: "Windows")
    monkeypatch.setattr("onnxocr.check_npu.subprocess.run", fake_run)

    return check_npu_driver_valid(logger), logger


def test_windows_accepts_current_intel_npu_driver(monkeypatch):
    valid, logger = run_windows_check(monkeypatch, "32.0.100.4724\n")

    assert valid is True
    assert (
        "info",
        "NPU driver version 32.0.100.4724 is > 32.0.100.4181",
    ) in logger.messages


def test_windows_rejects_old_intel_npu_driver(monkeypatch):
    valid, logger = run_windows_check(monkeypatch, "32.0.100.4181\n")

    assert valid is False
    assert any(
        level == "warning" and "Please update driver" in message
        for level, message in logger.messages
    )


def test_windows_rejects_missing_driver_version(monkeypatch):
    valid, logger = run_windows_check(monkeypatch, "")

    assert valid is False
    assert (
        "warning",
        "Could not detect NPU driver version on Windows, use cpu instead.",
    ) in logger.messages
