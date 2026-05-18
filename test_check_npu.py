import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from onnxocr.check_npu import check_npu_driver_valid


class _Logger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))


class CheckNpuDriverValidTests(unittest.TestCase):
    def _run_windows_check(self, stdout):
        logger = _Logger()

        def fake_run(cmd, **kwargs):
            self.assertEqual(os.path.basename(cmd[0]).lower(), "powershell.exe")
            self.assertEqual(cmd[1:3], ["-NoProfile", "-Command"])
            self.assertIn("Manufacturer -imatch 'Intel'", cmd[3])
            self.assertIn("DeviceName -imatch", cmd[3])
            return SimpleNamespace(stdout=stdout)

        with patch("onnxocr.check_npu.platform.system", return_value="Windows"), patch(
            "onnxocr.check_npu.subprocess.run", side_effect=fake_run
        ):
            return check_npu_driver_valid(logger), logger

    def test_windows_accepts_current_intel_npu_driver(self):
        valid, logger = self._run_windows_check("32.0.100.4724\n")

        self.assertTrue(valid)
        self.assertIn(("info", "NPU driver version 32.0.100.4724 is > 32.0.100.4181"), logger.messages)

    def test_windows_rejects_old_intel_npu_driver(self):
        valid, logger = self._run_windows_check("32.0.100.4181\n")

        self.assertFalse(valid)
        self.assertTrue(any(level == "warning" and "Please update driver" in message for level, message in logger.messages))

    def test_windows_rejects_missing_driver_version(self):
        valid, logger = self._run_windows_check("")

        self.assertFalse(valid)
        self.assertIn(("warning", "Could not detect NPU driver version on Windows, use cpu instead."), logger.messages)


if __name__ == "__main__":
    unittest.main()
