import unittest
from unittest.mock import MagicMock, patch

from onnxocr.check_npu import check_npu_driver_valid


class TestCheckNpuDriverValid(unittest.TestCase):
    def setUp(self):
        self.logger = MagicMock()

    @patch("onnxocr.check_npu.platform.system", return_value="Windows")
    @patch("onnxocr.check_npu.subprocess.run")
    def test_windows_valid_above_threshold(self, mock_run, _plat):
        mock_run.return_value = MagicMock(stdout="32.0.100.5000\n", returncode=0)
        self.assertTrue(check_npu_driver_valid(self.logger))

    @patch("onnxocr.check_npu.platform.system", return_value="Windows")
    @patch("onnxocr.check_npu.subprocess.run")
    def test_windows_invalid_below_threshold(self, mock_run, _plat):
        mock_run.return_value = MagicMock(stdout="31.0.100.0\n", returncode=0)
        self.assertFalse(check_npu_driver_valid(self.logger))


if __name__ == "__main__":
    unittest.main()
