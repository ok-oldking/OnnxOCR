import platform
import subprocess
import re

def check_npu_driver_valid(logger):
    npu_driver_valid = False
    sys_plat = platform.system()
    try:
        def parse_version(v):
            return tuple(map(int, re.findall(r'\d+', v)))
        
        if sys_plat == "Windows":
            cmd = ['powershell', '-NoProfile', '-Command', "Get-WmiObject Win32_PnPSignedDriver | Where-Object { $PSItem.DeviceName -match '\\bNPU\\b' } | Select-Object -ExpandProperty DriverVersion"]
            creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=creationflags)
            output = result.stdout.strip()
            if output:
                version_str = output.split('\n')[0].strip()
                if parse_version(version_str) > parse_version("32.0.100.4181"):
                    npu_driver_valid = True
                    logger.info(f"NPU driver version {version_str} is > 32.0.100.4181")
                else:
                    logger.warning(f"NPU driver version {version_str} is <= 32.0.100.4181, use cpu instead. Please update driver.")
            else:
                logger.warning("Could not detect NPU driver version on Windows, use cpu instead.")
        elif sys_plat == "Linux":
            cmd = ['dpkg-query', '-W', '-f=${Version}', 'intel-driver-compiler-npu']
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout.strip()
            if output:
                version_str = output.split('\n')[0].strip()
                if parse_version(version_str) >= parse_version("1.30.0"):
                    npu_driver_valid = True
                    logger.info(f"NPU driver version {version_str} is >= 1.30.0")
                else:
                    logger.warning(f"NPU driver version {version_str} is < 1.30.0, use cpu instead. Please update driver.")
            else:
                logger.warning("Could not detect NPU driver version on Linux, use cpu instead.")
        else:
            npu_driver_valid = True
    except Exception as e:
        logger.warning(f"Failed to check NPU driver version: {e}, use cpu instead.")
    
    return npu_driver_valid
