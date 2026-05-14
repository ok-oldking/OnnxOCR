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
            # Intel NPU only: require Intel OEM, but allow the different NPU
            # device names Windows exposes across driver/platform versions.
            ps = (
                "Get-WmiObject Win32_PnPSignedDriver | Where-Object { "
                "$_.Manufacturer -imatch 'Intel' -and $_.DeviceName -and "
                "$_.DeviceName -imatch '(\\bNPU\\b|AI Boost)' "
                "} | Select-Object -First 1 -ExpandProperty DriverVersion"
            )
            cmd = ['powershell', '-NoProfile', '-Command', ps]
            creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=creationflags)
            output = result.stdout.strip()
            if output:
                version_str = output.split('\n')[0].strip()
                if not parse_version(version_str):
                    logger.warning("Could not parse NPU driver version on Windows, use cpu instead.")
                elif parse_version(version_str) > parse_version("32.0.100.4181"):
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
