"""
This module handles flashing of a bitstream file to the FPGA.
"""

import os

CFG_FILE = "~/sft/share/trellis/misc/openocd/ecp5-evn.cfg"


def flash_ecp5(file_base_name):
    """Flashes a bitstream file to the ECP5 fpga. For best performance, use a file in ramdisk."""
    # os.system("openocd log_output /dev/null")
    os.system("openocd -f {0} -c \"transport select jtag; init; svf {1}.svf; exit\" >/dev/null 2>&1".format(CFG_FILE, file_base_name))

def flash_config_file(file_base_name):
    os.system("ecppack --svf {0}.svf {0}.config {0}.bit".format(file_base_name))
    flash_ecp5(file_base_name)

