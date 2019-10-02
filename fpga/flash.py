
import os


"""
This module handles flashing of a bitstream file to the FPGA.
"""

PROJ = "blinky"
TRELLIS = "~/sft/share/trellis"

def flash_ecp5(file):
    """Flashes a bitstream file to the ECP5 fpga. For best performance, use a file in ramdisk."""
    os.sys("openocd -f {0}/misc/openocd/ecp5-evn.cfc -c \"transport select jtag; init; svf {1}.svf; exit".format(TRELLIS, PROJ)

