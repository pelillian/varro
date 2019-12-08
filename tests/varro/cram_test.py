import time

'''
def printer(frame, event, arg):
  if event in ['exception', 'opcode']:
      #print(arg[0])
      print(frame, event, arg)
  return printer

sys.settrace(printer)
'''

from varro.cython.fast_cram import load_cram_fast
import numpy as np
import pytrellis
from varro.misc.variables import PRJTRELLIS_DATABASE, CHIP_NAME, CHIP_COMMENT

cram = pytrellis.CRAM(13294, 1136)

FPGA_BITSTREAM_SHAPE = (13294, 1136)
data = np.zeros(FPGA_BITSTREAM_SHAPE, dtype=np.int8)

start_time = time.time()
load_cram_fast(cram, data)
end_time = time.time()

print(end_time - start_time)
