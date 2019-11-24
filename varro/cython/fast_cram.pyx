import numpy as np
cimport numpy as np

import pytrellis

def load_cram_fast(cram, np.ndarray config_data):
    cdef int frames = cram.frames()
    cdef int bits_per_frame = cram.bits()
    
    for i in range(frames):
        for j in range(bits_per_frame):
            cram.set_bit(i, j, bool(config_data[i,j]))


