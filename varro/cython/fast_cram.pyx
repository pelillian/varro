import cython
cimport numpy as npx
import numpy as np
import pytrellis

DTYPE = np.bool
ctypedef npx.int8_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def load_cram_fast(cram, npx.ndarray[DTYPE_t, ndim=2] config_data):
    cdef int frames = cram.frames()
    cdef int bits_per_frame = cram.bits()
    
    for i in range(frames):
        for j in range(bits_per_frame):
            cram.set_bit(i, j, bool(config_data[i,j]))


