import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _make_cube(
    np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y,
    np.ndarray[np.int64_t, ndim=1] cidxs,
    int nx, int ny, int nc,
    np.float64_t dx, np.float64_t dy,
    double xmin, double ymin,
):
    cdef int i, ix, iy, ic, nnx, nny
    cdef np.ndarray[np.float64_t, ndim=3] data
    cdef int nevent = x.shape[0]

    data = np.zeros((nc, ny, nx), dtype=np.float64)
        
    for i in range(nevent):
        ix = int((x[i] - xmin) / dx)
        iy = int((y[i] - ymin) / dy)
        data[cidxs[i],iy,ix] += 1

    return data
