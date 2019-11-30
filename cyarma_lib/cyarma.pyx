import numpy as np
cimport numpy as np

import cython
from cython.operator cimport dereference as deref
from cyarma_lib.cyarma cimport Mat, cube, cx_mat, cx_cube, Col, cx_vec, real, imag, matrixdtype, vector_type
from libcpp cimport bool

cdef Col[vector_type] np2vec(np.ndarray[vector_type, ndim=1] x):
    if not (x.flags.f_contiguous or x.flags.owndata):
        x = x.copy()
    cdef Col[vector_type] *ar_p = new Col[vector_type](<vector_type*> x.data, x.shape[0], False, True)
    cdef Col[vector_type] ar = deref(ar_p)
    del ar_p
    return ar

cdef cx_vec np2cx_vec(np.ndarray[np.complex_t, ndim=1] X):
    cdef np.ndarray[dtype=double, ndim = 1] re = np.ascontiguousarray(np.real(X))
    cdef np.ndarray[dtype=double, ndim = 1] im = np.ascontiguousarray(np.imag(X))
    cdef cx_vec myvec = cx_vec(np2vec(re),np2vec(im))
    return myvec;

cdef Mat[matrixdtype] np2arma(np.ndarray[matrixdtype,ndim=2] X):
    if not (X.flags.f_contiguous or X.flags.owndata):
        X = X.copy(order="F")
    cdef Mat[matrixdtype] *aR_p  = new Mat[matrixdtype](<matrixdtype *> X.data, X.shape[1], X.shape[0], True, False)
    cdef Mat[matrixdtype] aR = deref(aR_p)
    return aR.t()



cdef cx_mat np2cx_mat(np.ndarray[np.complex_t, ndim=2] X):
    cdef np.ndarray[dtype=double, ndim = 2] re = np.real(X)
    cdef np.ndarray[dtype=double, ndim = 2] im = np.imag(X)
    cdef cx_mat mymat = cx_mat(np2arma(re) ,np2arma(im))
    return mymat.t();

cdef cube np2cube(np.ndarray[np.double_t, ndim=3] X):
    cdef cube *aR_p
    if not X.flags.c_contiguous:
        raise ValueError("For Cube, numpy array must be C contiguous")
    aR_p  = new cube(<double*> X.data, X.shape[2], X.shape[1], X.shape[0], False, True)

    cdef cube aR = deref(aR_p)
    del aR_p
    return aR

cdef cx_cube np2cx_cube(np.ndarray[np.complex_t, ndim=3] X):
    cdef cx_cube mycube = cx_cube(np2cube(np.ascontiguousarray(np.real(X))),np2cube((np.ascontiguousarray(np.imag(X)))))
    return mycube;

    

# #### Get subviews #####
# cdef vec * mat_col_view(Mat * x, int col) nogil:
#     cdef vec * ar_p = new vec(x.memptr()+x.n_rows*col, x.n_rows, False, True)
#     return ar_p

# cdef vec mat_col_view_d(Mat * x, int col) nogil:
#     cdef vec * ar_p = mat_col_view(x, col)
#     cdef vec ar = deref(ar_p)
#     del ar_p
#     return ar

# cdef Mat * cube_slice_view(cube * x, int slice) nogil:
#     cdef Mat *ar_p = new Mat(x.memptr() + x.n_rows*x.n_cols*slice,
#                            x.n_rows, x.n_cols, False, True)
#     return ar_p

# cdef Mat cube_slice_view_d(cube * x, int slice) nogil:
#     cdef Mat * ar_p = cube_slice_view(x, slice)
#     cdef Mat ar = deref(ar_p)
#     del ar_p
#     return ar



##### Converting back to python arrays, must pass preallocated memory or None
# all data will be copied since numpy doesn't own the data and can't clean up
# otherwise. Maybe this can be improved. #######
@cython.boundscheck(False)
cdef np.ndarray[np.complex_t, ndim=3] cube2np(const cube & X, np.ndarray[np.double_t, ndim=3] D):
    cdef const double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty((X.n_rows, X.n_cols,X.n_slices), dtype=np.double, order="F")
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_rows*X.n_cols*X.n_slices):
        Dptr[i] = Xptr[i]
    D = np.swapaxes(D,0,2)
    return np.swapaxes(D,1,2)

@cython.boundscheck(False)
cdef np.ndarray[np.complex_t, ndim=3] cx_cube2np(const cx_cube & X, np.ndarray[np.double_t, ndim=3] D):
    cdef cube real_part = real(X)
    cdef cube imag_part = imag(X)
    
    cdef np.ndarray[np.double_t, ndim=3] my_numpy_real= None
    cdef np.ndarray[np.double_t, ndim=3] my_numpy_imag= None

    my_numpy_real = cube2np(real_part, my_numpy_real)
    my_numpy_imag = cube2np(imag_part, my_numpy_imag)

    return my_numpy_real + 1j*my_numpy_imag

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=2] mat2np(const Mat[double] & X, np.ndarray[np.double_t, ndim=2] D):
    cdef const double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty((X.n_rows, X.n_cols), dtype=np.double, order="F")
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_rows*X.n_cols):
        Dptr[i] = Xptr[i]
    return D

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=2] cx_mat2np(const cx_mat & X, np.ndarray[np.double_t, ndim=2] D):
    cdef Mat[double] real_part = real(X)
    cdef Mat[double] imag_part = imag(X)

    cdef np.ndarray[np.double_t, ndim=2] my_numpy_real= None
    cdef np.ndarray[np.double_t, ndim=2] my_numpy_imag= None

    my_numpy_real = mat2np(real_part, my_numpy_real)
    my_numpy_imag = mat2np(imag_part, my_numpy_imag)

    return my_numpy_real + 1j*my_numpy_imag

@cython.boundscheck(False)
cdef np.ndarray[np.double_t, ndim=1] vec2np(const Col[double] & X, np.ndarray[np.double_t, ndim=1] D):
    cdef const double * Xptr = X.memptr()
    
    if D is None:
        D = np.empty(X.n_elem, dtype=np.double)
    cdef double * Dptr = <double*> D.data
    for i in range(X.n_elem):
        Dptr[i] = Xptr[i]
    return D

# ## A few wrappers for much needed numpy linalg functionality using armadillo
# cpdef np_chol(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     # wrap them up in armidillo arrays
#     cdef Mat *aX = new Mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef Mat *aR  = new Mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    
#     chol(deref(aR), deref(aX))
    
#     return R

# cpdef np_inv(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     # wrap them up in armidillo arrays
#     cdef Mat *aX = new Mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef Mat *aR  = new Mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
    
#     inv(deref(aR), deref(aX))
    
#     return R


# def np_eig_sym(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     cdef np.ndarray[np.double_t, ndim=1] v = \
#          np.empty(X.shape[0], dtype=np.double)
#     # wrap them up in armidillo arrays
#     cdef Mat *aX = new Mat(<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef Mat *aR  = new Mat(<double*> R.data, R.shape[0], R.shape[1], False, True)
#     cdef vec *av = new vec(<double*> v.data, v.shape[0], False, True)

#     eig_sym(deref(av), deref(aR), deref(aX))

#     return [v, R]

