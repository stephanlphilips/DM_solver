import numpy as np
cimport numpy as np
cimport cython

from cython.operator cimport dereference as deref

from libcpp cimport bool

cdef extern from "armadillo" namespace "arma" nogil:
    # matrix class (double)
    cdef cppclass Mat[T]:
        Mat(T * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict) nogil
        Mat(T * aux_mem, int n_rows, int n_cols) nogil
        Mat(int n_rows, int n_cols) nogil
        Mat() nogil
        void raw_print() nogil


        # attributes
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        int n_nonzero

        # fuctions
        Mat[T] i() nogil #inverse
        Mat[T] t() nogil #transpose
        vec diag() nogil
        vec diag(int) nogil
        fill(double) nogil
        void raw_print(char*) nogil
        void raw_print() nogil
        vec unsafe_col(int) nogil
        vec col(int) nogil
        #print(char)
        #management
        Mat[T] reshape(int, int) nogil
        Mat[T] resize(int, int) nogil
        double * memptr() nogil
        # opperators
        double& operator[](int) nogil
        double& operator[](int,int) nogil
        double& at(int,int) nogil
        double& at(int) nogil
        Mat[T] operator*(Mat[T]) nogil
        Mat[T] operator%(Mat[T]) nogil
        vec operator*(vec) nogil
        Mat[T] operator+(Mat[T]) nogil
        Mat[T] operator-(Mat[T]) nogil
        Mat[T] operator*(double) nogil
        Mat[T] operator-(double) nogil
        Mat[T] operator+(double) nogil
        Mat[T] operator/(double) nogil

    cdef cppclass cx_mat:
        cx_mat(Mat[double], Mat[double]) nogil
        cx_mat() nogil

        cx_mat t() nogil

    cdef cppclass cube:
        cube(double * aux_mem, int n_rows, int n_cols, int n_slices, bool copy_aux_mem, bool strict) nogil
        cube(double * aux_mem, int n_rows, int n_cols, int n_slices) nogil
        cube(int, int, int) nogil
        cube() nogil
        
        #attributes
        int n_rows
        int n_cols
        int n_elem
        int n_elem_slices
        int n_slices
        int n_nonzero
        double * memptr() nogil
        void raw_print(char*) nogil
        void raw_print() nogil
        
    cdef cppclass cx_cube:
        cx_cube(cube, cube) nogil
        cx_cube() nogil

    cdef cppclass cx_vec:
        cx_vec() nogil
        cx_vec(vec,vec) nogil
    # vector class (double)
    cdef cppclass vec:
        cppclass iterator:
            double& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(iterator)
            bint operator!=(iterator)
            bint operator<(iterator)
            bint operator>(iterator)
            bint operator<=(iterator)
            bint operator>=(iterator)
        cppclass reverse_iterator:
            double& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
            bint operator==(reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator>=(reverse_iterator)
        vec(double * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict) nogil
        vec(double * aux_mem, int number_of_elements) nogil
        vec(int) nogil
        vec() nogil
        # attributes
        int n_elem
        # opperators
        double& operator[](int)
        double& at(int)
        vec operator%(vec)
        vec operator+(vec)
        vec operator/(vec)
        vec operator*(Mat[double])
        vec operator*(double)
        vec operator-(double)
        vec operator+(double)
        vec operator/(double)
        iterator begin()
        iterator end()
        reverse_iterator rbegin()
        reverse_iterator rend()


        # functions
        double * memptr()
        void raw_print(char*) nogil
        void raw_print() nogil
        

    # Armadillo Linear Algebra tools
    cdef bool chol(Mat R, Mat X) nogil # preallocated result
    cdef Mat chol(Mat X) nogil # new result
    cdef bool inv(Mat R, Mat X) nogil
    cdef Mat inv(Mat X) nogil
    cdef bool solve(vec x, Mat A, vec b) nogil
    cdef vec solve(Mat A, vec b) nogil
    cdef bool solve(Mat X, Mat A, Mat B) nogil
    cdef Mat solve(Mat A, Mat B) nogil
    cdef bool eig_sym(vec eigval, Mat eigvec, Mat B) nogil
    cdef bool svd(Mat U, vec s, Mat V, Mat X, method) nogil
    cdef bool lu(Mat L, Mat U, Mat P, Mat X) nogil
    cdef bool lu(Mat L, Mat U, Mat X) nogil
    cdef Mat pinv(Mat A) nogil
    cdef bool pinv(Mat B, Mat A) nogil
    cdef bool qr(Mat Q, Mat R, Mat X) nogil
    cdef float dot(vec a, vec b) nogil
    cdef Mat arma_cov "cov"(Mat X) nogil
    cdef vec arma_mean "mean"(Mat X, int dim) nogil
    cdef Mat arma_var "var"(Mat X, int norm_type, int dim) nogil
    cdef Mat[double] real(cx_mat X) nogil
    cdef Mat[double] imag(cx_mat X) nogil
    cdef cube real(cx_cube X) nogil
    cdef cube imag(cx_cube X) nogil


# Use tempate based method // use fused data type to match both template options for numpy and armadillo
ctypedef fused matrixdtype:
    cython.int
    cython.double
    cython.float

cdef vec np2vec(np.ndarray[np.double_t, ndim=1] x):
    if not (x.flags.f_contiguous or x.flags.owndata):
        x = x.copy()
    cdef vec *ar_p = new vec(<double*> x.data, x.shape[0], False, True)
    cdef vec ar = deref(ar_p)
    del ar_p
    return ar

cdef cx_vec np2cx_vec(np.ndarray[np.complex_t, ndim=1] X):
    cdef cx_vec myvec = cx_vec(np2vec(np.ascontiguousarray(np.real(X))),np2vec((np.ascontiguousarray(np.imag(X)))))
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
# cdef vec * mat_col_view(Mat[T] * x, int col) nogil:
#     cdef vec * ar_p = new vec(x.memptr()+x.n_rows*col, x.n_rows, False, True)
#     return ar_p

# cdef vec mat_col_view_d(Mat[T] * x, int col) nogil:
#     cdef vec * ar_p = mat_col_view(x, col)
#     cdef vec ar = deref(ar_p)
#     del ar_p
#     return ar

# cdef Mat[T] * cube_slice_view(cube * x, int slice) nogil:
#     cdef Mat[T] *ar_p = new Mat[T](x.memptr() + x.n_rows*x.n_cols*slice,
#                            x.n_rows, x.n_cols, False, True)
#     return ar_p

# cdef Mat[T] cube_slice_view_d(cube * x, int slice) nogil:
#     cdef Mat[T] * ar_p = cube_slice_view(x, slice)
#     cdef Mat[T] ar = deref(ar_p)
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
cdef np.ndarray[np.double_t, ndim=1] vec2numpy(const vec & X, np.ndarray[np.double_t, ndim=1] D):
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
#     cdef Mat[T] *aX = new Mat[T](<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef Mat[T] *aR  = new Mat[T](<double*> R.data, R.shape[0], R.shape[1], False, True)
    
#     chol(deref(aR), deref(aX))
    
#     return R

# cpdef np_inv(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     # wrap them up in armidillo arrays
#     cdef Mat[T] *aX = new Mat[T](<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef Mat[T] *aR  = new Mat[T](<double*> R.data, R.shape[0], R.shape[1], False, True)
    
#     inv(deref(aR), deref(aX))
    
#     return R


# def np_eig_sym(np.ndarray[np.double_t, ndim=2] X):
#     # initialize result numpy array
#     cdef np.ndarray[np.double_t, ndim=2] R = \
#          np.empty((X.shape[0], X.shape[1]), dtype=np.double, order="F")
#     cdef np.ndarray[np.double_t, ndim=1] v = \
#          np.empty(X.shape[0], dtype=np.double)
#     # wrap them up in armidillo arrays
#     cdef Mat[T] *aX = new Mat[T](<double*> X.data, X.shape[0], X.shape[1], False, True)
#     cdef Mat[T] *aR  = new Mat[T](<double*> R.data, R.shape[0], R.shape[1], False, True)
#     cdef vec *av = new vec(<double*> v.data, v.shape[0], False, True)

#     eig_sym(deref(av), deref(aR), deref(aX))

#     return [v, R]

