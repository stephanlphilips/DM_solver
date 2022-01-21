from libcpp cimport bool
cimport numpy as np
cimport cython


ctypedef fused matrixdtype:
    cython.int
    cython.double
    cython.float

ctypedef fused vector_type:
    cython.int
    cython.double
    cython.float

cdef extern from "armadillo" namespace "arma" nogil:
    cdef cppclass Col[T]:
        Col(T * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict) nogil
        Col(T * aux_mem, int number_of_elements) nogil
        Col(int) nogil
        Col() nogil

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
        # attributes
        int n_elem
        # opperators
        double& operator[](int)
        double& at(int)
        Col operator%(Col)
        Col operator+(Col)
        Col operator/(Col)
        Col operator*(Mat)
        Col operator*(double)
        Col operator-(double)
        Col operator+(double)
        Col operator/(double)
        iterator begin()
        iterator end()
        reverse_iterator rbegin()
        reverse_iterator rend()

        # functions
        double * memptr()
        void raw_print(char*) nogil
        void raw_print() nogil

    cdef cppclass vec:
        vec(double * aux_mem, int number_of_elements, bool copy_aux_mem, bool strict) nogil
        vec(double * aux_mem, int number_of_elements) nogil
        vec(int) nogil
        vec() nogil

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
        # attributes
        int n_elem
        # opperators
        double& operator[](int)
        double& at(int)
        vec operator%(vec)
        vec operator+(vec)
        vec operator/(vec)
        vec operator*(Mat)
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

    cdef cppclass Mat[T]:
        Mat(T * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict) nogil
        Mat(T * aux_mem, int n_rows, int n_cols) nogil
        Mat(int n_rows, int n_cols) nogil
        Mat() nogil
        cx_Mat(Mat real_Mat, Mat im_Mat) nogil
        # attributes
        int n_rows
        int n_cols
        int n_elem
        int n_slices
        int n_nonzero
        # fuctions
        Mat i() nogil #inverse
        Mat t() nogil #transpose
        Col diag() nogil
        Col diag(int) nogil
        fill(double) nogil
        void raw_print(char*) nogil
        void raw_print() nogil
        Col unsafe_col(int) nogil
        Col col(int) nogil
        #management
        Mat reshape(int, int) nogil
        Mat resize(int, int) nogil
        double * memptr() nogil
        # opperators
        double& operator[](int) nogil
        double& operator[](int,int) nogil
        double& at(int,int) nogil
        double& at(int) nogil

        Mat operator*(Mat) nogil
        Mat operator%(Mat) nogil
        Col operator*(Col) nogil
        Mat operator+(Mat) nogil
        Mat operator-(Mat) nogil
        Mat operator*(double) nogil
        Mat operator-(double) nogil
        Mat operator+(double) nogil
        Mat operator/(double) nogil
        #etc

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
    
    cdef cppclass cx_mat:
        cx_mat(Mat[double], Mat[double]) nogil
        cx_mat() nogil

        cx_mat t() nogil

    cdef cppclass cx_vec:
        cx_vec(Col[double], Col[double]) nogil
        cx_vec() nogil

        cx_vec t() nogil

    cdef cppclass cx_cube:
        cx_cube(cube real_cube, cube im_cube) nogil
        cx_cube() nogil

    # Armadillo Linear Algebra tools
    cdef bool chol(Mat R, Mat X) nogil # preallocated result
    cdef Mat chol(Mat X) nogil # new result
    cdef bool inv(Mat R, Mat X) nogil
    cdef Mat inv(Mat X) nogil
    cdef bool solve(Col x, Mat A, Col b) nogil
    cdef Col solve(Mat A, Col b) nogil
    cdef bool solve(Mat X, Mat A, Mat B) nogil
    cdef Mat solve(Mat A, Mat B) nogil
    cdef bool eig_sym(Col eigval, Mat eigvec, Mat B) nogil
    cdef bool svd(Mat U, Col s, Mat V, Mat X, method) nogil
    cdef bool lu(Mat L, Mat U, Mat P, Mat X) nogil
    cdef bool lu(Mat L, Mat U, Mat X) nogil
    cdef Mat pinv(Mat A) nogil
    cdef bool pinv(Mat B, Mat A) nogil
    cdef bool qr(Mat Q, Mat R, Mat X) nogil
    cdef float dot(Col a, Col b) nogil
    cdef Mat arma_cov "cov"(Mat X) nogil
    cdef Col arma_mean "mean"(Mat X, int dim) nogil
    cdef Mat arma_var "var"(Mat X, int norm_type, int dim) nogil
    cdef Mat[double] real(cx_mat X) nogil
    cdef Mat[double] imag(cx_mat X) nogil
    cdef cube real(cx_cube X) nogil
    cdef cube imag(cx_cube X) nogil

# cdef Mat * numpy_to_Mat(np.ndarray[np.double_t, ndim=2] X)

# cdef Mat numpy_to_Mat_d(np.ndarray[np.double_t, ndim=2] X)

# # cdef cx_Mat numpy_to_cx_Mat_d(np.ndarray[np.complex_t, ndim=2] X)

# cdef cube * numpy_to_cube(np.ndarray[np.double_t, ndim=3] X)

# cdef cube numpy_to_cube_d(np.ndarray[np.double_t, ndim=3] X)

# # cdef cx_cube numpy_to_cxcube_d(np.ndarray[np.complex_t, ndim=3] X)

# cdef Col * numpy_to_vec(np.ndarray[np.double_t, ndim=1] x)

# cdef Col numpy_to_vec_d(np.ndarray[np.double_t, ndim=1] x)

# cdef Col * Mat_col_view(Mat * x, int col) nogil

# cdef Col Mat_col_view_d(Mat * x, int col) nogil

# cdef Mat * cube_slice_view(cube * x, int slice) nogil

# cdef Mat cube_slice_view_d(cube * x, int slice) nogil

# cdef np.ndarray[np.double_t, ndim=2] Mat_to_numpy(const Mat & X, np.ndarray[np.double_t, ndim=2] D)

# cdef np.ndarray[np.double_t, ndim=1] vec_to_numpy(const Col & X, np.ndarray[np.double_t, ndim=1] D)


cdef Col[vector_type] np2vec(np.ndarray[vector_type,ndim=1] x)

cdef cx_vec np2cx_vec(np.ndarray[np.complex_t, ndim=1] X)

cdef Mat[matrixdtype] np2arma(np.ndarray[matrixdtype,ndim=2] X)

cdef cx_mat np2cx_mat(np.ndarray[np.complex_t, ndim=2] X)

# cdef Mat[double] np2mat(np.ndarray[np.double_t, ndim=2] X)

cdef cube np2cube(np.ndarray[np.double_t, ndim=3] X)

cdef cx_cube np2cx_cube(np.ndarray[np.complex_t, ndim=3] X)

cdef np.ndarray[np.complex_t, ndim=3] cube2np(const cube & X, np.ndarray[np.double_t, ndim=3] D)

cdef np.ndarray[np.complex_t, ndim=3] cx_cube2np(const cx_cube & X, np.ndarray[np.double_t, ndim=3] D)

cdef np.ndarray[np.double_t, ndim=2] mat2np(const Mat[double] & X, np.ndarray[np.double_t, ndim=2] D)

cdef np.ndarray[np.double_t, ndim=2] cx_mat2np(const cx_mat & X, np.ndarray[np.double_t, ndim=2] D)

cdef np.ndarray[np.double_t, ndim=1] vec2np(const Col[double] & X, np.ndarray[np.double_t, ndim=1] D)