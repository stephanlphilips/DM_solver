from libcpp cimport bool
cimport numpy as np

cdef extern from "armadillo" namespace "arma" nogil:
    # Mat[T]rix class (double)
    cdef cppclass Mat[T]:
        Mat[T](double * aux_mem, int n_rows, int n_cols, bool copy_aux_mem, bool strict) nogil
        Mat[T](double * aux_mem, int n_rows, int n_cols) nogil
        Mat[T](int n_rows, int n_cols) nogil
        Mat[T]() nogil
        cx_Mat[T](Mat[T] real_Mat[T], Mat[T] im_Mat[T]) nogil
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
        #etc

    # cdef cppclass cx_Mat[T]:
    #     cx_Mat[T](Mat[T] real_Mat[T], Mat[T] im_Mat[T]) nogil
    #     cx_Mat[T]() nogil

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
        
    # cdef cppclass cx_cube:
    #     cx_cube(cube real_cube, cube im_cube) nogil
    #     cx_cube() nogil

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
        vec operator*(Mat[T])
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
    cdef bool chol(Mat[T] R, Mat[T] X) nogil # preallocated result
    cdef Mat[T] chol(Mat[T] X) nogil # new result
    cdef bool inv(Mat[T] R, Mat[T] X) nogil
    cdef Mat[T] inv(Mat[T] X) nogil
    cdef bool solve(vec x, Mat[T] A, vec b) nogil
    cdef vec solve(Mat[T] A, vec b) nogil
    cdef bool solve(Mat[T] X, Mat[T] A, Mat[T] B) nogil
    cdef Mat[T] solve(Mat[T] A, Mat[T] B) nogil
    cdef bool eig_sym(vec eigval, Mat[T] eigvec, Mat[T] B) nogil
    cdef bool svd(Mat[T] U, vec s, Mat[T] V, Mat[T] X, method) nogil
    cdef bool lu(Mat[T] L, Mat[T] U, Mat[T] P, Mat[T] X) nogil
    cdef bool lu(Mat[T] L, Mat[T] U, Mat[T] X) nogil
    cdef Mat[T] pinv(Mat[T] A) nogil
    cdef bool pinv(Mat[T] B, Mat[T] A) nogil
    cdef bool qr(Mat[T] Q, Mat[T] R, Mat[T] X) nogil
    cdef float dot(vec a, vec b) nogil

cdef Mat[T] * numpy_to_Mat[T](np.ndarray[np.double_t, ndim=2] X)

cdef Mat[T] numpy_to_Mat[T]_d(np.ndarray[np.double_t, ndim=2] X)

# cdef cx_Mat[T] numpy_to_cx_Mat[T]_d(np.ndarray[np.complex_t, ndim=2] X)

cdef cube * numpy_to_cube(np.ndarray[np.double_t, ndim=3] X)

cdef cube numpy_to_cube_d(np.ndarray[np.double_t, ndim=3] X)

# cdef cx_cube numpy_to_cxcube_d(np.ndarray[np.complex_t, ndim=3] X)

cdef vec * numpy_to_vec(np.ndarray[np.double_t, ndim=1] x)

cdef vec numpy_to_vec_d(np.ndarray[np.double_t, ndim=1] x)

cdef vec * Mat[T]_col_view(Mat[T] * x, int col) nogil

cdef vec Mat[T]_col_view_d(Mat[T] * x, int col) nogil

cdef Mat[T] * cube_slice_view(cube * x, int slice) nogil

cdef Mat[T] cube_slice_view_d(cube * x, int slice) nogil

cdef np.ndarray[np.double_t, ndim=2] Mat[T]_to_numpy(const Mat[T] & X, np.ndarray[np.double_t, ndim=2] D)

cdef np.ndarray[np.double_t, ndim=1] vec_to_numpy(const vec & X, np.ndarray[np.double_t, ndim=1] D)
