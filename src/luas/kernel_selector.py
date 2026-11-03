import luas.kernels.covtype as covtype
from luas.kernels.covtype import CovType, Outer
from luas.kronecker_fns import calc_data_shape
from luas.kernel_interface import read_K_list_2D
from luas.luas_types import Kernel, PyTree, JAXArray, Scalar
from luas import (
    WhiteNoiseKernel,
    SeparableKernel,
    LuasKernel,
    LuasKernelND,
    LuasLasrachKernel,
    MultiLowRankKernel,
    DoubleMultiLowRankKernel,
    LuasPlusMultiLowRankKernel,
    ExtendedLuasKernel,
    GeneralKernel,
)
from luas.kernels.lowrank import LowRank


quasisep_compatible_list = (covtype.GeneralQuasisepPlusNoise, covtype.GeneralQuasisep,
                            covtype.Diagonal, covtype.ScaledIdentity, covtype.Identity,
                            covtype.Outer, covtype.OuterPlusScaledIdentity)
quasisep_cov_list =  (covtype.GeneralQuasisepPlusNoise, covtype.GeneralQuasisep)
diag_cov_list = (covtype.Diagonal, covtype.ScaledIdentity, covtype.Identity)


def kernel_selector(
        p,
        X,
        kf = None,
        verbose = False,
        use_kernel = None,
        kernel_select_kwargs = {},
        **kernel_kwargs,
    ):
    data_shape = calc_data_shape(X)
    dim = len(data_shape)
    
    if use_kernel is not None:
        if kf is not None:
            cov_form = kf(p, X)

            if isinstance(cov_form, tuple): 
                build_kf = lambda p, X: use_kernel(*kf(p, X), **kernel_kwargs)
            else:
                build_kf = lambda p, X: use_kernel(kf(p, X), **kernel_kwargs)

        elif use_kernel == GeneralKernel and dim == 1:
            def general_kf(hp, X, **kwargs):
                cov_form = kf(hp, X)
                return covtype.General(lambda p, X, **kwargs: cov_form.evaluate(X, **kwargs), **kernel_kwargs)
                
            build_kf = general_kf

        elif kf is None:
            # Least squares or some other CovType without parameters
            build_kf = lambda p, X: use_kernel(**kernel_kwargs)

        else:
            raise Exception(f"use_kernel = {use_kernel} not recognised!")

    else:
        # use_kernel is None

        # kf specified exactly
        if isinstance(kf, covtype.CovType):
            # Just a fixed kernel not given as a function
            build_kf = lambda p, X: kf
        
        # or kf not specified at all
        elif kf is None:
            if verbose:
                print("No kernel function specified, defaulting to least squares")
            build_kf = lambda p, X: WhiteNoiseKernel(wn_diag = 1.0)

        else:
            # need to figure out best GP optimisation to use based on form of kernel
            cov_form = kf(p, X)
            
            if isinstance(cov_form, covtype.CovType):
                # kernel returns a GP optimisation object already
                build_kf = kf
            else:
                assert isinstance(cov_form, tuple) # kf must return a kernel object or tuple(s)
                use_kernel, new_kernel_kwargs = find_best_optimisation(X, cov_form,
                                                                       verbose = verbose, **kernel_select_kwargs)
                new_kernel_kwargs.update(kernel_kwargs)
                build_kf = lambda p, X: use_kernel(*kf(p, X), **new_kernel_kwargs)
            
    return build_kf


def find_best_optimisation(X, cov_form, verbose = True, **kwargs):
    data_shape = calc_data_shape(X)
    dim = len(data_shape)
    print_str = ""

    is_singlekronterm = True
    for d in range(dim):
        if not isinstance(cov_form[d], covtype.CovType):
            is_singlekronterm = False

    if is_singlekronterm:
        use_kernel, kernel_kwargs = SeparableKernel, {}
    else:
        Sigma, *args = cov_form

        # Ensure dimensions are correct
        assert (len(Sigma) <= dim)
        for arg in args:
            assert len(arg) == dim

        if dim == 1:
            raise Exception("""One regressor specified but kernel function doesn't return a kernel.
            For 1D data make sure the kernel function kf returns a Kernel object rather than a tuple.
            For >1D data make sure as many regressors are specified as kernel dimensions
            i.e. X should be a tuple the same length as the terms kf returns.""")
        
        elif dim == 2:
            dense_kron, alpha_terms, beta_terms = read_K_list_2D(cov_form[1:], X)


            N_alpha = 0 if alpha_terms is None else len(alpha_terms)
            N_beta = 0 if beta_terms is None else len(beta_terms)

            if data_shape[0] > data_shape[1]:
                longest_dim = 0
            else:
                longest_dim = 1

            # Check whether there is a valid "Celerite dimension", will do nothing if args is None
            dim0_quasi_compat = isinstance(Sigma[0], quasisep_compatible_list)
            dim1_quasi_compat = isinstance(Sigma[1], quasisep_compatible_list)

            for arg in args:
                dim0_quasi_compat = isinstance(arg[0], quasisep_compatible_list) and dim0_quasi_compat
                dim1_quasi_compat = isinstance(arg[1], quasisep_compatible_list) and dim1_quasi_compat

            valid_quasi_dim = (dim0_quasi_compat, dim1_quasi_compat)

            if valid_quasi_dim[longest_dim]:
                longest_fast_dim = longest_dim 
            elif valid_quasi_dim[1-longest_dim]:
                longest_fast_dim = 1-longest_dim
            else:
                longest_fast_dim = None

            # Determine whether Sigma is a kronecker product of diagonal matrices
            kron_diagonal = isinstance(Sigma[0], diag_cov_list) and isinstance(Sigma[1], diag_cov_list)
            
            # valid_kernel_list, valid_kernel_kwarg_list = valid_kernels(dense_kron, len(alpha_terms), len(beta_terms),
            #                                                            kron_diagonal = kron_diagonal, **kwargs)
            use_kernel, kernel_kwargs = fastest_opt_2D(dense_kron, N_alpha, N_beta, data_shape,
                                                    kron_diagonal = kron_diagonal, valid_quasi_dim = valid_quasi_dim,
                                                        longest_fast_dim = longest_fast_dim, verbose = verbose, **kwargs)
        else:
            # >2D GP
            # Could also do luaslasrach but often slower for >2D
            print_str += "For >2D with two Kronecker terms we currently default to LuasKernelND, but LuasLasrachKernel may be worth checking too!\n"
            use_kernel, kernel_kwargs = LuasKernelND, {}

    if verbose and print_str:
        print(print_str)
    
    return use_kernel, kernel_kwargs



def fastest_opt_2D(dense_kron, N_alpha, N_beta, data_shape,
                   valid_quasi_dim = (False, False), kron_diagonal = False,
                   longest_fast_dim = None, max_gen_cholesky_blocks = 10, verbose = True):
    if data_shape[0] > data_shape[1]:
        longest_dim = 0
    else:
        longest_dim = 1

    kernel_kwargs = {}
    if dense_kron:
        if N_alpha > 0 and N_beta > 0:
            use_kernel = ExtendedLuasKernel
        
        elif N_alpha == 0 and N_beta == 0:
            if longest_fast_dim is None:
                if data_shape[1-longest_dim] < max_gen_cholesky_blocks:
                    use_kernel = LuasLasrachKernel
                    kernel_kwargs = {"fast_dim":longest_dim}
                else:
                    use_kernel = LuasKernel

            elif longest_fast_dim == longest_dim:
                use_kernel = LuasLasrachKernel
                kernel_kwargs = {"fast_dim":longest_dim}
            else:
                use_kernel = LuasKernel

        else:
            use_kernel = LuasPlusMultiLowRankKernel
            kernel_kwargs = {"fast_dim":N_beta > 0, "eigen_both":longest_fast_dim != longest_dim}
    else:
        if N_alpha > 0 and N_beta > 0:
            use_kernel = DoubleMultiLowRankKernel

            fast_dim = N_alpha > 0
            if kron_diagonal and valid_quasi_dim[0] and valid_quasi_dim[1]:
                kernel_kwargs = {"use_quasi":True}
            else:
                kernel_kwargs = {"use_quasi":False}

        elif N_alpha == 0 and N_beta == 0:
            use_kernel = SeparableKernel

        else:
            use_kernel = MultiLowRankKernel

            fast_dim = N_alpha > 0
            if valid_quasi_dim[fast_dim]:
                kernel_kwargs = {"fast_dim":fast_dim, "use_quasi":True}
            else:
                kernel_kwargs = {"fast_dim":fast_dim, "use_quasi":False}

    return use_kernel, kernel_kwargs



def valid_kernels(dense_kron, N_alpha, N_beta, longest_fast_dim = None):
    kernel_kwarg_list = [{}]
    if dense_kron:
        if N_alpha > 0 and N_beta > 0:
            use_kernels = [ExtendedLuasKernel]
        elif N_alpha == 0 and N_beta == 0:
            use_kernels = [LuasLasrachKernel, LuasLasrachKernel, LuasKernel] # OldLuasKernel
            kernel_kwarg_list = [{"fast_dim":0}, {"fast_dim":1}, {}] # choice, also luas kernel
        else:
            use_kernels = [LuasPlusMultiLowRankKernel, LuasPlusMultiLowRankKernel]
            kernel_kwarg_list = [{"fast_dim":N_beta > 0, "eigen_both":False}, {"fast_dim":N_beta > 0, "eigen_both":True}]
    else:
        if N_alpha > 0 and N_beta > 0:
            use_kernels = [DoubleMultiLowRankKernel]
            kernel_kwarg_list = [{"use_quasi":False}] # {"use_quasi":True}, 
        elif N_alpha == 0 and N_beta == 0:
            use_kernels = [SeparableKernel]
        else:
            use_kernels = [MultiLowRankKernel, MultiLowRankKernel]
            kernel_kwarg_list = [{"fast_dim":N_alpha > 0, "use_quasi":True}, {"fast_dim":N_alpha > 0, "use_quasi":False}]

    return use_kernels, kernel_kwarg_list
