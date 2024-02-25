(things_to_know)=

# Working with JAX

`luas` is implemented in [`jax`](https://github.com/google/jax), which allows it to efficiently calculate gradients of the log-likelihood for use with gradient-based optimisation and MCMC methods. It also makes it easy to run code on a GPU which can provide significant computational advantages.

However, `jax` can be a bit challenging to work with without some initial context. ["How to Think in JAX"](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html) and ["JAX - The Sharp Bits"](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) are excellent places to learn the specifics of programming in `jax`. However, if your needs with `luas` are not too complex then perhaps going to the [`tutorials`](tutorials) section will be sufficient to get up and running. A quick summary of issues you may first encounter with `jax` is included below.

## A Brief Summary of Common Issues with JAX

`jax` is designed for pure functional programming and so functions shouldn't have side-effects i.e. don't modify global variables within functions. For example, classes such as [`GP`](api-gp) and [`LuasKernel`](api-luaskernel) have a few attributes which are never modified by any of their methods. You should also never modify these attributes but instead should initialise a new object if changing them.

`jax` defaults to single-point precision, but calculations of matrix decomposition typically require double precision. Therefore, it is always recommended to include `jax.config.update("jax_enable_x64", True)` at the start of running a script/notebook to ensure `jax` uses double-precision floats.

`jax.numpy` closely matches `NumPy` in many respects but there are some differences. One difference is that changing the values of array elements has a slightly different syntax. Instead of `mat[i] = value` you will need to use `mat = mat.at[i].set(value)`.

`jax` provides the option to compile functions with Just-in Time (JIT) compilation. This can provide significant runtime improvements as the same compilation may be used for multiple calculations but it comes with stricter requirements on how control flow is used. One particular example is that control flow cannot be based on the value of an array element but it is okay for it to depend on the shape of an array e.g. `if mat[i] == 0:` will fail to compile but `if mat.size == 10:` is fine.

`jax` makes it easy to calculate derivatives by using the `jax.grad` and `jax.hessian` functions provided the functions only return a `Scalar` value. For example, if you want to calculate the gradient of a function `f(x)` this can be done with `jax.grad(f)(x)`. This will work not just if `x` is an `array` but also if `x` is a PyTree (which is basically just a `dict`), return a PyTree of gradients in the same form as `x`. Note that if you want to take the gradient of a function which returns an `array` you can do this by modifying the function so that it only returns particular elements of the `array`.

`for` loops and `while` loops can be very slow to compile. If possible, it is generally faster to make use of `jax.vmap` to "vectorise" a function. This means for example if we have a function `f(x)` which takes in a Scalar value `x` and outputs a 1D array `y`, we can use `y_2D = jax.vmap(f, in_axes = 0)(x_1D)` to create a function which takes in a 1D array `x_1D` and outputs a 2D array `y_2D`, serving a similar function as a `for` loop.
