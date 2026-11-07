import luas.kernels
import tinygp.kernels
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

N_t = 512
ind = 256
x_t = jnp.linspace(0., 1., N_t)
scale = 0.1
double_nu = 3
sigma = 0.2
alpha = 0.5
tol = 1e-10
equal_kernel_dict = {}

expected_max = sigma**2
print(f"Expected max values = {expected_max}")

# Squared exponential kernel
kernel_squaredexp = luas.kernels.SquaredExp(scale = scale, sigma = sigma)
kernel_squaredexp_tinygp = sigma**2 * tinygp.kernels.ExpSquared(scale = scale)

equal_kernel_dict["squared_exp"] = [kernel_squaredexp, kernel_squaredexp_tinygp]

# Matern-1/2 aka Exponential kernel
kernel_maternhalfint_12_quasi = luas.kernels.quasisep.MaternHalfInt(scale = scale, double_nu = 1, sigma = sigma)
kernel_maternhalfint_12 = luas.kernels.MaternHalfInt(scale = scale, double_nu = 1, sigma = sigma)
kernel_exp_quasi = luas.kernels.quasisep.Exp(scale = scale, sigma = sigma)
kernel_exp = luas.kernels.Exp(scale = scale, sigma = sigma)

equal_kernel_dict["exp"] = [kernel_maternhalfint_12_quasi, kernel_maternhalfint_12, kernel_exp_quasi, kernel_exp]

# Matern-3/2
kernel_maternhalfint_32_quasi = luas.kernels.quasisep.MaternHalfInt(scale = scale, double_nu = 3, sigma = sigma)
kernel_maternhalfint_32 = luas.kernels.MaternHalfInt(scale = scale, double_nu = 3, sigma = sigma)
kernel_matern32_quasi = luas.kernels.quasisep.Matern32(scale = scale, sigma = sigma)
kernel_matern32 = luas.kernels.Matern32(scale = scale, sigma = sigma)

equal_kernel_dict["matern32"] = [kernel_maternhalfint_32_quasi, kernel_maternhalfint_32, kernel_matern32_quasi, kernel_matern32]

# Matern-5/2
kernel_maternhalfint_52_quasi = luas.kernels.quasisep.MaternHalfInt(scale = scale, double_nu = 5, sigma = sigma)
kernel_maternhalfint_52 = luas.kernels.MaternHalfInt(scale = scale, double_nu = 5, sigma = sigma)
kernel_matern52_quasi = luas.kernels.quasisep.Matern52(scale = scale, sigma = sigma)
kernel_matern52 = luas.kernels.Matern52(scale = scale, sigma = sigma)

equal_kernel_dict["matern52"] = [kernel_maternhalfint_52_quasi, kernel_maternhalfint_52, kernel_matern52_quasi, kernel_matern52]

# Rational quadratic
kernel_rat_quad = luas.kernels.RationalQuadratic(scale = scale, alpha = alpha, sigma = sigma)
kernel_rat_quad_tinygp = sigma**2 * tinygp.kernels.RationalQuadratic(scale = scale, alpha = alpha)

equal_kernel_dict["rational_quad"] = [kernel_rat_quad, kernel_rat_quad_tinygp]

for kernel_name, equal_kernels in equal_kernel_dict.items():
    Kt_eval = jnp.zeros((len(equal_kernels), N_t, N_t))
    for i in range(len(equal_kernels)):
        Kt_eval = Kt_eval.at[i, :, :].set(equal_kernels[i](x_t[ind:ind+1], x_t))

    devations_std = jnp.abs(Kt_eval.std(0)).max()
    assert devations_std < tol

    max_vals = Kt_eval.max((1, 2))
    assert jnp.all(max_vals - expected_max < tol)

    print(f"{kernel_name} max deviation from {len(equal_kernels)} kernels: {devations_std}")
    print(f"Diff. from expected max = {max_vals - expected_max}")
    print("")