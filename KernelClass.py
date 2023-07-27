import numpy as np
import jax.numpy as jnp
from kronecker_fn import kron_prod

class Kernel(object):
    def __init__(self, method = "rakitsch"):
        
        if method == "rakitsch":
            self.Kl = None
            self.Kt = None
            self.Sl = None
            self.St = None
            
            self.generate_noise = self.generate_noise_rakitsch
            
        elif method == "saatchi":
            self.Kl = None
            self.Kt = None
            self.generate_noise = self.generate_noise_saatchi
            
        elif method == "cholesky":
            self.K = None
            self.generate_noise = self.generate_noise_cholesky
            
        else:
            raise ValueError("Method not one of three available options (rakitsch, saatchi, cholesky)!")
            
        self.method = method
            
            
    def generate_noise_rakitsch(self, hp, x_l, x_t, size = 1, stable_const = 1e-6):
        
        Kl = self.Kl(hp, x_l, x_l)
        Kt = self.Kt(hp, x_t, x_t)
        Sl = self.Sl(hp, x_l, x_l)
        St = self.St(hp, x_t, x_t)
    
        Lam_Kl, Q_Kl = jnp.linalg.eigh(Kl)
        Lam_Kt, Q_Kt = jnp.linalg.eigh(Kt)

        Lam_mat_K = jnp.outer(Lam_Kl, Lam_Kt) + stable_const**2
        Lam_mat_sqrt_K = jnp.sqrt(Lam_mat_K)

        Lam_Sl, Q_Sl = jnp.linalg.eigh(Sl)
        Lam_St, Q_St = jnp.linalg.eigh(St)

        Lam_mat_S = jnp.outer(Lam_Sl, Lam_St) - stable_const**2
        Lam_mat_sqrt_S = jnp.sqrt(Lam_mat_S)

        z = np.random.normal(size = (Kl.shape[0], Kt.shape[0], 2, size))

        if size == 1:
            Lam_z1 = jnp.multiply(Lam_mat_sqrt_K, z[:, :, 0, 0])
            R = kron_prod(Q_Kl, Q_Kt, Lam_z1)

            Lam_z2 = jnp.multiply(Lam_mat_sqrt_S, z[:, :, 1, 0])
            R += kron_prod(Q_Sl, Q_St, Lam_z2)

        else:
            R = jnp.zeros((x_l.size, x_t.size, size))
            for i in range(size):
                Lam_z1 = jnp.multiply(Lam_mat_sqrt_K, z[:, :, 0, i])
                R[:, :, i] = kron_prod(Q_Kl, Q_Kt, Lam_z1)

                Lam_z2 = jnp.multiply(Lam_mat_sqrt_S, z[:, :, 1, i])
                R[:, :, i] += kron_prod(Q_Sl, Q_St, Lam_z2)

        return R