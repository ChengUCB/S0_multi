# S0_multi

This repository implements the matrix operations used in the extended S0
method for ** multi-component mixtures**.

Starting from the zero-wavevector limits of the partial static structure
factors, $S^0_{\alpha\beta}$, it computes derivatives of chemical potentials
with respect to a set of independent atomic fractions. 
The code also provides utilities for checking the statistical consistency of $S^0$ values obtained
from trajectory splits and for preparing gradient observations for Gaussian process (GP) integration (please refer to [`GPR_grad`](https://github.com/ChengUCB/GPR_grad)).

See [`test.ipynb`](test.ipynb) for a short executable example.

## Installation

```bash
git clone https://github.com/ChengUCB/S0_multi.git
cd S0_multi
python -m pip install numpy pandas matplotlib jupyter
jupyter lab test.ipynb
```

## From $S^0$ to chemical potential derivatives

For a system with $C$ components and atomic-fraction vector
$\mathbf{x}=[x_1,\ldots,x_C]^T$, the code constructs

$$
B_{\alpha\beta}=S^0_{\alpha\beta}\sqrt{x_\alpha x_\beta},
$$

followed by the chemical potential derivative matrix

$$
\mathbf{U}=k_{\rm B}T\left[
\mathbf{B}^{-1}-
\frac{\mathbf{B}^{-1}\mathbf{x}\mathbf{x}^T\mathbf{B}^{-1}}
{\mathbf{x}^T\mathbf{B}^{-1}\mathbf{x}}
\right].
$$

For a neutral mixture, normalization leaves $f=C-1$ independent atomic
fractions, $\tilde{\mathbf{x}}$. The projection matrix $\mathbf{M}$ satisfies

$$
M_{ij}=\frac{\partial x_i}{\partial\tilde{x}_j}.
$$

The total derivative matrix is

$$
\mathbf{\Gamma}=\mathbf{U}\mathbf{M}, \qquad
\Gamma_{\alpha\beta}=
\left(\frac{\partial\mu_\alpha}{\partial\tilde{x}_\beta}\right)
_{\tilde{x}_i\ne\tilde{x}_\beta},
$$

and satisfies the Gibbs–Duhem condition
$\mathbf{x}^T\mathbf{\Gamma}=\mathbf{0}^T$. 


## Example

The following example uses the independent atomic fractions
$\tilde{\mathbf{x}}=[x_{\rm Fe},x_{\rm Cu}]^T$ for a Fe–Cu–Ni mixture.

```python
from szero import s0_to_gamma

entry = {
    "x_Fe": 0.45, "x_Cu": 0.05, "x_Ni": 0.50,
    "S_FeFe": 1.20, "S_FeCu": 0.10, "S_FeNi": 0.05,
    "S_CuCu": 1.40, "S_CuNi": 0.12, "S_NiNi": 1.10,
}

gamma_ex, components, independent_components = s0_to_gamma(
    entry,
    components=["Fe", "Cu", "Ni"],
    independent_components=["Fe", "Cu"],
    temperature=3000.0,
    excess=True,
)
```

`gamma_ex[alpha, beta]` is
$\Gamma^{\rm ex}_{\alpha\beta}$; rows follow `components`, and columns follow
`independent_components`.

## Trajectory-split diagnostics

The plotting and cleaning utilities recognize columns such as
`split0_FeCu_S0`, `split1_FeCu_S0`, and optional uncertainty columns ending in
`_error`.

```python
import pandas as pd
from szero import find_s0_split_outlier_rows, plot_s0_split_residuals

df = pd.read_csv("s0-splits.csv")
fig, _ = plot_s0_split_residuals(df)
bad_mask, details = find_s0_split_outlier_rows(df)
clean_df = df.loc[~bad_mask]
```

Inspect `details` and choose consistency thresholds according to the
statistical uncertainties in the $S^0$ values before averaging trajectory
splits.


## Notes
- For GP integration of $\mathbf{\Gamma}$ or $\mathbf{\Gamma}^{\rm ex}$ using function-value and gradient observations, please refer to [`GPR_grad`](https://github.com/ChengUCB/GPR_grad).
- Use interior compositions for $\mathbf{\Gamma}^{\rm ex}$ because its ideal
  contribution contains $1/x_\alpha$.
- Pass `components` explicitly when matrix row order matters.
- `jitter` is a numerical stabilization for the inversion of $\mathbf{B}$; it
  does not correct an unphysical structure-factor matrix.
