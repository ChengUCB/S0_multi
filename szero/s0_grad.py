import numpy as np


KB_IN_EV = 8.617333262145e-5


def s0_to_gamma(
    entry,
    independent_components,
    components=None,
    charges=None,
    kb_t=None,
    temperature=None,
    excess=False,
    jitter=1e-12,
    return_labels=True,
):
    """
    Convert one multi-component S0 entry to a Gamma matrix.

    Parameters
    ----------
    entry
        Dictionary containing composition keys and S0 pair keys, for example:

            {
                "x_A": 0.2,
                "x_B": 0.3,
                "x_C": 0.5,
                "S_AA": ...,
                "S_AB": ...,
                "S_AC": ...,
                "S_BB": ...,
                "S_BC": ...,
                "S_CC": ...,
            }

        Pair keys may be supplied in either order, e.g. "S_AB" or "S_BA".

    independent_components
        Components used as x_tilde variables, e.g. ["A", "B"].

    components
        Optional full component order. If omitted, inferred from x_* keys and
        sorted alphabetically. Pass this explicitly if row/column order matters.

    charges
        Optional dict mapping component -> charge. If omitted, all charges are
        treated as zero. Nonzero charges add a charge-neutrality constraint.

    kb_t
        Thermal energy. If omitted, computed as k_B * temperature.

    temperature
        Temperature in K, used only when kb_t is omitted.

    excess
        If True, subtract ideal-solution gradients:
        kBT * d ln(x_alpha) / d x_tilde.

    jitter
        Diagonal stabilization for inverting B.

    return_labels
        If True, return (gamma, row_labels, col_labels). If False, return gamma.

    Returns
    -------
    gamma
        Array with shape (n_components, n_independent_components). Rows follow
        `components`; columns follow `independent_components`.
    """
    components = _resolve_components(entry, components)
    independent_components = list(independent_components)

    if kb_t is None:
        if temperature is None:
            raise ValueError("Provide either kb_t or temperature.")
        kb_t = KB_IN_EV * float(temperature)

    x = _composition_vector(entry, components)
    b = _b_matrix(entry, x, components)
    m = _projection_matrix(components, independent_components, charges=charges)

    a = np.linalg.inv(b + jitter * np.eye(len(components)))
    xc = x.reshape(-1, 1)
    xr = x.reshape(1, -1)
    alpha = float(xr @ a @ xc)
    u = kb_t * (a - (a @ xc @ xr @ a) / alpha)

    gamma = u @ m
    if excess:
        gamma -= kb_t * np.diag(1.0 / x) @ m

    if return_labels:
        return gamma, components, independent_components
    return gamma


def _resolve_components(entry, components):
    if components is not None:
        return list(components)

    components = sorted(key[2:] for key in entry if key.startswith("x_"))
    if not components:
        raise ValueError("Could not infer components. Provide x_* keys or components.")
    return components


def _composition_vector(entry, components):
    values = []
    missing = []
    for component in components:
        key = f"x_{component}"
        if key not in entry:
            missing.append(key)
        else:
            values.append(float(entry[key]))

    if missing:
        raise ValueError(f"Missing composition keys: {missing}")

    x = np.asarray(values, dtype=float)
    #if np.any(x <= 0):
    if np.any(x < 0):
        raise ValueError("All composition entries must be positive.")
    if not np.isclose(np.sum(x), 1.0):
        raise ValueError(f"Compositions must sum to 1. Got {np.sum(x)}.")
    return x


def _s0_value(entry, a, b):
    candidates = (
        f"S_{a}{b}",
        f"S_{b}{a}",
        f"S_{a}_{b}",
        f"S_{b}_{a}",
        f"S0_{a}{b}",
        f"S0_{b}{a}",
        f"S0_{a}_{b}",
        f"S0_{b}_{a}",
    )
    for key in candidates:
        if key in entry:
            return float(entry[key])
    raise ValueError(f"Missing S0 pair key for {a}-{b}. Tried {candidates}.")


def _b_matrix(entry, x, components):
    n = len(components)
    b = np.zeros((n, n), dtype=float)
    for i, comp_i in enumerate(components):
        for j, comp_j in enumerate(components):
            s0 = _s0_value(entry, comp_i, comp_j)
            b[i, j] = s0 * np.sqrt(x[i] * x[j])
    return b


def _projection_matrix(components, independent_components, charges=None):
    n = len(components)
    f = len(independent_components)
    component_index = {component: i for i, component in enumerate(components)}

    has_charge_constraint = False
    if charges is not None:
        q = np.array([float(charges.get(component, 0.0)) for component in components])
        has_charge_constraint = bool(np.any(q != 0.0))

    expected_f = n - 2 if has_charge_constraint else n - 1
    if f != expected_f:
        raise ValueError(
            f"Expected {expected_f} independent components for {n} components "
            f"with {'a charge' if has_charge_constraint else 'only the sum'} "
            f"constraint, got {f}."
        )

    p = np.zeros((f, n), dtype=float)
    for row, component in enumerate(independent_components):
        if component not in component_index:
            raise ValueError(f"Unknown independent component: {component}")
        p[row, component_index[component]] = 1.0

    constraint_rows = [np.ones(n, dtype=float)]
    if has_charge_constraint:
        constraint_rows.append(q)

    q_matrix = np.vstack([p, *constraint_rows])
    if np.linalg.matrix_rank(q_matrix) < n:
        raise ValueError(
            "Projection plus constraints do not determine all components. "
            "Provide enough independent components or charge constraints."
        )

    return np.linalg.pinv(q_matrix)[:, :f]


def gamma_dict(entry, independent_components, **kwargs):
    """
    Convenience wrapper returning nested dictionaries instead of an array.

    Returns
    -------
    {
        component: {
            independent_component: gamma_value,
            ...
        },
        ...
    }
    """
    gamma, row_labels, col_labels = s0_to_gamma(
        entry,
        independent_components,
        return_labels=True,
        **kwargs,
    )
    return {
        row_label: {
            col_label: float(gamma[i, j])
            for j, col_label in enumerate(col_labels)
        }
        for i, row_label in enumerate(row_labels)
    }


def s0_entry_from_row(row, column_map):
    """
    Build an s0_to_gamma entry dictionary from one dataframe row.

    column_map maps s0_to_gamma entry keys to dataframe columns, for example
    {"x_A": "comp_a", "S_AB": "split0_AB_S0"}.
    A mapped value can also be a list/tuple of columns, which are averaged.
    """
    return _mapped_entry_from_row(row, column_map)


def gamma_from_row(
    row,
    independent_components,
    column_map,
    components=None,
    **kwargs,
):
    """Compute Gamma from one dataframe row."""
    if components is None:
        components = _components_from_column_map(column_map)
    entry = s0_entry_from_row(row, column_map=column_map)
    return s0_to_gamma(
        entry,
        independent_components=independent_components,
        components=components,
        **kwargs,
    )


def prepare_gp_gradient_data(
    df,
    independent_components,
    column_map,
    components=None,
    species=None,
    excess=False,
    as_torch=False,
    torch_dtype=None,
    **gamma_kwargs,
):
    """
    Convert a dataframe of S0 rows into GP gradient observations.

    Parameters
    ----------
    df
        Dataframe containing composition and S0 columns.

    independent_components
        Composition variables used as GP inputs, e.g. ["Fe", "Cu"].

    column_map
        Explicit map from s0_to_gamma entry keys to dataframe columns.
        Keys should be composition/S0 entry names, e.g. "x_Fe", "S_FeCu".
        Values can be column names, or a list/tuple of column names to average.

    components
        Full component order. If omitted, inferred from the x_* keys in
        column_map insertion order.

    species
        Which chemical-potential component(s) to prepare. If a string, returns
        one (X_g, G_g, grad_indices) triplet. If a list/tuple of strings,
        returns a dictionary keyed by species. If None, returns all components
        in a dictionary.

    excess
        If True, subtract the ideal-solution part from Gamma. Defaults to False.

    as_torch
        If True, return torch tensors for direct use in GradientGP.fit.

    charges
        If provided, should be a dict mapping components to their charge. Used to determine the projection matrix when charge neutrality is a constraint.
        e.g. charges={"Na": 1, "Cl": -1, "O":0}

    Returns
    -------
    If species is a string:
        X_g, G_g, grad_indices

    If species is a list/tuple, or species is None:
        {
            component: (X_g, G_g, grad_indices),
            ...
        }

    X_g has shape (n_rows, n_independent_components).
    G_g has shape (n_rows, n_independent_components).
    grad_indices has shape (n_rows * n_independent_components, 2), enumerating
    all gradient components. It is optional for full-gradient observations, but
    returned for explicitness and compatibility with GradientGP.fit.
    """
    if components is None:
        components = _components_from_column_map(column_map)
    components = list(components)
    independent_components = list(independent_components)

    x_g = atomic_fraction_to_xtilde(
        df,
        column_map=column_map,
        independent_components=independent_components,
        components=components,
    )
    gamma_all = []

    for _, row in df.iterrows():
        gamma, row_labels, col_labels = gamma_from_row(
            row,
            independent_components=independent_components,
            components=components,
            column_map=column_map,
            excess=excess,
            return_labels=True,
            **gamma_kwargs,
        )
        if row_labels != components:
            raise RuntimeError("Internal component label mismatch.")
        if col_labels != independent_components:
            raise RuntimeError("Internal independent-component label mismatch.")
        gamma_all.append(gamma)

    gamma_all = np.stack(gamma_all, axis=0)
    grad_indices = _full_grad_indices(len(df), len(independent_components))

    if species is None:
        selected_species = components
        return_dict = True
    elif isinstance(species, str):
        selected_species = [species]
        return_dict = False
    else:
        selected_species = list(species)
        return_dict = True

    unknown_species = [
        component for component in selected_species if component not in components
    ]
    if unknown_species:
        raise ValueError(
            f"Unknown species {unknown_species}. Available: {components}"
        )

    result = {
        component: _maybe_torch_triplet(
            x_g,
            gamma_all[:, components.index(component), :],
            grad_indices,
            as_torch=as_torch,
            torch_dtype=torch_dtype,
        )
        for component in selected_species
    }

    if return_dict:
        return result
    return result[selected_species[0]]


def pure_component_anchor(
    species,
    independent_components,
    components,
    dtype=None,
    device=None,
):
    """
    Return X_f, Y_f anchoring mu_species = 0 at the pure-species point.

    X_f contains only the independent composition coordinates. For example,
    with species="Cu", independent_components=["Fe", "Cu"], and
    components=["Fe", "Cu", "Ni"], this returns X_f = [[0, 1]].
    """
    import torch

    if species not in components:
        raise ValueError(f"{species!r} is not in components={components}")

    dtype = dtype if dtype is not None else torch.get_default_dtype()

    x_full = {
        component: 1.0 if component == species else 0.0
        for component in components
    }

    x_f = torch.tensor(
        [[x_full[component] for component in independent_components]],
        dtype=dtype,
        device=device,
    )
    y_f = torch.tensor([0.0], dtype=dtype, device=device)
    return x_f, y_f


def atomic_fraction_to_xtilde(
    data,
    column_map,
    independent_components,
    components,
    as_torch=False,
    torch_dtype=None,
    device=None,
):
    """
    Convert atomic fractions to tilde{x} using the independent components.

    data can be a dataframe, a dataframe row/Series, or a dict-like object.
    column_map maps x_* entry keys to data columns, e.g. {"x_Fe": "Fe_frac"}.

    Returns shape (n_points, n_independent_components) for dataframe input, and
    shape (n_independent_components,) for one row/dict input.
    """
    x_cols = []
    for component in independent_components:
        if component not in components:
            raise ValueError(f"{component!r} is not in components={components}")

        entry_key = f"x_{component}"
        if entry_key not in column_map:
            raise ValueError(f"column_map is missing {entry_key!r}.")

        mapped_col = column_map[entry_key]
        if not isinstance(mapped_col, str):
            raise ValueError(
                f"column_map[{entry_key!r}] must map to one composition column."
            )
        x_cols.append(mapped_col)

    if hasattr(data, "loc") and hasattr(data, "columns"):
        missing_x = [col for col in x_cols if col not in data.columns]
        if missing_x:
            raise ValueError(f"Missing mapped independent columns: {missing_x}")
        x_tilde = data.loc[:, x_cols].to_numpy(dtype=float)
    else:
        x_tilde = np.asarray([float(data[col]) for col in x_cols], dtype=float)

    if not as_torch:
        return x_tilde

    import torch

    dtype = torch_dtype if torch_dtype is not None else torch.get_default_dtype()
    return torch.as_tensor(x_tilde, dtype=dtype, device=device)


def make_free_zero_gradient_points(
    species,
    independent_components,
    components=None,
    n_per_face=10,
    dtype=None,
    device=None,
    eps=0.0,
    random_state=None,
):
    """
    Make synthetic free-gradient constraints for one chemical-potential species.

    If `species` is an independent component x_j, this generates points with
    x_j = eps inside the composition simplex and constrains
    d mu_species^ex / d x_j = 0.

    If `species` is not an independent component, returns empty tensors because
    there is no corresponding derivative dimension in this coordinate system.

    Returns
    -------
    X_free
        Tensor with shape (n_per_face, n_independent_components), or
        (0, n_independent_components) when species is not independent.

    free_grad_dims
        Tensor with shape (len(X_free),), giving the derivative dimension for
        each row of X_free.

    G_free
        Zero tensor with shape (len(X_free),).
    """
    import torch

    independent_components = list(independent_components)
    if components is not None and species not in components:
        raise ValueError(f"{species!r} is not in components={components}")

    n_dims = len(independent_components)
    if n_dims < 1:
        raise ValueError("independent_components must contain at least one entry.")
    if n_per_face < 1:
        raise ValueError("n_per_face must be at least 1.")
    if eps < 0.0 or eps > 1.0:
        raise ValueError("eps must be between 0 and 1.")

    dtype = dtype if dtype is not None else torch.get_default_dtype()

    if species not in independent_components:
        return (
            torch.empty((0, n_dims), dtype=dtype, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )

    dim = independent_components.index(species)
    generator = None
    if random_state is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(random_state))

    remaining_total = 1.0 - eps
    x_free = torch.zeros(n_per_face, n_dims, dtype=dtype, device=device)
    x_free[:, dim] = eps

    other_dims = [other for other in range(n_dims) if other != dim]
    if other_dims:
        if len(other_dims) == 1:
            values = torch.linspace(
                0.0,
                remaining_total,
                n_per_face,
                dtype=dtype,
                device=device,
            ).reshape(-1, 1)
        else:
            raw = torch.rand(
                n_per_face,
                len(other_dims) + 1,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            weights = -torch.log(torch.clamp(raw, min=1e-12))
            weights = weights / weights.sum(dim=1, keepdim=True)
            values = remaining_total * weights[:, :-1]

        x_free[:, other_dims] = values

    free_grad_dims = torch.full(
        (n_per_face,),
        dim,
        dtype=torch.long,
        device=device,
    )
    g_free = torch.zeros(x_free.shape[0], dtype=dtype, device=device)
    return x_free, free_grad_dims, g_free


def append_partial_gradients(
    X_g,
    G_g,
    grad_indices,
    X_new,
    G_new,
    grad_dims_new,
):
    """
    Append selected derivative observations to gradient GP data.

    X_new[k] carries gradient value G_new[k] for derivative dimension
    grad_dims_new[k]. The returned G_g is flattened to match grad_indices.
    """
    import torch

    if torch.is_tensor(X_g):
        dtype = X_g.dtype
        device = X_g.device
    else:
        dtype = torch.get_default_dtype()
        device = None
        X_g = torch.as_tensor(X_g, dtype=dtype)

    G_g = torch.as_tensor(G_g, dtype=dtype, device=device)
    grad_indices = torch.as_tensor(grad_indices, dtype=torch.long, device=device)
    X_new = torch.as_tensor(X_new, dtype=dtype, device=device)
    G_new = torch.as_tensor(G_new, dtype=dtype, device=device).reshape(-1)
    grad_dims_new = torch.as_tensor(
        grad_dims_new,
        dtype=torch.long,
        device=device,
    ).reshape(-1)

    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    if X_new.shape[1] != X_g.shape[1]:
        raise ValueError(
            f"X_new has dimension {X_new.shape[1]}, expected {X_g.shape[1]}."
        )
    if X_new.shape[0] != G_new.numel():
        raise ValueError("X_new and G_new must have the same number of rows.")
    if X_new.shape[0] != grad_dims_new.numel():
        raise ValueError("X_new and grad_dims_new must have the same number of rows.")

    n_old = X_g.shape[0]
    point_ids_new = torch.arange(X_new.shape[0], device=device) + n_old
    grad_indices_new = torch.stack([point_ids_new, grad_dims_new], dim=1)

    if G_g.shape == X_g.shape:
        point_ids_old = grad_indices[:, 0]
        dim_ids_old = grad_indices[:, 1]
        G_old = G_g[point_ids_old, dim_ids_old]
    else:
        G_old = G_g.reshape(-1)
        if G_old.numel() != grad_indices.shape[0]:
            raise ValueError(
                "Flattened G_g must have one value per row of grad_indices."
            )

    X_g_aug = torch.cat([X_g, X_new], dim=0)
    G_g_aug = torch.cat([G_old, G_new], dim=0)
    grad_indices_aug = torch.cat([grad_indices, grad_indices_new], dim=0)

    return X_g_aug, G_g_aug, grad_indices_aug


def _mapped_entry_from_row(row, column_map):
    entry = {}
    for entry_key, columns in column_map.items():
        entry[entry_key] = _row_value_from_columns(row, columns)
    return entry


def _row_value_from_columns(row, columns):
    if isinstance(columns, str):
        if columns not in row.index:
            raise ValueError(f"Missing mapped dataframe column: {columns}")
        return float(row[columns])

    values = []
    missing = []
    for column in columns:
        if column not in row.index:
            missing.append(column)
        else:
            values.append(float(row[column]))

    if missing:
        raise ValueError(f"Missing mapped dataframe columns: {missing}")
    if not values:
        raise ValueError("Mapped column lists must contain at least one column.")

    return float(np.mean(values))


def _components_from_column_map(column_map):
    components = [key[2:] for key in column_map if key.startswith("x_")]
    if not components:
        raise ValueError("column_map must include x_* entry keys, e.g. x_Fe.")
    return components


def _full_grad_indices(n_points, n_dims):
    point_ids = np.repeat(np.arange(n_points), n_dims)
    dim_ids = np.tile(np.arange(n_dims), n_points)
    return np.column_stack([point_ids, dim_ids]).astype(np.int64)


def _maybe_torch_triplet(x_g, g_g, grad_indices, as_torch=False, torch_dtype=None):
    if not as_torch:
        return x_g.copy(), g_g.copy(), grad_indices.copy()

    import torch

    dtype = torch_dtype if torch_dtype is not None else torch.get_default_dtype()
    return (
        torch.as_tensor(x_g, dtype=dtype),
        torch.as_tensor(g_g, dtype=dtype),
        torch.as_tensor(grad_indices, dtype=torch.long),
    )
