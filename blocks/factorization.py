from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import ConfigDict

from backend.block import (
    BaseBlock,
    BlockOutput,
    BlockParams,
    BlockValidationError,
    InsufficientInputs,
    ProgressBar,
)


def _parse_int_list(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [item.strip() for item in text.split(",") if item.strip()]
        return [int(item) for item in items]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _parse_float_list(value: Any, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        items = [item.strip() for item in text.split(",") if item.strip()]
        return [float(item) for item in items]
    if isinstance(value, (list, tuple, set)):
        return [float(item) for item in value]
    return [float(value)]


def _weighted_als(
    M: np.ndarray,
    W: np.ndarray,
    k: int,
    lam: float,
    n_iters: int = 15,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_rows, n_cols = M.shape
    U = 0.01 * rng.standard_normal((n_rows, k))
    V = 0.01 * rng.standard_normal((n_cols, k))
    M0 = np.nan_to_num(M, nan=0.0)
    eye = np.eye(k)

    for _ in range(max(int(n_iters), 1)):
        for i in range(n_rows):
            w = W[i, :]
            if np.all(w == 0):
                continue
            Vw = V * w[:, None]
            A = (V.T @ Vw) + (lam + 1e-8) * eye
            b = V.T @ (w * M0[i, :])
            try:
                U[i, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                U[i, :] = np.linalg.lstsq(A, b, rcond=None)[0]

        for j in range(n_cols):
            w = W[:, j]
            if np.all(w == 0):
                continue
            Uw = U * w[:, None]
            A = (U.T @ Uw) + (lam + 1e-8) * eye
            b = U.T @ (w * M0[:, j])
            try:
                V[j, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                V[j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
    return U, V


def _fit_bias_terms(
    M: np.ndarray, W_bias: np.ndarray, n_iters: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    X = np.nan_to_num(M, nan=0.0)
    W = W_bias.copy()
    n_rows, n_cols = X.shape
    b = np.zeros(n_rows, dtype=float)
    c = np.zeros(n_cols, dtype=float)
    for _ in range(max(int(n_iters), 1)):
        resid = X - c[None, :]
        b = (W * resid).sum(axis=1) / np.maximum(W.sum(axis=1), 1e-12)
        resid = X - b[:, None]
        c = (W * resid).sum(axis=0) / np.maximum(W.sum(axis=0), 1e-12)
    return b, c


def _fit_global_rank1(R: np.ndarray, W: np.ndarray, g: np.ndarray) -> np.ndarray:
    g2 = g * g
    num = (W * (g[:, None] * R)).sum(axis=0)
    den = (W * (g2[:, None])).sum(axis=0)
    d = np.zeros(R.shape[1], dtype=float)
    mask = den > 1e-12
    d[mask] = num[mask] / den[mask]
    return d


def _parse_key_token_to_str(token: str, prefix: str, token_name: str) -> str:
    text = str(token)
    pref = str(prefix or "")
    if pref:
        if not text.startswith(pref):
            raise ValueError(
                f"{token_name} token '{text}' does not start with expected prefix '{pref}'."
            )
        text = text[len(pref) :]
    if not text:
        raise ValueError(
            f"{token_name} token is empty after stripping prefix '{pref}'."
        )
    return text


def _parse_key_token_to_int(token: str, prefix: str, token_name: str) -> int:
    text = str(token)
    pref = str(prefix or "")
    if pref:
        if not text.startswith(pref):
            raise ValueError(
                f"{token_name} token '{text}' does not start with expected prefix '{pref}'."
            )
        text = text[len(pref) :]
    if not text:
        raise ValueError(
            f"{token_name} token is empty after stripping prefix '{pref}'."
        )
    try:
        value = float(text)
    except Exception as exc:
        raise ValueError(f"{token_name} token '{token}' is not numeric.") from exc
    if not np.isfinite(value) or abs(value - round(value)) > 1e-9:
        raise ValueError(f"{token_name} token '{token}' is not an integer value.")
    return int(round(value))


def _parse_matrix_column_key(
    column: Any,
    separator: str,
    group_prefix: str,
    step_prefix: str,
) -> tuple[str, int]:
    text = str(column)
    if separator not in text:
        raise ValueError(f"column '{text}' is missing separator '{separator}'.")
    leading_part, step_part = text.split(separator, 1)
    group_key = _parse_key_token_to_str(leading_part, group_prefix, "group")
    step_id = _parse_key_token_to_int(step_part, step_prefix, "step")
    return group_key, step_id


def _validate_matrix_column_keys(
    columns: list[Any],
    separator: str,
    group_prefix: str,
    step_prefix: str,
    max_errors: int = 5,
) -> None:
    if not columns:
        raise ValueError("matrix has no columns.")
    errors: list[str] = []
    for col in columns:
        try:
            _parse_matrix_column_key(col, separator, group_prefix, step_prefix)
        except Exception as exc:
            errors.append(str(exc))
            if len(errors) >= max_errors:
                break
    if errors:
        expected = f"<{group_prefix}group>{separator}{step_prefix}<step>"
        joined = "; ".join(errors)
        raise ValueError(f"{joined}. Expected format: {expected}.")


def _make_group_block_holdout_mask(
    M_df: pd.DataFrame,
    holdout_frac: float,
    seed: int = 0,
    separator: str = "__",
    group_prefix: str = "",
    step_prefix: str = "Step",
) -> np.ndarray:
    """
    Generates a holdout mask such that for each row, a fraction of the
    observed groups (where groups are defined by column name parsing)
    are held out entirely (all steps for those groups).

    This simulates the scenario of predicting values for new groups not seen during training.
    The column names of M_df are expected to encode group and step information in a consistent
    format that can be parsed by _parse_matrix_column_key.

    The holdout_frac parameter controls the fraction of observed groups to hold out for each row,
    and the seed parameter ensures reproducibility of the random selection.
    """
    rng = np.random.default_rng(seed)
    cols = list(M_df.columns)
    col_group: list[str] = []
    for col in cols:
        group_name, _ = _parse_matrix_column_key(
            col, separator, group_prefix, step_prefix
        )
        col_group.append(group_name)
    col_group_arr = np.array(col_group, dtype=str)

    group_to_colidx: dict[str, list[int]] = {}
    for j, group_name in enumerate(col_group_arr):
        group_to_colidx.setdefault(group_name, []).append(j)

    values = M_df.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    holdout = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        observed_cols = np.where(np.isfinite(values[i, :]))[0]
        if observed_cols.size == 0:
            continue
        observed_groups = np.unique(col_group_arr[observed_cols])
        n_hold = int(np.ceil(float(holdout_frac) * len(observed_groups)))
        if n_hold <= 0:
            continue
        held_groups = rng.choice(observed_groups, size=n_hold, replace=False)
        for group_name in held_groups:
            idxs = np.array(group_to_colidx.get(group_name, []), dtype=int)
            if idxs.size == 0:
                continue
            holdout[i, idxs] = np.isfinite(values[i, idxs])
    return holdout


def _make_random_entry_holdout_mask(
    M_df: pd.DataFrame,
    holdout_frac: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = M_df.to_numpy(dtype=float)
    observed = np.isfinite(values)
    holdout = np.zeros_like(observed, dtype=bool)
    obs_idx = np.argwhere(observed)
    if obs_idx.size == 0:
        return holdout
    n_hold = int(np.ceil(float(holdout_frac) * obs_idx.shape[0]))
    if n_hold <= 0:
        return holdout
    chosen = rng.choice(
        obs_idx.shape[0], size=min(n_hold, obs_idx.shape[0]), replace=False
    )
    selected = obs_idx[chosen]
    holdout[selected[:, 0], selected[:, 1]] = True
    return holdout


def _rmse_weighted(M_true: np.ndarray, M_pred: np.ndarray, W_eval: np.ndarray) -> float:
    w = np.asarray(W_eval, dtype=float)
    denom = float(np.sum(w))
    if denom <= 0:
        return float("nan")
    target = np.nan_to_num(M_true, nan=0.0)
    err2 = (target - M_pred) ** 2
    return float(np.sqrt(np.sum(w * err2) / denom))


def _nuisance_als_components(
    M: np.ndarray,
    W_input: np.ndarray,
    k: int,
    lam: float,
    n_iters: int,
    seed: int,
    bias_weight_mode: str,
) -> dict[str, np.ndarray]:
    obs = np.isfinite(M)
    X = np.nan_to_num(M, nan=0.0)
    W = np.asarray(W_input, dtype=float) * obs

    if bias_weight_mode == "binary":
        W_bias = obs.astype(float)
    elif bias_weight_mode == "qc":
        W_bias = W
    else:
        raise ValueError("bias_weight_mode must be 'binary' or 'qc'.")

    b, c = _fit_bias_terms(M, W_bias=W_bias, n_iters=8)
    R0 = X - b[:, None] - c[None, :]
    g = b.copy()
    g = g - np.average(g, weights=np.maximum(W_bias.sum(axis=1), 1e-12))
    d = _fit_global_rank1(R0, W, g=g)
    R = R0 - np.outer(g, d)

    U, V = _weighted_als(R, W, k=k, lam=lam, n_iters=n_iters, seed=seed)
    pred = b[:, None] + c[None, :] + np.outer(g, d) + (U @ V.T)
    return {
        "U": U,
        "V": V,
        "b": b,
        "c": c,
        "g": g,
        "d": d,
        "pred": pred,
        "R": R,
        "obs": obs,
    }


def _align_components_by_v(
    V_ref: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    k = V.shape[1]
    C = np.zeros((k, k), dtype=float)
    for i in range(k):
        a = (V_ref[:, i] - V_ref[:, i].mean()) / (V_ref[:, i].std() + 1e-12)
        for j in range(k):
            b = (V[:, j] - V[:, j].mean()) / (V[:, j].std() + 1e-12)
            C[i, j] = float(np.dot(a, b) / max(len(a), 1))

    perm = [-1] * k
    used: set[int] = set()
    for i in range(k):
        j_best, score_best = None, -1.0
        for j in range(k):
            if j in used:
                continue
            score = abs(C[i, j])
            if score > score_best:
                j_best = j
                score_best = score
        if j_best is None:
            raise ValueError("Failed to align components.")
        perm[i] = j_best
        used.add(j_best)
    perm_arr = np.array(perm, dtype=int)
    signs = np.sign(np.array([C[i, perm_arr[i]] for i in range(k)], dtype=float))
    signs[signs == 0] = 1.0
    return perm_arr, signs


def _pick_best_row(sweep_df: pd.DataFrame) -> pd.Series:
    required = {"k", "lambda"}
    missing = required - set(sweep_df.columns)
    if missing:
        raise ValueError(f"Sweep data missing required columns: {sorted(missing)}")
    ranked = sweep_df.copy()
    if "rmse_holdout_mean" in ranked.columns:
        ranked = ranked.sort_values(
            ["rmse_holdout_mean", "k", "lambda"], ascending=[True, True, False]
        )
    else:
        ranked = ranked.sort_values(["k", "lambda"], ascending=[True, False])
    return ranked.iloc[0]


def _extract_scalar_from_frame(
    frame: pd.DataFrame,
    label: str,
    preferred_columns: list[str] | None = None,
) -> object:
    if not isinstance(frame, pd.DataFrame):
        raise BlockValidationError(f"{label} input must be a DataFrame.")
    if frame.empty:
        raise BlockValidationError(f"{label} input is empty.")

    preferred = [str(col) for col in (preferred_columns or [])]
    for col in preferred:
        if col in frame.columns:
            series = frame[col].dropna()
            if not series.empty:
                return series.iloc[0]

    if "value" in frame.columns:
        series = frame["value"].dropna()
        if not series.empty:
            return series.iloc[0]

    if frame.shape[0] == 1 and frame.shape[1] == 1:
        return frame.iloc[0, 0]

    non_null = frame.stack(dropna=True)  # pyright: ignore[reportCallIssue]
    if not non_null.empty:
        return non_null.iloc[0]

    raise BlockValidationError(f"{label} input has no non-null values.")


class WeightedALSFactorization(BaseBlock):
    name = "Weighted ALS Factorization"
    version = "1.1.0"
    category = "Factorization"
    description = (
        "Run weighted ALS matrix factorization using an observed-value weight matrix."
    )
    n_inputs = 2
    input_labels = ["Matrix", "Weights"]
    output_labels = ["U matrix", "V matrix"]

    class Params(BlockParams):
        n_components: int = 20
        lambda_value: float = 1.0
        n_iters: int = 15
        seed: int = 0
        output_prefix: str = "program_"

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("WeightedALSFactorization requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("WeightedALSFactorization requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "WeightedALSFactorization expects [matrix_df, weight_df]."
            )
        matrix_df, weight_df = data
        M = matrix_df.to_numpy(dtype=float)
        W = weight_df.reindex(
            index=matrix_df.index, columns=matrix_df.columns
        ).to_numpy(dtype=float)
        k = min(max(int(params.n_components), 1), matrix_df.shape[1])
        U, V = _weighted_als(
            M,
            W,
            k=k,
            lam=float(params.lambda_value),
            n_iters=int(params.n_iters),
            seed=int(params.seed),
        )

        cols = [f"{params.output_prefix}{i + 1}" for i in range(k)]
        u_df = pd.DataFrame(U, index=matrix_df.index.astype(str), columns=cols)
        v_df = pd.DataFrame(V, index=matrix_df.columns.astype(str), columns=cols)
        return BlockOutput(
            data=u_df,
            outputs={"output_0": u_df, "output_1": v_df},
            metadata={"v_matrix": v_df.to_dict(orient="split")},
        )


class NuisanceALSSweep(BaseBlock):
    name = "Nuisance ALS Sweep"
    version = "1.1.0"
    category = "Factorization"
    description = "Sweep k/lambda for nuisance+bias+weighted ALS and evaluate weighted RMSE on generated holdout masks."
    param_descriptions = {
        "k_values": "Comma-separated candidate latent dimensions to test (for example: 20,40,80).",
        "lambda_values": "Comma-separated regularization values to test (for example: 0.5,1.0,2.0).",
        "n_repeats": "Number of independent holdout/evaluation repeats per (k, lambda) pair.",
        "n_iters": "ALS iterations per training run.",
        "holdout_frac": "Fraction of observed entries to hold out for validation on each repeat.",
        "sample_rows": "Optional row subsample size for a faster sweep; null means use all rows.",
        "bias_weight_mode": "Bias fitting weights: 'binary' uses observed mask, 'qc' uses provided QC weights.",
        "seed_base": "Base random seed used to derive deterministic per-repeat seeds.",
        "column_separator": "Delimiter between group and step tokens in matrix column keys (for example '__').",
        "group_prefix": "Optional prefix before the group token in column keys.",
        "step_prefix": "Optional prefix before the numeric step token in column keys (for example 'Step').",
        "validate_column_keys": "Validate matrix columns match the configured key schema before sweep starts.",
        "holdout_strategy": "Holdout generator: group_block or random_entry.",
    }
    n_inputs = 2
    input_labels = ["Matrix", "Weights"]
    output_labels = ["Sweep table"]

    class Params(BlockParams):
        k_values: str = "20,30,40,80,120,240"
        lambda_values: str = "1.0"
        n_repeats: int = 2
        n_iters: int = 15
        holdout_frac: float = 0.05
        sample_rows: int | None = None
        bias_weight_mode: str = "binary"
        seed_base: int = 100000
        column_separator: str = "__"
        group_prefix: str = ""
        step_prefix: str = "Step"
        validate_column_keys: bool = True
        holdout_strategy: str = "group_block"

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 2 or any(d is None for d in data[:2]):
            raise InsufficientInputs("NuisanceALSSweep requires both inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("NuisanceALSSweep requires params.")
        if not isinstance(data, list) or len(data) != 2:
            raise BlockValidationError(
                "NuisanceALSSweep expects [matrix_df, weight_df]."
            )

        matrix_df, weight_df = data
        M_df = matrix_df.copy()
        W_df = weight_df.reindex(index=M_df.index, columns=M_df.columns).copy()

        sample_rows = getattr(params, "sample_rows", None)
        if (
            sample_rows is not None
            and int(sample_rows) > 0
            and int(sample_rows) < M_df.shape[0]
        ):
            rng = np.random.default_rng(0)
            keep = rng.choice(
                M_df.index.to_numpy(), size=int(sample_rows), replace=False
            )
            M_df = M_df.loc[keep].copy()
            W_df = W_df.loc[keep].copy()

        M = M_df.to_numpy(dtype=float)
        W = W_df.to_numpy(dtype=float)
        W = W * np.isfinite(M)

        k_values = _parse_int_list(getattr(params, "k_values", "20"), default=[20])
        lambda_values = _parse_float_list(
            getattr(params, "lambda_values", "1.0"), default=[1.0]
        )
        k_values = [k for k in k_values if k > 0 and k <= M_df.shape[1]]
        if not k_values:
            raise BlockValidationError("No valid k values for current matrix width.")

        n_repeats = max(int(getattr(params, "n_repeats", 2)), 1)
        holdout_frac = float(getattr(params, "holdout_frac", 0.05))
        seed_base = int(getattr(params, "seed_base", 100000))
        n_iters = int(getattr(params, "n_iters", 15))
        bias_weight_mode = str(getattr(params, "bias_weight_mode", "binary"))
        column_separator = str(getattr(params, "column_separator", "__"))
        group_prefix = str(getattr(params, "group_prefix", "") or "")
        step_prefix = str(getattr(params, "step_prefix", "Step") or "")
        validate_column_keys = bool(getattr(params, "validate_column_keys", True))
        holdout_strategy = (
            str(getattr(params, "holdout_strategy", "group_block")).strip().lower()
        )
        if validate_column_keys and holdout_strategy in {"group_block", "curve_block"}:
            try:
                _validate_matrix_column_keys(
                    list(M_df.columns),
                    separator=column_separator,
                    group_prefix=group_prefix,
                    step_prefix=step_prefix,
                )
            except Exception as exc:
                raise BlockValidationError(
                    f"Matrix columns do not match configured key format: {exc}"
                ) from exc

        tasks = [
            (float(lam), int(k), int(repeat))
            for lam in lambda_values
            for k in k_values
            for repeat in range(n_repeats)
        ]
        metrics: dict[tuple[int, float], dict[str, list[float]]] = {}
        for lam, k, repeat in ProgressBar(
            tasks,
            total=len(tasks),
            label="Sweeping nuisance ALS",
            throttle_seconds=0.2,
        ):
            key = (int(k), float(lam))
            bucket = metrics.setdefault(key, {"train": [], "holdout": []})
            seed = seed_base + repeat + 1000 * (k + int(round(1000 * lam)))

            if holdout_strategy in {"group_block", "curve_block"}:
                holdout_mask = _make_group_block_holdout_mask(
                    M_df,
                    holdout_frac=holdout_frac,
                    seed=seed,
                    separator=column_separator,
                    group_prefix=group_prefix,
                    step_prefix=step_prefix,
                )
            elif holdout_strategy == "random_entry":
                holdout_mask = _make_random_entry_holdout_mask(
                    M_df,
                    holdout_frac=holdout_frac,
                    seed=seed,
                )
            else:
                raise BlockValidationError(
                    "holdout_strategy must be one of: group_block, random_entry."
                )

            W_train = W.copy()
            W_train[holdout_mask] = 0.0
            W_hold = np.zeros_like(W)
            W_hold[holdout_mask] = W[holdout_mask]

            model = _nuisance_als_components(
                M=M,
                W_input=W_train,
                k=int(k),
                lam=float(lam),
                n_iters=n_iters,
                seed=seed,
                bias_weight_mode=bias_weight_mode,
            )
            pred = model["pred"]
            bucket["train"].append(_rmse_weighted(M, pred, W_train))
            bucket["holdout"].append(_rmse_weighted(M, pred, W_hold))

        rows: list[dict[str, float | int | str]] = []
        for lam in lambda_values:
            for k in k_values:
                key = (int(k), float(lam))
                bucket = metrics.get(key)
                if bucket is None:
                    continue
                rmse_train = bucket["train"]
                rmse_holdout = bucket["holdout"]
                rows.append(
                    {
                        "k": int(k),
                        "lambda": float(lam),
                        "n_iters": n_iters,
                        "holdout_frac": holdout_frac,
                        "n_repeats": n_repeats,
                        "sample_rows": int(M_df.shape[0]),
                        "bias_weight_mode": bias_weight_mode,
                        "column_separator": column_separator,
                        "group_prefix": group_prefix,
                        "step_prefix": step_prefix,
                        "rmse_train_mean": float(np.nanmean(rmse_train)),
                        "rmse_holdout_mean": float(np.nanmean(rmse_holdout)),
                        "rmse_train_std": float(np.nanstd(rmse_train)),
                        "rmse_holdout_std": float(np.nanstd(rmse_holdout)),
                    }
                )

        result = (
            pd.DataFrame(rows)
            .sort_values(["rmse_holdout_mean", "k", "lambda"])
            .reset_index(drop=True)
        )
        best = result.iloc[0]
        holdout_strategy_out = (
            "group_block" if holdout_strategy == "curve_block" else holdout_strategy
        )
        return BlockOutput(
            data=result,
            metadata={
                "best_k": int(best["k"]),
                "best_lambda": float(best["lambda"]),
                "best_rmse_holdout_mean": float(best["rmse_holdout_mean"]),
                "best_rmse_train_mean": float(best["rmse_train_mean"]),
                "holdout_strategy": holdout_strategy_out,
                "column_separator": column_separator,
                "group_prefix": group_prefix,
                "step_prefix": step_prefix,
            },
        )


class NuisanceALS(BaseBlock):
    name = "Nuisance ALS"
    aliases = [
        "NuisanceALSConsensus",
        "NuisanceALSResidualMatrix",
        "Nuisance ALS Consensus",
        "Nuisance ALS Background-Corrected Matrix",
        "Nuisance ALS Residual Matrix",
    ]
    version = "1.1.0"
    category = "Factorization"
    description = "Fit nuisance + weighted ALS using provided k/lambda inputs and emit consensus U/V plus background-corrected matrix."
    n_inputs = 4
    input_labels = ["Matrix", "Weights", "K", "Lambda"]
    output_labels = ["Consensus U", "Consensus V", "Background Corrected M"]
    usage_notes = [
        "Inputs 3 and 4 provide the k and lambda scalars used for fitting.",
        "Legacy pipeline params like use_best_from_sweep, k, and lambda_value are ignored for compatibility.",
    ]

    class Params(BlockParams):
        model_config = ConfigDict(extra="ignore", validate_assignment=True)

        n_iters: int = 15
        seeds: str = "0,1,2"
        bias_weight_mode: str = "binary"
        row_bias_column: str = "row_bias"
        output_prefix: str = "program_"

    @classmethod
    def normalize_params_payload(
        cls, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = super().normalize_params_payload(params)
        payload.pop("use_best_from_sweep", None)
        payload.pop("k", None)
        payload.pop("lambda_value", None)
        return payload

    def validate(self, data: Any) -> None:
        if not isinstance(data, list) or len(data) < 4 or any(d is None for d in data[:4]):
            raise InsufficientInputs("NuisanceALS requires all four inputs.")

    def execute(
        self, data: list[pd.DataFrame], params: Params | None = None
    ) -> BlockOutput:
        if params is None:
            raise BlockValidationError("NuisanceALS requires params.")
        if not isinstance(data, list) or len(data) != 4:
            raise BlockValidationError(
                "NuisanceALS expects [matrix_df, weight_df, k_df, lambda_df]."
            )
        matrix_df, weight_df, k_df, lambda_df = data

        k_raw = _extract_scalar_from_frame(
            k_df, label="k", preferred_columns=["k", "best_k"]
        )
        lambda_raw = _extract_scalar_from_frame(
            lambda_df, label="lambda", preferred_columns=["lambda", "best_lambda"]
        )
        try:
            k = int(round(float(k_raw)))  # type: ignore
        except Exception as exc:
            raise BlockValidationError(
                f"Failed to parse k from value {k_raw!r}."
            ) from exc
        try:
            lam = float(lambda_raw)  # type: ignore
        except Exception as exc:
            raise BlockValidationError(
                f"Failed to parse lambda from value {lambda_raw!r}."
            ) from exc

        if not np.isfinite(k) or k <= 0:
            raise BlockValidationError("k must be a positive integer.")
        if not np.isfinite(lam) or lam <= 0:
            raise BlockValidationError("lambda must be a positive number.")

        n_iters = int(params.n_iters)
        bias_weight_mode = params.bias_weight_mode

        k = min(max(k, 1), matrix_df.shape[1])
        seed_values = _parse_int_list(params.seeds, default=[0, 1, 2])
        if not seed_values:
            raise BlockValidationError("No seeds provided for nuisance ALS.")

        M = matrix_df.to_numpy(dtype=float)
        W = weight_df.reindex(
            index=matrix_df.index, columns=matrix_df.columns
        ).to_numpy(dtype=float)

        runs = []
        for seed in ProgressBar(
            seed_values,
            label="Fitting nuisance ALS seed runs",
            throttle_seconds=0.2,
        ):
            runs.append(
                _nuisance_als_components(
                    M=M,
                    W_input=W,
                    k=k,
                    lam=lam,
                    n_iters=n_iters,
                    seed=seed,
                    bias_weight_mode=bias_weight_mode,
                )
            )

        V_ref = runs[0]["V"]
        aligned_U = [runs[0]["U"]]
        aligned_V = [V_ref]
        corrs: list[np.ndarray] = []
        for run in ProgressBar(
            runs[1:],
            label="Aligning nuisance ALS components",
            throttle_seconds=0.2,
        ):
            perm, signs = _align_components_by_v(V_ref, run["V"])
            aligned_V_i = run["V"][:, perm] * signs[None, :]
            aligned_U_i = run["U"][:, perm] * signs[None, :]
            aligned_V.append(aligned_V_i)
            aligned_U.append(aligned_U_i)

            corr_vals = []
            for j in range(k):
                a = (V_ref[:, j] - V_ref[:, j].mean()) / (V_ref[:, j].std() + 1e-12)
                b = (aligned_V_i[:, j] - aligned_V_i[:, j].mean()) / (
                    aligned_V_i[:, j].std() + 1e-12
                )
                corr_vals.append(float(np.dot(a, b) / max(len(a), 1)))
            corrs.append(np.array(corr_vals, dtype=float))

        U_cons = np.mean(np.stack(aligned_U, axis=0), axis=0)
        V_cons = np.mean(np.stack(aligned_V, axis=0), axis=0)
        b_cons = np.mean(np.stack([run["b"] for run in runs], axis=0), axis=0)
        c_cons = np.mean(np.stack([run["c"] for run in runs], axis=0), axis=0)
        d_cons = np.mean(np.stack([run["d"] for run in runs], axis=0), axis=0)
        g_cons = b_cons - np.mean(b_cons)

        X = np.nan_to_num(M, nan=0.0)
        obs = np.isfinite(M)
        residual = X - b_cons[:, None] - c_cons[None, :] - np.outer(g_cons, d_cons)
        residual[~obs] = np.nan

        cols = [f"{params.output_prefix}{i + 1}" for i in range(k)]
        u_df = pd.DataFrame(U_cons, index=matrix_df.index.astype(str), columns=cols)
        u_df[params.row_bias_column] = b_cons
        v_df = pd.DataFrame(V_cons, index=matrix_df.columns.astype(str), columns=cols)
        residual_df = pd.DataFrame(
            residual,
            index=matrix_df.index.astype(str),
            columns=matrix_df.columns.astype(str),
        )

        stability = None
        if corrs:
            corr_mat = np.abs(np.stack(corrs, axis=1))
            stability = {
                "mean_abs_corr_to_seed0": corr_mat.mean(axis=1).tolist(),
                "min_abs_corr_to_seed0": corr_mat.min(axis=1).tolist(),
            }

        metadata = {
            "k": k,
            "lambda": lam,
            "n_iters": n_iters,
            "bias_weight_mode": bias_weight_mode,
            "v_columns": list(matrix_df.columns.astype(str)),
            "v_matrix": V_cons.tolist(),
            "col_bias": c_cons.tolist(),
            "global_direction": d_cons.tolist(),
            "global_row_direction": g_cons.tolist(),
            "stability": stability,
            "row_bias_column": params.row_bias_column,
        }
        return BlockOutput(
            data=u_df,
            outputs={
                "output_0": u_df,
                "output_1": v_df,
                "output_2": residual_df,
            },
            metadata=metadata,
        )
