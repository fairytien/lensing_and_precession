# Replace `use_opt_match`/`compare_both` with `MatchMethod` Enum

Replace the two-boolean match-method selection (`use_opt_match`, `compare_both`) with a single `MatchMethod` enum across `match_utils.py`, `bank_io.py`, and all non-legacy caller scripts.

## Enum Definition

```python
class MatchMethod(Enum):
    MATCH = "match"                          # pycbc.filter.match (FFT discrete)
    OPTIMIZED_BRENT = "optimized_brent"      # pycbc.filter.optimized_match (Brent)
    OPTIMIZED_BOUNDED = "optimized_bounded"  # optimized_match_bounded() (scipy bounded)
    COMPARE_BOTH = "compare_both"            # run MATCH + OPTIMIZED_BRENT, take best
```

**Rationale for `COMPARE_BOTH`**: The Brent minimizer in `OPTIMIZED_BRENT` can return numerically unreliable results. `COMPARE_BOTH` runs both `MATCH` and `OPTIMIZED_BRENT` and takes the best, hedging against Brent failures. `OPTIMIZED_BOUNDED` resolves this issue entirely, so it stands alone as the default.

## Proposed Changes

### `match_utils.py` — Enum definition and core API

#### [MODIFY] [match_utils.py](../modules/match_utils.py)

**1. Add `MatchMethod` enum and import `optimized_match`** (top of file, after existing imports):

```python
from enum import Enum
from pycbc.filter import match, optimized_match

class MatchMethod(Enum):
    MATCH = "match"
    OPTIMIZED_BRENT = "optimized_brent"
    OPTIMIZED_BOUNDED = "optimized_bounded"
    COMPARE_BOTH = "compare_both"
```

**2. Update [mismatch_from_strains](../modules/match_utils.py#L271-L332)** — replace `use_opt_match`/`compare_both` with `match_method`:

```python
def mismatch_from_strains(
    t_strain, s_strain,
    f_min=20, delta_f=0.25,
    psd=None,
    match_method: MatchMethod = MatchMethod.OPTIMIZED_BOUNDED,
    sn_func=None,
) -> dict:
```

Dispatch logic:

```python
    _COMPARE_BOTH_METHODS = {
        MatchMethod.MATCH: match,
        MatchMethod.OPTIMIZED_BRENT: optimized_match,
    }

    if match_method is MatchMethod.COMPARE_BOTH:
        results = []
        for method, func in _COMPARE_BOTH_METHODS.items():
            try:
                match_val, index, phi = func(
                    t_strain, s_strain, psd, return_phase=True
                )
                results.append({
                    "mismatch": 1 - match_val,
                    "index": index,
                    "phi": phi,
                    "match_val": match_val,
                    "match_method": method.value,
                })
            except Exception:
                continue
        if not results:
            raise RuntimeError("Both match and optimized_match failed.")
        best = max(results, key=lambda x: x["match_val"])
        return {k: v for k, v in best.items() if k != "match_val"}

    _DISPATCH = {
        MatchMethod.MATCH: lambda t, s, p: match(t, s, p, return_phase=True),
        MatchMethod.OPTIMIZED_BRENT: lambda t, s, p: optimized_match(t, s, psd=p, return_phase=True),
        MatchMethod.OPTIMIZED_BOUNDED: lambda t, s, p: optimized_match_bounded(t, s, psd=p, return_phase=True),
    }
    match_val, index, phi = _DISPATCH[match_method](t_strain, s_strain, psd)
    return {"mismatch": 1 - match_val, "index": index, "phi": phi}
```

> [!NOTE]
> `match()` takes `psd` positionally, while `optimized_match` and `optimized_match_bounded` take `psd=` as keyword. The dispatch lambdas handle this difference (same as existing code lines 324–330).

**3. Update 5 forwarding functions** — each drops `use_opt_match` + `compare_both`, gains `match_method`:

| Function | Lines | Change |
|----------|-------|--------|
| [mismatch_from_params](../modules/match_utils.py#L335-L384) | 343–384 | Replace 2 params → 1; forward `match_method` to `mismatch_from_strains` |
| [optimize_mismatch_mcz](../modules/match_utils.py#L392-L479) | 400–468 | Same |
| [optimize_mismatch_gammaP](../modules/match_utils.py#L482-L682) | 491–675 | Same |
| [find_optimized_coalescence_params](../modules/match_utils.py#L685-L803) | 693–803 | Same; inner `_evaluate_current_mismatch` passes `match_method` |

**4. Update multiprocessing worker globals** — collapse `_COMPARE_BOTH` + `_USE_OPT_MATCH` into single `_MATCH_METHOD`:

| Symbol | Lines | Change |
|--------|-------|--------|
| Globals `_COMPARE_BOTH`, `_USE_OPT_MATCH` | 881–882 | → `_MATCH_METHOD: Optional[MatchMethod] = None` |
| [init_mismatch_worker](../modules/match_utils.py#L912-L956) | 912–956 | Accept `match_method` instead of `compare_both, use_opt_match` |
| [_require_worker_state](../modules/match_utils.py#L889-L909) | 889–909 | Return `_MATCH_METHOD` instead of both globals |
| [mismatch_gamma_job](../modules/match_utils.py#L959-L1015) | 959–1015 | Pass `match_method=` to `mismatch_from_strains` |

---

### `bank_io.py` — Provenance attrs

#### [MODIFY] [bank_io.py](../modules/bank_io.py)

Update [write_match_provenance_attrs](../modules/bank_io.py#L276-L294):

```python
def write_match_provenance_attrs(
    h5: h5py.File,
    *,
    match_method: "MatchMethod",
) -> None:
    from modules.match_utils import MatchMethod
    _MINIMIZER = {
        MatchMethod.MATCH: "discrete",
        MatchMethod.OPTIMIZED_BRENT: "brent",
        MatchMethod.OPTIMIZED_BOUNDED: "bounded",
        MatchMethod.COMPARE_BOTH: "brent_and_discrete",
    }
    _write_attrs(h5, {
        "match_method": match_method.value,
        "minimizer": _MINIMIZER[match_method],
    })
```

---
### `cli_utils.py` — Centralized CLI Argument Helper

#### [NEW] [add_match_method_arg](../modules/cli_utils.py#L243-L254)

Add centralized CLI argument helper function to attach the `--match_method` option to an argparse parser:

```python
def add_match_method_arg(parser: ArgumentParser) -> ArgumentParser:
    """Attach the --match_method choice argument."""
    from modules.match_utils import MatchMethod

    parser.add_argument(
        "--match_method",
        type=str,
        choices=[m.value for m in MatchMethod],
        default=MatchMethod.OPTIMIZED_BOUNDED.value,
        help="Match method to use. Default: optimized_bounded",
    )
    return parser
```

---

### Caller scripts

Instead of declaring `--match_method` inline, all caller scripts import `add_match_method_arg` from `modules.cli_utils` to register the new argument, replacing the two legacy boolean flags (`--compare_both` and `--use_opt_match`). They then parse with `match_method=MatchMethod(args.match_method)`.

#### [MODIFY] [compute_mismatch_cubes.py](../scripts/mismatch_mcz_td/compute_mismatch_cubes.py)

- Function signature: `compare_both: bool, use_opt_match: bool` → `match_method: MatchMethod`
- CLI: replace the two flags with `add_match_method_arg(p)`
- `init_mismatch_worker` call: pass `match_method` instead of `compare_both, use_opt_match`
- `write_match_provenance_attrs` call: pass `match_method=match_method`

#### [MODIFY] [v4_indiv_contour_otf.py](../scripts/contour_omega_theta/v4_indiv_contour_otf.py)

- Collapse the `if compare_both: ... else: ...` branch in `_compute_cell_min_ep` into a single `optimize_mismatch_gammaP` call with `match_method=...`
- Function signature + CLI: same pattern as `compute_mismatch_cubes.py` (using `add_match_method_arg`)

#### [MODIFY] [v3_indiv_contour_otf.py](../scripts/contour_omega_theta/v3_indiv_contour_otf.py)

- Same: collapse branch in `_compute_cell_min_ep`, update signature + CLI (using `add_match_method_arg`)

#### [MODIFY] [v3_indiv_contour_otf_v2prec.py](../scripts/contour_omega_theta/v3_indiv_contour_otf_v2prec.py)

- Same: collapse branch in `_compute_cell_min_ep`, update signature + CLI (using `add_match_method_arg`)

#### [MODIFY] [contour_L_NP_mcz_td.py](../scripts/contour_mcz_td/contour_L_NP_mcz_td.py)

- Function signature + CLI: same pattern (using `add_match_method_arg`)
- `_compute_mismatch_row`: unpack `match_method` instead of `compare_both`; pass to `mismatch_from_params`/`optimize_mismatch_mcz`

#### [MODIFY] `mismatch_I_td` and `np_fast` Sweep/Pipeline Scripts

The same CLI refactoring pattern (using `add_match_method_arg`) is applied to:
- [compute_mismatch_cubes.py (I_td)](../scripts/mismatch_I_td/compute_mismatch_cubes.py)
- [compute_mismatch_mcz_td.py (np_fast)](../scripts/np_fast/compute_mismatch_mcz_td.py)
- [compute_mismatch_I_td.py (np_fast)](../scripts/np_fast/compute_mismatch_I_td.py)

---

### Compatibility shims

#### [MODIFY] [functions.py](../modules/functions.py)

Add `MatchMethod` to the re-export list:

```python
from modules.match_utils import (
    MatchMethod,
    find_optimized_coalescence_params,
    ...
)
```

#### [MODIFY] [functions_v3.py](../modules/functions_v3.py)

Same — add `MatchMethod` to the re-export list.

---

### Files NOT changed

- **Legacy scripts** (`legacy/scripts/`, `legacy/modules/`): use their own copy of functions, not touched.
- **`waveform_plotting.py`**: passes `**kwargs` to `find_optimized_coalescence_params` — no signature change needed. Existing callers rely on defaults (OPTIMIZED_BOUNDED), so no breakage.

## Verification Plan

### Automated

1. `get_errors` on every touched file
2. Import smoke test:
   ```bash
   python -c "from modules.match_utils import MatchMethod; print(list(MatchMethod))"
   ```
3. CLI `--help` for each modified script:
   ```bash
   python -m scripts.mismatch_mcz_td.compute_mismatch_cubes --help
   python -m scripts.contour_omega_theta.v3_indiv_contour_otf --help
   python -m scripts.contour_omega_theta.v3_indiv_contour_otf_v2prec --help
   python -m scripts.contour_omega_theta.v4_indiv_contour_otf --help
   python -m scripts.contour_mcz_td.contour_L_NP_mcz_td --help
   ```

### Manual

- Existing HDF5 outputs remain readable — `match_method` and `minimizer` attrs are informational strings; readers (`read_source_attrs`) propagate them without validation.
