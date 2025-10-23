# Polarization Guard Analysis: tan(ψ) Singularities

**Date:** October 23, 2025  
**Analysis of:** Asymptotic guard necessity for `Classes_v2.py` and `Classes_v3.py`

---

## Executive Summary

Summary for two regimes:

- Small precession (e.g., θ̃ ≲ 0.5): The asymptotic guard is not necessary; algebraic formulas are naturally stable and empirically identical to the guarded version.
- Moderate to large precession (θ̃ up to 15): The asymptotic guard is recommended to prevent NaNs when den_psi crosses zero (tanψ → ±∞). The code has been updated to include a safe guard in both v2 and v3.

---

## Background

### The Polarization Angle Formula

The polarization angle ψ is defined by:

```
tan(ψ) = num_psi / den_psi
```

where:
```python
num_psi = sin(θ_LJ) * [cos(φ_LJ)*sin(o_XH) + sin(φ_LJ)*cos(i_JN)*cos(o_XH)]
          - cos(θ_LJ) * sin(i_JN) * cos(o_XH)

den_psi = sin(θ_LJ) * [cos(φ_LJ)*cos(o_XH) - sin(φ_LJ)*cos(i_JN)*sin(o_XH)]
          + cos(θ_LJ) * sin(i_JN) * sin(o_XH)
```

### Potential Singularity

When `den_psi → 0`, `tan(ψ) → ±∞`, which could cause numerical issues in downstream calculations:
```python
sin(2ψ+α) = (2*cos(α)*T + sin(α)*(1-T²)) / (1+T²)
cos(2ψ+α) = (cos(α)*(1-T²) - 2*sin(α)*T) / (1+T²)
```
where `T = tan(ψ)`.

### Asymptotic Behavior

Mathematical analysis shows that as `T → ±∞`:
- sin(2ψ+α) → -sin(α)
- cos(2ψ+α) → -cos(α)

This motivated consideration of an explicit guard:
```python
T_bad = ~np.isfinite(T) | (np.abs(T) > 1e12) | (np.abs(den_psi) < 1e-12)
sin_2pa = np.where(T_bad, -sin_alpha, sin_2pa_algebraic)
cos_2pa = np.where(T_bad, -cos_alpha, cos_2pa_algebraic)
```

---

## Empirical Testing

### Test Setup

Script: `diagnostic_scripts/compare_polarization_guard.py`

**Parameters:**
- Frequency: f = 50 Hz
- Fixed: θ_S = π/2, φ_S = 0, φ_J = 0
- Swept: θ_J from 0° to 180° (500 points)
- Precession: θ_tilde = 0.3, ω_tilde = 0.1
- System: m_chirp = 30 M☉, η = 0.25

**Comparison:**
1. **Without guard**: Raw algebraic formulas (current implementation)
2. **With guard**: Asymptotic fallback when |T| > 10¹² or |den_psi| < 10⁻¹²

### Results (θ̃ = 0.3)

![Polarization Guard Comparison (θ̃ = 0.3)](../figures/polarization_guard_compare/polarization_guard_comparison_thetaT_0.3_f_50.png)

#### Key Observations:

1. **tan(ψ) remains bounded**: -40 < tan(ψ) < +40
   - No infinities observed across the full parameter sweep
   
2. **Denominator stays safe**: |den_psi| > 10⁻¹¹
   - Well above the proposed guard threshold (10⁻¹²)
   - No zero-crossings in this parameter regime

3. **Identical results**: 
   - Blue (without guard) and orange (with guard) curves **perfectly overlap**
   - No visible differences in middle comparison panels
   
4. **Zero differences**:
   - Bottom panels show: `with_guard - without_guard ≈ 0` everywhere
   - No numerical artifacts, spikes, or discontinuities

5. **Sharp feature is physical**:
   - The rapid transition around θ_J ≈ 90-100° is a legitimate physical effect
   - Both guarded and unguarded versions handle it identically

---

## Mathematical Explanation

### Why tan(ψ) Stays Bounded

#### 1. Small θ_LJ at f = 50 Hz

The precession opening angle is given by:
```python
θ_LJ = (0.1 / (4*η)) * θ_tilde * (f/f_cut)^(1/3)
```

With our parameters:
- η = 0.25 → prefactor = 0.1
- θ_tilde = 0.3
- f = 50 Hz, f_cut ≈ few hundred Hz → (f/f_cut)^(1/3) ≈ 0.5

**Result:** θ_LJ ≈ 0.015 rad ≈ 0.86° (very small!)

#### 2. Both Numerator and Denominator Scale Similarly

When θ_LJ ≪ 1:
- sin(θ_LJ) ≈ θ_LJ ≈ 0.015 (tiny)
- cos(θ_LJ) ≈ 1

Both `num_psi` and `den_psi` have terms proportional to sin(θ_LJ), so:
- num_psi ~ O(0.01)
- den_psi ~ O(0.01)

**Result:** tan(ψ) = O(0.01) / O(0.01) = O(1) (finite!)

#### 3. Algebraic Formulas Have Natural Cancellation

The formulas for sin(2ψ+α) and cos(2ψ+α) have **built-in numerical stability**:

```python
sin(2ψ+α) = (2*cos(α)*T + sin(α)*(1-T²)) / (1+T²)
```

When T → ±∞:
- **Numerator:** 2*cos(α)*T + sin(α)*(1-T²) → 2*cos(α)*T - sin(α)*T² → **-sin(α)*T²** (T² dominates)
- **Denominator:** 1 + T² → **T²**
- **Ratio:** -sin(α)*T² / T² = **-sin(α)** ✓ (correct asymptotic limit!)

The algebraic cancellation **automatically** produces the correct asymptotic behavior through floating-point arithmetic, without requiring explicit guards.

#### 4. No Geometric Cancellation in This Sweep

For den_psi to cross zero, you need special alignments where:
```
sin(θ_LJ) * [...] = -cos(θ_LJ) * sin(i_JN) * sin(o_XH)
```

In the tested parameter regime (θ_S = π/2, φ_S = 0, φ_J = 0, varying θ_J), this **cancellation doesn't occur**.

---

## When Would Singularities Appear?

### Conditions Required for tan(ψ) → ±∞:

1. **Higher frequency**: f ≫ 100 Hz
   - Makes θ_LJ larger: θ_LJ ∝ f^(1/3)
   - Example: f = 500 Hz → θ_LJ ~ 0.025 rad

2. **Larger precession amplitude**: θ_tilde ≫ 0.3
   - Directly scales θ_LJ: θ_LJ ∝ θ_tilde
   - Examples:
     - θ_tilde = 3.0 → θ_LJ 10× larger
     - θ_tilde = 15 → θ_LJ ~ 50× larger than θ̃=0.3 case; θ_LJ ≈ O(1) rad at 50 Hz

3. **Special geometric alignments**:
   - Parameter combinations where den_psi numerically cancels
   - Would require careful tuning of (θ_S, φ_S, θ_J, φ_J)

### Current Physical Regime:

- **Inspiral frequencies**: 20 Hz < f < 300 Hz (most of the signal)
- **Typical precession**: θ_tilde ~ 0.1-0.5 (moderate precession)
- **Random orientations**: Generic source-observer geometries

In these physically relevant scenarios, θ_LJ stays small enough that singularities are unlikely to develop.

---

## Recommendations

### ✅ Use the asymptotic guard (now implemented in v2 and v3)

Rationale for enabling the guard by default:

1. For θ̃ up to 15, den_psi can cross zero for certain orientations, making tanψ → ±∞ and leading to NaNs in the algebraic expressions due to inf − inf cancellation in the numerators.
2. The guard replaces such pathological evaluations with the correct mathematical limits: sin(2ψ+α) → −sinα and cos(2ψ+α) → −cosα.
3. For moderate precession (θ̃ ≲ 0.5), the guard is a no-op (results match exactly with and without it), so there’s no downside.

The updated implementation keeps the hybrid face-on handling and adds the guard:

```python
# After computing tan_psi and den_psi
sin_2pa_alg = (2*cos_alpha*tan_psi + sin_alpha*(1 - tan_psi**2)) / (1 + tan_psi**2)
cos_2pa_alg = (cos_alpha*(1 - tan_psi**2) - 2*sin_alpha*tan_psi) / (1 + tan_psi**2)

den_small = np.abs(den_psi) < 1e-12
T = tan_psi
T_bad = ~np.isfinite(T) | (np.abs(T) > 1e12) | den_small

sin_2pa = np.where(T_bad, -sin_alpha, sin_2pa_alg)
cos_2pa = np.where(T_bad, -cos_alpha, cos_2pa_alg)
```

### 📝 Documentation

The current inline comments adequately explain:
- The hybrid approach rationale
- Face-on special case handling
- Correction term in integrand_delta_phi

No further documentation needed for the guard decision.

---

## Testing Scripts

### Available Comparison Scripts:

1. **`diagnostic_scripts/compare_polarization_guard.py`** (this analysis)
   - Tests with/without asymptotic guard
   - Sweeps θ_J from 0° to 180°
   - Output: `diagnostic_scripts/figures/polarization_guard_compare/polarization_guard_comparison.png`

2. **`diagnostic_scripts/compare_polarization_faceon.py`**
   - Visualizes polarization near face-on/face-off
   - Angle sweeps, frequency sweeps, narrow zooms
   - Validates hybrid implementation

3. **`diagnostic_scripts/compare_integrand_delta_phi.py`**
   - Shows impact of correction term
   - Frequency and angle sweeps
   - Validates integrand changes

4. **`diagnostic_scripts/compare_phase_delta_phi.py`**
   - Cumulative phase differences
   - Validates phase accumulation

### Running the Guard Test:

```bash
cd /Users/fairytien/Documents/TEXAS_Bridge_2324/code/lensing_and_precession
python diagnostic_scripts/compare_polarization_guard.py --theta-tilde 15 --freq 50 --save
```

---

## References

### Code Files:
- `lensing_and_precession/modules/Classes_v2.py`: Main implementation (uses `odeint`)
- `lensing_and_precession/modules/Classes_v3.py`: Vectorized version (uses `cumulative_trapezoid`)

### Key Constants:
- `NEAR_ZERO_THRESHOLD = 1e-10` (v2) or `1e-8` (v3): Face-on detection tolerance
- `FMIN = 20` Hz: Minimum frequency for integration
- Guard threshold (tested but not used): |den_psi| < 10⁻¹², |T| > 10¹²

### Physics References:
- Regular precession model: arXiv:2509.10628 [gr-qc]
- Correction term source: Taman's implementation in `Taman/regular_precession.py`

---

## Revision History

- **2025-10-23**: Initial analysis based on empirical testing
  - Created comparison script testing with/without guard
  - Ran parameter sweep: θ_J ∈ [0°, 180°] at f = 50 Hz
  - Conclusion: Guard not necessary; algebraic formulas naturally stable
  - Decision: Keep current hybrid implementation without adding guard

---

## Contact

For questions about this analysis or the implementation, refer to:
- This document: `diagnostic_scripts/docs/POLARIZATION_GUARD_ANALYSIS.md`
- Main implementation: `modules/Classes_v2.py` (lines 720-770)
- Test script: `diagnostic_scripts/compare_polarization_guard.py`
