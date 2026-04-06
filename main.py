"""
2D Steady-State Thermal Analysis — FastAPI Backend
Supports:
  • Fixed temperature (Dirichlet)  on each of the 4 boundaries
  • Convective (Robin / Newton's cooling) on any / all boundaries

FDM  : SOR Red-Black with ghost-node convective BC
Analytical : Fourier series with transcendental eigenvalues for
             convective sides (bisection); classic sin-series for fixed sides.
"""

import os
import math
import time
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="2D Steady-State Thermal Analysis Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

progress_store = {"current": 0}


# ══════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════

class BCSpec(BaseModel):
    """Boundary condition for one side of the plate."""
    type: Literal["fixed", "convective"] = "fixed"
    T:    float = 0.0          # fixed temperature OR T_inf for convective
    h:    float = 10.0         # convection coefficient  [W/(m²·K)]
    k:    float = 50.0         # thermal conductivity    [W/(m·K)]


class CalculateRequest(BaseModel):
    m:      int   = Field(50, ge=5, le=200)
    top:    BCSpec = BCSpec(type="fixed", T=100.0)
    bottom: BCSpec = BCSpec(type="fixed", T=0.0)
    left:   BCSpec = BCSpec(type="fixed", T=0.0)
    right:  BCSpec = BCSpec(type="fixed", T=0.0)


# ══════════════════════════════════════════════════════════════════
#  HELPER: Biot number for a boundary
# ══════════════════════════════════════════════════════════════════

def biot(bc: BCSpec, dx: float) -> float:
    """Bi = h·Δx / k  (used in ghost-node convective discretisation)"""
    return bc.h * dx / bc.k


# ══════════════════════════════════════════════════════════════════
#  FDM — SOR Solver with convective ghost-node BCs
# ══════════════════════════════════════════════════════════════════

def sor_solver(top: BCSpec, bottom: BCSpec, left: BCSpec, right: BCSpec,
               n: int, tol: float = 1e-3, max_iter: int = 10000):
    """
    Solve on an n×n interior grid.
    Boundaries are handled via ghost-node (one-sided FD) approach:

      Fixed:      T_boundary = T_wall
      Convective: T_boundary = (T_interior + Bi · T_inf) / (1 + Bi)
                  where Bi = h·Δx / k
    """
    global progress_store
    progress_store["current"] = 0

    dx = 1.0 / (n + 1)  # grid spacing on unit square

    # Initial guess: average of all BC reference temperatures
    T_refs = [bc.T for bc in [top, bottom, left, right]]
    T = torch.full((n, n), float(np.mean(T_refs)), dtype=torch.float64)

    # Optimal SOR relaxation factor for fixed BCs (good approximation)
    omega = 2.0 / (1.0 + math.sin(math.pi / (n + 1)))

    I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    red_mask   = ((I + J) % 2 == 0)
    black_mask = ~red_mask

    # Precompute Biot numbers
    Bi_top    = biot(top,    dx)
    Bi_bottom = biot(bottom, dx)
    Bi_left   = biot(left,   dx)
    Bi_right  = biot(right,  dx)

    def get_boundary_value(T_interior_row_or_col, bc: BCSpec, Bi: float):
        """Return boundary temperature given interior neighbour."""
        if bc.type == "fixed":
            return float(bc.T)
        else:
            # Robin: T_b = (T_interior + Bi * T_inf) / (1 + Bi)
            return (T_interior_row_or_col + Bi * bc.T) / (1.0 + Bi)

    start_time = time.perf_counter()
    initial_diff = None
    last_progress = 0

    for it in range(max_iter):
        T_old = T.clone()

        for mask in [red_mask, black_mask]:
            # ── North neighbour (row above, or top boundary) ──
            north = torch.zeros_like(T)
            north[1:, :] = T[:-1, :]
            if top.type == "fixed":
                north[0, :] = top.T
            else:
                north[0, :] = (T[0, :] + Bi_top * top.T) / (1.0 + Bi_top)

            # ── South neighbour (row below, or bottom boundary) ──
            south = torch.zeros_like(T)
            south[:-1, :] = T[1:, :]
            if bottom.type == "fixed":
                south[-1, :] = bottom.T
            else:
                south[-1, :] = (T[-1, :] + Bi_bottom * bottom.T) / (1.0 + Bi_bottom)

            # ── West neighbour (col left, or left boundary) ──
            west = torch.zeros_like(T)
            west[:, 1:] = T[:, :-1]
            if left.type == "fixed":
                west[:, 0] = left.T
            else:
                west[:, 0] = (T[:, 0] + Bi_left * left.T) / (1.0 + Bi_left)

            # ── East neighbour (col right, or right boundary) ──
            east = torch.zeros_like(T)
            east[:, :-1] = T[:, 1:]
            if right.type == "fixed":
                east[:, -1] = right.T
            else:
                east[:, -1] = (T[:, -1] + Bi_right * right.T) / (1.0 + Bi_right)

            T_new = 0.25 * (north + south + west + east)
            T[mask] = (1 - omega) * T[mask] + omega * T_new[mask]

        diff = torch.max(torch.abs(T - T_old)).item()
        if initial_diff is None:
            initial_diff = max(diff, 1e-9)

        if diff > tol:
            ratio = diff / tol
            prog = int(max(0, min(99,
                100 * (1 - math.log(ratio) / math.log(initial_diff / tol))
            )))
            if prog > last_progress:
                last_progress = prog
                progress_store["current"] = prog
        else:
            break

    progress_store["current"] = 100

    # ── Build full (m+1)×(m+1) grid including boundary rows/cols ──
    m = n + 1
    full = np.zeros((m + 1, m + 1))
    full[1:-1, 1:-1] = T.numpy()

    # Boundary rows/cols using the same ghost-node formula
    # Top row (visual top → high y index after flipud)
    if bottom.type == "fixed":
        full[-1, :] = bottom.T
    else:
        full[-1, :] = (full[-2, :] + Bi_bottom * bottom.T) / (1.0 + Bi_bottom)

    # Bottom row
    if top.type == "fixed":
        full[0, :] = top.T
    else:
        full[0, :] = (full[1, :] + Bi_top * top.T) / (1.0 + Bi_top)

    # Left col
    if left.type == "fixed":
        full[:, 0] = left.T
    else:
        full[:, 0] = (full[:, 1] + Bi_left * left.T) / (1.0 + Bi_left)

    # Right col
    if right.type == "fixed":
        full[:, -1] = right.T
    else:
        full[:, -1] = (full[:, -2] + Bi_right * right.T) / (1.0 + Bi_right)

    # SOR stores row-0=top; flipud → row-0=bottom (matches Fourier/Plotly)
    full = np.flipud(full)

    return full, time.perf_counter() - start_time, it + 1


# ══════════════════════════════════════════════════════════════════
#  ANALYTICAL — Fourier series with convective eigenvalues
# ══════════════════════════════════════════════════════════════════

def _bisect(f, a, b, tol=1e-12, max_iter=300):
    """Simple bisection root finder."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return None
    for _ in range(max_iter):
        m = (a + b) / 2.0
        fm = f(m)
        if abs(fm) < tol or (b - a) / 2.0 < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2.0


def _eigenvalues_1d(Bi_start: float, Bi_end: float, L: float, N: int):
    """
    Compute first N positive eigenvalues λ_n for the 1-D Sturm-Liouville
    problem on [0, L] with Robin conditions at both ends:

      X'' + λ²X = 0
      X'(0)  = Bi_start · X(0)   (convective: Bi_start = h·L/k; fixed: Bi→∞)
      X'(L)  = -Bi_end  · X(L)

    Special cases:
      Both fixed (Bi→∞): λ_n = n·π/L
      Mixed / both convective: solve transcendental equation numerically.

    Returns list of N eigenvalues.
    """
    INF = 1e15

    if Bi_start >= INF and Bi_end >= INF:
        # Classic: λ_n = nπ/L
        return [(n * math.pi / L) for n in range(1, N + 1)]

    eigs = []
    # Search interval: roots lie between k·π/(2L) approximately
    search_density = 4000
    dl = math.pi / (2.0 * L)

    for k in range(search_density):
        if len(eigs) >= N:
            break
        lam_lo = k * dl + 1e-10
        lam_hi = (k + 1) * dl - 1e-10

        def eq(lam):
            # X = A·cos(λx) + B·sin(λx)
            # BC at x=0: X'(0) = Bi_start·X(0) → λ·B = Bi_start·A
            #   → B = (Bi_start/λ)·A  (if Bi→∞ use sin only)
            # BC at x=L:
            # X'(L) = -λ·A·sin(λL) + λ·B·cos(λL) = -Bi_end·(A·cos(λL)+B·sin(λL))
            if Bi_start >= INF:
                # X = sin(λx)  →  λcos(λL) = -Bi_end·sin(λL)
                return lam * math.cos(lam * L) + Bi_end * math.sin(lam * L)
            elif Bi_end >= INF:
                # X = cos(λx)+(Bi_start/λ)sin(λx)  →  X(L)=0
                return math.cos(lam * L) + (Bi_start / lam) * math.sin(lam * L)
            else:
                # General Robin-Robin
                A = 1.0
                B = Bi_start / lam
                lhs = -lam * A * math.sin(lam * L) + lam * B * math.cos(lam * L)
                rhs = -Bi_end * (A * math.cos(lam * L) + B * math.sin(lam * L))
                return lhs - rhs

        root = _bisect(eq, lam_lo, lam_hi)
        if root is not None:
            if not eigs or abs(root - eigs[-1]) > 1e-8:
                eigs.append(root)

    return eigs[:N]


def _X_fn(lam: float, x: float, Bi_start: float) -> float:
    """Eigenfunction X_n(x) satisfying X'(0)=Bi_start·X(0)."""
    INF = 1e15
    if Bi_start >= INF:
        return math.sin(lam * x)
    return math.cos(lam * x) + (Bi_start / lam) * math.sin(lam * x)


def _norm2_X(lam: float, L: float, Bi_start: float, M: int = 500) -> float:
    """||X_n||² = ∫₀ᴸ X_n(x)² dx  (Simpson's rule)."""
    h = L / M
    s = sum(
        (1 if i in (0, M) else (2 if i % 2 == 0 else 4))
        * _X_fn(lam, i * h, Bi_start) ** 2
        for i in range(M + 1)
    )
    return s * h / 3.0


def _fourier_coeff(lam: float, L: float, Bi_start: float,
                   f_vals: np.ndarray, xs: np.ndarray) -> float:
    """
    A_n = (1/||X_n||²) ∫₀ᴸ f(x)·X_n(x) dx
    f_vals and xs are arrays of length M+1 on [0,L].
    """
    Xn = np.array([_X_fn(lam, x, Bi_start) for x in xs])
    norm2 = _norm2_X(lam, L, Bi_start, len(xs) - 1)
    # Simpson's rule
    h = xs[1] - xs[0]
    integrand = f_vals * Xn
    s = (integrand[0] + integrand[-1]
         + 4 * integrand[1:-1:2].sum()
         + 2 * integrand[2:-2:2].sum()) * h / 3.0
    return s / norm2 if norm2 > 1e-30 else 0.0


def fourier_solution_general(
    top: BCSpec, bottom: BCSpec, left: BCSpec, right: BCSpec,
    m: int, N_terms: int = 60
) -> np.ndarray:
    """
    Analytical Fourier-series solution for a unit-square plate with
    arbitrary fixed / convective BCs on all four sides.

    Strategy — superposition of 4 sub-problems, each with one
    non-homogeneous side and the rest made homogeneous (T=0 or dT/dn=0).

    For each sub-problem the solution has the form:
      θ(x,y) = Σ_n  A_n · X_n(x) · Y_n(y)
    where X_n solves a Sturm-Liouville ODE with the x-BCs,
    and Y_n is the corresponding hyperbolic function satisfying
    the homogeneous y-BCs.

    Returns (m+1)×(m+1) array, row-0=bottom (y=0), row-m=top (y=1).
    """
    INF = 1e15
    L   = 1.0      # unit square

    # Biot numbers (use large number for fixed to keep equations uniform)
    def bc_Bi(bc: BCSpec) -> float:
        if bc.type == "fixed":
            return INF
        return bc.h / bc.k   # per-unit-length Biot (Δx=1 normalised)

    Bi_t = bc_Bi(top)
    Bi_b = bc_Bi(bottom)
    Bi_l = bc_Bi(left)
    Bi_r = bc_Bi(right)

    T_t = top.T
    T_b = bottom.T
    T_l = left.T
    T_r = right.T

    # Reference temperature (constant particular solution): weighted mean
    T_ref = 0.0
    weight_sum = 0.0
    for bc in [top, bottom, left, right]:
        w = 1.0
        T_ref += w * bc.T
        weight_sum += w
    T_ref /= weight_sum

    # Shift: θ = T - T_ref  (all shifted BCs; some become zero)
    θ_t = T_t - T_ref
    θ_b = T_b - T_ref
    θ_l = T_l - T_ref
    θ_r = T_r - T_ref

    xs = np.linspace(0, L, m + 1)
    ys = np.linspace(0, L, m + 1)
    X2D, Y2D = np.meshgrid(xs, ys)   # shape (m+1, m+1), row-0 = y=0

    theta = np.zeros((m + 1, m + 1))

    # ──────────────────────────────────────────────────────────────
    #  Sub-problem helper
    #  non_zero_side: 'bottom','top','left','right'
    #  theta_val:     the non-zero shifted boundary value (constant)
    #  The other three sides are homogeneous.
    # ──────────────────────────────────────────────────────────────

    def add_sub_problem(non_zero_side: str, theta_val: float):
        nonlocal theta
        if abs(theta_val) < 1e-12:
            return  # nothing to add

        # Eigenvalues in x-direction for sub-problems with non-zero top/bottom
        # Eigenvalues in y-direction for sub-problems with non-zero left/right
        if non_zero_side in ("bottom", "top"):
            # x-direction: homogeneous left and right BCs
            lams = _eigenvalues_1d(Bi_l, Bi_r, L, N_terms)
            for lam in lams:
                A_n = _fourier_coeff(lam, L, Bi_l,
                                     np.full(m + 1, theta_val), xs)
                Xn  = np.array([_X_fn(lam, x, Bi_l) for x in xs])

                # Y direction: sinh form satisfying homogeneous BC on the
                # opposite side and value 1 at the non-zero side.
                if non_zero_side == "bottom":
                    # Y(0)=1, homogeneous top
                    if Bi_t >= INF:
                        # Y=sinh(λ(L-y))/sinh(λL)
                        denom = math.sinh(lam * L)
                        if denom < 1e-15:
                            continue
                        Yn = np.sinh(lam * (L - ys)) / denom
                    else:
                        # Y'(L) = -Bi_t · Y(L); Y(0)=1
                        # Y = cosh(λy)·A + sinh(λy)·B, A=1
                        # λ(sinh(λL)+B cosh(λL)) = -Bi_t(cosh(λL)+B sinh(λL))
                        sL = math.sinh(lam * L); cL = math.cosh(lam * L)
                        denom = lam * cL + Bi_t * sL
                        B = -(lam * sL + Bi_t * cL) / denom if abs(denom) > 1e-15 else 0.0
                        Yn = np.cosh(lam * ys) + B * np.sinh(lam * ys)
                else:  # top
                    # Y(L)=1, homogeneous bottom
                    if Bi_b >= INF:
                        denom = math.sinh(lam * L)
                        if denom < 1e-15:
                            continue
                        Yn = np.sinh(lam * ys) / denom
                    else:
                        sL = math.sinh(lam * L); cL = math.cosh(lam * L)
                        denom = lam * cL + Bi_b * sL
                        B = -(lam * sL + Bi_b * cL) / denom if abs(denom) > 1e-15 else 0.0
                        # Y = cosh(λ(L-y)) + B sinh(λ(L-y))  (normalised at y=L)
                        Yn = np.cosh(lam * (L - ys)) + B * np.sinh(lam * (L - ys))

                # outer product: theta += A_n * X_n(x) * Y_n(y)
                theta += A_n * np.outer(Yn, Xn)

        else:  # left or right — swap x↔y
            mus = _eigenvalues_1d(Bi_b, Bi_t, L, N_terms)
            for mu in mus:
                A_n = _fourier_coeff(mu, L, Bi_b,
                                     np.full(m + 1, theta_val), ys)
                Yn  = np.array([_X_fn(mu, y, Bi_b) for y in ys])

                if non_zero_side == "left":
                    # X(0)=1, homogeneous right
                    if Bi_r >= INF:
                        denom = math.sinh(mu * L)
                        if denom < 1e-15:
                            continue
                        Xn = np.sinh(mu * (L - xs)) / denom
                    else:
                        sL = math.sinh(mu * L); cL = math.cosh(mu * L)
                        denom = mu * cL + Bi_r * sL
                        B = -(mu * sL + Bi_r * cL) / denom if abs(denom) > 1e-15 else 0.0
                        Xn = np.cosh(mu * xs) + B * np.sinh(mu * xs)
                else:  # right
                    # X(L)=1, homogeneous left
                    if Bi_l >= INF:
                        denom = math.sinh(mu * L)
                        if denom < 1e-15:
                            continue
                        Xn = np.sinh(mu * xs) / denom
                    else:
                        sL = math.sinh(mu * L); cL = math.cosh(mu * L)
                        denom = mu * cL + Bi_l * sL
                        B = -(mu * sL + Bi_l * cL) / denom if abs(denom) > 1e-15 else 0.0
                        Xn = np.cosh(mu * (L - xs)) + B * np.sinh(mu * (L - xs))

                theta += A_n * np.outer(Yn, Xn)

    # Add the 4 sub-problems
    add_sub_problem("bottom", θ_b)
    add_sub_problem("top",    θ_t)
    add_sub_problem("left",   θ_l)
    add_sub_problem("right",  θ_r)

    T_grid = theta + T_ref

    # Enforce boundary values exactly for fixed BCs
    if top.type    == "fixed": T_grid[-1, :] = top.T
    if bottom.type == "fixed": T_grid[0,  :] = bottom.T
    if left.type   == "fixed": T_grid[:,  0] = left.T
    if right.type  == "fixed": T_grid[:, -1] = right.T

    return T_grid


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "online"}


@app.get("/api/progress")
def get_progress():
    return {"progress": progress_store["current"]}


@app.post("/api/calculate")
def calculate(req: CalculateRequest):
    # ── FDM (SOR) ──
    fdm, solve_time, iters = sor_solver(
        req.top, req.bottom, req.left, req.right,
        n=req.m - 1
    )

    # ── Analytical (Fourier + transcendental eigenvalues) ──
    ana = fourier_solution_general(
        req.top, req.bottom, req.left, req.right,
        m=req.m, N_terms=200
    )

    # ── Accuracy metrics ──
    internal_fdm = fdm[1:-1, 1:-1]
    internal_ana = ana[1:-1, 1:-1]
    rmse = float(np.sqrt(np.mean((internal_fdm - internal_ana) ** 2)))

    all_T = [bc.T for bc in [req.top, req.bottom, req.left, req.right]]
    temp_range = max(max(all_T) - min(all_T), 1.0)
    v_score = float(max(0.0, 100.0 * (1.0 - rmse / temp_range)))

    # ── Detect if any boundary is convective (affects DB log) ──
    has_convective = any(
        bc.type == "convective"
        for bc in [req.top, req.bottom, req.left, req.right]
    )

    # ── Persist to Supabase ──
    try:
        supabase.table("thermal_logs").insert({
            "grid_size":        req.m,
            "t_top":            req.top.T,
            "t_bottom":         req.bottom.T,
            "t_left":           req.left.T,
            "t_right":          req.right.T,
            "iterations":       iters,
            "time_taken":       solve_time,
            "validation_score": v_score,
            "has_convective":   has_convective,
        }).execute()
    except Exception as e:
        print(f"DB Error: {e}")

    return {
        "fdm":           fdm.tolist(),
        "analytic":      ana.tolist(),
        "time":          solve_time,
        "iters":         iters,
        "rmse":          rmse,
        "v_score":       v_score,
        "has_convective": has_convective,
    }


@app.get("/api/regression")
def get_regression():
    try:
        res = (
            supabase.table("thermal_logs")
            .select("grid_size", "iterations", "time_taken")
            .execute()
        )
        if not res.data:
            return {"error": "no_data"}

        df = pd.DataFrame(res.data).sort_values("grid_size")
        m_arr    = df["grid_size"].values.reshape(-1, 1).astype(float)
        it_arr   = df["iterations"].values.astype(float)
        tm_arr   = df["time_taken"].values.astype(float)

        lin = LinearRegression().fit(m_arr, it_arr)
        iter_m = float(lin.coef_[0])
        iter_c = float(lin.intercept_)

        poly   = PolynomialFeatures(degree=3)
        m_poly = poly.fit_transform(m_arr)
        cub    = LinearRegression().fit(m_poly, tm_arr)
        time_coeffs = cub.coef_.tolist()
        time_c      = float(cub.intercept_)

        m_range    = np.linspace(float(m_arr.min()), float(m_arr.max()), 200)
        iter_curve = iter_m * m_range + iter_c
        time_curve = cub.predict(poly.transform(m_range.reshape(-1, 1)))

        return {
            "m_values":    df["grid_size"].tolist(),
            "iter_values": df["iterations"].tolist(),
            "time_values": df["time_taken"].tolist(),
            "m_curve":     m_range.tolist(),
            "iter_curve":  iter_curve.tolist(),
            "time_curve":  time_curve.tolist(),
            "coeffs": {
                "iter_m":      iter_m,
                "iter_c":      iter_c,
                "time_coeffs": time_coeffs,
                "time_c":      time_c,
            },
        }
    except Exception as e:
        print(f"DB Fetch Error: {e}")
        return {"error": "db_error"}