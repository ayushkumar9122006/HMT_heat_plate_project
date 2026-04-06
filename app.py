import streamlit as st
import torch
import time
import math
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.title("2-D Temperature Analysis Using Finite Difference Method")

# -------------------- GRID SIZE --------------------
m = st.number_input("Enter Number of Grids",
                    min_value=3,
                    step=1,
                    format="%d")

# -------------------- RED-BLACK SOR SOLVER --------------------
def red_black_sor_2d_vectorized(Tu, Td, Tl, Tr, n, tol=1e-6, max_iter=5000):

    start_time = time.perf_counter()
    T = torch.zeros((n, n), dtype=torch.float32)
    omega = 2.0 / (1.0 + math.sin(math.pi / n))

    I, J = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    red = (I + J) % 2 == 0
    black = ~red

    for k in range(max_iter):
        T_old = T.clone()

        north = torch.zeros_like(T)
        south = torch.zeros_like(T)
        west  = torch.zeros_like(T)
        east  = torch.zeros_like(T)

        north[1:, :] = T[:-1, :]
        north[0, :] = Td

        south[:-1, :] = T[1:, :]
        south[-1, :] = Tu

        west[:, 1:] = T[:, :-1]
        west[:, 0] = Tl

        east[:, :-1] = T[:, 1:]
        east[:, -1] = Tr

        T_new = 0.25 * (north + south + west + east)
        T[red] = (1 - omega) * T[red] + omega * T_new[red]

        north[1:, :] = T[:-1, :]
        south[:-1, :] = T[1:, :]
        west[:, 1:] = T[:, :-1]
        east[:, :-1] = T[:, 1:]

        T_new = 0.25 * (north + south + west + east)
        T[black] = (1 - omega) * T[black] + omega * T_new[black]

        if torch.max(torch.abs(T - T_old)) < tol:
            break

    elapsed_time = time.perf_counter() - start_time
    return T, elapsed_time, k + 1


# -------------------- ANALYTICAL SOLUTION --------------------
def analytical_square_solution(x, y, k, T1, T2, T3, T4, terms=50):

    value = 0.0

    for n in range(1, 2*terms, 2):  # odd terms only

        coeff = 4.0 / (n * math.pi * math.sinh(n * math.pi))

        # Bottom + Top contribution
        part_x = math.sin(n * math.pi * x / k)
        bottom_top = (
            T1 * math.sinh(n * math.pi * (k - y) / k) +
            T2 * math.sinh(n * math.pi * y / k)
        )

        value += coeff * part_x * bottom_top

        # Left + Right contribution
        part_y = math.sin(n * math.pi * y / k)
        left_right = (
            T3 * math.sinh(n * math.pi * (k - x) / k) +
            T4 * math.sinh(n * math.pi * x / k)
        )

        value += coeff * part_y * left_right

    return value


# -------------------- PERFORMANCE LOG --------------------
def log_run_data(grid_size, iterations, time_taken):

    file_name = "solver_runs.csv"

    new_data = pd.DataFrame([{
        "grid_size": grid_size,
        "iterations": iterations,
        "time_taken": time_taken
    }])

    if os.path.exists(file_name):
        old_data = pd.read_csv(file_name)
        updated = pd.concat([old_data, new_data], ignore_index=True)
    else:
        updated = new_data

    updated.to_csv(file_name, index=False)


# -------------------- HEATMAP --------------------
def build_full_matrix(Temp, Tu, Td, Tl, Tr, m):

    Tfull = torch.zeros((m+1, m+1))
    Tfull[1:-1, 1:-1] = Temp

    Tfull[0, :] = Tu
    Tfull[-1, :] = Td
    Tfull[:, 0] = Tl
    Tfull[:, -1] = Tr

    return Tfull


def plot_temperature(Tfull):

    fig = px.imshow(
        Tfull.numpy(),
        color_continuous_scale="inferno",
        origin="lower",
        aspect="equal"
    )

    fig.update_layout(height=700)

    fig.update_traces(
        hovertemplate="X: %{x}<br>Y: %{y}<br>T: %{z:.2f}<extra></extra>"
    )

    return fig


# -------------------- BOUNDARY INPUT --------------------
st.subheader("Boundary Conditions")

col1, col2 = st.columns(2)

with col1:
    Td = st.number_input("Top Temperature (K)", min_value=0, step=10)
    Tu = st.number_input("Bottom Temperature (K)", min_value=0, step=10)

with col2:
    Tl = st.number_input("Left Temperature (K)", min_value=0, step=10)
    Tr = st.number_input("Right Temperature (K)", min_value=0, step=10)


# -------------------- CALCULATE --------------------
if st.button("Calculate Temperature", type="primary"):

    Temp, solve_time, iters = red_black_sor_2d_vectorized(
        Tu=Td,
        Td=Tu,
        Tl=Tl,
        Tr=Tr,
        n=m-1,
        tol=0.001
    )

    Tfull = build_full_matrix(Temp, Tu, Td, Tl, Tr, m)

    st.session_state["Temp"] = Temp
    st.session_state["Tfull"] = Tfull
    st.session_state["time"] = solve_time
    st.session_state["iters"] = iters
    st.session_state["Tu"] = Tu
    st.session_state["Td"] = Td
    st.session_state["Tl"] = Tl
    st.session_state["Tr"] = Tr

    log_run_data(m, iters, solve_time)


# -------------------- DISPLAY --------------------
if "Temp" in st.session_state:

    st.success("Calculation complete!")

    st.metric("Solver Time (s)", f"{st.session_state['time']:.4f}")
    st.metric("Iterations", st.session_state["iters"])

    fig = plot_temperature(st.session_state["Tfull"])
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- POINTWISE COMPARISON ----------------
    st.subheader("Compare Analytical vs FDM at a Point")

    colx, coly = st.columns(2)

    with colx:
        px_point = st.number_input("X coordinate",
                                   min_value=0,
                                   max_value=m,
                                   value=m//2,
                                   step=1)

    with coly:
        py_point = st.number_input("Y coordinate",
                                   min_value=0,
                                   max_value=m,
                                   value=m//2,
                                   step=1)

    if st.button("Compare at Point"):

        fdm_value = st.session_state["Tfull"][int(py_point), int(px_point)].item()

        analytic_value = analytical_square_solution(
            x=px_point,
            y=py_point,
            k=m,
            T1=st.session_state["Tu"],
            T2=st.session_state["Td"],
            T3=st.session_state["Tl"],
            T4=st.session_state["Tr"],
            terms=50
        )

        error = abs(fdm_value - analytic_value)

        st.write(f"FDM Value: {fdm_value:.6f}")
        st.write(f"Analytical Value: {analytic_value:.6f}")
        st.write(f"Absolute Error: {error:.6f}")

    # ---------------- GLOBAL ERROR ----------------
    st.subheader("Global L2 Error")

    Tfull = st.session_state["Tfull"]
    analytic_matrix = torch.zeros_like(Tfull)

    for i in range(m+1):
        for j in range(m+1):
            analytic_matrix[j, i] = analytical_square_solution(
                x=i,
                y=j,
                k=m,
                T1=st.session_state["Tu"],
                T2=st.session_state["Td"],
                T3=st.session_state["Tl"],
                T4=st.session_state["Tr"],
                terms=50
            )

    error_matrix = Tfull - analytic_matrix
    L2_error = torch.sqrt(torch.mean(error_matrix**2)).item()

    st.write(f"L2 Norm Error: {L2_error:.6f}")