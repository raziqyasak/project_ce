import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
import time

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="Diet Meal Planning Optimisation (PSO)", layout="wide")

st.markdown("""
# 🍽️ Diet Meal Planning Optimisation  
### Using Particle Swarm Optimisation (PSO)

This system selects a daily meal plan that satisfies calorie requirements
while minimising total cost (Price in RM).
""")
st.divider()

# =========================
# Load Dataset
# =========================
data = pd.read_csv("Food_and_Nutrition_with_Price.csv")
# Pastikan dataset ada kolum: Food, Calories, Protein, Price_RM
NUM_MEALS = len(data)

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ PSO Parameters")
TARGET_CALORIES = st.sidebar.slider("Target Calories (kcal)", 1500, 3000, 1900)
NUM_PARTICLES = st.sidebar.slider("Number of Particles", 10, 50, 30)
MAX_ITER = st.sidebar.slider("Iterations", 50, 300, 100)

W = st.sidebar.slider("Inertia (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Parameter (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Parameter (C2)", 0.5, 2.5, 1.5)

# =========================
# Fitness Function
# =========================
def fitness_function(particle):
    if particle.sum() == 0:
        particle[random.randint(0, len(particle)-1)] = 1

    selected = data[particle.astype(bool)]
    total_calories = selected['Calories'].sum()
    total_cost = selected['Price_RM'].sum()
    total_protein = selected['Protein'].sum()

    penalty = 0
    if total_calories < TARGET_CALORIES:
        penalty += (TARGET_CALORIES - total_calories) * 10
    if total_calories > TARGET_CALORIES * 1.1:
        penalty += (total_calories - TARGET_CALORIES) * 5
    if total_protein < 50:
        penalty += (50 - total_protein) * 20

    return total_cost + penalty

# =========================
# Run Button
# =========================
st.markdown("### 🚀 Run Optimisation")
run = st.button("Start PSO Optimisation")

# =========================
# PSO Execution
# =========================
if run:
    start_time = time.time()
    particles = (np.random.rand(NUM_PARTICLES, NUM_MEALS) < 0.3).astype(int)
    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, NUM_MEALS))

    pbest = particles.copy()
    pbest_fitness = np.array([fitness_function(p) for p in particles])
    gbest = pbest[np.argmin(pbest_fitness)]
    convergence = []

    for _ in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1, r2 = random.random(), random.random()
            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (pbest[i] - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )
            particles[i] = 1 / (1 + np.exp(-velocities[i]))
            particles[i] = (particles[i] > 0.5).astype(int)

            fitness = fitness_function(particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness

        gbest = pbest[np.argmin(pbest_fitness)]
        convergence.append(min(pbest_fitness))

    end_time = time.time()
    runtime = round(end_time - start_time, 2)

    # Pilih hidangan berdasarkan gbest
    selected_indices = np.where(gbest == 1)[0]
    selected_data = data.iloc[selected_indices]

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## ✅ Optimisation Results")

    total_calories = int(selected_data['Calories'].sum())
    total_cost = round(selected_data['Price_RM'].sum(), 2)
    total_protein = round(selected_data['Protein'].sum(), 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Calories (kcal)", total_calories)
    c2.metric("Total Cost (RM)", total_cost)
    c3.metric("Total Protein (g)", total_protein)

    st.markdown("### 🥗 Selected Daily Meal Plan")
    st.dataframe(selected_data[['Food','Calories','Protein','Price_RM']], use_container_width=True)

    # =========================
    # Convergence Curve
    # =========================
    st.markdown("## 📈 PSO Convergence Curve")
    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence)+1),
        "Best Fitness Value": convergence
    })
    chart = (
        alt.Chart(convergence_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("Iteration", title="Iteration"),
            y=alt.Y("Best Fitness Value", title="Fitness Value")
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown(f"**Runtime:** {runtime} seconds")
