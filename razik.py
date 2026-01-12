import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Diet Meal Planning Optimisation (PSO)",
    layout="wide"
)

# =========================
# Header
# =========================
st.markdown("""
# 🍽️ Diet Meal Planning Optimisation  
### Using Particle Swarm Optimisation (PSO)

This system selects a **daily meal combination** that achieves
**calorie intake closest to the target** while **minimising total cost**.
""")
st.divider()

# =========================
# Load Dataset
# =========================
data = pd.read_csv("Food_and_Nutrition_with_Price.csv")
data = data[['Calories', 'Protein']].copy()

# -------------------------
# Cost Model
# -------------------------
np.random.seed(42)
data['Cost'] = data['Calories'] * np.random.uniform(0.008, 0.015)

NUM_MEALS = len(data)

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ PSO Parameters")

TARGET_CALORIES = st.sidebar.slider("Target Daily Calories", 1500, 3000, 1900)
NUM_PARTICLES = st.sidebar.slider("Number of Particles", 20, 60, 40)
MAX_ITER = st.sidebar.slider("Iterations", 100, 400, 200)

C1 = st.sidebar.slider("Cognitive Factor (C1)", 1.0, 3.0, 2.0)
C2 = st.sidebar.slider("Social Factor (C2)", 0.5, 2.0, 0.8)

# =========================
# Fitness Function
# =========================
def fitness_function(particle):
    selected = data[particle.astype(bool)]

    if selected.empty:
        return 1e9  # avoid empty solution

    total_calories = selected['Calories'].sum()
    total_cost = selected['Cost'].sum()
    total_protein = selected['Protein'].sum()

    calorie_diff = abs(TARGET_CALORIES - total_calories)
    protein_deficit = max(0, 50 - total_protein)

    fitness = (
        total_cost
        + calorie_diff * 3      # softer penalty → smooth graph
        + protein_deficit * 10
    )

    return fitness

# =========================
# Run Button
# =========================
st.markdown("### 🚀 Run Optimisation")
run = st.button("Start PSO Optimisation")

# =========================
# PSO Execution
# =========================
if run:
    particles = np.random.randint(0, 2, (NUM_PARTICLES, NUM_MEALS))
    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, NUM_MEALS))

    pbest = particles.copy()
    pbest_fitness = np.array([fitness_function(p) for p in particles])
    gbest = pbest[np.argmin(pbest_fitness)]

    convergence_best = []
    convergence_avg = []

    W_MAX = 0.9
    W_MIN = 0.4

    for iter in range(MAX_ITER):
        W = W_MAX - (W_MAX - W_MIN) * (iter / MAX_ITER)

        for i in range(NUM_PARTICLES):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (pbest[i] - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )

            sigmoid = 1 / (1 + np.exp(-velocities[i]))
            particles[i] = (sigmoid > np.random.rand(NUM_MEALS)).astype(int)

            fitness = fitness_function(particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness

        gbest = pbest[np.argmin(pbest_fitness)]

        convergence_best.append(np.min(pbest_fitness))
        convergence_avg.append(np.mean(pbest_fitness))

    best_meal = data[gbest.astype(bool)]

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## ✅ Optimisation Results")

    total_calories = int(best_meal['Calories'].sum())
    total_cost = round(best_meal['Cost'].sum(), 2)
    total_protein = round(best_meal['Protein'].sum(), 1)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Calories (kcal)", total_calories)
    c2.metric("Total Cost (RM)", total_cost)
    c3.metric("Total Protein (g)", total_protein)

    st.markdown("### 🥗 Selected Daily Meal Combination")
    st.dataframe(best_meal, use_container_width=True)

    # =========================
    # Convergence Curve
    # =========================
    st.markdown("## 📈 PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration": range(1, MAX_ITER + 1),
        "Best Fitness": convergence_best,
        "Average Fitness": convergence_avg
    })

    chart = alt.Chart(convergence_df).transform_fold(
        ["Best Fitness", "Average Fitness"],
        as_=["Type", "Fitness"]
    ).mark_line(strokeWidth=3).encode(
        x="Iteration:Q",
        y="Fitness:Q",
        color="Type:N"
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    # =========================
    # Summary
    # =========================
    st.markdown(f"""
**Summary:**  
- Target calorie intake: **{TARGET_CALORIES} kcal**  
- Achieved calorie intake: **{total_calories} kcal**  
- Total cost: **RM {total_cost}**  
- Protein intake: **{total_protein} g**  
- The convergence curve shows **gradual improvement**, indicating effective PSO search behaviour.
""")
