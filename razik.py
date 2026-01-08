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

This system selects a daily meal plan that satisfies calorie requirements
while minimising total cost.
""")

st.divider()

# =========================
# Load Dataset
# =========================
data = pd.read_csv("Food_and_Nutrition__.csv")
data = data[['Calories', 'Protein']].copy()

np.random.seed(42)
data['Cost'] = np.random.uniform(2, 10, size=len(data))

# =========================
# Sidebar Parameters (UNCHANGED)
# =========================
st.sidebar.header("⚙️ PSO Parameters")

TARGET_CALORIES = st.sidebar.slider("Target Calories", 1500, 3000, 2000)
MEALS_PER_DAY = st.sidebar.slider("Meals per Day", 2, 5, 3)
NUM_PARTICLES = st.sidebar.slider("Number of Particles", 10, 50, 30)
MAX_ITER = st.sidebar.slider("Iterations", 50, 300, 100)

W = st.sidebar.slider("Inertia Weight (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Parameter (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Parameter (C2)", 0.5, 2.5, 1.5)

# =========================
# Fitness Function
# =========================
def fitness_function(particle):
    indices = np.clip(particle.astype(int), 0, len(data) - 1)
    selected = data.iloc[indices]

    total_calories = selected['Calories'].sum()
    total_cost = selected['Cost'].sum()

    penalty = 0
    if total_calories < TARGET_CALORIES:
        penalty = (TARGET_CALORIES - total_calories) * 10

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
    particles = np.random.randint(
        0, len(data), (NUM_PARTICLES, MEALS_PER_DAY)
    ).astype(float)

    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, MEALS_PER_DAY))

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

            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], 0, len(data) - 1)

            fitness = fitness_function(particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i].copy()
                pbest_fitness[i] = fitness

        gbest = pbest[np.argmin(pbest_fitness)]
        convergence.append(min(pbest_fitness))

    best_meal = data.iloc[gbest.astype(int)]

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## ✅ Optimisation Results")

    c1, c2 = st.columns(2)
    c1.metric("Total Calories", int(best_meal['Calories'].sum()))
    c2.metric("Total Cost (RM)", round(best_meal['Cost'].sum(), 2))

    st.markdown("### 🥗 Selected Daily Meal Plan")
    st.dataframe(best_meal, use_container_width=True)

    # =========================
    # STATIC GRAPH 1: Convergence Curve
    # =========================
    st.markdown("## 📈 PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence) + 1),
        "Best Fitness (Cost)": convergence
    })

    chart1 = alt.Chart(convergence_df).mark_line(point=True).encode(
        x="Iteration",
        y="Best Fitness (Cost)"
    ).properties(height=350)

    st.altair_chart(chart1, use_container_width=True)

    # =========================
    # STATIC GRAPH 2: Fitness Improvement
    # =========================
    st.markdown("## 📉 Fitness Improvement per Iteration")

    improvement = [0] + [
        convergence[i-1] - convergence[i]
        for i in range(1, len(convergence))
    ]

    improvement_df = pd.DataFrame({
        "Iteration": range(1, len(improvement) + 1),
        "Fitness Improvement": improvement
    })

    chart2 = alt.Chart(improvement_df).mark_line(point=True).encode(
        x="Iteration",
        y="Fitness Improvement"
    ).properties(height=350)

    st.altair_chart(chart2, use_container_width=True)
