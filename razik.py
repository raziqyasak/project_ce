import streamlit as st
import pandas as pd
import numpy as np
import random

# =========================
# Streamlit Page Setup
# =========================
st.set_page_config(page_title="Diet Meal Planning PSO", layout="wide")

st.title("🍽️ Diet Meal Planning Optimisation using PSO")

st.write("""
This application applies **Particle Swarm Optimization (PSO)** to select
a daily meal plan that satisfies calorie requirements while minimizing total cost.
""")

# =========================
# Load Dataset
# =========================
data = pd.read_csv("data/Food_and_Nutrition__.csv")

# Use required columns only
data = data[['Calories', 'Protein']].copy()

# Add dummy cost (since dataset has no price)
np.random.seed(42)
data['Cost'] = np.random.uniform(2, 10, size=len(data))

# =========================
# Sidebar Parameters
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
# Run PSO Button
# =========================
if st.button("🚀 Run PSO Optimisation"):

    # Initialize swarm
    particles = np.random.randint(0, len(data), (NUM_PARTICLES, MEALS_PER_DAY))
    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, MEALS_PER_DAY))

    pbest = particles.copy()
    pbest_fitness = np.array([fitness_function(p) for p in particles])

    gbest = pbest[np.argmin(pbest_fitness)]
    gbest_fitness = min(pbest_fitness)

    convergence = []

    # =========================
    # PSO Main Loop
    # =========================
    for iteration in range(MAX_ITER):
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
        gbest_fitness = min(pbest_fitness)

        convergence.append(gbest_fitness)

    # =========================
    # Results
    # =========================
    best_indices = gbest.astype(int)
    best_meal = data.iloc[best_indices]

    st.subheader("✅ Optimal Daily Meal Plan")
    st.dataframe(best_meal, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Total Calories", int(best_meal['Calories'].sum()))
    col2.metric("Total Cost (RM)", round(best_meal['Cost'].sum(), 2))

    # =========================
    # Convergence Plot (NO matplotlib)
    # =========================
    st.subheader("📈 Convergence Curve")
    st.line_chart(convergence)
