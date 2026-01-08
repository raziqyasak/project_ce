import streamlit as st
import pandas as pd
import numpy as np
import random
import os

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Diet Meal Planning Optimisation (PSO)", layout="centered")

st.title("🍽️ Diet Meal Planning Optimisation using PSO")
st.write("This system selects a daily meal plan that meets nutritional requirements at minimum cost using Particle Swarm Optimisation (PSO).")

# =========================
# Load Dataset (Auto Detect)
# =========================
if os.path.exists("Food_and_Nutrition__.csv"):
    data = pd.read_csv("Food_and_Nutrition__.csv")
elif os.path.exists("data/Food_and_Nutrition__.csv"):
    data = pd.read_csv("data/Food_and_Nutrition__.csv")
else:
    st.error("❌ Dataset file not found. Please upload Food_and_Nutrition__.csv")
    st.stop()

# =========================
# Select Required Columns
# =========================
data = data[['Calories', 'Protein']].dropna().reset_index(drop=True)

# Add dummy cost (since dataset has no price)
np.random.seed(42)
data['Cost'] = np.random.uniform(2, 10, size=len(data))

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# =========================
# Problem Parameters
# =========================
TARGET_CALORIES = 2000
MEALS_PER_DAY = 3
NUM_PARTICLES = 30
MAX_ITER = 100

W = 0.7
C1 = 1.5
C2 = 1.5

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
# PSO Initialisation
# =========================
particles = np.random.randint(0, len(data), (NUM_PARTICLES, MEALS_PER_DAY))
velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, MEALS_PER_DAY))

pbest = particles.copy()
pbest_fitness = np.array([fitness_function(p) for p in particles])

gbest = pbest[np.argmin(pbest_fitness)]
gbest_fitness = min(pbest_fitness)

convergence = []

# =========================
# Run PSO Button
# =========================
if st.button("🚀 Run PSO Optimisation"):

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
    st.success("✅ Optimisation Completed")

    best_indices = gbest.astype(int)
    best_meal = data.iloc[best_indices]

    st.subheader("🥗 Optimal Daily Meal Plan")
    st.dataframe(best_meal)

    st.metric("🔥 Total Calories", int(best_meal['Calories'].sum()))
    st.metric("💰 Total Cost (RM)", round(best_meal['Cost'].sum(), 2))

    # =========================
    # Convergence Graph (NO matplotlib)
    # =========================
    st.subheader("📈 PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration": list(range(1, len(convergence) + 1)),
        "Best Fitness (Cost)": convergence
    })

    st.line_chart(convergence_df.set_index("Iteration"))

    # =========================
    # Show convergence table (optional)
    # =========================
    st.subheader("📋 Convergence Data (First 10 Iterations)")
    st.dataframe(convergence_df.head(10))
