import streamlit as st
import pandas as pd
import numpy as np
import random

# ========================= 
# Page Configuration
# =========================
st.set_page_config(
    page_title="Diet Meal Planning (PSO Structure Only)",
    layout="wide"
)

# =========================
# Header
# =========================
st.markdown("""
# 🍽️ Diet Meal Planning  
### PSO Structure (Without Fitness Function)

This version demonstrates particle movement and meal selection
without optimisation evaluation.
""")

st.divider()

# =========================
# Load Dataset
# =========================
data = pd.read_csv("Food_and_Nutrition_with_Price.csv")
data = data[['Calories', 'Protein']].copy()

np.random.seed(42)
data['Cost'] = np.random.uniform(2, 10, size=len(data))

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("⚙️ PSO Parameters")

MEALS_PER_DAY = st.sidebar.slider("Meals per Day", 2, 5, 3)
NUM_PARTICLES = st.sidebar.slider("Number of Particles", 10, 50, 30)
MAX_ITER = st.sidebar.slider("Iterations", 50, 300, 100)

W = st.sidebar.slider("Inertia Weight (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Parameter (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Parameter (C2)", 0.5, 2.5, 1.5)

# =========================
# Run Button
# =========================
st.markdown("### 🚀 Run PSO Simulation")
run = st.button("Start Simulation")

# =========================
# PSO Simulation (No Fitness)
# =========================
if run:
    particles = np.random.randint(
        0, len(data), (NUM_PARTICLES, MEALS_PER_DAY)
    ).astype(float)

    velocities = np.random.uniform(-1, 1, (NUM_PARTICLES, MEALS_PER_DAY))

    gbest = particles[0]

    for _ in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1, r2 = random.random(), random.random()

            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (gbest - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, len(data) - 1)

    selected_meal = data.iloc[particles[0].astype(int)]

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## 📋 Selected Meal Combination (Sample Particle)")

    c1, c2 = st.columns(2)
    c1.metric("Total Calories", int(selected_meal['Calories'].sum()))
    c2.metric("Total Cost (RM)", round(selected_meal['Cost'].sum(), 2))

    st.dataframe(selected_meal, use_container_width=True)
