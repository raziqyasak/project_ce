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
# Diet Meal Planning Optimisation  
### Using Particle Swarm Optimisation (PSO)

This system selects a daily meal plan that satisfies calorie requirements
while minimising total cost.
""")
st.divider()

# =========================
# Upload Dataset
# =========================
st.markdown("## Upload Food Dataset")

uploaded_file = st.file_uploader(
    "Upload Food and Nutrition CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload the dataset to continue.")
    st.stop()

data = pd.read_csv(uploaded_file)

# =========================
# Detect Food Column
# =========================
possible_food_cols = ['Food', 'Menu', 'Item', 'Dish', 'Name']
food_col = None

for col in possible_food_cols:
    if col in data.columns:
        food_col = col
        break

required_cols = [food_col, 'Calories', 'Protein']

if food_col is None or not all(col in data.columns for col in required_cols):
    st.error("Dataset must contain Food/Menu, Calories, and Protein columns.")
    st.stop()

# Keep required columns only
data = data[[food_col, 'Calories', 'Protein']].copy()
data.rename(columns={food_col: 'Food'}, inplace=True)

# =========================
# Logical Cost Model
# =========================
np.random.seed(42)
data['Cost'] = data['Calories'] * np.random.uniform(0.008, 0.015, size=len(data))
data['Cost'] = data['Cost'].round(2)

NUM_MEALS = len(data)

# =========================
# Display Available Menu
# =========================
st.markdown("## Available Food Menu")

st.dataframe(
    data[['Food', 'Calories', 'Protein', 'Cost']],
    use_container_width=True,
    height=400
)

st.divider()

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("PSO Parameters")

TARGET_CALORIES = st.sidebar.slider(
    "Target Calories (kcal)", 1500, 3000, 1900
)

NUM_PARTICLES = st.sidebar.slider(
    "Number of Particles", 10, 50, 30
)

MAX_ITER = st.sidebar.slider(
    "Iterations", 50, 300, 100
)

W = st.sidebar.slider("Inertia Weight (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Parameter (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Parameter (C2)", 0.5, 2.5, 1.5)

# =========================
# Fitness Function
# =========================
def fitness_function(particle):
    if particle.sum() == 0:
        particle[random.randint(0, len(particle) - 1)] = 1

    selected = data[particle.astype(bool)]

    total_calories = selected['Calories'].sum()
    total_cost = selected['Cost'].sum()
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
st.markdown("### Run Optimisation")
run = st.button("Start PSO Optimisation")

# =========================
# PSO Execution
# =========================
if run:
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

            sigmoid = 1 / (1 + np.exp(-velocities[i]))
