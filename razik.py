import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
import time

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
#  Diet Meal Planning Optimisation  
### Using Particle Swarm Optimisation (PSO)

This system selects **one daily diet plan** that achieves calories
closest to the target while minimising total cost (RM).
""")
st.divider()

# =========================
# Load Dataset
# =========================
data = pd.read_csv("Food_and_Nutrition_with_Price.csv")

# =========================
# Sidebar Parameters
# =========================
st.sidebar.header("PSO Parameters")
TARGET_CALORIES = st.sidebar.slider("Target Calories (kcal)", 1500, 3000, 1900)
NUM_PARTICLES = st.sidebar.slider("Number of Particles", 10, 50, 30)
MAX_ITER = st.sidebar.slider("Iterations", 50, 300, 100)
W = st.sidebar.slider("Inertia Weight (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Parameter (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Parameter (C2)", 0.5, 2.5, 1.5)

# =========================
# Realistic Meal Price Generator
# =========================
def generate_meal_prices(total_price):
    total_price = round(float(total_price), 2)

    # Ratio harga logik per meal
    ratios = {
        "Breakfast": 0.25,
        "Lunch": 0.35,
        "Dinner": 0.30,
        "Snack": 0.10
    }

    prices = {}
    for meal, ratio in ratios.items():
        raw_price = total_price * ratio
        rounded_price = round(raw_price * 2) / 2
        prices[meal] = max(1.0, rounded_price)

    # Adjust to match total price exactly
    diff = round(total_price - sum(prices.values()), 2)
    for meal in ["Lunch", "Dinner"]:
        if diff == 0:
            break
        adjustment = min(diff, 0.5) if diff > 0 else max(diff, -0.5)
        prices[meal] = round(prices[meal] + adjustment, 2)
        diff = round(total_price - sum(prices.values()), 2)

    return prices

# =========================
# Fitness Function
# =========================
def fitness_function(index):
    row = data.iloc[int(index)]
    calorie_diff = abs(row['Calories'] - TARGET_CALORIES)
    price = row['Price_RM']
    return calorie_diff * 0.7 + price * 0.3

# =========================
# Run Button
# =========================
st.markdown("### Run Optimisation")
run = st.button("Start PSO Optimisation")

# =========================
# PSO Execution
# =========================
if run:
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    start_time = time.time()

    # Initialize particles and velocities
    particles = np.random.randint(0, len(data), NUM_PARTICLES).astype(float)
    velocities = np.random.uniform(-1, 1, NUM_PARTICLES)

    # Personal best
    pbest = particles.copy()
    pbest_fitness = np.array([fitness_function(p) for p in particles])

    # Global best
    gbest = pbest[np.argmin(pbest_fitness)]
    convergence = []

    # PSO main loop
    for _ in range(MAX_ITER):
        for i in range(NUM_PARTICLES):
            r1, r2 = random.random(), random.random()
            velocities[i] = (
                W * velocities[i]
                + C1 * r1 * (pbest[i] - particles[i])
                + C2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, len(data) - 1)

            fitness = fitness_function(particles[i])
            if fitness < pbest_fitness[i]:
                pbest[i] = particles[i]
                pbest_fitness[i] = fitness

        gbest = pbest[np.argmin(pbest_fitness)]
        convergence.append(min(pbest_fitness))

    runtime = round(time.time() - start_time, 3)
    best_plan = data.iloc[int(gbest)]
    meal_prices = generate_meal_prices(best_plan['Price_RM'])

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## Optimisation Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories (kcal)", int(best_plan['Calories']))
    c2.metric("Total Price (RM)", round(best_plan['Price_RM'], 2))
    c3.metric("Protein (g)", best_plan['Protein'])
    c4.metric("Best Fitness Value", round(min(pbest_fitness), 3))  # Display Best Fitness

    st.markdown("### Daily Meal Suggestions & Prices")
    meal_df = pd.DataFrame({
        "Meal": ["Breakfast", "Lunch", "Dinner", "Snack"],
        "Suggestion": [
            best_plan['Breakfast Suggestion'],
            best_plan['Lunch Suggestion'],
            best_plan['Dinner Suggestion'],
            best_plan['Snack Suggestion']
        ],
        "Price (RM)": [
            meal_prices["Breakfast"],
            meal_prices["Lunch"],
            meal_prices["Dinner"],
            meal_prices["Snack"]
        ]
    })
    st.dataframe(meal_df, use_container_width=True)

    # =========================
    # Convergence Curve
    # =========================
    st.markdown("## PSO Convergence Curve")
    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence) + 1),
        "Best Fitness Value": convergence
    })
    chart = (
        alt.Chart(convergence_df)
        .mark_line(strokeWidth=3)
        .encode(x="Iteration", y="Best Fitness Value")
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown(f"**Optimisation completed in {runtime} seconds.**")
