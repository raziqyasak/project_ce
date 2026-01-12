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
# 🍽️ Diet Meal Planning Optimisation  
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
st.sidebar.header("⚙️ PSO Parameters")

TARGET_CALORIES = st.sidebar.slider(
    "Target Calories (kcal)", 1500, 3000, 1900
)

NUM_PARTICLES = st.sidebar.slider("Number of Particles", 10, 50, 30)
MAX_ITER = st.sidebar.slider("Iterations", 50, 300, 100)

W = st.sidebar.slider("Inertia Weight (W)", 0.1, 1.0, 0.7)
C1 = st.sidebar.slider("Cognitive Parameter (C1)", 0.5, 2.5, 1.5)
C2 = st.sidebar.slider("Social Parameter (C2)", 0.5, 2.5, 1.5)

# =========================
# Fixed Meal Price Generator
# =========================
def generate_meal_prices_fixed(total_price):
    min_price = 2
    n_meals = 4
    # pastikan total_price cukup untuk min_price semua meal
    if total_price < min_price * n_meals:
        total_price = min_price * n_meals

    ratios = np.random.dirichlet([1]*n_meals)
    prices = ratios * (total_price - min_price*n_meals)
    prices = [p + min_price for p in prices]

    # betulkan supaya jumlah = total_price
    prices[-1] += total_price - sum(prices)
    prices = [round(p, 2) for p in prices]

    return {
        "Breakfast": prices[0],
        "Lunch": prices[1],
        "Dinner": prices[2],
        "Snack": prices[3]
    }

# =========================
# Fitness Function
# =========================
def fitness_function(index):
    row = data.iloc[int(index)]
    calorie_diff = abs(row['Calories'] - TARGET_CALORIES)
    price = row['Price_RM']
    # weighted fitness
    return calorie_diff * 0.7 + price * 0.3

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

    particles = np.random.randint(0, len(data), NUM_PARTICLES).astype(float)
    velocities = np.random.uniform(-1, 1, NUM_PARTICLES)

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
    meal_prices = generate_meal_prices_fixed(best_plan['Price_RM'])

    # =========================
    # Results
    # =========================
    st.divider()
    st.markdown("## ✅ Optimisation Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("Calories (kcal)", int(best_plan['Calories']))
    c2.metric("Total Price (RM)", round(best_plan['Price_RM'], 2))
    c3.metric("Protein (g)", best_plan['Protein'])

    st.markdown("### 🍳 Daily Meal Prices (≥ RM 2 each)")

    meal_df = pd.DataFrame({
        "Meal": ["Breakfast", "Lunch", "Dinner", "Snack"],
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
    st.markdown("## 📈 PSO Convergence Curve")
    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence)+1),
        "Best Fitness Value": convergence
    })

    chart = (
        alt.Chart(convergence_df)
        .mark_line(strokeWidth=3)
        .encode(
            x="Iteration",
            y="Best Fitness Value"
        )
        .properties(height=350)
    )

    st.altair_chart(chart, use_container_width=True)

    # =========================
    # Summary
    # =========================
    st.markdown(f"""
### 📝 Summary
- Target Calories: **{TARGET_CALORIES} kcal**
- Selected Plan Calories: **{int(best_plan['Calories'])} kcal**
- Total Daily Price: **RM {round(best_plan['Price_RM'],2)}**
- PSO Runtime: **{runtime} seconds**
- Each meal has **≥ RM 2** and total price matches dataset.
""")
