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
data = pd.read_csv("Food_and_Nutrition_with_Price.csv")
data = data[['Calories', 'Protein']].copy()

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

    for iter_num in range(1, MAX_ITER + 1):
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
    total_calories = int(best_meal['Calories'].sum())
    total_cost = round(best_meal['Cost'].sum(), 2)
    c1.metric("Total Calories", total_calories)
    c2.metric("Total Cost (RM)", total_cost)

    st.markdown("### 🥗 Selected Daily Meal Plan")
    st.dataframe(best_meal, use_container_width=True)

    # =========================
    # GRAPH: Convergence Curve
    # =========================
    st.markdown("## 📈 PSO Convergence Curve")

    convergence_df = pd.DataFrame({
        "Iteration": range(1, len(convergence) + 1),
        "Best Fitness (Cost)": convergence
    })

    chart1 = (
        alt.Chart(convergence_df)
        .mark_line(color="#1f77b4", strokeWidth=3)
        .encode(
            x=alt.X("Iteration", title="Iteration"),
            y=alt.Y("Best Fitness (Cost)", title="Best Fitness Value")
        )
        .properties(
            title="Convergence Behaviour of PSO",
            height=350
        )
        .configure_axis(
            grid=True,
            labelFontSize=12,
            titleFontSize=13
        )
    )

    st.altair_chart(chart1, use_container_width=True)

    # =========================
    # Dynamic Convergence Summary
    # =========================
    final_iteration = len(convergence)
    final_fitness = round(convergence[-1], 2)
    target_met = "✅ Target calories achieved" if total_calories >= TARGET_CALORIES else "⚠️ Target calories not achieved"

    st.markdown(f"""
**Summary:**  
- PSO ran for **{final_iteration} iterations**.  
- Final fitness value (cost + penalty) is **{final_fitness}**.  
- Total calories in selected meal plan: **{total_calories}** kcal.  
- Total cost of selected meal plan: **RM {total_cost}**.  
- {target_met}  
- The convergence curve shows how the algorithm gradually improved solutions over iterations.
""")
