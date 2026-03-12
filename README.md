# Q-LEARNING IMPLEMENTATION GUIDE - PHASE 1
## Simulation-Optimization for Multi-Depot Online Dynamic Pickup-and-Delivery

---

## 📦 PACKAGE CONTENTS

Bạn nhận được **2 files Python** với tổng cộng **~52KB** code và documentation:

1. **`q_learning_guide.py`** (~26KB)
   - Complete Q-learning implementation
   - StateEncoder (3500 states)
   - ActionEncoder (16 actions)
   - QLearningAgent class
   - Reward function
   - Training & evaluation examples

2. **`integration_guide.py`** (~26KB)
   - Integration với code hiện có
   - SimulationEnvironment wrapper
   - Unified simulation loop
   - Training workflow
   - Evaluation framework
   - Statistical analysis helpers

---

## 🎯 QUICK START (5 STEPS)

### **STEP 1: Copy files vào project**

```bash
# Đặt 2 files Python vào cùng folder với code hiện tại
project/
├── q_learning_guide.py          # NEW
├── integration_guide.py          # NEW
├── your_simulation.py            # Your existing code
└── ...
```

---

### **STEP 2: Test state encoding**

```python
from q_learning_guide import StateEncoder

encoder = StateEncoder()

# Test với 1 order mẫu
test_state = encoder.encode_state(
    time_bin=50,        # 10:00 AM
    pickup_lat=10.77,
    pickup_lng=106.70,
    dropoff_lat=10.78,
    dropoff_lng=106.71,
    num_available=10,
    total_shippers=16
)

print(f"State index: {test_state}")  
# Expected: number between 0-3499

# Nếu ra ngoài range → fix encode_zone() logic
```

---

### **STEP 3: Train Q-learning**

```python
from integration_guide import train_qlearning, check_state_coverage

# Train 50,000 episodes (~5-10 phút)
agent = train_qlearning(
    n_episodes=50000,
    scenario="nominal",
    save_interval=5000
)

# Kiểm tra Q-table coverage
check_state_coverage(agent)
# Expected: >50% coverage

# Q-table được save tự động:
# - q_table_episode_5000.pkl
# - q_table_episode_10000.pkl
# - ...
# - q_table_final.pkl
```

---

### **STEP 4: Evaluate all 3 policies**

```python
from integration_guide import evaluate_all_policies, statistical_analysis

# Run 100 replications × 3 scenarios × 3 policies
# (~10-15 phút)
results_df = evaluate_all_policies(
    n_replications=100,
    scenarios=["moderate", "nominal", "stress"],
    qlearning_model_path="q_table_final.pkl"
)

# Results saved to: evaluation_results.csv
print(results_df)
```

---

### **STEP 5: Statistical analysis**

```python
# Compare policies
statistical_analysis(results_df)

# Output example:
# Scenario: NOMINAL
# ----------------------------------------------------------------------
#   Q-learning vs Greedy:  t=-2.451, p=0.0147
#   Q-learning improvement over Greedy: +8.3%
```

---

## 📊 EXPECTED RESULTS

### **Training (50K episodes):**
- Q-table coverage: 70-80%
- Epsilon decay: 1.0 → 0.05
- Training time: 5-10 minutes

### **Evaluation:**

| Policy | Distance (km) | Lateness (min) | Improvement |
|--------|---------------|----------------|-------------|
| Greedy | 1250 ± 120 | 12.3 ± 2.1 | baseline |
| BALANCE | 1180 ± 115 | 11.8 ± 2.0 | +5.6% |
| Q-learning | 1145 ± 110 | 11.5 ± 1.9 | **+8.4%** |

*Note: Actual numbers depend on your data*

---

## 🔧 CUSTOMIZATION GUIDE

### **Adjust State Space (if needed):**

```python
# In q_learning_guide.py, modify:
N_TIME_STATES = 50      # Try 30-70
N_ZONE_STATES = 14      # Match your districts
N_WORKLOAD_STATES = 5   # Try 3-7
```

### **Tune Hyperparameters:**

```python
# In q_learning_guide.py:
ALPHA = 0.1             # Learning rate (try 0.05-0.2)
GAMMA = 0.95            # Discount (try 0.9-0.99)
EPSILON_DECAY = 0.9995  # Exploration decay (try 0.999-0.9999)
```

### **Adjust Reward Function:**

```python
# In q_learning_guide.py:
def compute_reward(assignment_cost, lateness, lambda_penalty=1.0):
    # Current: reward = -(cost + λ*lateness)
    # Try different λ values:
    lambda_penalty = 0.5  # Less penalty on lateness
    lambda_penalty = 2.0  # More penalty on lateness
```

---

## 🐛 TROUBLESHOOTING

### **Problem: "State index out of range"**
```python
# Solution: Check encode_zone() logic
encoder = StateEncoder()
# Add print statements:
def encode_zone(self, pickup_lat, pickup_lng, ...):
    zone = ...  # your logic
    print(f"Zone: {zone}, range: 0-{N_ZONE_STATES-1}")
    return np.clip(zone, 0, N_ZONE_STATES-1)  # Add safety clip
```

### **Problem: "Q-learning worse than Greedy"**
```python
# Possible causes:
# 1. Reward scale wrong → Normalize costs
# 2. Not enough training → Increase to 100K episodes
# 3. State too sparse → Reduce N_TIME_STATES to 30

# Check learning:
from integration_guide import visualize_learning_curve
visualize_learning_curve("training_rewards.npy")
# Should show upward trend
```

### **Problem: "Training too slow"**
```python
# Solution: Reduce episodes first
agent = train_qlearning(n_episodes=10000)  # Quick test
# If results look good, scale to 50K
```

---

## 📈 GENERATING CHAPTER 5 CONTENT

### **Table 5.1: Performance Comparison**

```python
import pandas as pd

results = pd.read_csv("evaluation_results.csv")

# Pivot table for thesis
table = results.pivot_table(
    index="policy",
    columns="scenario",
    values=["mean_distance", "mean_lateness"],
    aggfunc="mean"
)

# Export for Word
table.to_csv("table_5_1_performance.csv")

# Or LaTeX
print(table.to_latex(float_format="%.1f"))
```

### **Figure 5.1: Distance Comparison**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot by scenario
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, scenario in enumerate(["moderate", "nominal", "stress"]):
    data = results[results["scenario"] == scenario]
    sns.boxplot(data=data, x="policy", y="mean_distance", ax=axes[i])
    axes[i].set_title(f"Scenario: {scenario.capitalize()}")
    axes[i].set_ylabel("Total Distance (km)")

plt.tight_layout()
plt.savefig("figure_5_1_distance.png", dpi=300)
```

### **Figure 5.2: Learning Curve**

```python
from integration_guide import visualize_learning_curve
visualize_learning_curve("training_rewards.npy")
# Saves to: learning_curve.png
```

---

## 💡 INTEGRATION WITH YOUR CODE

### **Minimal changes needed:**

Your existing code already has:
- ✅ `greedy_dispatch(order, current_time)` - line 672
- ✅ `choose_shipper(order, current_time)` - line 1436 (BALANCE)
- ✅ `shippers` list
- ✅ Synthetic order generator

**You only need to:**

1. Import Q-learning components:
```python
from q_learning_guide import StateEncoder, ActionEncoder, QLearningAgent
from integration_guide import run_episode, train_qlearning
```

2. Add policy selector:
```python
if policy == "greedy":
    shipper = greedy_dispatch(order, current_time)
elif policy == "balance":
    shipper = choose_shipper(order, current_time)
elif policy == "qlearning":
    shipper = qlearning_dispatch(order, ...)  # From integration_guide
```

3. Run training & evaluation (see QUICK START above)

---

## 📝 CHAPTER 5 OUTLINE SUGGESTION

```
Chapter 5: Results and Discussion

5.1 Introduction
    - Recap 3 policies: Greedy, BALANCE, Q-learning
    - Evaluation setup: 100 reps × 3 scenarios

5.2 Training Results (Q-learning only)
    - Figure 5.1: Learning curve
    - Table 5.1: Q-table statistics
    - Convergence analysis

5.3 Performance Comparison
    - Table 5.2: Mean ± std for all metrics
    - Figure 5.2: Distance comparison (box plots)
    - Figure 5.3: Lateness comparison

5.4 Statistical Significance
    - t-test results
    - Effect sizes
    - Practical significance

5.5 Scenario Analysis
    - How do policies perform under stress?
    - Trade-offs between distance and lateness

5.6 Discussion
    - Why does Q-learning outperform baselines?
    - When does BALANCE work well?
    - Practical implications for small operators

5.7 Limitations
    - Simulation vs reality gap
    - State space assumptions
    - Training data requirements
```

---

## ⏱️ TIME ESTIMATES

| Task | Time | Output |
|------|------|--------|
| Copy files & test | 15 min | State encoding works |
| Train Q-learning | 10 min | q_table_final.pkl |
| Run evaluation | 15 min | evaluation_results.csv |
| Statistical analysis | 5 min | t-test results |
| Generate figures | 10 min | 3-4 PNG files |
| Write Chapter 5 | 2-3 hours | Draft chapter |
| **TOTAL** | **~3-4 hours** | Complete Chapter 5 |

---

## 📞 SUPPORT & DEBUGGING

**If you encounter issues:**

1. Check state encoding first:
```python
encoder = StateEncoder()
# Test 10 random states
for _ in range(10):
    state = encoder.encode_state(...)
    assert 0 <= state < 3500, f"Invalid state: {state}"
```

2. Check action encoding:
```python
action_encoder = ActionEncoder(shippers, strategy="dynamic")
valid_actions = action_encoder.get_valid_actions(current_time, order)
print(f"Valid actions: {valid_actions}")  # Should be non-empty list
```

3. Monitor training:
```python
# Add to training loop:
if episode % 100 == 0:
    print(f"Episode {episode}: Q-updates = {agent.total_updates}")
```

---

## ✅ SUCCESS CRITERIA

**You'll know it's working when:**

✅ State encoding produces values in [0, 3499]  
✅ Training shows epsilon decay from 1.0 → 0.05  
✅ Q-table coverage > 50%  
✅ Learning curve shows upward trend  
✅ Q-learning beats Greedy by 5-15%  
✅ All 3 policies produce similar service rates (≈100%)


