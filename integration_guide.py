"""
================================================================================
INTEGRATION GUIDE: Kết nối Q-Learning với Code Hiện Có
================================================================================

File này hướng dẫn CHI TIẾT cách tích hợp Q-learning vào simulation code

STRUCTURE:
1. Modify existing code structure
2. Add Q-learning dispatch function
3. Unified simulation loop for all 3 policies
4. Episode/replication management
5. Results collection for Chapter 5

================================================================================
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pickle
import json

# Import your existing code
# from your_simulation import (
#     generate_synthetic_orders,
#     road_distance,
#     greedy_dispatch,
#     choose_shipper,  # BALANCE policy
#     shippers,
#     ...
# )

# Import Q-learning components from q_learning_guide.py
from q_learning_guide import (
    StateEncoder,
    ActionEncoder, 
    QLearningAgent,
    compute_reward,
    TOTAL_STATES,
    N_ACTIONS
)


# =============================================================================
# STEP 1: WRAP EXISTING FUNCTIONS
# =============================================================================

class SimulationEnvironment:
    """
    Wrapper around your existing simulation code
    
    This class provides a clean interface for:
    - Resetting simulation state
    - Stepping through orders
    - Computing rewards
    - Tracking metrics
    """
    
    def __init__(self, shippers_list: List[Dict], 
                 scenario: str = "nominal",
                 lambda_penalty: float = 1.0):
        """
        Args:
            shippers_list: Your shipper objects
            scenario: "moderate", "nominal", or "stress"
            lambda_penalty: Lateness penalty weight
        """
        self.shippers = shippers_list
        self.scenario = scenario
        self.lambda_penalty = lambda_penalty
        
        # Track metrics
        self.reset_metrics()
    
    def reset_metrics(self):
        """Reset all tracking variables for new episode/replication"""
        self.total_distance = 0.0
        self.total_lateness = 0.0
        self.num_served = 0
        self.num_orders = 0
        
        # Reset shipper states
        for shipper in self.shippers:
            shipper["available_time"] = 0.0
            shipper["workload"] = 0
            shipper["current_location"] = shipper["node"]  # Reset to depot
    
    def assign_order(self, order: Dict, shipper: Dict, 
                    current_time: float) -> Tuple[float, float]:
        """
        Execute order assignment and return cost metrics
        
        Args:
            order: Order dict
            shipper: Selected shipper dict
            current_time: Current simulation time
        
        Returns:
            (assignment_cost, lateness): Cost and lateness in minutes
        """
        # Use your existing road_distance function
        pickup_dist = road_distance(shipper["node"], order["pickup_node"])
        delivery_dist = road_distance(order["pickup_node"], order["dropoff_node"])
        
        total_dist = pickup_dist + delivery_dist
        
        # Time calculations (simplified - adjust to your model)
        SPEED_KM_PER_HOUR = 20  # Average speed
        travel_time_hours = total_dist / SPEED_KM_PER_HOUR
        travel_time_minutes = travel_time_hours * 60
        
        # Expected delivery time (from order["expectedDeliveryTime"])
        expected_delivery_minutes = 60  # 3h or 5h service → use actual value
        
        # Compute lateness
        lateness = max(0, travel_time_minutes - expected_delivery_minutes)
        
        # Update shipper state
        shipper["available_time"] = current_time + travel_time_minutes
        shipper["workload"] += 1
        shipper["current_location"] = order["dropoff_node"]
        
        # Update metrics
        self.total_distance += total_dist
        self.total_lateness += lateness
        self.num_served += 1
        
        # Assignment cost (for reward)
        assignment_cost = total_dist  # Could add other costs
        
        return assignment_cost, lateness
    
    def get_metrics(self) -> Dict[str, float]:
        """Return episode metrics"""
        return {
            "total_distance_km": self.total_distance,
            "avg_lateness_min": self.total_lateness / max(self.num_served, 1),
            "service_rate": self.num_served / max(self.num_orders, 1),
            "num_served": self.num_served,
            "num_orders": self.num_orders
        }


# =============================================================================
# STEP 2: Q-LEARNING DISPATCH FUNCTION
# =============================================================================

def qlearning_dispatch(order: Dict, current_time: float, time_bin: int,
                      agent: QLearningAgent, 
                      state_encoder: StateEncoder,
                      action_encoder: ActionEncoder,
                      shippers: List[Dict],
                      training: bool = False) -> Dict:
    """
    Q-learning-based order-shipper assignment
    
    This function mirrors your greedy_dispatch() and choose_shipper()
    
    Args:
        order: Order to assign
        current_time: Current simulation time
        time_bin: Current time bin [0-95]
        agent: Trained QLearningAgent
        state_encoder, action_encoder: Encoding objects
        shippers: List of all shippers
        training: If True, use epsilon-greedy; if False, pure greedy
    
    Returns:
        selected_shipper: Shipper object to assign order to
    """
    
    # 1. Encode current state
    num_available = sum(1 for s in shippers if s["available_time"] <= current_time)
    state = state_encoder.encode_state(
        time_bin,
        order["pickup_node"][0],  # pickup_lat
        order["pickup_node"][1],  # pickup_lng
        order["dropoff_node"][0], # dropoff_lat
        order["dropoff_node"][1], # dropoff_lng
        num_available,
        len(shippers)
    )
    
    # 2. Get valid actions
    valid_actions = action_encoder.get_valid_actions(current_time, order)
    
    if not valid_actions:
        print(f"WARNING: No valid actions for order {order.get('id', 'unknown')}")
        return None
    
    # 3. Select action using Q-table
    action = agent.select_action(state, valid_actions, training=training)
    
    # 4. Convert action to shipper
    selected_shipper = action_encoder.action_to_shipper(action, current_time, order)
    
    return selected_shipper


# =============================================================================
# STEP 3: UNIFIED SIMULATION LOOP
# =============================================================================

def run_episode(orders: List[Dict],
               policy: str,
               env: SimulationEnvironment,
               agent: QLearningAgent = None,
               state_encoder: StateEncoder = None,
               action_encoder: ActionEncoder = None,
               training: bool = False) -> Dict[str, float]:
    """
    Run ONE episode (one day of orders) with specified policy
    
    Args:
        orders: List of orders for this episode (synthetic or real)
        policy: "greedy", "balance", or "qlearning"
        env: SimulationEnvironment object
        agent: QLearningAgent (required for qlearning)
        state_encoder, action_encoder: Required for qlearning
        training: If True, perform Q-updates (only for qlearning)
    
    Returns:
        metrics: Dict with performance metrics
    """
    
    env.reset_metrics()
    env.num_orders = len(orders)
    
    current_time = 7.0  # Start at 7:00 AM
    
    for order_idx, order in enumerate(orders):
        
        # Convert order timestamp to time_bin
        # (Adjust this based on how your orders are timestamped)
        order_hour = order.get("hour", 12)  # Replace with actual field
        time_bin = int((order_hour - 7) * 6)  # 7am = bin 0, each hour = 6 bins
        time_bin = np.clip(time_bin, 0, 95)
        
        # Update current_time to order arrival time
        current_time = order_hour
        
        # === POLICY SELECTION ===
        
        if policy == "greedy":
            # Use your existing greedy_dispatch function
            selected_shipper = greedy_dispatch(order, current_time)
        
        elif policy == "balance":
            # Use your existing choose_shipper function (BALANCE heuristic)
            selected_shipper = choose_shipper(order, current_time)
        
        elif policy == "qlearning":
            # Use Q-learning
            if agent is None:
                raise ValueError("Q-learning requires agent!")
            
            # Get current state (before assignment)
            num_available = sum(1 for s in env.shippers 
                              if s["available_time"] <= current_time)
            state = state_encoder.encode_state(
                time_bin,
                order["pickup_node"][0],
                order["pickup_node"][1],
                order["dropoff_node"][0],
                order["dropoff_node"][1],
                num_available,
                len(env.shippers)
            )
            
            # Get valid actions
            valid_actions = action_encoder.get_valid_actions(current_time, order)
            
            # Select action
            action = agent.select_action(state, valid_actions, training=training)
            selected_shipper = action_encoder.action_to_shipper(
                action, current_time, order
            )
            
        else:
            raise ValueError(f"Unknown policy: {policy}")
        
        # === EXECUTE ASSIGNMENT ===
        
        if selected_shipper is None:
            # No shipper available - skip this order
            continue
        
        assignment_cost, lateness = env.assign_order(
            order, selected_shipper, current_time
        )
        
        # === Q-LEARNING UPDATE (if training) ===
        
        if policy == "qlearning" and training:
            
            # Compute reward
            reward = compute_reward(assignment_cost, lateness, 
                                   env.lambda_penalty)
            
            # Get next state (after assignment)
            if order_idx < len(orders) - 1:
                next_order = orders[order_idx + 1]
                next_hour = next_order.get("hour", current_time + 0.5)
                next_time_bin = int((next_hour - 7) * 6)
                next_time_bin = np.clip(next_time_bin, 0, 95)
                
                num_available_next = sum(1 for s in env.shippers 
                                        if s["available_time"] <= next_hour)
                next_state = state_encoder.encode_state(
                    next_time_bin,
                    next_order["pickup_node"][0],
                    next_order["pickup_node"][1],
                    next_order["dropoff_node"][0],
                    next_order["dropoff_node"][1],
                    num_available_next,
                    len(env.shippers)
                )
                
                next_valid_actions = action_encoder.get_valid_actions(
                    next_hour, next_order
                )
                done = False
            else:
                next_state = state
                next_valid_actions = []
                done = True
            
            # Q-update
            agent.update(state, action, reward, next_state, 
                        next_valid_actions, done)
    
    # Return episode metrics
    return env.get_metrics()


# =============================================================================
# STEP 4: TRAINING LOOP
# =============================================================================

def train_qlearning(n_episodes: int = 50000,
                   scenario: str = "nominal",
                   save_interval: int = 5000):
    """
    Train Q-learning agent over multiple episodes
    
    Args:
        n_episodes: Number of training episodes
        scenario: Demand scenario for training
        save_interval: Save Q-table every N episodes
    
    Returns:
        agent: Trained QLearningAgent
    """
    
    print("="*70)
    print(f"Q-LEARNING TRAINING: {n_episodes} episodes")
    print("="*70)
    
    # 1. Initialize components
    # (Assume you have shippers list from your simulation)
    # shippers = load_shippers()  # Your function
    
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder(shippers, strategy="dynamic")
    agent = QLearningAgent(state_encoder, action_encoder)
    env = SimulationEnvironment(shippers, scenario=scenario)
    
    # 2. Training loop
    episode_rewards = []
    
    for episode in range(n_episodes):
        
        # Generate synthetic orders for this episode
        # orders = generate_synthetic_orders(scenario, seed=episode)
        
        # Run episode with Q-learning (training=True)
        metrics = run_episode(
            orders, 
            policy="qlearning",
            env=env,
            agent=agent,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            training=True  # Enable Q-updates
        )
        
        # Track progress
        avg_reward = -(metrics["total_distance_km"] + 
                      env.lambda_penalty * metrics["avg_lateness_min"])
        episode_rewards.append(avg_reward)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Logging
        if episode % 1000 == 0:
            recent_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode:5d}/{n_episodes} | "
                  f"Reward: {recent_reward:7.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Q-updates: {agent.total_updates}")
        
        # Save checkpoint
        if episode > 0 and episode % save_interval == 0:
            agent.save(f"q_table_episode_{episode}.pkl")
    
    # 3. Save final Q-table
    agent.save("q_table_final.pkl")
    
    # 4. Save training history
    np.save("training_rewards.npy", episode_rewards)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total Q-updates: {agent.total_updates}")
    print(f"Q-table sparsity: {np.sum(agent.Q != 0) / agent.Q.size * 100:.1f}%")
    print("="*70)
    
    return agent


# =============================================================================
# STEP 5: EVALUATION FRAMEWORK
# =============================================================================

def evaluate_all_policies(n_replications: int = 100,
                         scenarios: List[str] = ["moderate", "nominal", "stress"],
                         qlearning_model_path: str = "q_table_final.pkl"):
    """
    Compare all 3 policies across scenarios
    
    This generates data for Chapter 5: Results & Discussion
    
    Args:
        n_replications: Number of Monte Carlo replications per scenario
        scenarios: List of demand scenarios to test
        qlearning_model_path: Path to trained Q-table
    
    Returns:
        results_df: Pandas DataFrame with all results
    """
    
    print("="*70)
    print("POLICY EVALUATION")
    print("="*70)
    
    # Load trained Q-learning agent
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder(shippers, strategy="dynamic")
    qlearning_agent = QLearningAgent(state_encoder, action_encoder)
    qlearning_agent.load(qlearning_model_path)
    
    # Results storage
    all_results = []
    
    policies = ["greedy", "balance", "qlearning"]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario.upper()}")
        print("-" * 70)
        
        env = SimulationEnvironment(shippers, scenario=scenario)
        
        for policy in policies:
            
            policy_metrics = {
                "distance": [],
                "lateness": [],
                "service_rate": []
            }
            
            for rep in range(n_replications):
                
                # Generate orders for this replication
                # orders = generate_synthetic_orders(scenario, seed=rep)
                
                # Run episode (training=False for all policies)
                if policy == "qlearning":
                    metrics = run_episode(
                        orders, 
                        policy="qlearning",
                        env=env,
                        agent=qlearning_agent,
                        state_encoder=state_encoder,
                        action_encoder=action_encoder,
                        training=False  # Pure exploitation
                    )
                else:
                    metrics = run_episode(orders, policy=policy, env=env)
                
                # Store metrics
                policy_metrics["distance"].append(metrics["total_distance_km"])
                policy_metrics["lateness"].append(metrics["avg_lateness_min"])
                policy_metrics["service_rate"].append(metrics["service_rate"])
            
            # Aggregate statistics
            result_row = {
                "scenario": scenario,
                "policy": policy,
                "mean_distance": np.mean(policy_metrics["distance"]),
                "std_distance": np.std(policy_metrics["distance"]),
                "mean_lateness": np.mean(policy_metrics["lateness"]),
                "std_lateness": np.std(policy_metrics["lateness"]),
                "mean_service_rate": np.mean(policy_metrics["service_rate"]),
            }
            
            all_results.append(result_row)
            
            print(f"  {policy.upper():12s} | "
                  f"Distance: {result_row['mean_distance']:6.1f} ± {result_row['std_distance']:4.1f} km | "
                  f"Lateness: {result_row['mean_lateness']:5.1f} ± {result_row['std_lateness']:4.1f} min")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv("evaluation_results.csv", index=False)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("Results saved to: evaluation_results.csv")
    print("="*70)
    
    return results_df


# =============================================================================
# STEP 6: STATISTICAL TESTS (for Chapter 5)
# =============================================================================

def statistical_analysis(results_df: pd.DataFrame):
    """
    Perform statistical tests to compare policies
    
    Returns p-values for:
    - Q-learning vs Greedy
    - Q-learning vs BALANCE
    - BALANCE vs Greedy
    """
    from scipy import stats
    
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TESTS (t-tests)")
    print("="*70)
    
    for scenario in results_df["scenario"].unique():
        
        print(f"\nScenario: {scenario.upper()}")
        print("-" * 70)
        
        scenario_data = results_df[results_df["scenario"] == scenario]
        
        # Get raw data for each policy (you need to store this separately)
        # This is simplified - you need actual replication-level data
        
        greedy_dist = scenario_data[scenario_data["policy"] == "greedy"]["mean_distance"].values
        balance_dist = scenario_data[scenario_data["policy"] == "balance"]["mean_distance"].values
        ql_dist = scenario_data[scenario_data["policy"] == "qlearning"]["mean_distance"].values
        
        # t-tests (paired, since same orders)
        # NOTE: You need replication-level arrays, not just means
        
        # t_ql_vs_greedy, p_ql_vs_greedy = stats.ttest_rel(ql_dist, greedy_dist)
        # t_ql_vs_balance, p_ql_vs_balance = stats.ttest_rel(ql_dist, balance_dist)
        
        # print(f"  Q-learning vs Greedy:  t={t_ql_vs_greedy:.3f}, p={p_ql_vs_greedy:.4f}")
        # print(f"  Q-learning vs BALANCE: t={t_ql_vs_balance:.3f}, p={p_ql_vs_balance:.4f}")
        
        # Improvement percentage
        greedy_mean = scenario_data[scenario_data["policy"] == "greedy"]["mean_distance"].values[0]
        ql_mean = scenario_data[scenario_data["policy"] == "qlearning"]["mean_distance"].values[0]
        
        improvement_pct = (greedy_mean - ql_mean) / greedy_mean * 100
        
        print(f"  Q-learning improvement over Greedy: {improvement_pct:+.1f}%")


# =============================================================================
# USAGE WORKFLOW
# =============================================================================

"""
COMPLETE WORKFLOW TO GET CHAPTER 5 RESULTS:
-------------------------------------------

1. TRAINING PHASE (once):
   
   agent = train_qlearning(
       n_episodes=50000,
       scenario="nominal",
       save_interval=5000
   )
   
   → This creates: q_table_final.pkl
   → Time: ~5-10 minutes

2. EVALUATION PHASE:
   
   results_df = evaluate_all_policies(
       n_replications=100,
       scenarios=["moderate", "nominal", "stress"],
       qlearning_model_path="q_table_final.pkl"
   )
   
   → This creates: evaluation_results.csv
   → Time: ~10-15 minutes

3. STATISTICAL ANALYSIS:
   
   statistical_analysis(results_df)
   
   → Prints t-test results and improvement percentages

4. VISUALIZATION (for Chapter 5):
   
   import matplotlib.pyplot as plt
   import seaborn as sns
   
   # Box plot: Distance comparison
   fig, axes = plt.subplots(1, 3, figsize=(15, 5))
   for i, scenario in enumerate(["moderate", "nominal", "stress"]):
       data = results_df[results_df["scenario"] == scenario]
       data.boxplot(column="mean_distance", by="policy", ax=axes[i])
       axes[i].set_title(f"Scenario: {scenario}")
   plt.tight_layout()
   plt.savefig("chapter5_distance_comparison.png", dpi=300)
   
   # Similar plots for lateness, service_rate

5. GENERATE TABLES FOR THESIS:
   
   # Table 5.1: Performance comparison
   pivot = results_df.pivot_table(
       index="policy",
       columns="scenario",
       values=["mean_distance", "mean_lateness"],
       aggfunc="mean"
   )
   
   print(pivot.to_latex())  # For LaTeX thesis
   # Or: pivot.to_csv("table_5_1.csv") for Word

"""


# =============================================================================
# DEBUGGING HELPERS
# =============================================================================

def check_state_coverage(agent: QLearningAgent):
    """
    Verify that Q-table is well-populated
    """
    total_entries = agent.Q.size
    nonzero_entries = np.sum(agent.Q != 0)
    coverage = nonzero_entries / total_entries * 100
    
    print(f"Q-table coverage: {coverage:.1f}%")
    print(f"Nonzero entries: {nonzero_entries} / {total_entries}")
    
    if coverage < 50:
        print("⚠️  WARNING: Low coverage - consider training longer")
    elif coverage > 80:
        print("✅ Good coverage - most states visited")
    
    # Check state visitation histogram
    state_visits = np.sum(agent.Q != 0, axis=1)  # Count actions per state
    
    print(f"\nState visitation distribution:")
    print(f"  Never visited: {np.sum(state_visits == 0)} states")
    print(f"  Visited 1-5 actions: {np.sum((state_visits > 0) & (state_visits <= 5))} states")
    print(f"  Visited 6-10 actions: {np.sum((state_visits > 5) & (state_visits <= 10))} states")
    print(f"  Visited 11+ actions: {np.sum(state_visits > 10)} states")


def visualize_learning_curve(rewards_file: str = "training_rewards.npy"):
    """
    Plot learning progress
    """
    import matplotlib.pyplot as plt
    
    rewards = np.load(rewards_file)
    
    # Smooth with moving average
    window = 100
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label="Raw")
    plt.plot(smoothed, label=f"Smoothed (window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Q-Learning Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png", dpi=300)
    plt.show()
    
    print("Learning curve saved to: learning_curve.png")


# =============================================================================
# FINAL CHECKLIST
# =============================================================================

"""
INTEGRATION CHECKLIST FOR STUDENT:
----------------------------------

✅ Step 1: Copy q_learning_guide.py to your project

✅ Step 2: Copy this file (integration_guide.py) to your project

✅ Step 3: Modify your existing simulation code:
   - Ensure shippers list is accessible
   - Ensure greedy_dispatch() and choose_shipper() work as functions
   - Ensure generate_synthetic_orders() returns list of order dicts

✅ Step 4: Test state encoding:
   encoder = StateEncoder()
   test_state = encoder.encode_state(50, 10.77, 106.70, 10.78, 106.71, 10, 16)
   print(f"Test state index: {test_state}")  # Should be 0-3499

✅ Step 5: Train Q-learning:
   agent = train_qlearning(n_episodes=50000)
   check_state_coverage(agent)

✅ Step 6: Evaluate all policies:
   results = evaluate_all_policies(n_replications=100)
   print(results)

✅ Step 7: Statistical analysis:
   statistical_analysis(results)

✅ Step 8: Generate Chapter 5 figures and tables

EXPECTED TIME:
- Training: 5-10 minutes
- Evaluation: 10-15 minutes
- Total: ~20-25 minutes to get full results

IF ISSUES:
- State encoding errors → Check district detection logic
- Q-table not learning → Check reward scale (normalize!)
- Training too slow → Reduce n_episodes to 20000 first
- Results look wrong → Verify greedy/balance implementations match

GOOD LUCK! 🚀
"""
