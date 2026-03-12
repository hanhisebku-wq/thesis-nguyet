"""
================================================================================
Q-LEARNING IMPLEMENTATION GUIDE
Simulation-Optimization for Multi-Depot Online Dynamic Pickup-and-Delivery
================================================================================

AUTHOR: Implementation guide for undergraduate thesis
DATE: March 2026
PURPOSE: Complete Q-learning algorithm for online order-shipper matching

This guide provides:
1. State space encoding (3500 states)
2. Action space definition (16 actions)
3. Q-table initialization and updates
4. Epsilon-greedy exploration
5. Training loop structure
6. Integration with existing simulation code

================================================================================
PART 1: STATE SPACE DESIGN (3500 states)
================================================================================

CONCEPTUAL BREAKDOWN:
---------------------
Total states = Time_bins × Zone_pairs × Workload_levels
3500 ≈ 50 × 14 × 5

State components:
1. Temporal component (50 bins)
   - 96 raw time bins → aggregate to 50 temporal states
   - Group nearby bins to reduce sparsity
   
2. Spatial component (14 zones)
   - 13 district hotspots + 1 "other" zone
   - Represents pickup-delivery zone pair type
   
3. System state component (5 levels)
   - Current workload/congestion level
   - Number of available shippers
"""

import numpy as np
from typing import Tuple, Dict, List
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================

# State space dimensions
N_TIME_STATES = 50      # Aggregate 96 bins into 50 temporal states
N_ZONE_STATES = 14      # 13 districts + 1 other
N_WORKLOAD_STATES = 5   # Low, Med-Low, Medium, Med-High, High

TOTAL_STATES = N_TIME_STATES * N_ZONE_STATES * N_WORKLOAD_STATES  # = 3500

# Action space
N_ACTIONS = 16          # Top 16 shippers by historical volume

# Q-learning hyperparameters
ALPHA = 0.1             # Learning rate (0.05-0.2 typical)
GAMMA = 0.95            # Discount factor (0.9-0.99 for long-term planning)
EPSILON_START = 1.0     # Initial exploration rate
EPSILON_MIN = 0.05      # Minimum exploration rate
EPSILON_DECAY = 0.9995  # Decay per episode

# Training parameters
N_EPISODES = 50000      # Number of training episodes
EPISODE_LENGTH = 100    # Max orders per episode (or until day ends)


# =============================================================================
# STATE ENCODING
# =============================================================================

class StateEncoder:
    """
    Encodes continuous state variables into discrete state index [0, 3499]
    
    State = (time_state, zone_state, workload_state)
    State_index = time_state * (14 * 5) + zone_state * 5 + workload_state
    """
    
    def __init__(self):
        # Time bin to time state mapping (96 bins → 50 states)
        # Strategy: Merge adjacent bins, keep lunch peak fine-grained
        self.time_bin_to_state = self._build_time_mapping()
        
        # District hotspots (from your config)
        self.districts = [
            "quan_1", "quan_2", "quan_3", "quan_4", "quan_8", "quan_9",
            "quan_10", "tan_binh", "tan_phu", "binh_thanh", "go_vap",
            "phu_nhuan", "thu_duc", "other"
        ]
        self.district_to_idx = {d: i for i, d in enumerate(self.districts)}
    
    def _build_time_mapping(self) -> np.ndarray:
        """
        Map 96 raw time bins to 50 time states
        
        Strategy:
        - Morning (bins 0-20): 1 bin = 1 state → 21 states
        - Lunch peak (bins 21-65): keep fine-grained → 45 states  
        - Afternoon-evening (bins 66-95): merge 2 bins → 15 states
        
        Total: 21 + 24 + 5 = 50 states
        """
        mapping = np.zeros(96, dtype=int)
        
        # Morning: bins 0-20 → states 0-20 (1:1)
        mapping[0:21] = np.arange(21)
        
        # Lunch peak: bins 21-44 → states 21-44 (1:1 - critical period)
        mapping[21:45] = np.arange(21, 45)
        
        # Late afternoon: bins 45-95 → states 45-49 (merge 10-11 bins each)
        chunk_size = (96 - 45) // 5
        for i in range(5):
            start_bin = 45 + i * chunk_size
            end_bin = 45 + (i + 1) * chunk_size if i < 4 else 96
            mapping[start_bin:end_bin] = 45 + i
        
        return mapping
    
    def encode_time(self, time_bin: int) -> int:
        """
        Convert raw time bin [0-95] to time state [0-49]
        
        Args:
            time_bin: Raw 10-minute bin (0 = 7:00, 95 = 22:50)
        
        Returns:
            time_state: Aggregated time state [0-49]
        """
        if time_bin < 0 or time_bin >= 96:
            time_bin = np.clip(time_bin, 0, 95)
        return int(self.time_bin_to_state[time_bin])
    
    def encode_zone(self, pickup_lat: float, pickup_lng: float,
                    dropoff_lat: float, dropoff_lng: float) -> int:
        """
        Encode zone based on pickup location
        
        Args:
            pickup_lat, pickup_lng: Pickup coordinates
            dropoff_lat, dropoff_lng: Dropoff coordinates (for future use)
        
        Returns:
            zone_state: District index [0-13]
        """
        # Find nearest district hotspot to pickup location
        # (You already have DISTRICT_HOTSPOTS in your config)
        
        # Simple approach: quantize lat/lng to district
        # For robustness, use your existing hotspot proximity logic
        
        # Placeholder: map to district based on lat/lng range
        # Replace with your actual district detection logic
        if 10.77 <= pickup_lat <= 10.78 and 106.69 <= pickup_lng <= 106.71:
            return self.district_to_idx["quan_1"]
        elif 10.78 <= pickup_lat <= 10.79 and 106.74 <= pickup_lng <= 106.76:
            return self.district_to_idx["quan_2"]
        # ... (add all 13 districts)
        else:
            return self.district_to_idx["other"]  # Fallback
    
    def encode_workload(self, num_available_shippers: int, 
                       total_shippers: int) -> int:
        """
        Encode system workload level
        
        Args:
            num_available_shippers: Currently idle shippers
            total_shippers: Total fleet size
        
        Returns:
            workload_state: [0-4] from low to high congestion
        """
        utilization = 1.0 - (num_available_shippers / max(total_shippers, 1))
        
        # 5 levels: [0-20%), [20-40%), [40-60%), [60-80%), [80-100%]
        if utilization < 0.2:
            return 0  # Low workload
        elif utilization < 0.4:
            return 1  # Medium-low
        elif utilization < 0.6:
            return 2  # Medium
        elif utilization < 0.8:
            return 3  # Medium-high
        else:
            return 4  # High workload
    
    def encode_state(self, time_bin: int, pickup_lat: float, pickup_lng: float,
                    dropoff_lat: float, dropoff_lng: float,
                    num_available: int, total_shippers: int) -> int:
        """
        Convert continuous state to discrete state index [0, 3499]
        
        Returns:
            state_idx: Integer in range [0, 3499]
        """
        t = self.encode_time(time_bin)
        z = self.encode_zone(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
        w = self.encode_workload(num_available, total_shippers)
        
        # Linear index: state = t * (14 * 5) + z * 5 + w
        state_idx = t * (N_ZONE_STATES * N_WORKLOAD_STATES) + z * N_WORKLOAD_STATES + w
        
        return int(state_idx)


# =============================================================================
# ACTION ENCODING
# =============================================================================

class ActionEncoder:
    """
    Maps discrete action [0-15] to actual shipper assignment
    
    Two strategies:
    1. Fixed mapping: Action i → assign to shipper_id i (if i < 16)
    2. Dynamic mapping: Action i → assign to i-th nearest available shipper
    
    Recommendation: Use dynamic mapping for better generalization
    """
    
    def __init__(self, shippers_list: List[Dict], strategy: str = "dynamic"):
        """
        Args:
            shippers_list: List of shipper dicts (from your simulation)
            strategy: "fixed" or "dynamic"
        """
        self.shippers = shippers_list
        self.strategy = strategy
        
        if strategy == "fixed":
            # Pre-select top 16 shippers by historical volume
            # (You need to track this in your simulation)
            self.top_16_ids = [s["id"] for s in shippers_list[:16]]
    
    def get_valid_actions(self, current_time: float, 
                         order: Dict) -> List[int]:
        """
        Get list of valid actions for current order
        
        Args:
            current_time: Current simulation time
            order: Order dict with pickup/dropoff info
        
        Returns:
            valid_actions: List of action indices [0-15] that are feasible
        """
        available_shippers = [
            s for s in self.shippers 
            if s["available_time"] <= current_time
        ]
        
        if self.strategy == "fixed":
            # Actions correspond to top 16 shippers
            valid_actions = [
                i for i, sid in enumerate(self.top_16_ids)
                if any(s["id"] == sid for s in available_shippers)
            ]
        else:  # dynamic
            # Actions correspond to nearest N available shippers
            # Sort by distance to pickup
            from math import sqrt
            
            def distance(s):
                dx = s["node"][0] - order["pickup_node"][0]
                dy = s["node"][1] - order["pickup_node"][1]
                return sqrt(dx**2 + dy**2)
            
            available_shippers.sort(key=distance)
            
            # First min(16, len(available)) are valid actions
            valid_actions = list(range(min(16, len(available_shippers))))
        
        return valid_actions if valid_actions else [0]  # Always allow at least action 0
    
    def action_to_shipper(self, action: int, current_time: float,
                         order: Dict) -> Dict:
        """
        Convert action index to actual shipper object
        
        Args:
            action: Action index [0-15]
            current_time: Current time
            order: Order being assigned
        
        Returns:
            shipper: Shipper dict to assign this order to
        """
        available_shippers = [
            s for s in self.shippers 
            if s["available_time"] <= current_time
        ]
        
        if not available_shippers:
            return None  # No valid assignment
        
        if self.strategy == "fixed":
            target_id = self.top_16_ids[action]
            shipper = next((s for s in available_shippers if s["id"] == target_id), None)
            return shipper if shipper else available_shippers[0]  # Fallback
        
        else:  # dynamic
            # Sort by distance, return action-th nearest
            from math import sqrt
            
            def distance(s):
                dx = s["node"][0] - order["pickup_node"][0]
                dy = s["node"][1] - order["pickup_node"][1]
                return sqrt(dx**2 + dy**2)
            
            available_shippers.sort(key=distance)
            
            if action < len(available_shippers):
                return available_shippers[action]
            else:
                return available_shippers[0]  # Fallback to nearest


# =============================================================================
# Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """
    Tabular Q-learning for online order-shipper matching
    
    Q-table: shape (3500, 16)
    Updates: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(self, state_encoder: StateEncoder, 
                 action_encoder: ActionEncoder,
                 alpha: float = ALPHA,
                 gamma: float = GAMMA,
                 epsilon_start: float = EPSILON_START,
                 epsilon_min: float = EPSILON_MIN,
                 epsilon_decay: float = EPSILON_DECAY):
        
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        
        # Q-table: initialize to zeros (optimistic init alternative: small negative)
        self.Q = np.zeros((TOTAL_STATES, N_ACTIONS))
        
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Training statistics
        self.episode_count = 0
        self.total_updates = 0
        
    def get_state(self, time_bin: int, order: Dict, 
                 num_available: int, total_shippers: int) -> int:
        """
        Encode current situation to state index
        """
        return self.state_encoder.encode_state(
            time_bin,
            order["pickup_node"][0],  # lat
            order["pickup_node"][1],  # lng
            order["dropoff_node"][0],
            order["dropoff_node"][1],
            num_available,
            total_shippers
        )
    
    def select_action(self, state: int, valid_actions: List[int],
                     training: bool = True) -> int:
        """
        Epsilon-greedy action selection
        
        Args:
            state: Current state index
            valid_actions: List of feasible actions
            training: If False, always exploit (greedy)
        
        Returns:
            action: Selected action index
        """
        if not valid_actions:
            return 0
        
        # Exploration: random action
        if training and np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Exploitation: best Q-value among valid actions
        q_values = self.Q[state, valid_actions]
        best_idx = np.argmax(q_values)
        return valid_actions[best_idx]
    
    def update(self, state: int, action: int, reward: float, 
              next_state: int, next_valid_actions: List[int],
              done: bool):
        """
        Q-learning update rule
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward received
            next_state: Resulting state
            next_valid_actions: Valid actions in next_state
            done: True if episode ended
        """
        current_q = self.Q[state, action]
        
        if done or not next_valid_actions:
            # Terminal state: no future reward
            target = reward
        else:
            # Bootstrap from best next action
            next_q_values = self.Q[next_state, next_valid_actions]
            max_next_q = np.max(next_q_values)
            target = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.Q[state, action] += self.alpha * (target - current_q)
        self.total_updates += 1
    
    def decay_epsilon(self):
        """
        Decay exploration rate after each episode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
    
    def save(self, filepath: str):
        """Save Q-table and training state"""
        state = {
            'Q': self.Q,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'total_updates': self.total_updates
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table and training state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.Q = state['Q']
        self.epsilon = state['epsilon']
        self.episode_count = state['episode_count']
        self.total_updates = state['total_updates']
        print(f"Q-table loaded from {filepath}")
        print(f"  Episodes trained: {self.episode_count}")
        print(f"  Total updates: {self.total_updates}")
        print(f"  Current epsilon: {self.epsilon:.4f}")


# =============================================================================
# REWARD FUNCTION
# =============================================================================

def compute_reward(assignment_cost: float, lateness: float, 
                  lambda_penalty: float = 1.0) -> float:
    """
    Reward function for Q-learning
    
    Reward = -(cost + λ * lateness)
    
    Lower cost + lower lateness → higher (less negative) reward
    
    Args:
        assignment_cost: Total cost of this assignment (distance-based)
        lateness: Max(0, delivery_time - expected_delivery_time) in minutes
        lambda_penalty: Weight for lateness penalty (tune this!)
    
    Returns:
        reward: Negative cost (higher is better)
    
    IMPORTANT: Normalize costs to similar scale
    - Typical assignment_cost: 5-20 km → normalize by dividing by 10
    - Typical lateness: 0-60 minutes → normalize by dividing by 30
    """
    normalized_cost = assignment_cost / 10.0
    normalized_lateness = lateness / 30.0
    
    reward = -(normalized_cost + lambda_penalty * normalized_lateness)
    
    return reward


# =============================================================================
# USAGE EXAMPLE (PSEUDOCODE)
# =============================================================================

def example_training_loop():
    """
    PSEUDOCODE: How to integrate Q-learning with your simulation
    
    This is NOT runnable code - it shows the structure
    You need to adapt this to your existing simulation loop
    """
    
    # 1. Initialize components
    state_encoder = StateEncoder()
    action_encoder = ActionEncoder(shippers_list, strategy="dynamic")
    agent = QLearningAgent(state_encoder, action_encoder)
    
    # 2. Training loop
    for episode in range(N_EPISODES):
        
        # Reset simulation for new episode
        # orders = generate_synthetic_orders(scenario="nominal")
        # reset_shippers()
        # current_time = START_HOUR
        
        episode_reward = 0
        
        # Process each order in this episode
        for order in orders:
            
            # Get current state
            num_available = count_available_shippers(current_time)
            time_bin = time_to_bin(current_time)
            state = agent.get_state(time_bin, order, num_available, len(shippers))
            
            # Get valid actions
            valid_actions = action_encoder.get_valid_actions(current_time, order)
            
            # Select action (epsilon-greedy)
            action = agent.select_action(state, valid_actions, training=True)
            
            # Execute action: assign order to selected shipper
            shipper = action_encoder.action_to_shipper(action, current_time, order)
            
            # Compute immediate reward
            assignment_cost = compute_assignment_cost(shipper, order)
            lateness = compute_lateness(shipper, order)
            reward = compute_reward(assignment_cost, lateness)
            
            episode_reward += reward
            
            # Update shipper state
            # update_shipper_state(shipper, order)
            # current_time += some_delta
            
            # Get next state (after assignment)
            num_available_next = count_available_shippers(current_time)
            next_state = agent.get_state(time_to_bin(current_time), order,
                                        num_available_next, len(shippers))
            
            # Q-learning update
            next_valid_actions = action_encoder.get_valid_actions(current_time, order)
            done = (order == orders[-1])  # Last order in episode
            
            agent.update(state, action, reward, next_state, next_valid_actions, done)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Log progress
        if episode % 1000 == 0:
            print(f"Episode {episode}/{N_EPISODES}, "
                  f"Avg Reward: {episode_reward/len(orders):.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
    
    # 3. Save trained Q-table
    agent.save("q_table_trained.pkl")
    
    print("Training complete!")


# =============================================================================
# EVALUATION: Compare Q-learning vs Greedy vs BALANCE
# =============================================================================

def evaluate_policy(policy_name: str, agent=None):
    """
    Evaluate a single policy over multiple replications
    
    Args:
        policy_name: "qlearning", "greedy", or "balance"
        agent: QLearningAgent (only needed for qlearning)
    
    Returns:
        metrics: Dict with avg distance, lateness, service rate, etc.
    """
    
    results = {
        "total_distance": [],
        "avg_lateness": [],
        "service_rate": [],
        "num_served": []
    }
    
    n_replications = 100
    scenarios = ["moderate", "nominal", "stress"]
    
    for scenario in scenarios:
        for rep in range(n_replications):
            
            # Generate orders for this replication
            # orders = generate_synthetic_orders(scenario, seed=rep)
            # reset_shippers()
            
            total_dist = 0
            total_late = 0
            num_served = 0
            
            for order in orders:
                
                if policy_name == "qlearning":
                    # Use trained Q-table (NO exploration)
                    state = agent.get_state(...)
                    valid_actions = action_encoder.get_valid_actions(...)
                    action = agent.select_action(state, valid_actions, training=False)
                    shipper = action_encoder.action_to_shipper(action, ...)
                
                elif policy_name == "greedy":
                    # Use your existing greedy_dispatch() function
                    shipper = greedy_dispatch(order, current_time)
                
                elif policy_name == "balance":
                    # Use your existing choose_shipper() function
                    shipper = choose_shipper(order, current_time)
                
                # Track metrics
                if shipper is not None:
                    num_served += 1
                    total_dist += compute_distance(shipper, order)
                    total_late += compute_lateness(shipper, order)
            
            # Store replication results
            results["total_distance"].append(total_dist)
            results["avg_lateness"].append(total_late / max(num_served, 1))
            results["service_rate"].append(num_served / len(orders))
            results["num_served"].append(num_served)
    
    # Aggregate statistics
    metrics = {
        "mean_distance": np.mean(results["total_distance"]),
        "std_distance": np.std(results["total_distance"]),
        "mean_lateness": np.mean(results["avg_lateness"]),
        "mean_service_rate": np.mean(results["service_rate"])
    }
    
    return metrics


# =============================================================================
# FINAL NOTES FOR STUDENT
# =============================================================================

"""
INTEGRATION CHECKLIST:
----------------------

1. ✅ Add StateEncoder to your code
   - Adjust encode_zone() to use your actual district detection logic
   - Test: Print state indices for a few sample orders

2. ✅ Add ActionEncoder 
   - Choose "dynamic" strategy (more robust)
   - Test: Verify valid_actions list is correct

3. ✅ Add QLearningAgent
   - Start with default hyperparameters
   - You can tune α, γ, ε later

4. ✅ Modify your simulation loop
   - Add Q-learning as 3rd policy option
   - Call agent.update() after each assignment
   - Save Q-table every 5000 episodes

5. ✅ Training phase
   - Run 50,000 episodes (takes ~5 minutes)
   - Monitor: epsilon decay, average reward trend
   - Save final Q-table

6. ✅ Evaluation phase
   - Load trained Q-table
   - Run 100 reps × 3 scenarios × 3 policies
   - Collect metrics for Chapter 5

COMMON BUGS TO AVOID:
---------------------

❌ Forgetting to turn off exploration during evaluation
   → Always use training=False when testing

❌ Not clipping state/action indices
   → Add bounds checks: state ∈ [0, 3499], action ∈ [0, 15]

❌ Reward scale mismatch
   → Normalize costs to similar magnitude (divide by 10-30)

❌ Invalid actions not filtered
   → Always pass valid_actions list to select_action()

❌ Q-table not saved
   → Save every 5000 episodes + at end of training

HYPERPARAMETER TUNING TIPS:
----------------------------

If Q-learning performs poorly:

1. Check state visitation frequency
   → print(np.sum(Q != 0)) should be > 50% of Q-table

2. Try different α (learning rate)
   → Start 0.1, try 0.05 (slower) or 0.2 (faster)

3. Try different λ (lateness penalty)
   → Start 1.0, try 0.5 (less penalty) or 2.0 (more penalty)

4. Extend training
   → 50K episodes might not be enough, try 100K

5. Adjust epsilon decay
   → Slower decay = more exploration = better coverage

EXPECTED RESULTS:
-----------------

After 50K episodes:
- Q-table should have ~70-80% non-zero entries
- Epsilon should decay to ~0.05-0.10
- Average reward should show upward trend (less negative)

Evaluation (vs Greedy baseline):
- Q-learning should reduce total distance by 5-15%
- Lateness should be similar or slightly better
- Service rate should be equal (all policies serve same orders)

If Q-learning is WORSE than Greedy:
- Check reward function (might be mis-scaled)
- Check state encoding (states might be too sparse)
- Increase training episodes to 100K

"""
