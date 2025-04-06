# marl_script.py
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from pettingzoo.mpe import simple_tag_v3
import os

# Function to register the PettingZoo environment with RLlib
def env_creator(config):
    # simple_tag_v3.env() creates the environment instance
    # PettingZooEnv wraps it for RLlib compatibility
    return PettingZooEnv(simple_tag_v3.env(render_mode=None, continuous_actions=False))

# Function to map agents to policies
# In simple_tag: 'adversary_0' is the predator, 'agent_0', 'agent_1' are prey
# We'll use two policies: one for adversaries, one for agents (prey)
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id.startswith("adversary"):
        return "adversary_policy"
    else:
        return "agent_policy"

if __name__ == "__main__":
    # --- Ray Initialization ---
    # 'auto' connects to the Ray cluster managed by KubeRay when run via 'ray job submit'
    # If running locally for testing, it starts a local Ray instance.
    # Set dashboard_host to allow access from outside the container/pod if needed
    ray.init(
        address='auto',
        # _temp_dir="/tmp/ray_marl_script" # Optional: Specify temp dir if needed
    )

    print("Ray initialized.")
    print("Nodes in cluster:", ray.nodes())

    # --- Environment Setup ---
    env_name = "simple_tag_v3_marl"
    tune.register_env(env_name, env_creator)

    # Get dummy env instance to extract observation and action spaces
    # Needed for defining policies
    temp_env = env_creator({})
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close() # Close the dummy env

    # --- Algorithm Configuration ---
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch") # Or "tf"
        # Configure for multi-agent setup
        .multi_agent(
            # Define the policies (key is the policy ID)
            policies={
                "adversary_policy": (None, obs_space, act_space, {}), # Use default PPO policy class
                "agent_policy": (None, obs_space, act_space, {}),     # Use default PPO policy class
            },
            # Function mapping agent IDs to policy IDs
            policy_mapping_fn=policy_mapping_fn,
            # List policies that should be trained
            policies_to_train=["adversary_policy", "agent_policy"],
        )
        # Parallelism: Distribute rollout collection across workers
        # Set num_workers >= 1 for distributed execution
        # This should correspond to the number of Ray worker pods you intend to use
        .rollouts(num_rollout_workers=2) # Adjust based on your cluster size
        # Add minimal resource requirements if needed, often handled by KubeRay pod specs
        # .resources(num_gpus=0) # Example: specify 0 GPUs explicitly
    )

    # --- Training ---
    algo = config.build()
    print("Algorithm built. Starting training...")

    # Run N training iterations
    results = None
    for i in range(5): # Keep this small for a minimal example
        results = algo.train()
        print(f"Iteration: {i+1}, Mean Episode Reward: {results['episode_reward_mean']}")

        # You can access metrics for individual policies like this:
        # print(f"  Adversary Policy Reward: {results['policy_reward_mean']['adversary_policy']}")
        # print(f"  Agent Policy Reward: {results['policy_reward_mean']['agent_policy']}")

    print("Training finished.")
    if results:
        print("Final Result Snippet:", results)

    # --- Cleanup ---
    algo.stop()
    ray.shutdown()
    print("Ray shutdown.")