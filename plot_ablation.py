import re
import matplotlib.pyplot as plt

def parse_rewards(log_file):
    steps = []
    rewards = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if 'step:' in line and 'critic/rewards/mean:' in line:
                    try:
                        step_match = re.search(r'step:(\d+)', line)
                        if step_match:
                            step = int(step_match.group(1))
                            
                        reward_match = re.search(r'critic/rewards/mean:([-\d\.]+)', line)
                        if reward_match:
                            reward = float(reward_match.group(1))
                        
                        steps.append(step)
                        rewards.append(reward)
                    except Exception:
                        pass
    except FileNotFoundError:
        print(f"Warning: {log_file} not found yet.")
        
    return steps, rewards

if __name__ == '__main__':
    base_steps, base_rewards = parse_rewards('ablation_baseline.log')
    gated_steps, gated_rewards = parse_rewards('ablation_gated.log')

    plt.figure(figsize=(10, 6))
    if base_steps:
        plt.plot(base_steps, base_rewards, 'b-', marker='o', label='Baseline (100% Teacher evaluation)')
    if gated_steps:
        plt.plot(gated_steps, gated_rewards, 'g-', marker='s', label='Gated SDPO (student_probs < 0.90)')
        
    plt.title('SDPO Reward Convergence: Baseline vs Gated Ablation')
    plt.xlabel('PPO Validation Step')
    plt.ylabel('Mean Reward (GSM8k Mathematical Evaluation)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ablation_results.png')
    print("Ablation graph generated gracefully: ablation_results.png")
