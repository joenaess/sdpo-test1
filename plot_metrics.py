import re
import matplotlib.pyplot as plt

def parse_log_and_plot(log_file):
    steps = []
    rewards_mean = []
    pg_loss = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'step:' in line and 'critic/rewards/mean:' in line:
                try:
                    # Extract step
                    step_match = re.search(r'step:(\d+)', line)
                    if step_match:
                        step = int(step_match.group(1))
                        if step == 1:
                            steps = []
                            rewards_mean = []
                            pg_loss = []
                        
                    # Extract rewards mean
                    reward_match = re.search(r'critic/rewards/mean:([-\d\.]+)', line)
                    if reward_match:
                        reward = float(reward_match.group(1))
                        
                    # Extract pg_loss
                    loss_match = re.search(r'actor/pg_loss:np\.float64\(([-\d\.]+)\)', line)
                    if loss_match:
                        loss = float(loss_match.group(1))
                    elif 'actor/pg_loss:' in line:
                        loss_match = re.search(r'actor/pg_loss:([-\d\.]+)', line)
                        loss = float(loss_match.group(1))
                        
                    steps.append(step)
                    rewards_mean.append(reward)
                    pg_loss.append(loss)
                except Exception as e:
                    pass

    if not steps:
        print("No metrics found in log file.")
        return

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.plot(steps, rewards_mean, 'b-', marker='o')
    ax1.set_title('Critic Reward Mean over Steps')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Reward')
    ax1.grid(True)
    
    # Plot pg_loss
    ax2.plot(steps, pg_loss, 'r-', marker='o')
    ax2.set_title('Actor Policy Gradient Loss over Steps')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('PG Loss')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Graph saved to training_metrics.png")

if __name__ == '__main__':
    parse_log_and_plot('poc_training.log')
