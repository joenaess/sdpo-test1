# SDPO 12GB Proof-of-Concept: Experiment Results

## Objective
The primary objective of this experiment was to validate whether the `verl` framework's advanced multi-node hybrid RLHF architecture could be aggressively condensed to fit within the VRAM limitations of a single, consumer-grade 12GB GPU.

## Performance Analysis & Results
The pipeline successfully completed 25 iterations of the full rollout generation and SDPO (Self-Distillation Policy Optimization) PPO update loops without triggering PyTorch CUDA Out-Of-Memory exceptions or Ray System RAM limit kills.

However, **the model's reasoning performance did not actually increase.** 

We can confirm this by examining the terminal metrics appended to the `poc_training.log` file at the final iteration (Step 25):

```text
step:25 
- actor/pg_loss: 0.0 
- actor/kl_loss: 0.0 
- actor/grad_norm: 0.0
- critic/rewards/mean: 0.0 
- critic/advantages/mean: 0.0 
```

### Why did the gradients flatline?
The recorded metrics above reveal that the environment was providing absolute zero rewards for all generated answers (`critic/rewards/mean: 0.0`). 

In the execution shell script (`run_12gb_poc_sdpo.sh`), the execution specifies `algorithm.adv_estimator=grpo` and inherently spins up a default `naive` reward loop process. However, we did not wire a domain-specific mathematical reward function into the pipeline (such as a python function that parses the model's output for `<answer>XX</answer>` and logically compares it against the GSM8K dataset's literal ground truths).

Because the reward manager could not yield any relative success/failure signals, the Generalized Advantage Estimation (GAE/GRPO) calculated every generated response's "advantage" over the baseline as precisely `0.0`. 

Since Policy Gradient Loss (`pg_loss`) is the mathematical product of the advantage scalar and the log-probabilities of the generated tokens, the loss ultimately computed to exactly `0.0`, resulting in a `grad_norm: 0.0`. When gradients are zero, the backpropagation step naturally skips applying any substantive modifications to the LoRA weights.

## Next Steps for Iteration
To transition this PoC from "architecturally stable" to "actively training," the following integration is required:
1. **Implement a Custom Reward Function**: A reward script must be added to parse the generated inference output, extract the mathematical answer, evaluate it against the dataset, and return `1.0` (Correct), `-1.0` (Incorrect), or partial parsing rewards. 
2. **Wire the Reward Manager**: The script parameters (`reward_model.reward_manager.reward_fn`) must point to this function within the `verl` instantiation to provide the missing mathematical reward signal.
