import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def profile_sdpo():
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-0.5B-Instruct" 
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    
    # 2. Dummy Data Creation
    batch_size = 2
    seq_len = 512
    vocab_size = model.config.vocab_size
    
    print("Creating dummy data...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device="cuda")
    
    # Dummy student representation (log-probs) - pre-computed to avoid timing overhead
    dummy_student_logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device="cuda")
    dummy_student_logprobs = F.log_softmax(dummy_student_logits, dim=-1)
    
    del dummy_student_logits
    torch.cuda.empty_cache()

    # The theoretical router selection mask (exactly 10% True)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device="cuda")
    num_true = int(seq_len * 0.1)
    for i in range(batch_size):
        indices = torch.randperm(seq_len, device="cuda")[:num_true]
        mask[i, indices] = True

    def run_baseline():
        model.train()
        hidden_states = model.model(input_ids)[0]
        
        # Dense LM Head Computation
        teacher_logits = model.lm_head(hidden_states)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)
        target_logprobs = dummy_student_logprobs
        
        # Calculate full KL divergence
        kl_loss = F.kl_div(teacher_logprobs, target_logprobs, reduction="batchmean", log_target=True)
        kl_loss.backward()
        
        model.zero_grad(set_to_none=True)

    def run_optimized():
        model.train()
        hidden_states = model.model(input_ids)[0]
        
        # Sparse Selection Execution Layer
        selected_hidden_states = hidden_states[mask] 
        
        # LM Head on mathematically pruned tokens only
        selected_teacher_logits = model.lm_head(selected_hidden_states)
        selected_teacher_logprobs = F.log_softmax(selected_teacher_logits, dim=-1)
        
        # Extract corresponding student logprobs
        selected_dummy_student_logprobs = dummy_student_logprobs[mask]
        
        # Calculate gated KL divergence
        kl_loss = F.kl_div(selected_teacher_logprobs, selected_dummy_student_logprobs, reduction="batchmean", log_target=True)
        kl_loss.backward()
        
        model.zero_grad(set_to_none=True)

    def benchmark(func, name, num_iters=10, warmup=3):
        # Warmup Iterations
        for _ in range(warmup):
            func()
            
        torch.cuda.synchronize()
        
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_iters):
            start_events[i].record()
            func()
            end_events[i].record()
            
        torch.cuda.synchronize()
        
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        avg_time = sum(times) / num_iters
        return avg_time

    print("\nBenchmarking Baseline SDPO...")
    baseline_time = benchmark(run_baseline, "Baseline")
    
    print("Benchmarking Optimized Gated SDPO...")
    optimized_time = benchmark(run_optimized, "Optimized")
    
    print("\n==================================")
    print("      SDPO BENCHMARK RESULTS      ")
    print("==================================")
    print(f"Batch Size      : {batch_size}")
    print(f"Sequence Length : {seq_len}")
    print(f"Gating Ratio    : 10%")
    print(f"----------------------------------")
    print(f"Baseline Time   : {baseline_time:.2f} ms")
    print(f"Optimized Time  : {optimized_time:.2f} ms")
    print(f"Speedup Ratio   : {baseline_time / optimized_time:.2f}x")
    print("==================================\n")

if __name__ == "__main__":
    profile_sdpo()
