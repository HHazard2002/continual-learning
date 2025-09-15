# Continual Learning with Surprise Replay & Dual Learner

This repository implements methods for reducing catastrophic forgetting in large language models (LLMs) via *surprise-based replay* and a *dual learner architecture*.  

---

## What’s in This Code

- **Surprise Replay**: a replay buffer that retains the *most surprising* sequences rather than using uniform sampling.  
- **Dual Learner Architecture**: fast & slow LoRA heads, where the slow learner consolidates knowledge from the fast learner via exponential moving average (EMA).  
- Variants / ablations:  
  1. Surprise computed on full sequences vs. just labels  
  2. Surprise & buffer update before vs. after training on each dataset  
  3. Aging mechanism: updating surprise when samples are replayed  
- Benchmarks: experiments on the *Standard CL Benchmark* and *Large Number of Tasks* settings.  

---

## ⚙️ Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/HHazard2002/continual-learning.git
cd continual-learning

# Python environment (example)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# continual-learning
