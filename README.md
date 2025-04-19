# policy_gradient_comparison
This experiment compares three classic policy gradient variants using PyTorch

- ✅ Vanilla REINFORCE
- ✅ REINFORCE with Reward-to-Go
- ✅ REINFORCE with Baseline (Advantage Estimation)

## 🚀 Results (100 Evaluation Episodes)

| Method         | Mean Reward | Std Dev | Success (≥195) |
|----------------|-------------|---------|----------------|
| Vanilla        | 42.61       | 14.45   | 0/100          |
| Reward-to-Go   | 42.15       | 12.21   | 0/100          |
| Baseline       | 219.76      | 21.15   | 88/100         |

## 📊 Outputs

- `training_rewards_log.xlsx` – average episode reward across training
- `evaluation_results.xlsx` – evaluation stats after training

## 📦 Dependencies

- `gym`
- `torch`
- `matplotlib`
- `pandas`

Install with:
```bash
pip install gym torch matplotlib pandas
