"""run_titan.py —— TorchTitan 训练入口包装（供 torchrun 分发）。

用法：
    torchrun ... run_titan.py --module parity_titan --config parity_tiny
"""

from torchtitan.train import main

if __name__ == "__main__":
    main()
