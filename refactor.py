import os
import shutil

# 定义目录结构
structure = {
    "alphazero_gomoku": [
        "game.py",
        "mcts_alphaZero.py",
        "mcts_pure.py",
        "policy_value_net.py",
        "policy_value_net_keras.py",
        "policy_value_net_numpy.py",
        "policy_value_net_pytorch.py",
        "policy_value_net_tensorflow.py",
    ],
    "models": [
        "best_policy_6_6_4.model",
        "best_policy_6_6_4.model2",
        "best_policy_8_8_5.model",
        "best_policy_8_8_5.model2",
        "current_policy.model",
        "best_policy.model",
    ],
    "assets": ["playout400.gif"],
}

# 创建目录并移动文件
for folder, files in structure.items():
    if not os.path.exists(folder):
        os.makedirs(folder)

    for file in files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(folder, file))
            print(f"Moved {file} to {folder}")
        else:
            print(f"File {file} not found (skipping)")

# 创建 __init__.py
init_file = "alphazero_gomoku/__init__.py"
if not os.path.exists(init_file):
    open(init_file, "a").close()
    print(f"Created {init_file}")

print("Project restructuring complete.")
