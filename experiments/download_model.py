import os
from huggingface_hub import snapshot_download

# 目标路径: NuSy-Edge/models/qwen3-0.6b
# 确保你是在 NuSy-Edge 根目录下运行，或者根据需要调整 target_dir
# target_dir = os.path.join(os.getcwd(), "models", "qwen3-0.6b")
target_dir = os.path.join(os.getcwd(), "models", "qwen3-1.7b")

print(f"Downloading Qwen/Qwen3-1.7B to {target_dir} ...")

snapshot_download(
    repo_id="Qwen/Qwen3-1.7B",
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # 确保是实体文件
    ignore_patterns=["*.msgpack", "*.h5", ".gitattributes"], # 过滤无关文件
    token=None # 如果是公开模型不需要 token，如果是受限模型请填写
)

print("✅ Download complete.")