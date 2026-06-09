# Video → 3D Reconstruction Pipelines

从视频自动重建 3D 场景或物体网格，输出 Z-up 对齐的 USDZ，直接导入 Isaac Sim / Isaac Lab。

## 环境配置

### 硬件

| 组件  | 要求                                           |
| --- | -------------------------------------------- |
| GPU | NVIDIA RTX 系列（4090 / 24 GB）                  |
| 显存  | ≥ 16 GB（`-d 1` 全分辨率需 >24 GB，默认 `-d 2`） |
| 内存  | ≥ 32 GB                                      |
| 磁盘  | 单次重建 50-200 GB（含中间文件）                    |
| 系统  | Ubuntu 22.04, CUDA 12.4, NVIDIA Driver ≥ 550 |

### 快速安装

```bash
# 从备份文件恢复环境（推荐）
conda env create -f vid2sim_env.yml
conda activate vid2sim

# 设置 GPU 架构
conda env config vars set -n vid2sim TORCH_CUDA_ARCH_LIST=8.9
conda activate vid2sim
```

### 手动安装（从零构建）

```bash
# 1. 基础环境
conda create -n vid2sim python=3.11 -y
conda activate vid2sim

# 2. PyTorch CUDA 12.4（3DGUT 官方测试版本）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. 核心依赖
pip install gsplat plyfile open3d trimesh opencv-python scikit-learn

# 4. CUDA 编译的 COLMAP 3.13.0（GPU Bundle Adjustment）
# 需从源码编译 Ceres + COLMAP，参见下方 [编译 COLMAP with CUDA]
conda install -c conda-forge colmap  # 或使用系统包管理器安装基础版

# 5. 可选依赖
pip install --no-build-isolation \
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21"
pip install --no-build-isolation \
    "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.1"

# 6. GPU 架构
conda env config vars set -n vid2sim TORCH_CUDA_ARCH_LIST=8.9

# 7. FFmpeg（系统级）
# Ubuntu: sudo apt install ffmpeg
```

### 编译 COLMAP with CUDA

conda-forge 的 COLMAP 不带 GPU Bundle Adjustment。需从源码编译：

```bash
conda activate vid2sim

# 编译 Ceres Solver 2.2.0 with CUDA
git clone --depth 1 --branch 2.2.0 https://github.com/ceres-solver/ceres-solver /tmp/ceres
cd /tmp/ceres
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
cmake --build build -j$(nproc) && cmake --install build

# 编译 COLMAP 3.13.0 with CUDA
git clone --depth 1 --branch 3.13.0 https://github.com/colmap/colmap /tmp/colmap
cd /tmp/colmap
# 安装缺失依赖
conda install -c conda-forge -y boost-cpp libboost-devel cgal mesa-libgl-devel-cos7-x86_64
ln -sf $CONDA_PREFIX/lib/libGLX.so.0 $CONDA_PREFIX/lib/libGLX.so
ln -sf $CONDA_PREFIX/lib/libGL.so.1 $CONDA_PREFIX/lib/libGL.so
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCeres_DIR=$CONDA_PREFIX/lib/cmake/Ceres \
    -DBoost_DIR=$CONDA_PREFIX/lib/cmake/Boost-1.84.0
cmake --build build -j$(nproc) && cmake --install build
```

### 3DGUT 内核补丁

3DGUT 有两个边界条件 bug 需要手动修复：

**`threedgut_tracer/src/gutRenderer.cu`**（`numParticles=0` 时非法内存访问）：
```cpp
// 在 "fetch total number of particle/tile intersections" 之前添加：
if (numParticles == 0) {
    return Status();
}
```

**`threedgrut/strategy/gs.py`**（空梯度 / 字典访问错误）：
```python
# update_gradient_buffer 方法中：
params_grad = self.model.positions.grad
assert params_grad is not None  # 移到 mask 之前
if params_grad.numel() == 0: return
mask = (params_grad != 0).max(dim=1)[0]
if not mask.any(): return

# prune_gaussians_scale 方法中：
intr = list(dataset.intrinsics.values())[0]  # 曾是 dataset.intrinsic[0]
max_focal = torch.as_tensor(intr[0]['focal_length']).float().max()
```

### 验证

```bash
conda activate vid2sim
python -c "
import torch; print(f'Torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
import gsplat; print(f'gsplat {gsplat.__version__}')
import ppisp; print(f'ppisp {ppisp.__version__}')
import fused_ssim; print('fused-ssim OK')
"
colmap help 2>&1 | grep CUDA  # 应显示 "(Commit ... with CUDA)"
```

### 目录结构

```
simulation_reconstruct/
├── videos/                   # 输入视频（自动查找）
├── results/                  # 重建输出（自动生成）
├── 3dgrut/                   # 3DGUT 源码
├── 2dgs/                     # 2DGS 源码
├── video_to_scene.sh         # 场景重建入口
├── video_to_mesh.sh          # 物体网格入口
└── README.md
```

## 两个 Pipeline

|     | `video_to_scene.sh`  | `video_to_mesh.sh`        |
| --- | -------------------- | ------------------------- |
| 目标  | 场景级 3D Gaussian 重建   | 物体级 3D 网格重建               |
| 方法  | 3DGUT (GS, SH=4)     | 2DGS (SH=4) + TSDF + 几何优化 |
| 环境  | 保留完整场景               | DBSCAN 自动剔除 / 手动交互筛选      |
| 对齐  | RANSAC 地板+天花板 → Z-up | RANSAC top-k 倾角最小 → Z-up  |
| 地面  | 自动碰撞体                | 底面自动封闭                    |
| 输出  | USDZ + 碰撞地面          | USDA + PLY (含碰撞 API)      |
| 体素  | —                    | 4mm TSDF                  |

**环境**：`conda activate vid2sim`

## 共享流程

两个脚本共用阶段 1-2，阶段 3-4 各自不同：

```
FFmpeg (4K原图) → CLAHE增强 → COLMAP SfM → [3DGUT | 2DGS] → 清理 + 旋转 → USDZ
```

- 视频放 `videos/` 目录，脚本自动查找
- 输出在 `results/` 目录下
- `-c` 跳过 FFmpeg，`-S` 跳过 COLMAP，`-T` 跳过训练

## video_to_scene.sh — 场景重建

```bash
./video_to_scene.sh -v video.mp4               # 全流程
./video_to_scene.sh -v video.mp4 -c -S          # 仅重训练
./video_to_scene.sh -v video.mp4 -f 10          # 高帧率（精细重建）
```

| 参数   | 说明                     | 默认                    |
| ---- | ---------------------- | --------------------- |
| `-v` | 视频文件名或路径               | 优先 `videos/`          |
| `-o` | 输出目录                   | `results/{视频名}_scene` |
| `-f` | 抽帧 FPS                 | `5`                   |
| `-i` | 训练迭代数                  | `60000`               |
| `-d` | 训练下采样 (`1`=4K, `2`=2K) | `2` (24GB 推荐)         |
| `-g` | GPU ID                 | `0`                   |
| `-u` | 跳过 USDZ 导出             | 否                     |

输出结构：

```
results/<video>_scene/runs/<experiment>/<experiment>-MMDD_HHMMSS/
├── scene_nurec.usdz            # ★ 视觉场景 (Z-up 对齐)
├── ground_collision.usda       # ★ 地面碰撞体 (同坐标系)
├── rotation.json               # 旋转变换记录
├── ckpt_last.pt                # 模型 checkpoint (续训用)
├── ours_*/                     # 各阶段 checkpoint (每 5000 步)
└── metrics.json                # PSNR/SSIM/LPIPS
```

**Isaac Lab 导入**：

```python
from isaaclab.sim.spawners.from_files import UsdFileCfg

visual = UsdFileCfg(usd_path=".../scene_nurec.usdz")       # 视觉场景
ground = UsdFileCfg(usd_path=".../ground_collision.usda")    # 地面碰撞
```

视觉与地面在同一坐标系中，RANSAC 自动检测地板 + 天花板，统一旋转到 Z-up。

## video_to_mesh.sh — 物体网格重建

```bash
./video_to_mesh.sh -v video.mp4               # 全流程
./video_to_mesh.sh -v video.mp4 -c -S          # 仅重训练
```

| 参数   | 说明                     | 默认                   |
| ---- | ---------------------- | -------------------- |
| `-v` | 视频文件名或路径               | 优先 `videos/`         |
| `-o` | 输出目录                   | `results/{视频名}_mesh` |
| `-f` | 抽帧 FPS                 | `5`                  |
| `-i` | 训练迭代数                  | `60000`              |
| `-d` | 训练下采样 (`1`=4K, `2`=2K) | `2` (24GB 推荐)        |
| `-b` | 背景剔除系数                 | `1.5`                |
| `-V` | 交互筛选点云                 | 否                    |
| `-u` | 跳过 USDA 导出             | 否                     |
| `-g` | GPU ID                 | `0`                  |
| `-c` | 跳过 FFmpeg              | 否                     |
| `-S` | 跳过 COLMAP              | 否                     |
| `-T` | 跳过训练                   | 否                     |

输出结构：

```
results/<video>_mesh/runs/mesh_2dgs/<experiment>-MMDD_HHMMSS/
├── mesh_<video>.usda            # ★ 网格 (Z-up, 碰撞已集成, 底面已封闭)
├── mesh_<video>.ply             # PLY 网格
├── mesh_metrics.json            # 网格质量指标
├── ckpt_*.pt                    # 模型 checkpoint (每 5000 步)
├── train.log                    # 训练日志
└── point_cloud/                 # 2DGS 模型
```

**Isaac Lab 导入**：网格自带碰撞几何，可直接 spawn

```python
from isaaclab.sim.spawners.from_files import UsdFileCfg

object = UsdFileCfg(
    usd_path=".../mesh_<video>.usda",
    rigid_props=UsdFileCfg.RigidBodyPropertiesCfg(rigid_body_enabled=True),
    collision_props=UsdFileCfg.CollisionPropertiesCfg(collision_enabled=True),
)
```

RANSAC top-k 倾角最小平面自动对齐 Z 轴，几何优化平滑平面区、保留杆件细节、自动封闭底面。`-V` 可交互框选目标区域，终端摘要显示网格顶点/面数、密闭性、平坦/细节比例。

# 

## 参考

- [3DGRUT](https://github.com/nv-tlabs/3dgrut) — 3D Gaussian Ray Tracing & Unscented Transform
- [2D Gaussian Splatting](https://surfsplatting.github.io/) — Geometrically Accurate Radiance Fields
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian Splatting 库
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion
