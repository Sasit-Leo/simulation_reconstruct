# Video → 3D Reconstruction Pipelines

从视频自动重建 3D 场景或物体网格，输出 Z-up 对齐的 USDZ，直接导入 Isaac Sim / Isaac Lab。

## 环境配置

### 硬件

| 组件  | 要求                                           |
| --- | -------------------------------------------- |
| GPU | NVIDIA RTX 系列（4090 / 24 GB）                  |
| 显存  | ≥ 16 GB（4K 全分辨率训练需 24 GB）                    |
| 内存  | ≥ 32 GB                                      |
| 磁盘  | 单次重建 50-200 GB（含中间文件）                        |
| 系统  | Ubuntu 22.04, CUDA 12.4, NVIDIA Driver ≥ 550 |

### Conda 环境

```bash
# 创建环境
conda create -n vid2sim python=3.11 -y
conda activate vid2sim

# PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 核心依赖
pip install gsplat plyfile open3d trimesh opencv-python scikit-learn

# COLMAP (系统级安装, 3.13.0+)
# Ubuntu: sudo apt install colmap
# 或 conda: conda install -c conda-forge colmap

# FFmpeg
# Ubuntu: sudo apt install ffmpeg
```

### 手动编译包

以下包无法通过 pip 直接安装，需从源码编译。编译时注意设置 `TORCH_CUDA_ARCH_LIST` 匹配 GPU 架构（RTX 4090 = `8.9`）。

```bash
conda activate vid2sim
export TORCH_CUDA_ARCH_LIST=8.9

# fused-ssim — SSIM 损失 CUDA 加速 (3DGUT 依赖)
pip install --no-build-isolation \
    "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim@1272e21a282342e89537159e4bad508b19b34157"

# diff-surfel-rasterization — 2DGS 微分面元渲染器
cd simulation_reconstruct/2dgs && git submodule update --init
pip install --no-build-isolation submodules/diff-surfel-rasterization

# simple-knn — 2DGS KNN 初始化
pip install --no-build-isolation submodules/simple-knn
```

### 验证

```bash
conda activate vid2sim
python -c "
import torch; print(f'Torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
import gsplat; print(f'gsplat {gsplat.__version__}')
import open3d; print(f'open3d {open3d.__version__}')
import trimesh; print(f'trimesh {trimesh.__version__}')
from diff_surfel_rasterization import GaussianRasterizer; print('diff-surfel OK')
from fused_ssim import fused_ssim; print('fused-ssim OK')
from simple_knn._C import distCUDA2; print('simple-knn OK')
"
ffmpeg -version | head -1
colmap --version | head -1
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
./video_to_scene.sh -v video.mp4 -f 5           # 降 FPS（困难场景）
```

| 参数   | 说明             | 默认                    |
| ---- | -------------- | --------------------- |
| `-v` | 视频文件名或路径       | 优先 `videos/`          |
| `-o` | 输出目录           | `results/{视频名}_scene` |
| `-f` | 抽帧 FPS         | `10`                  |
| `-i` | 训练迭代数          | `45000`               |
| `-d` | 训练下采样 (`1`=4K) | `1`                   |
| `-g` | GPU ID         | `0`                   |
| `-u` | 跳过 USDZ 导出     | 否                     |

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

| 参数   | 说明             | 默认                   |
| ---- | -------------- | -------------------- |
| `-v` | 视频文件名或路径       | 优先 `videos/`         |
| `-o` | 输出目录           | `results/{视频名}_mesh` |
| `-f` | 抽帧 FPS         | `10`                 |
| `-i` | 训练迭代数          | `30000`              |
| `-d` | 训练下采样 (`1`=4K) | `1`                  |
| `-b` | 背景剔除系数         | `1.5`                |
| `-V` | 交互筛选点云         | 否                    |
| `-g` | GPU ID         | `0`                  |

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

## 参考

- [3DGRUT](https://github.com/nv-tlabs/3dgrut) — 3D Gaussian Ray Tracing & Unscented Transform
- [2D Gaussian Splatting](https://surfsplatting.github.io/) — Geometrically Accurate Radiance Fields
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian Splatting 库
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion
