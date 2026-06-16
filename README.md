# Video → 3D Reconstruction Pipelines

从视频自动重建 3D 场景或物体网格，输出 Y-up 对齐的 USDZ，直接导入 Isaac Sim / Isaac Lab。

## 环境配置

### 硬件

| 组件  | 要求                                           |
| --- | -------------------------------------------- |
| GPU | NVIDIA RTX 系列（4090 / 24 GB）                  |
| 显存  | ≥ 16 GB（`-d 1` 全分辨率需 >24 GB，默认 `-d 2`）       |
| 内存  | ≥ 32 GB                                      |
| 磁盘  | 单次重建 50-200 GB（含中间文件）                        |
| 系统  | Ubuntu 22.04, CUDA 12.4, NVIDIA Driver ≥ 550 |

### 环境搭建

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

# 7. FFmpeg
sudo apt install ffmpeg
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

3DGUT 有三个边界条件 bug 需要手动修复（已在本仓库中应用）。

**1. `threedgut_tracer/src/gutRenderer.cu` — forward render: `numParticles=0` 时非法 kernel launch**

在 `renderForward` 函数中，原来的 `numParticles == 0` 检查在 `projectOnTiles` kernel 之后，导致对 0 粒子发起 CUDA kernel 报 `cudaErrorInvalidConfiguration`。修复：将检查移到函数开头（kernel launch 之前）：

```cpp
const uint32_t numParticles = parameters.values.numParticles;
// 在 kernel launch 之前提前返回
if (numParticles == 0) {
    return Status();
}
```

**2. `threedgut_tracer/src/gutRenderer.cu` — backward render: `numParticles=0` 时仅打日志不返回**

`renderBackward` 中原来只 `LOG_ERROR` 后继续执行，改为 `RETURN_ERROR` 真正退出：

```cpp
if (numParticles == 0) {
    RETURN_ERROR(m_logger, ErrorCode::Runtime,
        "[GUTRenderer] number of particles is 0, cannot render backward.");
}
```

**3. `threedgrut/strategy/gs.py` — 防止剪枝至 0 粒子的安全兜底**

`prune_gaussians_scale` 和 `prune_gaussians_opacity` 中新增最小粒子数保护：当 mask 会剔除所有粒子时，强制保留 ratio 最小 / density 最高的 `MIN_PARTICLES=16` 个粒子，避免 CUDA 端崩溃。

> **注意**：`prune_scale` 默认是**禁用**的（`start_iteration: -1`）。若启用，阈值 `threshold` 含义是「投影像素尺寸 ≥ 阈值则剔除」。默认值 1.0 极激进，建议 ≥ 10.0 且 `start_iteration ≥ 3000`，否则早期高斯会被全量剔除（100% → 0 particle → CUDA crash）。

### 配置验证

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
├── utils/                    # 共享工具模块
│   └── align_to_isaac.py     #   对齐 + USDA 导出
├── 3dgrut/                   # 3DGUT 源码（需单独 clone）
├── 2dgs/                     # 2DGS 源码（需单独 clone）
├── video_to_scene.sh         # 场景重建入口
├── video_to_mesh.sh          # 物体网格入口
└── README.md
```

## 两个 Pipeline

|     | `video_to_scene.sh`                                                     | `video_to_mesh.sh`        |
| --- | ----------------------------------------------------------------------- | ------------------------- |
| 目标  | 场景级 3D Gaussian 重建                                                      | 物体级 3D 网格重建               |
| 方法  | 3DGUT (GS, SH=4)                                                        | 2DGS (SH=4) + TSDF + 几何优化 |
| 环境  | 保留完整场景                                                                  | DBSCAN 自动剔除 / 手动交互筛选      |
| 对齐  | PCA Manhattan → Z-up → flip → Y-up       (共享 `utils/align_to_isaac.py`) |                           |
| 地面  | 自动碰撞体 + 组合场景                                                            | 底面自动封闭 + 网格自带碰撞           |
| 输出  | USDZ + 碰撞地面 + 组合 USDA                                                   | USDA                      |
| 体素  | —                                                                       | 4mm TSDF                  |

**环境**：`conda activate vid2sim`

## 共享流程

两个脚本共用阶段 1-2，阶段 3-4 不同，对齐与导出共用 `utils/align_to_isaac.py`：

```
FFmpeg → 边缘增强 → COLMAP SfM → [3DGUT | 2DGS] → 对齐 → 导出
```

- 视频放 `videos/` 目录，脚本自动查找
- 输出在 `results/` 目录下
- `-c` 跳过 FFmpeg，`-S` 跳过 COLMAP，`-T` 跳过训练
- 图像增强：Laplacian 高通滤波 — 增强几何边缘（默认启用，`-A` 禁用）
- 高反射优化：严格 SIFT 筛选 + 禁用密度衰减/尺度剪枝 + 降低 specular LR + 多尺度 DBSCAN
- `utils/align_to_isaac.py` 提供 PCA Manhattan 3 轴对齐、flip 检测、USDA 导出等共享函数

## video_to_scene.sh — 场景重建

```bash
./video_to_scene.sh -v video.mp4               # 全流程
./video_to_scene.sh -v video.mp4 -c -S          # 仅重训练
```

| 参数   | 说明                     | 默认                    |
| ---- | ---------------------- | --------------------- |
| `-v` | 视频文件名或路径               | 优先 `videos/`          |
| `-o` | 输出目录                   | `results/{视频名}_scene` |
| `-f` | 抽帧 FPS                 | `5`                   |
| `-i` | 训练迭代数                  | `80000`               |
| `-d` | 训练下采样 (`1`=4K, `2`=2K) | `2` (24GB 推荐)         |
| `-g` | GPU ID                 | `0`                   |
| `-A` | 禁用边缘增强                 | 否                     |
| `-u` | 跳过 USDZ 导出             | 否                     |
| `-c` | 跳过 FFmpeg              | 否                     |
| `-S` | 跳过 COLMAP              | 否                     |
| `-T` | 跳过训练                   | 否                     |

输出结构：

```
results/<video>_scene/runs/<experiment>/<experiment>-MMDD_HHMMSS/
├── scene_nurec.usdz            # ★ 视觉场景
├── ground_collision.usda       # ★ 地面碰撞体
├── reconstruction.json         # 运行记录信息
├── ckpt_last.pt                # 模型 checkpoint (续训用)
└── ours_*/                     # 各阶段 checkpoint 
```

**Isaac Lab 导入**：

```python
from isaaclab.sim.spawners.from_files import UsdFileCfg

visual = UsdFileCfg(usd_path=".../scene_nurec.usdz")       # 视觉场景
ground = UsdFileCfg(usd_path=".../ground_collision.usda")    # 地面碰撞
```

对齐使用 PCA Manhattan 世界模型：最小方差轴 → 高度方向（Y），最大方差轴 → 最长墙面（X），中间方差轴 → 另一水平方向（Z），自动密度翻转检测确保场景不倒置。地面碰撞体沿 X×Z 平面展开，Y 位置在地板高度。视觉模型与地面碰撞体在同一坐标系中。

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
| `-g` | GPU ID                 | `0`                  |
| `-V` | 禁用交互筛选                 | 否                    |
| `-b` | 背景剔除系数                 | `1.5`                |
| `-A` | 禁用边缘增强                 | 否                    |
| `-u` | 跳过 USDA 导出             | 否                    |
| `-c` | 跳过 FFmpeg              | 否                    |
| `-S` | 跳过 COLMAP              | 否                    |
| `-T` | 跳过训练                   | 否                    |

输出结构：

```
results/<video>_mesh/runs/<experiment>/<experiment>-MMDD_HHMMSS/
├── mesh_<video>.usda            # ★ 网格模型
├── mesh_<video>.ply             # PLY 网格
├── reconstruction.json          # 运行记录信息
├── ckpt_*.pt                    # 模型 checkpoint
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

对齐使用与场景管道相同，确保 3 轴与 Isaac Sim 坐标系一致。几何优化平滑平面区、保留杆件细节、自动封闭底面。终端摘要显示网格顶点/面数、密闭性、平坦/细节比例。

# 

## 参考

- [3DGRUT](https://github.com/nv-tlabs/3dgrut) — 3D Gaussian Ray Tracing & Unscented Transform
- [2D Gaussian Splatting](https://surfsplatting.github.io/) — Geometrically Accurate Radiance Fields
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian Splatting 库
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion
