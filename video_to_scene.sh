#!/usr/bin/env bash
# video_to_scene.sh — Video → 3D Gaussian → USDZ, one command
#
# Stages:  1) FFmpeg extract  2) COLMAP SfM  3) 3DGUT train  4) USDZ export
# Stages can be skipped: -c (skip FFmpeg), -S (skip COLMAP), -T (skip train)
set -euo pipefail

# --- defaults ---
VIDEO_PATH=""
OUTPUT_DIR=""
EXPERIMENT_NAME=""
FPS=5
MAX_IMAGE_SIZE=2560
GPU_ID=0
TRAIN_ITERATIONS=80000
DOWNSAMPLE_FACTOR=2
SKIP_USDZ=false
SKIP_FFMPEG=false
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_ENHANCE=false

CONDA_ENV="vid2sim"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"; THREEDGRUT_DIR="$PROJECT_DIR/3dgrut"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()    { echo -e "${GREEN}[*]${NC} $(date '+%H:%M:%S') $*"; }
warn()   { echo -e "${YELLOW}[!]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[X]${NC} $(date '+%H:%M:%S') $*"; }
step()   { echo -e "\n${BLUE}▶${NC} $(date '+%H:%M:%S') $*"; }

usage() { head -36 "$0" | tail -20; exit 0; }

while getopts "v:o:n:f:s:g:i:d:uAcSTh" opt; do
    case $opt in
        v) VIDEO_PATH="$OPTARG" ;;  o) OUTPUT_DIR="$OPTARG" ;;
        n) EXPERIMENT_NAME="$OPTARG" ;; f) FPS="$OPTARG" ;;
        s) MAX_IMAGE_SIZE="$OPTARG" ;; g) GPU_ID="$OPTARG" ;;
        i) TRAIN_ITERATIONS="$OPTARG" ;; d) DOWNSAMPLE_FACTOR="$OPTARG" ;;
        u) SKIP_USDZ=true ;;  A) SKIP_ENHANCE=true ;;  c) SKIP_FFMPEG=true ;;
        S) SKIP_COLMAP=true ;; T) SKIP_TRAINING=true ;;  h) usage ;;  *) usage ;;
    esac
done

# --- validate & derive ---
[ -z "$VIDEO_PATH" ] && { err "必须指定 -v <video>"; usage; }

# 如果只给文件名，优先在 videos/ 下查找
if [[ "$VIDEO_PATH" != */* ]]; then
    VIDEO_DIR="$(dirname "$0")/videos"
    [ -f "$VIDEO_DIR/$VIDEO_PATH" ] && VIDEO_PATH="$VIDEO_DIR/$VIDEO_PATH"
fi
[ "$SKIP_FFMPEG" = false ] && [ ! -f "$VIDEO_PATH" ] && { err "视频不存在: $VIDEO_PATH"; exit 1; }

VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$(dirname "$0")/results/${VIDEO_NAME}_scene"
[ -z "$EXPERIMENT_NAME" ] && EXPERIMENT_NAME="${VIDEO_NAME}_3dgut"
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")

# 清理可能干扰的旧进程
pkill -f "colmap mapper.*$OUTPUT_DIR" 2>/dev/null || true
pkill -f "train.py.*$OUTPUT_DIR" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# --- env setup ---
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$CONDA_ENV" || { err "激活 conda 环境失败: $CONDA_ENV"; exit 1; }; set -u
export TORCH_CUDA_ARCH_LIST="8.9"  # must be after conda activate

for tool in ffmpeg colmap; do
    command -v "$tool" &>/dev/null || { err "未找到 $tool"; exit 1; }
done

log "conda=$CONDA_ENV | torch=$(python -c 'import torch;print(torch.__version__)') | gsplat=$(python -c 'import gsplat;print(gsplat.__version__)')"
python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null || warn "无 CUDA GPU!"
GPU_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')
GPU_MEM=$(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")')
log "GPU: $GPU_NAME ($GPU_MEM)"

# --- config summary ---
echo ""
echo "═══════════════════════════════════════════════"
printf "  %-16s %s\n" "视频" "$(basename "$VIDEO_PATH")"
printf "  %-16s %s\n" "输出" "$OUTPUT_DIR"
printf "  %-16s %s\n" "实验" "$EXPERIMENT_NAME"
printf "  %-16s %s\n" "FPS" "$FPS"
printf "  %-16s %s\n" "图片边长" "$MAX_IMAGE_SIZE"
printf "  %-16s %s\n" "迭代" "$TRAIN_ITERATIONS"
printf "  %-16s %s\n" "下采样" "${DOWNSAMPLE_FACTOR}x"
printf "  %-16s %s\n" "USDZ" "$([ "$SKIP_USDZ" = true ] && echo 否 || echo 是)"
printf "  %-16s %s %s %s\n" "跳过" "$([ "$SKIP_FFMPEG" = true ] && echo FFmpeg,)" "$([ "$SKIP_COLMAP" = true ] && echo COLMAP,)" "$([ "$SKIP_TRAINING" = true ] && echo Train)"
echo "═══════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

# ================================================================================================
# Step 1 — Extract frames
# ================================================================================================
IMAGE_DIR="$OUTPUT_DIR/images"

if [ "$SKIP_FFMPEG" = true ]; then
    step "Step 1/4: 跳过 FFmpeg"
    [ ! -d "$IMAGE_DIR" ] && { err "图片目录不存在: $IMAGE_DIR"; exit 1; }
    FRAME_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)
else
    step "Step 1/4: 抽帧 (FPS=$FPS)"
    rm -rf "$IMAGE_DIR" && mkdir -p "$IMAGE_DIR"
    # 不缩放——保留 4K 原图，COLMAP SIFT 内部 Lanczos 降采样质量更高
    ffmpeg -i "$VIDEO_PATH" -vf "fps=$FPS" \
        -q:v 2 -frame_pts 1 "$IMAGE_DIR/frame_%05d.jpg" -loglevel warning -stats
    FRAME_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -name "*.jpg" | wc -l)
    [ "$FRAME_COUNT" -eq 0 ] && { err "抽帧失败"; exit 1; }
fi
log "图片: $FRAME_COUNT 帧"

# 边缘增强 (Laplacian 高通滤波) — 增强几何边缘，抑制镜面渐变
if [ "$SKIP_FFMPEG" = false ] || [ "$FRAME_COUNT" -gt 0 ]; then
    EDGE_FLAG="${IMAGE_DIR}/.edge_enhanced"
    if [ "$SKIP_ENHANCE" = true ]; then
        touch "$EDGE_FLAG" 2>/dev/null || true
        log "跳过边缘增强"
    elif [ ! -f "$EDGE_FLAG" ]; then
        log "Laplacian 边缘增强 (抑制镜面渐变)..."
        python -c "
import cv2, numpy as np
from pathlib import Path
for p in sorted(Path('$IMAGE_DIR').glob('*.jpg')):
    img = cv2.imread(str(p))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    enhanced = np.clip(gray + 2.0 * lap, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(p), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f'{sum(1 for _ in Path(\"$IMAGE_DIR\").glob(\"*.jpg\"))} images')
" 2>&1 && touch "$EDGE_FLAG" && log "边缘增强完成"
    fi
fi

# auto-generate downsampled images if needed
if [ "$DOWNSAMPLE_FACTOR" -gt 1 ]; then
    DOWNSAMPLED_DIR="${IMAGE_DIR}_${DOWNSAMPLE_FACTOR}"
    if [ ! -d "$DOWNSAMPLED_DIR" ]; then
        log "生成 ${IMAGE_DIR}_${DOWNSAMPLE_FACTOR} ..."
        mkdir -p "$DOWNSAMPLED_DIR"
        python -c "
import torch, torchvision.transforms.functional as TF
from PIL import Image; from pathlib import Path
src,dst,f = Path('$IMAGE_DIR'), Path('$DOWNSAMPLED_DIR'), $DOWNSAMPLE_FACTOR
jpgs = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for p in jpgs:
    img = TF.to_tensor(Image.open(p)).unsqueeze(0).to(device)
    h, w = img.shape[2], img.shape[3]
    img = TF.resize(img, [h//f, w//f], antialias=True)
    TF.to_pil_image(img.squeeze(0).cpu()).save(dst/p.name, quality=95)
print(f'{len(jpgs)} images')
" || { err "降采样失败"; exit 1; }
    fi
fi

# ================================================================================================
# Step 2 — COLMAP
# ================================================================================================
SPARSE_DIR="$OUTPUT_DIR/sparse"
DATABASE_PATH="$OUTPUT_DIR/database.db"

if [ "$SKIP_COLMAP" = true ]; then
    step "Step 2/4: 跳过 COLMAP"
    NUM_IMAGES_REG=$(python -c "import struct;f=open('$SPARSE_DIR/0/images.bin','rb');print(struct.unpack('<Q',f.read(8))[0])" 2>/dev/null || echo "?")
    [ ! -f "$SPARSE_DIR/0/cameras.bin" ] || [ ! -f "$SPARSE_DIR/0/images.bin" ] && { err "COLMAP 结果不完整: $SPARSE_DIR/0/"; exit 1; }
else
    step "Step 2/4: COLMAP SfM"
    rm -f "$DATABASE_PATH"; rm -rf "$SPARSE_DIR"

    colmap feature_extractor \
        --database_path "$DATABASE_PATH" --image_path "$IMAGE_DIR" \
        --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1 \
        --FeatureExtraction.use_gpu 1 --SiftExtraction.max_image_size "$MAX_IMAGE_SIZE" \
        --SiftExtraction.max_num_features 16384 --SiftExtraction.peak_threshold 0.008 \
        --SiftExtraction.edge_threshold 15 --SiftExtraction.num_octaves 6 --SiftExtraction.estimate_affine_shape 1 \
        --SiftExtraction.domain_size_pooling 1 2>&1 | tail -2

    [ ! -f "$DATABASE_PATH" ] && { err "特征提取失败"; exit 1; }

    colmap sequential_matcher \
        --database_path "$DATABASE_PATH" --FeatureMatching.use_gpu 1 --FeatureMatching.max_num_matches 65536 \
        --SiftMatching.max_ratio 0.65 --SiftMatching.max_distance 0.45 --SiftMatching.cross_check 1 \
        --SequentialMatching.overlap 10 2>&1 | tail -2

    mkdir -p "$SPARSE_DIR"

    colmap mapper \
        --database_path "$DATABASE_PATH" --image_path "$IMAGE_DIR" --output_path "$SPARSE_DIR" \
        --Mapper.ba_global_function_tolerance 1e-6 --Mapper.ba_use_gpu 1 --Mapper.ba_refine_principal_point 1 2>&1 | \
        awk -v total="$FRAME_COUNT" '/Registering image/ {
            count++; pct=int(count/total*100);
            printf "\r  [Mapper] %d/%d frames (%d%%)", count, total, pct
        } /ERROR|WARN|Elapsed|Discard|Keeping|No good/ { print "\n" $0 }
        END { if(count>0) printf "\n" }'

    NUM_IMAGES_REG=$(python -c "import struct;f=open('$SPARSE_DIR/0/images.bin','rb');print(struct.unpack('<Q',f.read(8))[0])")
    log "注册 $NUM_IMAGES_REG/$FRAME_COUNT 帧"

    if [ ! -f "$SPARSE_DIR/0/cameras.bin" ] || [ ! -f "$SPARSE_DIR/0/images.bin" ]; then
        err "COLMAP 失败: 无有效重建"
        exit 1
    fi
fi

# ================================================================================================
# Step 3 — 3DGUT train
# ================================================================================================
RUNS_DIR="$OUTPUT_DIR/runs"

# Check for completed checkpoint before deciding to train
PREV_CKPT=$(find "$RUNS_DIR/$EXPERIMENT_NAME" -name "ckpt_last.pt" -type f 2>/dev/null | sort | tail -1 || true)
if [ -n "$PREV_CKPT" ]; then
    SKIP_TRAINING=true
    log "发现已完成 checkpoint，自动跳过训练"
fi

if [ "$SKIP_TRAINING" = true ]; then
    step "Step 3/4: 跳过训练"
else
    step "Step 3/4: 训练 (${TRAIN_ITERATIONS} iter)"

    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$THREEDGRUT_DIR"
    TRAIN_CMD=(python train.py --config-name apps/colmap_3dgut.yaml
        "path=$OUTPUT_DIR" "out_dir=$RUNS_DIR" "experiment_name=$EXPERIMENT_NAME"
        "dataset.downsample_factor=$DOWNSAMPLE_FACTOR"
        "n_iterations=$TRAIN_ITERATIONS" "scheduler.positions.max_steps=$TRAIN_ITERATIONS"
        "render.particle_radiance_sph_degree=4" \
        "model.progressive_training.max_n_features=4" \
        "model.progressive_training.increase_frequency=1000" \
        "optimizer.params.features_specular.lr=0.0002" \
        "loss.use_l1=true" "loss.lambda_l1=0.8" \
        "loss.use_l2=true" "loss.lambda_l2=0.2" \
        "loss.use_scale=true" "loss.lambda_scale=0.1" \
        "strategy.density_decay.start_iteration=-1" "strategy.density_decay.end_iteration=-1" \
        "strategy.prune_scale.start_iteration=500" "strategy.prune_scale.end_iteration=$TRAIN_ITERATIONS" \
        "strategy.prune_scale.frequency=500" "strategy.prune_scale.threshold=2.0" \
        "strategy.densify.end_iteration=40000" \
        "strategy.densify.clone_grad_threshold=0.0002" \
        "strategy.densify.split_grad_threshold=0.0002" \
        "strategy.reset_density.frequency=5000" \
        "strategy.reset_density.new_max_density=0.02" \
        "strategy.reset_density.end_iteration=$TRAIN_ITERATIONS" \
        "strategy.prune.end_iteration=$TRAIN_ITERATIONS" \
        "strategy.prune.density_threshold=0.005" \
        "post_processing.method=ppisp" "post_processing.n_distillation_steps=5000" \
        "num_workers=4" \
        "export_usd.enabled=true" "export_usd.format=nurec" "export_usd.apply_normalizing_transform=false")

    # auto-resume: add resume flag for incomplete training
    if [ -n "$PREV_CKPT" ] && [ ! -d "$(dirname "$PREV_CKPT")/ours_${TRAIN_ITERATIONS}" ]; then
        if [ "$SPARSE_DIR/0/cameras.bin" -nt "$PREV_CKPT" ]; then
            warn "COLMAP 数据已更新，忽略旧 checkpoint"
        else
            TRAIN_CMD+=("resume=$PREV_CKPT"); log "续训: $PREV_CKPT"
        fi
    fi

    mkdir -p "$RUNS_DIR/$EXPERIMENT_NAME"
    set +o pipefail
    "${TRAIN_CMD[@]}" 2>&1 | tee "$RUNS_DIR/$EXPERIMENT_NAME/train.log" | grep -vE "━|┃|┏|┓|┗|┛|Training Statistics|Test Metrics" || true
    TRAIN_EXIT=${PIPESTATUS[0]}
    set -o pipefail
    [ "$TRAIN_EXIT" -ne 0 ] && { err "训练异常退出 (exit $TRAIN_EXIT)"; exit 1; }

    TRAIN_OUTDIR=$(find "$RUNS_DIR/$EXPERIMENT_NAME" -maxdepth 2 -name "ckpt_last.pt" -type f -printf "%h\n" 2>/dev/null | sort | tail -1 || true)
    log "完成: $TRAIN_OUTDIR"
fi

# ================================================================================================
# Step 4 — Clean Gaussians + PCA Z-up + Export USDZ
# ================================================================================================
if [ "$SKIP_USDZ" = false ]; then
    step "Step 4/4: 清理 + 旋转 + USDZ 导出"
    [ -z "${TRAIN_OUTDIR:-}" ] && TRAIN_OUTDIR=$(find "$RUNS_DIR/$EXPERIMENT_NAME" -maxdepth 2 -name "ckpt_last.pt" -type f -printf "%h\n" 2>/dev/null | sort | tail -1 || true)
    CKPT_SRC="${TRAIN_OUTDIR}/ckpt_last.pt"
    CKPT_CLEAN="${TRAIN_OUTDIR}/ckpt_clean.pt"
    USDZ_FILE="${TRAIN_OUTDIR}/scene_nurec.usdz"

    if [ -f "$CKPT_SRC" ]; then
        log "清理浮空高斯..."
        python -c "
import torch, numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.spatial import cKDTree

ckpt = torch.load('$CKPT_SRC', map_location='cpu', weights_only=False)
pos = ckpt['positions'].detach().numpy()
density = ckpt['density'].detach().numpy().squeeze()
scales = ckpt['scale'].detach().numpy()
N = len(pos)

# 1. Opacity filter
opacity = 1/(1+np.exp(-density))
keep = opacity >= 0.008

# 2. Scale filter
scale_norm = np.linalg.norm(scales, axis=1)
keep &= scale_norm < np.nanmedian(scale_norm) * 3

# 3. Multi-scale DBSCAN: preserve all significant spatial clusters
n_sample = min(N//10, 100000)
rng = np.random.RandomState(42)
idx = rng.choice(N, n_sample, replace=False)
sample = pos[idx]

# Multi-scale clustering: fine → medium → coarse
keep_cluster = np.zeros(N, dtype=bool)
for eps in [0.3, 0.8, 2.0]:
    cl = DBSCAN(eps=eps, min_samples=20).fit(sample)
    ls = cl.labels_
    cnt = Counter(ls[ls >= 0])
    if not cnt: continue
    tree = cKDTree(sample); _, nn = tree.query(pos, k=3)
    fl = np.array([ls[nn[i][ls[nn[i]]>=0][0]] if np.any(ls[nn[i]]>=0) else -1 for i in range(N)])
    for label_id, count in cnt.items():
        if count > N * 0.02:
            keep_cluster |= (fl == label_id)
    print(f'  eps={eps}: {len(cnt)} clusters, kept {keep_cluster.sum():,} pts so far')

if keep_cluster.any():
    keep &= keep_cluster
else:
    # Fallback: keep largest cluster (conservative)
    cl = DBSCAN(eps=5.0, min_samples=30).fit(sample)
    ls = cl.labels_; cnt = Counter(ls[ls>=0])
    if cnt:
        largest = cnt.most_common(1)[0][0]
        tree = cKDTree(sample); _, nn = tree.query(pos, k=3)
        fl = np.array([ls[nn[i][ls[nn[i]]>=0][0]] if np.any(ls[nn[i]]>=0) else -1 for i in range(N)])
        keep &= fl == largest
for k in ['positions','rotation','scale','density','features_albedo','features_specular']:
    if k in ckpt: ckpt[k] = ckpt[k][keep]
print(f'{keep.sum():,} / {N:,} ({keep.mean()*100:.0f}%)')

# Rotation: PCA → Z-up → flip → Y-up (Isaac Sim) — uses shared align_to_isaac module
import sys, json
sys.path.insert(0, '$PROJECT_DIR/utils')
from align_to_isaac import compute_manhattan_rotation, apply_rotation_to_tensor, save_rotation_json

pos = ckpt['positions'].detach().numpy()
result = compute_manhattan_rotation(pos, do_flip_check=True)
R = result['R']
ckpt['positions'] = apply_rotation_to_tensor(ckpt['positions'], R)
print(f'Z-up: {result[\"angle_deg\"]:.1f}deg, flipped={result[\"flipped\"]}, Y-up OK')
if result['flipped']:
    print(f'  Flip (top={result[\"flip_top_count\"]} > bot={result[\"flip_bot_count\"]})')
save_rotation_json(R, '${TRAIN_OUTDIR}/rotation.json')

for k in list(ckpt.keys()):
    if isinstance(ckpt[k], torch.Tensor) and not isinstance(ckpt[k], torch.nn.Parameter):
        ckpt[k] = torch.nn.Parameter(ckpt[k])
torch.save(ckpt, '$CKPT_CLEAN')
" 2>&1 && log "清理完成: $CKPT_CLEAN"

        log "导出 NuRec USDZ ..."
        cd "$THREEDGRUT_DIR"
        python -m threedgrut.export.scripts.export_usd \
            --checkpoint "$CKPT_CLEAN" --output "$USDZ_FILE" --format nurec \
            --no-transform --no-cameras --no-background 2>&1 | tail -2
        [ -f "$USDZ_FILE" ] && log "USDZ: $USDZ_FILE ($(du -h "$USDZ_FILE" | cut -f1))"

        # Combined scene (Gaussians + ground) — uses shared module
        python -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/utils')
from align_to_isaac import export_combined_scene_usda
export_combined_scene_usda('${TRAIN_OUTDIR}/scene_combined.usda', './scene_nurec.usdz', './ground_collision.usda')
" 2>&1
        log "组合场景: ${TRAIN_OUTDIR}/scene_combined.usda"

        rm -f "$CKPT_CLEAN"
        find "$TRAIN_OUTDIR" -name "export_*.usdz" ! -name "scene_nurec.usdz" -delete 2>/dev/null || true
        find "$TRAIN_OUTDIR" \( -name "parsed.yaml" -o -name "events.out.*" -o -name "train.log" -o -name "metrics.json" \) -delete 2>/dev/null || true
        rm -rf "$TRAIN_OUTDIR/ppisp_report" 2>/dev/null || true
    else
        warn "未找到 checkpoint"
    fi
fi

# ================================================================================================

# ================================================================================================
# Ground collision — bounding box in original coordinate frame
# ================================================================================================
GROUND_FILE="${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}/ground_collision.usda"
ROTATION_FILE="${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}/rotation.json"
if [ -f "$SPARSE_DIR/0/points3D.bin" ]; then
    python -c "
import sys; sys.path.insert(0, '$PROJECT_DIR/utils')
from align_to_isaac import load_colmap_points, load_rotation_json, export_ground_collision_usda
import numpy as np

pts = load_colmap_points('$SPARSE_DIR/0/points3D.bin')
R = load_rotation_json('$ROTATION_FILE')
pts_rotated = (R @ pts.T).T
export_ground_collision_usda(pts_rotated, '$GROUND_FILE')
" 2>&1 && log "地面碰撞: $GROUND_FILE"
fi
# Done
# ================================================================================================
ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_M=$((ELAPSED / 60))
ELAPSED_S=$((ELAPSED % 60))

# Read metrics directly
METRICS_FILE=$(find "${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}" -name "metrics.json" -type f 2>/dev/null | tail -1 || true)
if [ -f "$METRICS_FILE" ]; then
    PSNR=$(python -c "import json;d=json.load(open('$METRICS_FILE'));print(f\"{d['mean_psnr']:.1f}\")" 2>/dev/null || echo "")
    SSIM=$(python -c "import json;d=json.load(open('$METRICS_FILE'));print(f\"{d['mean_ssim']:.3f}\")" 2>/dev/null || echo "")
    LPIPS=$(python -c "import json;d=json.load(open('$METRICS_FILE'));print(f\"{d['mean_lpips']:.3f}\")" 2>/dev/null || echo "")
fi

step "Pipeline 完成 (${ELAPSED_M}m ${ELAPSED_S}s)"

echo ""
echo "╔══════════════════════════════════════════╗"
printf "║  📷 图片      %-4s 帧  ║\n" "$FRAME_COUNT"
printf "║  📐 COLMAP    %-4s 帧注册 ║\n" "${NUM_IMAGES_REG:-?}"
[ -n "${PSNR:-}" ] && printf "║  🎯 PSNR      %-7s    ║\n" "$PSNR"
[ -n "${SSIM:-}" ] && printf "║  🎯 SSIM      %-7s    ║\n" "$SSIM"
[ -n "${LPIPS:-}" ] && printf "║  🎯 LPIPS     %-7s    ║\n" "$LPIPS"
echo "╠══════════════════════════════════════════╣"
printf "║  ⏱  总耗时    %-5s     ║\n" "${ELAPSED_M}m${ELAPSED_S}s"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "续跑: $0 -v '$VIDEO_PATH' -c -S"
