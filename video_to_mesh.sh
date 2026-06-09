#!/usr/bin/env bash
# video_to_mesh.sh — Video → Object Mesh via 2DGS + TSDF Fusion
#  1) FFmpeg extract  2) COLMAP SfM  3) 2DGS train  4) TSDF mesh + export
set -euo pipefail

VIDEO_PATH=""; OUTPUT_DIR=""; EXPERIMENT_NAME=""
FPS=5; MAX_IMAGE_SIZE=1920; GPU_ID=0; TRAIN_ITERATIONS=60000; DOWNSAMPLE_FACTOR=2
SKIP_FFMPEG=false; SKIP_COLMAP=false; SKIP_TRAINING=false; SKIP_USDZ=false
VISUAL_FILTER=false  # -V: interactive point cloud crop before training
CONDA_ENV="vid2sim"; TWODGS_DIR="$(cd "$(dirname "$0")" && pwd)/2dgs"
CULL_FACTOR=1.5   # IQR multiplier for spatial culling (larger = looser)
VOXEL_SIZE=0.004  # TSDF voxel — 4mm, 2cm gap = 5 voxels

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[*]${NC} $(date '+%H:%M:%S') $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $(date '+%H:%M:%S') $*"; }
err()  { echo -e "${RED}[X]${NC} $(date '+%H:%M:%S') $*"; }
step() { echo -e "\n${BLUE}▶${NC} $(date '+%H:%M:%S') $*"; }

usage() { head -32 "$0" | tail -16; exit 0; }

while getopts "v:o:n:f:s:g:i:d:b:cSThVu" opt; do
    case $opt in
        v) VIDEO_PATH="$OPTARG" ;;  o) OUTPUT_DIR="$OPTARG" ;;
        n) EXPERIMENT_NAME="$OPTARG" ;; f) FPS="$OPTARG" ;;
        s) MAX_IMAGE_SIZE="$OPTARG" ;; g) GPU_ID="$OPTARG" ;;
        i) TRAIN_ITERATIONS="$OPTARG" ;; d) DOWNSAMPLE_FACTOR="$OPTARG" ;;
        b) CULL_FACTOR="$OPTARG" ;;
        c) SKIP_FFMPEG=true ;;  S) SKIP_COLMAP=true ;;  T) SKIP_TRAINING=true ;;  u) SKIP_USDZ=true ;;  V) VISUAL_FILTER=true ;;
        h) usage ;;  *) usage ;;
    esac
done

[ -z "$VIDEO_PATH" ] && { err "必须指定 -v <video>"; usage; }
# 如果只给文件名，优先在 videos/ 下查找
if [[ "$VIDEO_PATH" != */* ]]; then
    VIDEO_DIR="$(dirname "$0")/videos"
    [ -f "$VIDEO_DIR/$VIDEO_PATH" ] && VIDEO_PATH="$VIDEO_DIR/$VIDEO_PATH"
fi
[ "$SKIP_FFMPEG" = false ] && [ ! -f "$VIDEO_PATH" ] && { err "视频不存在: $VIDEO_PATH"; exit 1; }

VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$(dirname "$0")/results/${VIDEO_NAME}_mesh"
[ -z "$EXPERIMENT_NAME" ] && EXPERIMENT_NAME="${VIDEO_NAME}_2dgs"
OUTPUT_DIR=$(realpath -m "$OUTPUT_DIR")

# 清理可能干扰的旧进程
pkill -f "colmap mapper.*$OUTPUT_DIR" 2>/dev/null || true
pkill -f "train.py.*$OUTPUT_DIR" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# --- env setup ---
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u; conda activate "$CONDA_ENV" || { err "激活 conda 环境失败: $CONDA_ENV"; exit 1; }; set -u
export TORCH_CUDA_ARCH_LIST="8.9"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"

for tool in ffmpeg colmap; do
    command -v "$tool" &>/dev/null || { err "未找到 $tool"; exit 1; }
done
log "env=$CONDA_ENV | 2DGS=$TWODGS_DIR"
python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null || warn "无 CUDA GPU!"
GPU_NAME=$(python -c 'import torch; print(torch.cuda.get_device_name(0))')
GPU_MEM=$(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")')
log "GPU: $GPU_NAME ($GPU_MEM)"

echo ""
echo "═══════════════════════════════════════════════"
printf "  %-16s %s\n" "视频" "$(basename "$VIDEO_PATH")"
printf "  %-16s %s\n" "输出" "$OUTPUT_DIR"
printf "  %-16s %s\n" "实验" "$EXPERIMENT_NAME"
printf "  %-16s %s\n" "迭代" "$TRAIN_ITERATIONS"
printf "  %-16s %s\n" "剔除系数" "${CULL_FACTOR}x"
printf "  %-16s %s\n" "体素大小" "$VOXEL_SIZE"
printf "  %-16s %s %s %s\n" "跳过" "$([ "$SKIP_FFMPEG" = true ] && echo FFmpeg,)" "$([ "$SKIP_COLMAP" = true ] && echo COLMAP,)" "$([ "$SKIP_TRAINING" = true ] && echo Train)"
echo "═══════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

# ================================================================================================
# Step 1 — Extract frames (same as scene pipeline)
# ================================================================================================
IMAGE_DIR="$OUTPUT_DIR/images"

if [ "$SKIP_FFMPEG" = true ]; then
    step "Step 1/4: 跳过 FFmpeg"
    [ ! -d "$IMAGE_DIR" ] && { err "图片目录不存在: $IMAGE_DIR"; exit 1; }
    FRAME_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)
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

# CLAHE 对比度增强
if [ "$SKIP_FFMPEG" = false ]; then
    CLAHE_FLAG="${IMAGE_DIR}/.clahe_done"
    if [ ! -f "$CLAHE_FLAG" ]; then
        log "CLAHE 增强..."
        python -c "
import cv2; from pathlib import Path
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for p in sorted(Path('$IMAGE_DIR').glob('*.jpg')):
    img = cv2.imread(str(p)); lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab); l = clahe.apply(l)
    cv2.imwrite(str(p), cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f'{sum(1 for _ in Path(\"$IMAGE_DIR\").glob(\"*.jpg\"))} images')
" 2>&1 && touch "$CLAHE_FLAG" && log "CLAHE 完成"
    fi
fi

# auto downsampled images for 2DGS
if [ "$DOWNSAMPLE_FACTOR" -gt 1 ]; then
    DOWNSAMPLED_DIR="${IMAGE_DIR}_${DOWNSAMPLE_FACTOR}"
    if [ ! -d "$DOWNSAMPLED_DIR" ]; then
        log "生成 ${DOWNSAMPLED_DIR} ..."
        mkdir -p "$DOWNSAMPLED_DIR"
        python -c "
from PIL import Image; from pathlib import Path
import torch, torchvision.transforms.functional as TF
src,dst,f = Path('$IMAGE_DIR'), Path('$DOWNSAMPLED_DIR'), $DOWNSAMPLE_FACTOR
jpgs = sorted(src.glob('*.jpg')) + sorted(src.glob('*.png'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for p in jpgs:
    img = TF.to_tensor(Image.open(p)).unsqueeze(0).to(device)
    h, w = img.shape[2], img.shape[3]
    img = TF.resize(img, [h//f, w//f], antialias=True)
    TF.to_pil_image(img.squeeze(0).cpu()).save(dst/p.name, quality=95)
" || { err "降采样失败"; exit 1; }
    fi
fi

# ================================================================================================
# Step 2 — COLMAP (same as scene pipeline)
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
        --SiftExtraction.max_num_features 65536 --SiftExtraction.peak_threshold 0.002 \
        --SiftExtraction.edge_threshold 5 --SiftExtraction.num_octaves 5 --SiftExtraction.estimate_affine_shape 0 \
        --SiftExtraction.domain_size_pooling 0 2>&1 | tail -2

    [ ! -f "$DATABASE_PATH" ] && { err "特征提取失败"; exit 1; }

    colmap vocab_tree_matcher \
        --database_path "$DATABASE_PATH" --FeatureMatching.use_gpu 1 --FeatureMatching.max_num_matches 65536 \
        --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 --SiftMatching.cross_check 1 2>&1 | tail -2

    mkdir -p "$SPARSE_DIR"

    MAX_MAPPER_RETRIES=3
    MAPPER_TRY=0
    REG_THRESHOLD=0.50
    while [ "$MAPPER_TRY" -lt "$MAX_MAPPER_RETRIES" ]; do
        MAPPER_TRY=$((MAPPER_TRY + 1))
        log "Mapper 尝试 $MAPPER_TRY/$MAX_MAPPER_RETRIES ..."

        rm -rf "$SPARSE_DIR"/*/
        colmap mapper \
            --database_path "$DATABASE_PATH" --image_path "$IMAGE_DIR" --output_path "$SPARSE_DIR" \
            --Mapper.ba_global_function_tolerance 1e-6 --Mapper.ba_use_gpu 1 --Mapper.ba_refine_principal_point 1 2>&1 | \
            awk -v total="$FRAME_COUNT" '/Registering image/ {
                count++; pct=int(count/total*100);
                printf "\r  [Mapper] %d/%d frames (%d%%)", count, total, pct
            } /ERROR|WARN|Elapsed|Discard|Keeping|No good/ { print "\n" $0 }
            END { if(count>0) printf "\n" }'

        BEST_SPARSE=$(python -c "
import struct, os, glob
best_n, best_dir = 0, ''
for d in sorted(glob.glob('$SPARSE_DIR/*/')):
    im = os.path.join(d, 'images.bin')
    if os.path.exists(im):
        with open(im, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
        if n > best_n: best_n, best_dir = n, d
print(best_dir.rstrip('/') if best_n > 0 else '')
")
        [ -z "$BEST_SPARSE" ] && continue
        BEST_IDX=$(basename "$BEST_SPARSE")
        if [ "$BEST_IDX" != "0" ]; then
            rm -rf "$SPARSE_DIR/0"
            mv "$BEST_SPARSE" "$SPARSE_DIR/0"
        fi

        NUM_IMAGES_REG=$(python -c "import struct;f=open('$SPARSE_DIR/0/images.bin','rb');print(struct.unpack('<Q',f.read(8))[0])")
        REG_RATIO=$(python -c "print($NUM_IMAGES_REG/$FRAME_COUNT)")
        log "注册 $NUM_IMAGES_REG/$FRAME_COUNT 帧 ($(python -c "print(f'{$REG_RATIO*100:.0f}')")%)"

        if python -c "exit(0 if $REG_RATIO >= $REG_THRESHOLD else 1)"; then
            log "注册率达标 (≥${REG_THRESHOLD//0./}%)，继续训练"
            break
        fi
        warn "注册率 < ${REG_THRESHOLD//0./}%，重试..."
    done

    if [ ! -f "$SPARSE_DIR/0/cameras.bin" ] || [ ! -f "$SPARSE_DIR/0/images.bin" ]; then
        err "COLMAP 失败: 多次重试后仍无有效重建"
        exit 1
    fi
    if python -c "exit(0 if $REG_RATIO < $REG_THRESHOLD else 1)"; then
        warn "多次重试后注册率仍 < ${REG_THRESHOLD//0./}%，但继续训练 (可用 $NUM_IMAGES_REG 帧)"
    fi
fi

# ================================================================================================
# Optional — Visual point cloud filter (interactive crop)
# ================================================================================================
CROP_FILE="$OUTPUT_DIR/crop_bounds.json"
if [ "$VISUAL_FILTER" = true ] && [ -f "$SPARSE_DIR/0/points3D.bin" ]; then
    step "交互筛选"

    echo ""
    echo "┌─────────────────────────────────────────────┐"
    echo "│  操作说明                                    │"
    echo "│  鼠标左键拖动 → 旋转视角                      │"
    echo "│  鼠标滚轮     → 缩放                          │"
    echo "│  Ctrl+左键    → 框选点云 (矩形选择)            │"
    echo "│  K 键         → 切换裁剪框 / 确认裁剪           │"
    echo "│  关闭窗口     → 保存并继续                     │"
    echo "│  建议: 旋转到俯视图，先框掉天花板和远处噪声       │"
    echo "└─────────────────────────────────────────────┘"
    echo ""

    python -c "
import struct, numpy as np, json, open3d as o3d

path = '$SPARSE_DIR/0/points3D.bin'
pts = []
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n):
        data = f.read(43); x,y,z = struct.unpack_from('<ddd', data, 8)
        pts.append([x,y,z])
        track_len = struct.unpack('<Q', f.read(8))[0]; f.seek(track_len * 8, 1)
pts = np.array(pts)
print(f'Loaded {len(pts):,} COLMAP points — rotate to top view, then Ctrl+drag to select')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)
pcd.paint_uniform_color([0.5, 0.5, 0.5])

cropped = o3d.visualization.draw_geometries_with_editing([pcd], window_name='Select object region')

if isinstance(cropped, o3d.geometry.PointCloud) and len(cropped.points) > 10:
    cpts = np.asarray(cropped.points)
    lower = cpts.min(axis=0).tolist()
    upper = cpts.max(axis=0).tolist()
    center = cpts.mean(axis=0).tolist()
    json.dump({'lower': lower, 'upper': upper, 'center': center, 'n': len(cpts)},
              open('$CROP_FILE', 'w'))
    print(f'Crop saved: {len(cpts):,} pts')
else:
    print('Crop skipped, using full point cloud')
    json.dump({'lower': None, 'upper': None, 'center': None, 'n': 0},
              open('$CROP_FILE', 'w'))
" 2>&1
    log "手动筛选完成: $CROP_FILE"

    # Ask if user wants additional auto-filtering on the cropped region
    echo ""
    read -r -p "是否对筛选后的区域再进行自动去噪 (DBSCAN)? [Y/n] " AUTO_AFTER_CROP
    if [ "${AUTO_AFTER_CROP:-y}" = "y" ] || [ "${AUTO_AFTER_CROP:-y}" = "Y" ]; then
        CROP_FILE_TMP="${CROP_FILE}.tmp"
        mv "$CROP_FILE" "$CROP_FILE_TMP" 2>/dev/null || true
        log "将在手动筛选基础上叠加自动 DBSCAN 去噪"
    fi
fi

# ================================================================================================
# Step 3 — 2DGS Train
# ================================================================================================
TWODGS_BASE="$OUTPUT_DIR/runs/mesh_2dgs"
TIMESTAMP=$(date +%m%d_%H%M%S)
TWODGS_OUT="${TWODGS_BASE}/${EXPERIMENT_NAME}-${TIMESTAMP}"

# Check for completed training in previous runs, or find checkpoint to resume
PREV_MESH=$(find "$TWODGS_BASE" -name "iteration_${TRAIN_ITERATIONS}" -type d 2>/dev/null | sort | tail -1 || true)
RESUME_CKPT=""
if [ -n "$PREV_MESH" ]; then
    SKIP_TRAINING=true
    TWODGS_OUT=$(dirname "$(dirname "$PREV_MESH")")
    log "发现已完成训练: $TWODGS_OUT，跳过训练"
else
    # Look for latest incomplete checkpoint to resume
    LATEST_ITER=$(find "$TWODGS_BASE" -name "iteration_*" -type d 2>/dev/null | sed 's/.*iteration_//' | sort -n | tail -1 || true)
    if [ -n "$LATEST_ITER" ] && [ "$LATEST_ITER" -lt "$TRAIN_ITERATIONS" ]; then
        RESUME_DIR=$(find "$TWODGS_BASE" -name "iteration_${LATEST_ITER}" -type d 2>/dev/null | sort | tail -1 || true)
        RESUME_CKPT=$(dirname "$RESUME_DIR")/chkpnt${LATEST_ITER}.pth
        if [ -f "$RESUME_CKPT" ]; then
            TWODGS_OUT=$(dirname "$(dirname "$RESUME_DIR")")
            log "续训: $TWODGS_OUT (从 $LATEST_ITER/$TRAIN_ITERATIONS)"
        fi
    fi
fi

mkdir -p "$TWODGS_OUT"

if [ "$SKIP_TRAINING" = true ]; then
    step "Step 3/4: 跳过训练"
else
    step "Step 3/4: 2DGS 训练 (${TRAIN_ITERATIONS} iter)"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$TWODGS_DIR"

    # 2DGS: -s <source_path> (containing sparse/0/), -m <model_output>
    # --resolution matches DOWNSAMPLE_FACTOR: 1=4K full res, 2=2K, etc.
    # SH degree 4 for metal specular, lambda_normal for bar surface smoothness
    RESUME_FLAG=""
    [ -n "$RESUME_CKPT" ] && RESUME_FLAG="--start_checkpoint $RESUME_CKPT"
    set +o pipefail
    python train.py \
        -s "$OUTPUT_DIR" \
        -m "$TWODGS_OUT" \
        $RESUME_FLAG \
        --iterations "$TRAIN_ITERATIONS" \
        --resolution "$DOWNSAMPLE_FACTOR" \
        --sh_degree 4 \
        --lambda_normal 0.05 \
        --lambda_dist 1.0 \
        --data_device cuda \
        --save_iterations $(seq -s ' ' 5000 5000 $TRAIN_ITERATIONS) \
        2>&1 | tee "$TWODGS_OUT/train.log"
    TRAIN_EXIT=${PIPESTATUS[0]}
    set -o pipefail
    [ "$TRAIN_EXIT" -ne 0 ] && { err "训练异常退出 (exit $TRAIN_EXIT)"; exit 1; }

    log "训练完成: $TWODGS_OUT"
fi

# ================================================================================================
# Step 4 — Spatial Culling (auto-detect object bounds from COLMAP density)
# ================================================================================================
log "  ◈ 空间剔除 (cull_factor=${CULL_FACTOR})"

if [ ! -f "$SPARSE_DIR/0/points3D.bin" ]; then
    err "无 COLMAP 点云，跳过剔除"
else
    BOUNDS_FILE="$TWODGS_OUT/object_bounds.json"
    CROP_FILE="$OUTPUT_DIR/crop_bounds.json"
    CROP_FILE_TMP="${CROP_FILE}.tmp"

    # If user manually cropped: use those bounds, optionally + DBSCAN
    if [ -f "$CROP_FILE" ] || [ -f "$CROP_FILE_TMP" ]; then
        log "使用手动筛选区域"
        CROP_INPUT="${CROP_FILE_TMP:-$CROP_FILE}"
        [ ! -f "$CROP_INPUT" ] && CROP_INPUT="$CROP_FILE"
        python -c "
import json, numpy as np
crop = json.load(open('$CROP_INPUT'))
if crop.get('lower') and crop['n'] > 10:
    center = crop['center']; lower = crop['lower']; upper = crop['upper']
    radius = float(np.linalg.norm(np.array(upper) - np.array(center)))
    json.dump({'center': center, 'radius': radius, 'lower': lower, 'upper': upper, 'manual': True},
              open('$BOUNDS_FILE', 'w'))
    print(f'Manual crop: {crop[\"n\"]} pts, box={lower} → {upper}')
else:
    print('No valid crop found, will auto-detect')
" 2>&1
    fi

    if [ ! -f "$BOUNDS_FILE" ] || [ -f "$CROP_FILE_TMP" ]; then
        # If user wants auto-DBSCAN on top of manual crop, use cropped bounds as pre-filter
        CROP_PREFILTER="$CROP_FILE"
        [ -f "$CROP_FILE_TMP" ] && CROP_PREFILTER="$CROP_FILE_TMP"
        python -c "
import struct, json, numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

path = '$SPARSE_DIR/0/points3D.bin'
pts = []
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n):
        data = f.read(43); x,y,z = struct.unpack_from('<ddd', data, 8)
        pts.append([x,y,z])
        track_len = struct.unpack('<Q', f.read(8))[0]; f.seek(track_len * 8, 1)
pts = np.array(pts)

# Pre-filter: if manual crop exists, restrict to crop box before DBSCAN
crop_pre = '${CROP_PREFILTER:-}'
if crop_pre:
    try:
        c = json.load(open(crop_pre))
        if c.get('lower') and c['n'] > 10:
            lo, hi = np.array(c['lower']), np.array(c['upper'])
            mask = (pts >= lo).all(axis=1) & (pts <= hi).all(axis=1)
            pts = pts[mask]
            print(f'Pre-filtered to crop box: {mask.sum():,}/{n} pts')
    except: pass

# DBSCAN: keep the densest spatial cluster → object (handles non-convex shapes)
cl = DBSCAN(eps=0.5, min_samples=30).fit(pts)
labels = cl.labels_
cnt = Counter(labels[labels >= 0])
if cnt:
    largest = cnt.most_common(1)[0][0]
    obj_mask = labels == largest
    obj_pts = pts[obj_mask]
    center = obj_pts.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(obj_pts - center, axis=1), 95))
    lower = obj_pts.min(axis=0); upper = obj_pts.max(axis=0)
    print(f'Object cluster: {obj_mask.sum():,}/{n} pts, {len(cnt)} clusters total')
else:
    obj_pts = pts
    center = pts.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 95))
    lower = pts.min(axis=0); upper = pts.max(axis=0)

print(f'Center: {center}')
print(f'Radius: {radius:.3f}')
print(f'Box: [{lower[0]:.2f},{lower[1]:.2f},{lower[2]:.2f}] — [{upper[0]:.2f},{upper[1]:.2f},{upper[2]:.2f}]')

json.dump({'center': center.tolist(), 'radius': radius,
           'lower': lower.tolist(), 'upper': upper.tolist()},
          open('$BOUNDS_FILE', 'w'))
" 2>&1

    log "物体边界: $BOUNDS_FILE"
rm -f "$CROP_FILE_TMP" 2>/dev/null || true
    fi
fi

# ================================================================================================
# Step 4 — TSDF Mesh Extraction + Geometry Optimization + Export
# ================================================================================================
# Step 4 runs if training output exists (even if training was skipped via -T)
TWODGS_ITER=$(find "$TWODGS_OUT/point_cloud" -name "iteration_*" -type d 2>/dev/null | sort | tail -1 || true)
if [ -z "$TWODGS_ITER" ] && [ "$SKIP_TRAINING" = true ]; then
    step "Step 4/4: 跳过网格重建 (无训练输出)"
elif [ -z "$TWODGS_ITER" ] && [ "$SKIP_TRAINING" != true ]; then
    warn "Step 4/4: 无训练输出，跳过网格重建"
else
    step "Step 4/4: 网格重建 + 导出"

TWODGS_ITER=$(find "$TWODGS_OUT/point_cloud" -name "iteration_*" -type d 2>/dev/null | sort | tail -1 || true)
if [ -z "$TWODGS_ITER" ]; then
    TWODGS_ITER="$TWODGS_OUT"
    warn "未找到 2DGS iteration 目录，使用 $TWODGS_OUT"
fi
ITER_NUM=$(basename "$TWODGS_ITER" | sed 's/iteration_//')

cd "$TWODGS_DIR"
MESH_OUT="$TWODGS_OUT/mesh_${VIDEO_NAME}.ply"
python render.py \
    -m "$TWODGS_OUT" \
    --iteration "$ITER_NUM" \
    --skip_train \
    --skip_test \
    --unbounded \
    --voxel_size "$VOXEL_SIZE" \
    --depth_trunc 8.0 \
    --sdf_trunc 0.01 \
    2>&1 | tail -5

MESH_FILE=$(find "$TWODGS_OUT" -name "mesh.ply" -type f 2>/dev/null | head -1)

# Apply spatial culling to the extracted mesh
if [ -n "$MESH_FILE" ] && [ -f "$BOUNDS_FILE" ]; then
    python -c "
import json, numpy as np, trimesh
bounds = json.load(open('$BOUNDS_FILE'))
center = np.array(bounds['center'])
radius = bounds['radius']
mesh = trimesh.load('$MESH_FILE')
dists = np.linalg.norm(mesh.vertices - center, axis=1)
mesh.update_vertices(dists < radius)
mesh.remove_unreferenced_vertices()
mesh.export('$MESH_OUT')
print(f'Filtered: {len(mesh.vertices)} verts, {len(mesh.faces)} faces')
" 2>&1 && log "  ◈ 背景剔除完成"
elif [ -n "$MESH_FILE" ]; then
    cp "$MESH_FILE" "$MESH_OUT"
    log "网格: $MESH_OUT (无剔除)"
else
    err "未找到 TSDF 输出网格"
fi

    log "  ◈ Z-up 对齐..."
# Z-axis alignment: rotate mesh to align floor plane with world Z-up
if [ -f "$MESH_OUT" ] && [ -f "$SPARSE_DIR/0/points3D.bin" ]; then
    python -c "
import struct, numpy as np, trimesh
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull

path = '$SPARSE_DIR/0/points3D.bin'
pts = []
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n):
        data = f.read(43); x,y,z = struct.unpack_from('<ddd', data, 8)
        pts.append([x,y,z])
        track_len = struct.unpack('<Q', f.read(8))[0]; f.seek(track_len * 8, 1)
pts = np.array(pts)

# Find top-k planes by area, pick the one closest to horizontal → floor
candidates = []
remaining = np.ones(len(pts), dtype=bool)
for _ in range(5):
    if remaining.sum() < 100: break
    p = pts[remaining]
    r = RANSACRegressor(residual_threshold=0.2, max_trials=500)
    r.fit(np.column_stack([p[:,0], p[:,1]]), p[:,2])
    a,b = r.estimator_.coef_
    normal = np.array([-a, -b, 1.0]); normal /= np.linalg.norm(normal)
    inliers = r.inlier_mask_
    remaining[remaining] = ~inliers
    pi = p[inliers]
    if len(pi) > 50:
        try:
            area = ConvexHull(pi[:, :2]).volume
            angle = np.degrees(np.arccos(np.clip(np.dot(normal, [0,0,1]), -1, 1)))
            candidates.append({'normal': normal, 'area': area, 'angle': angle})
        except: pass

# Among top-3 by area, pick the one with smallest Z-angle (most horizontal = floor)
candidates.sort(key=lambda x: -x['area'])
top_k = candidates[:min(3, len(candidates))]
best = min(top_k, key=lambda x: x['angle'])
best_normal = best['normal']
angle = best['angle']
print('Floor plane: area={:.0f}m2, angle={:.1f}deg (among top-{})'.format(best['area'], angle, len(top_k)))
for c in top_k:
    print('  candidate: area={:.0f}m2, angle={:.1f}deg'.format(c['area'], c['angle']))

if best_normal is not None:
    if angle > 2:
        v = np.cross(best_normal, [0,0,1]); s = np.linalg.norm(v)
        v /= s; c0 = np.dot(best_normal, [0,0,1])
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx @ vx * ((1-c0)/(s*s))
        mesh = trimesh.load('$MESH_OUT')
        mesh.vertices = (R @ mesh.vertices.T).T
        mesh.export('$MESH_OUT')
        print('Mesh rotated {:.1f}deg to Z-up'.format(angle))
" 2>&1 && log "  ◈ Z-up 对齐完成"
fi

    log "  ◈ 几何优化..."
# Geometry optimization: curvature-guided plane fitting + bottom sealing
if [ -f "$MESH_OUT" ]; then
    python -c "
import numpy as np, open3d as o3d, trimesh

mesh = trimesh.load('$MESH_OUT')
print(f'Input: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces')

# 1. Compute curvature to separate flat regions from detail
om = mesh.as_open3d
om.compute_vertex_normals()
# Mean curvature via discrete Laplacian
om.estimate_vertex_normals()
# Use vertex normal consistency as proxy for curvature
verts = np.asarray(om.vertices)
faces = np.asarray(om.triangles)
norms = np.asarray(om.vertex_normals)

# Curvature proxy: angular deviation from local neighborhood average
from scipy.spatial import cKDTree
tree = cKDTree(verts)
k = min(20, len(verts)-1)
_, idx = tree.query(verts, k=k+1)
local_norms = norms[idx[:, 1:]].mean(axis=1)  # average neighbor normals
local_norms = local_norms / (np.linalg.norm(local_norms, axis=1, keepdims=True) + 1e-8)
curvature = np.arccos(np.clip(np.abs((norms * local_norms).sum(axis=1)), -1, 1))
curv_median = np.median(curvature)

# 2. Classify: flat (low curvature) vs detail (high curvature)
CURV_THRESH = max(0.08, curv_median * 2.0)  #  radians
is_flat = curvature < CURV_THRESH
is_detail = ~is_flat
print(f'Flat regions: {is_flat.sum():,} ({is_flat.mean()*100:.0f}%), Detail: {is_detail.sum():,} ({is_detail.mean()*100:.0f}%)')

# 3. Iterative RANSAC plane fitting on flat regions only
flat_verts = verts[is_flat]
flat_idx = np.where(is_flat)[0]
remaining = np.ones(len(flat_verts), dtype=bool)
from sklearn.linear_model import RANSACRegressor

for plane_pass in range(8):
    if remaining.sum() < 100: break
    p = flat_verts[remaining]
    r = RANSACRegressor(residual_threshold=0.05, max_trials=300)
    r.fit(np.column_stack([p[:,0], p[:,1]]), p[:,2])
    a,b = r.estimator_.coef_; ci = r.estimator_.intercept_
    normal = np.array([-a, -b, 1.0]); normal /= np.linalg.norm(normal)
    inliers_local = r.inlier_mask_
    n_in = inliers_local.sum()
    if n_in < 50: break

    # Project inliers to the fitted plane
    global_idx = flat_idx[remaining][inliers_local]
    pts_proj = verts[global_idx]
    dists = pts_proj[:,2] - (a*pts_proj[:,0] + b*pts_proj[:,1] + ci)
    verts[global_idx, 2] -= dists

    remaining[remaining] = ~inliers_local
    print(f'  Plane {plane_pass+1}: {n_in} verts smoothed')

# 4. Bottom sealing: detect floor plane and cap
# Use the lowest 10% of Z as floor candidates
z = verts[:,2]
floor_z = np.percentile(z, 5)
floor_mask = z < floor_z + 0.1  # within 10cm of floor
floor_verts = verts[floor_mask]
if len(floor_verts) > 50:
    # Find boundary edges near the floor
    om2 = o3d.geometry.TriangleMesh()
    om2.vertices = o3d.utility.Vector3dVector(verts)
    om2.triangles = o3d.utility.Vector3iVector(faces)
    om2.compute_vertex_normals()

    # Get boundary edges (edges with only one adjacent face)
    edges = {}
    for f in faces:
        for e in [(f[0],f[1]), (f[1],f[2]), (f[2],f[0])]:
            e_key = (min(e), max(e))
            edges[e_key] = edges.get(e_key, 0) + 1
    boundary_edges = [e for e, c in edges.items() if c == 1]

    # Find boundary vertices near the floor
    boundary_verts = set()
    for e in boundary_edges:
        boundary_verts.add(e[0])
        boundary_verts.add(e[1])
    boundary_verts = np.array(list(boundary_verts))
    boundary_z = verts[boundary_verts, 2]
    floor_boundary = boundary_verts[boundary_z < floor_z + 0.15]

    if len(floor_boundary) > 10:
        # Triangulate the floor cap
        from scipy.spatial import Delaunay
        xy = verts[floor_boundary, :2]
        # Project boundary to 2D and triangulate
        try:
            tri = Delaunay(xy)
            # Keep only triangles that face downward (or are roughly horizontal)
            new_faces = []
            for t in tri.simplices:
                v0, v1, v2 = floor_boundary[t[0]], floor_boundary[t[1]], floor_boundary[t[2]]
                # Move these vertices to the floor plane
                verts[v0, 2] = min(verts[v0, 2], floor_z)
                verts[v1, 2] = min(verts[v1, 2], floor_z)
                verts[v2, 2] = min(verts[v2, 2], floor_z)
                new_faces.append([v0, v1, v2])

            if new_faces:
                all_faces = np.vstack([faces, np.array(new_faces)])
                faces = all_faces
                print(f'Bottom sealed: {len(new_faces)} cap faces')
        except:
            # Fallback: simple planar cap using convex hull
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(xy)
                new_faces = [[floor_boundary[t[0]], floor_boundary[t[1]], floor_boundary[t[2]]]
                            for t in hull.simplices]
                for v0, v1, v2 in new_faces:
                    verts[v0, 2] = min(verts[v0, 2], floor_z)
                    verts[v1, 2] = min(verts[v1, 2], floor_z)
                    verts[v2, 2] = min(verts[v2, 2], floor_z)
                all_faces = np.vstack([faces, np.array(new_faces)])
                faces = all_faces
                print(f'Bottom sealed (convex): {len(new_faces)} cap faces')
            except:
                print('Bottom sealing skipped (degenerate boundary)')

# 5. Rebuild mesh with cleaned vertices
clean_om = o3d.geometry.TriangleMesh()
clean_om.vertices = o3d.utility.Vector3dVector(verts)
clean_om.triangles = o3d.utility.Vector3iVector(faces)
clean_om.remove_unreferenced_vertices()
clean_om.remove_degenerate_triangles()

# Convert back to trimesh for export
out_verts = np.asarray(clean_om.vertices)
out_faces = np.asarray(clean_om.triangles)
import json
clean_mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces)
clean_mesh.export('$MESH_OUT')
is_watertight = clean_mesh.is_watertight
print(f'Output: {len(out_verts):,} verts, {len(out_faces):,} faces, watertight={is_watertight}')
json.dump({'method': '2dgs', 'mesh': {
    'vertices': len(out_verts), 'faces': len(out_faces), 'watertight': is_watertight,
    'flat_pct': float(is_flat.mean()*100), 'detail_pct': float(is_detail.mean()*100),
    'planes_smoothed': plane_pass}},
    open('${TWODGS_OUT}/reconstruction.json', 'w'))
" 2>&1 && log "  ◈ 几何优化完成"
fi

if [ "$SKIP_USDZ" = true ]; then
    log "  ◈ 跳过 USDA 导出"
elif [ -f "$MESH_OUT" ]; then
    log "  ◈ USDA 导出..."
# PLY → USDA (Isaac Lab native format, with collision API)
    USDA_MESH="${MESH_OUT%.ply}.usda"
    python -c "
from pxr import Usd, UsdGeom, UsdPhysics
import trimesh, numpy as np

m = trimesh.load('$MESH_OUT')

stage = Usd.Stage.CreateNew('$USDA_MESH')
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
mesh = UsdGeom.Mesh.Define(stage, '/World/mesh')
mesh.CreatePointsAttr().Set(m.vertices.astype(float).tolist())
mesh.CreateFaceVertexCountsAttr().Set([3] * len(m.faces))
mesh.CreateFaceVertexIndicesAttr().Set(m.faces.flatten().tolist())

# Collision + rigid body API for Isaac Lab
UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
body = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
body.CreateKinematicEnabledAttr().Set(True)

stage.GetRootLayer().Save()
import os
print(f'{os.path.getsize(\"$USDA_MESH\")/1024/1024:.0f} MB')
" 2>&1 && log "USDA: $USDA_MESH"
fi  # SKIP_USDZ

# Cleanup intermediate files
rm -f "$DATABASE_PATH" 2>/dev/null || true              # COLMAP DB (~2GB, no longer needed)
rm -f "$MESH_FILE" 2>/dev/null || true                   # raw TSDF mesh (filtered version kept)
rm -f "$BOUNDS_FILE" "$CROP_FILE" "$CROP_FILE_TMP" 2>/dev/null || true
find "$TWODGS_OUT" -name "mesh.ply" -delete 2>/dev/null || true  # any remaining raw meshes

# ================================================================================================
fi
# Done
# ================================================================================================
ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_M=$((ELAPSED / 60))
ELAPSED_S=$((ELAPSED % 60))

RECON_JSON="${TWODGS_OUT}/reconstruction.json"
if [ -f "$RECON_JSON" ]; then
    # Add colmap info + timing
    python -c "
import json
r = json.load(open('$RECON_JSON'))
r['colmap'] = {'registered': ${NUM_IMAGES_REG:-0}, 'total': $FRAME_COUNT}
r['timing'] = {'elapsed_s': $ELAPSED}
json.dump(r, open('$RECON_JSON', 'w'), indent=2)
" 2>/dev/null
    MESH_VERTS=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(f\"{d['mesh']['vertices']:,}\")" 2>/dev/null)
    MESH_FACES=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(f\"{d['mesh']['faces']:,}\")" 2>/dev/null)
    MESH_WATER=$(python -c "import json;d=json.load(open('$RECON_JSON'));print('是' if d['mesh']['watertight'] else '否')" 2>/dev/null)
    MESH_FLAT=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(f\"{d['mesh']['flat_pct']:.0f}%\")" 2>/dev/null)
    MESH_PLANES=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(d['mesh']['planes_smoothed'])" 2>/dev/null)
fi

# Cleanup
find "$TWODGS_OUT" -name "train.log" -delete 2>/dev/null || true

step "Pipeline 完成 (${ELAPSED_M}m ${ELAPSED_S}s)"

echo ""
echo "╔══════════════════════════════════════════╗"
printf "║  📷 图片      %-4s 帧  ║\n" "$FRAME_COUNT"
printf "║  📐 COLMAP    %-4s 帧注册 ║\n" "${NUM_IMAGES_REG:-?}"
[ -n "${MESH_VERTS:-}" ] && printf "║  🔺 顶点      %-10s ║\n" "$MESH_VERTS"
[ -n "${MESH_FACES:-}" ] && printf "║  🔺 三角面    %-10s ║\n" "$MESH_FACES"
[ -n "${MESH_WATER:-}" ] && printf "║  💧 封闭      %-10s ║\n" "$MESH_WATER"
[ -n "${MESH_FLAT:-}" ] && printf "║  📏 平滑      %-10s ║\n" "${MESH_FLAT} (${MESH_PLANES:-0} 平面)"
echo "╠══════════════════════════════════════════╣"
printf "║  ⏱  总耗时    %-5s     ║\n" "${ELAPSED_M}m${ELAPSED_S}s"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "输出目录: $TWODGS_OUT"
[ -f "${MESH_OUT:-}" ] && echo "★ PLY : $MESH_OUT"
[ -n "${USDA_MESH:-}" ] && echo "★ USDA: $USDA_MESH"
echo ""
echo "续跑: $0 -v '$VIDEO_PATH' -c -S"
