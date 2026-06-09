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
MAX_IMAGE_SIZE=1920
GPU_ID=0
TRAIN_ITERATIONS=60000
DOWNSAMPLE_FACTOR=2
SKIP_USDZ=false
SKIP_FFMPEG=false
SKIP_COLMAP=false
SKIP_TRAINING=false
SKIP_CLAHE=false

CONDA_ENV="vid2sim"
THREEDGRUT_DIR="$(cd "$(dirname "$0")" && pwd)/3dgrut"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()    { echo -e "${GREEN}[*]${NC} $(date '+%H:%M:%S') $*"; }
warn()   { echo -e "${YELLOW}[!]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[X]${NC} $(date '+%H:%M:%S') $*"; }
step()   { echo -e "\n${BLUE}▶${NC} $(date '+%H:%M:%S') $*"; }

usage() { head -36 "$0" | tail -20; exit 0; }

while getopts "v:o:n:f:s:g:i:d:ucSTh" opt; do
    case $opt in
        v) VIDEO_PATH="$OPTARG" ;;  o) OUTPUT_DIR="$OPTARG" ;;
        n) EXPERIMENT_NAME="$OPTARG" ;; f) FPS="$OPTARG" ;;
        s) MAX_IMAGE_SIZE="$OPTARG" ;; g) GPU_ID="$OPTARG" ;;
        i) TRAIN_ITERATIONS="$OPTARG" ;; d) DOWNSAMPLE_FACTOR="$OPTARG" ;;
        u) SKIP_USDZ=true ;;  A) SKIP_CLAHE=true ;;  c) SKIP_FFMPEG=true ;;
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

# CLAHE 对比度增强 — 金属/透明表面特征更明显
if [ "$SKIP_FFMPEG" = false ] || [ "$FRAME_COUNT" -gt 0 ]; then
    CLAHE_FLAG="${IMAGE_DIR}/.clahe_done"
    if [ "$SKIP_CLAHE" = true ]; then
        touch "$CLAHE_FLAG" 2>/dev/null || true
        log "跳过 CLAHE"
    elif [ ! -f "$CLAHE_FLAG" ]; then
        log "CLAHE 增强..."
        python -c "
import cv2
from pathlib import Path
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
for p in sorted(Path('$IMAGE_DIR').glob('*.jpg')):
    img = cv2.imread(str(p))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    cv2.imwrite(str(p), cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f'{sum(1 for _ in Path(\"$IMAGE_DIR\").glob(\"*.jpg\"))} images')
" 2>&1 && touch "$CLAHE_FLAG" && log "CLAHE 完成"
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
        --SiftExtraction.max_num_features 65536 --SiftExtraction.peak_threshold 0.0033 \
        --SiftExtraction.edge_threshold 5 --SiftExtraction.num_octaves 5 --SiftExtraction.estimate_affine_shape 0 \
        --SiftExtraction.domain_size_pooling 0 2>&1 | tail -2

    [ ! -f "$DATABASE_PATH" ] && { err "特征提取失败"; exit 1; }

    colmap vocab_tree_matcher \
        --database_path "$DATABASE_PATH" --FeatureMatching.use_gpu 1 --FeatureMatching.max_num_matches 65536 \
        --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 --SiftMatching.cross_check 1 2>&1 | tail -2

    mkdir -p "$SPARSE_DIR"

    # Mapper + 自动重试: 对困难场景，COLMAP 初始像对选取有随机性
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

        # 选注册图像最多的 sparse/N/
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
            log "注册率达标 (≥50%)，继续训练"
            break
        fi
        warn "注册率 < 50%，重试..."
    done

    if [ ! -f "$SPARSE_DIR/0/cameras.bin" ] || [ ! -f "$SPARSE_DIR/0/images.bin" ]; then
        err "COLMAP 失败: 多次重试后仍无有效重建"
        exit 1
    fi
    if python -c "exit(0 if $REG_RATIO < $REG_THRESHOLD else 1)"; then
        warn "多次重试后注册率仍 < 50%，但继续训练 (可用 $NUM_IMAGES_REG 帧)"
    fi
fi

# ================================================================================================
# Step 3 — 3DGUT train
# ================================================================================================
RUNS_DIR="$OUTPUT_DIR/runs"

# Check for completed checkpoint before deciding to train
PREV_CKPT=$(find "$RUNS_DIR/$EXPERIMENT_NAME" -name "ckpt_last.pt" -type f 2>/dev/null | sort | tail -1 || true)
if [ -n "$PREV_CKPT" ] && [ -d "$(dirname "$PREV_CKPT")/ours_${TRAIN_ITERATIONS}" ]; then
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
        "checkpoint.iterations=[10000,20000,30000,40000,50000,60000]" \
        "render.particle_radiance_sph_degree=4" \
        "model.progressive_training.max_n_features=4" \
        "model.progressive_training.increase_frequency=500" \
        "optimizer.params.features_specular.lr=0.001" \
        "loss.use_l2=true" \
        "loss.lambda_l2=0.3" \
        "post_processing.method=ppisp" \
        "post_processing.n_distillation_steps=5000" \
        "strategy.density_decay.start_iteration=500" \
        "strategy.density_decay.end_iteration=60000" \

        "strategy.prune_scale.start_iteration=500" \
        "strategy.prune_scale.end_iteration=30000" \
        "strategy.prune_scale.frequency=500" \
        "strategy.prune_scale.threshold=1.0" \
        "num_workers=4")
    [ "$SKIP_USDZ" = false ] && TRAIN_CMD+=("export_usd.enabled=true" "export_usd.format=nurec" "export_usd.apply_normalizing_transform=false")

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
# Step 4 — Clean Gaussians + Export USDZ
# ================================================================================================
if [ "$SKIP_USDZ" = false ]; then
    step "Step 4/4: 清理 + USDZ 导出"
    [ -z "${TRAIN_OUTDIR:-}" ] && TRAIN_OUTDIR=$(find "$RUNS_DIR/$EXPERIMENT_NAME" -maxdepth 2 -name "ckpt_last.pt" -type f -printf "%h\n" 2>/dev/null | sort | tail -1 || true)
    CKPT_SRC="${TRAIN_OUTDIR}/ckpt_last.pt"
    CKPT_CLEAN="${TRAIN_OUTDIR}/ckpt_clean.pt"
    USDZ_FILE="${TRAIN_OUTDIR}/scene_nurec.usdz"

    if [ -f "$CKPT_SRC" ]; then
        log "清理浮空高斯..."
        python -c "
import torch, numpy as np, struct
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.spatial import cKDTree

ckpt = torch.load('$CKPT_SRC', map_location='cpu', weights_only=False)
pos = ckpt['positions'].detach().numpy()
density = ckpt['density'].detach().numpy().squeeze()
scales = ckpt['scale'].detach().numpy()
N = len(pos)

# 1. Opacity filter: remove nearly invisible Gaussians
opacity = 1/(1+np.exp(-density))
keep = opacity >= 0.008

# 2. Scale filter: remove extremely stretched Gaussians
scale_norm = np.linalg.norm(scales, axis=1)
keep &= scale_norm < np.median(scale_norm) * 3

# 3. Z-range: remove ceiling/underground floaters
z = pos[:,2]
z05, z98 = np.percentile(z, 5), np.percentile(z, 98)
keep &= (z >= z05) & (z <= z98)

# 4. DBSCAN spatial clustering: keep largest connected component
n_sample = min(N//10, 100000)
idx = np.random.choice(N, n_sample, replace=False)
sample = pos[idx]
cl = DBSCAN(eps=1.0, min_samples=30).fit(sample)
ls = cl.labels_
cnt = Counter(ls[ls>=0])
if cnt:
    largest = cnt.most_common(1)[0][0]
    tree = cKDTree(sample)
    _, nn = tree.query(pos, k=3)
    fl = np.array([ls[nn[i][ls[nn[i]]>=0][0]] if np.any(ls[nn[i]]>=0) else -1 for i in range(N)])
    keep &= fl == largest

for k in ['positions','rotation','scale','density','features_albedo','features_specular']:
    if k in ckpt: ckpt[k] = ckpt[k][keep]
pos_f = ckpt['positions'].detach().numpy()

# 5. Z-axis alignment: rotate Gaussians to make floor Z-up
from sklearn.linear_model import RANSACRegressor
from scipy.spatial import ConvexHull

# Read COLMAP points and find largest plane → floor normal
cpts = []
with open('$SPARSE_DIR/0/points3D.bin', 'rb') as f:
    cn = struct.unpack('<Q', f.read(8))[0]
    for _ in range(cn):
        d = f.read(43); cx,cy,cz = struct.unpack_from('<ddd', d, 8)
        cpts.append([cx,cy,cz])
        tl = struct.unpack('<Q', f.read(8))[0]; f.seek(tl * 8, 1)
cpts = np.array(cpts)

best_area, best_normal = 0, None
crem = np.ones(len(cpts), dtype=bool)
for _ in range(5):
    if crem.sum() < 100: break
    cp = cpts[crem]
    rr = RANSACRegressor(residual_threshold=0.2, max_trials=500)
    rr.fit(np.column_stack([cp[:,0], cp[:,1]]), cp[:,2])
    aa,bb = rr.estimator_.coef_
    nrm = np.array([-aa, -bb, 1.0]); nrm /= np.linalg.norm(nrm)
    inl = rr.inlier_mask_
    crem[crem] = ~inl
    cpi = cp[inl]
    if len(cpi) > 50:
        try:
            area = ConvexHull(cpi[:, :2]).volume
            if area > best_area: best_area, best_normal = area, nrm
        except: pass

if best_normal is not None:
    angle = np.degrees(np.arccos(np.clip(np.dot(best_normal, [0,0,1]), -1, 1)))
    if angle > 2:
        v = np.cross(best_normal, [0,0,1]); s = np.linalg.norm(v)
        v = v / s; c0 = np.dot(best_normal, [0,0,1])
        vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R = np.eye(3) + vx + vx @ vx * ((1-c0)/(s*s))
        ckpt['positions'] = torch.from_numpy((R @ pos_f.T).T).float()
        # Transform rotation quaternions via numpy: q' = q_R * q (wxyz format)
        from scipy.spatial.transform import Rotation as Rot
        q_R = Rot.from_matrix(R).as_quat()  # scipy: [x,y,z,w]
        rw, rx, ry, rz = q_R[3], q_R[0], q_R[1], q_R[2]  # → wxyz
        qo = ckpt['rotation'].detach().numpy()  # [N,4] wxyz
        ow, ox, oy, oz = qo[:,0], qo[:,1], qo[:,2], qo[:,3]
        nw = rw*ow - rx*ox - ry*oy - rz*oz
        nx = rw*ox + rx*ow + ry*oz - rz*oy
        ny = rw*oy - rx*oz + ry*ow + rz*ox
        nz = rw*oz + rx*oy - ry*ox + rz*ow
        ckpt['rotation'] = torch.from_numpy(np.stack([nw, nx, ny, nz], axis=1)).float()
        print('Floor normal Z={:.3f}, rotated {:.1f}deg to Z-up'.format(best_normal[2], angle))
        import json
        json.dump({'method': '3dgut', 'transform': {'angle_deg': angle, 'floor_z': float(np.percentile(pos_f[:,2], 5)), 'ceiling_z': 0}},
                  open('${TRAIN_OUTDIR}/reconstruction.json', 'w'))
    else:
        print('Floor already Z-aligned ({:.1f}deg)'.format(angle))
        import json
        json.dump({'method': '3dgut', 'transform': {'angle_deg': angle, 'floor_z': float(np.percentile(pos_f[:,2], 5)), 'ceiling_z': 0}},
                  open('${TRAIN_OUTDIR}/reconstruction.json', 'w'))
else:
    import json
    json.dump({'method': '3dgut', 'transform': {'angle_deg': 0.0, 'floor_z': float(np.percentile(pos_f[:,2], 5)), 'ceiling_z': 0}},
              open('${TRAIN_OUTDIR}/reconstruction.json', 'w'))

for k in list(ckpt.keys()):
    if isinstance(ckpt[k], torch.Tensor) and not isinstance(ckpt[k], torch.nn.Parameter):
        ckpt[k] = torch.nn.Parameter(ckpt[k])
torch.save(ckpt, '$CKPT_CLEAN')
print(f'{keep.sum():,} / {N:,} ({keep.mean()*100:.0f}%)')
" 2>&1 && log "清理完成: $CKPT_CLEAN"

        # Export cleaned NuRec USDZ
        log "导出 NuRec USDZ ..."
        cd "$THREEDGRUT_DIR"
        if python -m threedgrut.export.scripts.export_usd \
            --checkpoint "$CKPT_CLEAN" --output "$USDZ_FILE" --format nurec \
            --no-transform --no-cameras --no-background 2>&1; then
            log "USDZ: $USDZ_FILE ($(du -h "$USDZ_FILE" | cut -f1))"
        else
            warn "NuRec 导出失败, 使用训练导出的 USDZ"
            EXPORT_USDZ=$(find "$TRAIN_OUTDIR" -maxdepth 1 -name "export_*.usdz" -print -quit 2>/dev/null)
            [ -n "$EXPORT_USDZ" ] && cp "$EXPORT_USDZ" "$USDZ_FILE" && log "USDZ: $USDZ_FILE (from training export)"
        fi
        rm -f "$CKPT_CLEAN"
        # Note: do NOT delete export_*.usdz here — scene_nurec.usdz may be a copy of it
    else
        warn "未找到 checkpoint, 跳过清理导出"
    fi
fi

# ================================================================================================

# ================================================================================================
# Ground collision — same rotation as visual scene, read from rotation.json
# ================================================================================================
GROUND_FILE="${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}/ground_collision.usda"
RECON_JSON="${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}/reconstruction.json"
if [ -f "$SPARSE_DIR/0/points3D.bin" ] && [ ! -f "$GROUND_FILE" ]; then
    python -c "
import struct, numpy as np, json
from pxr import Usd, UsdGeom, UsdPhysics, Gf

path = '$SPARSE_DIR/0/points3D.bin'
pts = []
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n):
        data = f.read(43); x,y,z = struct.unpack_from('<ddd', data, 8)
        pts.append([x,y,z])
        track_len = struct.unpack('<Q', f.read(8))[0]; f.seek(track_len * 8, 1)
pts = np.array(pts)

# Apply same rotation as visual scene (if rotation.json exists)
R = np.eye(3)
try:
    rot_data = json.load(open('$RECON_JSON'))
    R = np.array(rot_data['R'])
except: pass
pts = (R @ pts.T).T

zs = np.sort(pts[:,2])
floor_z = zs[len(zs)//20]
ceiling_z = zs[-len(zs)//20]
xmin, xmax = np.percentile(pts[:,0], 5), np.percentile(pts[:,0], 95)
ymin, ymax = np.percentile(pts[:,1], 5), np.percentile(pts[:,1], 95)
cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
w = (xmax - xmin) + 4.0; d = (ymax - ymin) + 4.0

stage = Usd.Stage.CreateNew('$GROUND_FILE')
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
ground = UsdGeom.Cube.Define(stage, '/World/ground')
ground.AddScaleOp().Set(Gf.Vec3f(w/2, d/2, 0.02))
ground.AddTranslateOp().Set(Gf.Vec3f(cx, cy, floor_z))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
body = UsdPhysics.RigidBodyAPI.Apply(ground.GetPrim())
body.CreateKinematicEnabledAttr().Set(True)
stage.GetRootLayer().Save()
print('Floor Z={:.2f}  Room H={:.1f}m  Ground={:.1f}x{:.1f}m'.format(floor_z, ceiling_z-floor_z, w, d))
" 2>&1 && log "地面碰撞: $GROUND_FILE"
fi
# Done
# ================================================================================================
ELAPSED=$(( $(date +%s) - START_TIME ))
ELAPSED_M=$((ELAPSED / 60))
ELAPSED_S=$((ELAPSED % 60))

# Merge metrics into reconstruction.json
RECON_JSON="${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}/reconstruction.json"
METRICS_FILE=$(find "${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}" -name "metrics.json" -type f 2>/dev/null | tail -1 || true)
if [ -f "$METRICS_FILE" ] && [ -f "$RECON_JSON" ]; then
    python -c "
import json
r = json.load(open('$RECON_JSON'))
m = json.load(open('$METRICS_FILE'))
r['quality'] = {'psnr': round(m['mean_psnr'],1), 'ssim': round(m['mean_ssim'],3), 'lpips': round(m['mean_lpips'],3)}
r['colmap'] = {'registered': ${NUM_IMAGES_REG:-0}, 'total': $FRAME_COUNT}
r['timing'] = {'elapsed_s': $ELAPSED}
json.dump(r, open('$RECON_JSON', 'w'), indent=2)
"
    rm -f "$METRICS_FILE"
fi

# Cleanup: remove non-essential artifacts
find "${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}" -name "train.log" -delete 2>/dev/null || true
find "${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}" -name "parsed.yaml" -delete 2>/dev/null || true
find "${TRAIN_OUTDIR:-$RUNS_DIR/$EXPERIMENT_NAME}" -name "events.out.*" -delete 2>/dev/null || true

# Read metrics for summary
if [ -f "$RECON_JSON" ]; then
    PSNR=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(d['quality']['psnr'])" 2>/dev/null || echo "")
    SSIM=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(d['quality']['ssim'])" 2>/dev/null || echo "")
    LPIPS=$(python -c "import json;d=json.load(open('$RECON_JSON'));print(d['quality']['lpips'])" 2>/dev/null || echo "")
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
echo "输出: $RECON_JSON"
echo "续跑: $0 -v '$VIDEO_PATH' -c -S"
