# Revo3 Hand MuJoCo Self-Collision Toolkit

URDF to MJCF 转换 + 自碰撞检测可视化工具集。

## 依赖

- Python 3.8+
- mujoco >= 3.0
- numpy

## 脚本说明

### 1. `urdf_to_mjcf_visualization.py` — 保留 visual 网格的转换

删除 URDF 中所有 `<collision>`，将 `<visual>` mesh 提升为 MJCF 的 `<geom>`（仅用于可视化，不参与碰撞检测），移除惯量张量只保留 mass。

```bash
python urdf_to_mjcf_visualization.py revo3_system/urdf/revo3_right.urdf
# 输出: revo3_system/urdf/revo3_right.xml

# 指定输出路径
python urdf_to_mjcf_visualization.py revo3_system/urdf/revo3_right.urdf -o output.xml
```

输出的 MJCF 中：
- 22 个 geom（每个 link 对应一个 visual mesh）
- geom 标记为 visual-only（`contype=0, conaffinity=0, group=1`）
- 无惯量张量，mass 写在 geom 上

### 2. `urdf_to_mjcf_collision.py` — 保留 collision 网格的转换

删除 URDF 中所有 `<visual>`，保留 `<collision>` mesh 作为 MJCF 的 `<geom>`（用于碰撞检测），移除惯量张量只保留 mass。

```bash
python urdf_to_mjcf_collision.py revo3_system/urdf/revo3_right.urdf
# 输出: revo3_system/urdf/revo3_right.xml

# 指定输出路径
python urdf_to_mjcf_collision.py revo3_system/urdf/revo3_right.urdf -o output_collision.xml
```

输出的 MJCF 中：
- 49 个 geom（collision mesh 数量多于 visual）
- geom 参与碰撞检测（默认 `contype=1, conaffinity=1`）
- 无惯量张量，mass 写在 geom 上

### 3. `mujoco_self_collision_viewer.py` — 自碰撞检测与可视化

加载 MJCF XML 文件，逐关节扫过整个运动范围，检测并可视化自碰撞接触点和接触力。

```bash
# 带 viewer 可视化（需要 GUI 环境）
python mujoco_self_collision_viewer.py revo3_system/urdf/revo3_right_collision.xml

# 无头模式，只输出碰撞数据
python mujoco_self_collision_viewer.py revo3_system/urdf/revo3_right_collision.xml --no-viewer

# 导出碰撞记录到 CSV
python mujoco_self_collision_viewer.py revo3_system/urdf/revo3_right_collision.xml --no-viewer --csv contacts.csv

# 调整参数
python mujoco_self_collision_viewer.py revo3_system/urdf/revo3_right_collision.xml \
    --steps-per-joint 120 \
    --fps 30 \
    --top-k 5
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `xml` | (必填) | MJCF .xml 文件路径 |
| `--steps-per-joint` | 60 | 每个关节的扫描步数 |
| `--fps` | 60 | viewer 刷新率 |
| `--print-every` | 12 | 每 N 帧打印一次最强碰撞 |
| `--top-k` | 3 | CSV 中每帧记录的最大碰撞数 |
| `--csv` | 无 | 碰撞记录 CSV 输出路径 |
| `--no-viewer` | 否 | 无头模式，不启动 viewer |

## 典型工作流

```bash
# 步骤 1: URDF 转 MJCF（保留 collision 网格）
python urdf_to_mjcf_collision.py revo3_system/urdf/revo3_right.urdf

# 步骤 2: 运行自碰撞检测
python mujoco_self_collision_viewer.py revo3_system/urdf/revo3_right_collision.xml

# 可选: 同时生成 visual 版本用于渲染查看
python urdf_to_mjcf_visualization.py revo3_system/urdf/revo3_right.urdf \
    -o revo3_system/urdf/revo3_right_visual.xml
```


将tip的质量和惯量都调为0；将palm的质量和惯量都调为0