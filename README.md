# ComfyUI-AutoShot 插件

基于深度学习的视频镜头自动分割与关键帧提取 ComfyUI 插件。

核心算法改造自 [Shot_Detection](https://github.com/PhucNguyenLamp/Shot_Detection) 仓库，使用 **TransNetV2Supernet** 模型进行镜头边界检测。

## 功能特性

- ✅ **深度学习镜头检测**：基于 TransNetV2Supernet 模型，准确率远高于简单帧差算法
- ✅ **关键帧自动提取**：每个镜头自动提取一帧关键帧（首帧/中间帧/末帧可选）
- ✅ **前端可视化多选**：节点下方预览所有关键帧缩略图，点击选择需要的帧
- ✅ **标准 ComfyUI 输出**：输出标准 IMAGE 图片列表，可直接接入后续工作流
- ✅ **参数可调**：检测阈值、最小镜头长度、关键帧位置均可自定义

<img width="1918" height="898" alt="image" src="https://github.com/user-attachments/assets/967e1d5d-b225-48d1-910c-7104394afefb" />

## 安装方法

### 1. 下载插件

将本插件文件夹 `ComfyUI-AutoShot` 复制到 ComfyUI 的 `custom_nodes` 目录下：

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-AutoShot/
        ├── __init__.py
        ├── nodes.py
        ├── autoshot_core/
        │   ├── __init__.py
        │   ├── model.py
        │   ├── supernet.py
        │   ├── linear.py
        │   ├── utils.py
        │   └── model_weight/
        │       └── ckpt_0_200_0.pth
        └── web/
            └── js/
                └── autoshot_ui.js
```

### 2. 下载模型权重（重要！）

由于模型权重文件较大（约 57MB），使用 Git LFS 管理。

**方法一：使用 Git LFS（推荐）**

如果是通过 git clone 下载的插件，请确保已安装 git-lfs，然后执行：

```bash
cd ComfyUI-AutoShot
git lfs pull
```

**方法二：手动下载**

从原仓库 Releases 或 HuggingFace 下载模型权重文件 `ckpt_0_200_0.pth`，放到：
`ComfyUI-AutoShot/autoshot_core/model_weight/` 目录下。

> ⚠️ 注意：如果模型权重文件只有几百字节，说明是 Git LFS 指针文件，不是真实权重。
> 请确保下载完整的 57MB 权重文件，否则插件无法运行。

### 3. 安装依赖

插件依赖以下 Python 库（ComfyUI 环境通常已预装）：

- torch
- numpy
- opencv-python
- einops
- tqdm

如果缺少依赖，可在 ComfyUI 虚拟环境中执行：

```bash
pip install torch numpy opencv-python einops tqdm
```

### 4. 重启 ComfyUI

重启 ComfyUI 服务，插件将自动加载。

## 使用方法

### 节点位置

在节点搜索框输入 `Auto Shot Detection` 或在分类 `Video/Analysis` 中找到节点。

### 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| video | VIDEO | - | 视频输入，从 Load Video 节点接入，支持 mp4/avi/mov 等常见格式 |
| threshold | FLOAT | 0.5 | 镜头边界检测阈值（0.1~0.9），值越大越保守，越不容易判定为镜头切换 |
| min_shot_frames | INT | 10 | 最小镜头帧数，过滤掉极短的闪镜 |
| keyframe_position | ENUM | middle | 每个镜头取哪一帧作为关键帧：first（首帧）/ middle（中间帧）/ last（末帧） |
| selected_indices | STRING | "" | 手动选择的关键帧帧号（逗号分隔），留空则输出全部。前端预览面板点击选择后自动填充。 |

### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| 关键帧图片列表 | IMAGE (LIST) | 标准 ComfyUI IMAGE 格式图片列表，可直接接入后续节点 |
| 关键帧帧号列表 | INT_LIST | 每个关键帧对应的原始视频帧号 |
| 预览图路径列表 | STRING_LIST | 预览图文件路径列表 |

### 前端多选操作

1. **运行节点**：填入视频路径，点击 Queue Prompt 运行
2. **查看预览**：节点下方会显示所有检测到的关键帧缩略图
3. **选择帧**：点击缩略图切换选中状态（蓝色边框 + ✓ 标记 = 选中）
4. **批量操作**：使用「全选」「取消全选」按钮批量操作
5. **应用选择**：点击「应用选择」按钮重新运行节点，仅输出选中的关键帧

## 工作流示例

### 典型用法：Load Video → 镜头分割 → 关键帧 → 后续处理

```
[Load Video] → [Auto Shot Detection] → [Save Image] / [其他处理节点]
```

1. 使用 Load Video 节点加载视频文件
2. 将视频输出接入 Auto Shot Detection 节点
3. 自动检测所有镜头边界，提取关键帧
4. 在前端预览面板中选择需要的关键帧
5. 点击「应用选择」重新运行
6. 输出的 IMAGE 列表可直接接入 Save Image、放大、风格化等后续节点

## 技术说明

### 算法原理

基于 TransNetV2Supernet 深度学习模型，通过分析视频帧序列的视觉特征变化来检测镜头边界。

模型输入：48×27 分辨率的 RGB 帧序列
模型输出：每帧是镜头边界的概率（0~1）
后处理：通过阈值二值化 + 边界检测，得到每个镜头的起止帧

### 性能说明

- 模型推理在 GPU 上运行较快，CPU 上可能较慢
- 视频越长，处理时间越长（主要耗时在帧提取和模型推理）
- 首次运行会加载模型权重（约 57MB），后续运行复用模型实例

## 常见问题

### Q: 报错 "模型权重文件不存在或未正确下载"
A: 请检查 `autoshot_core/model_weight/ckpt_0_200_0.pth` 文件大小。如果只有几百字节，说明是 LFS 指针文件，需要执行 `git lfs pull` 下载真实权重。

### Q: 检测到的镜头太多/太少怎么办？
A: 调整 `threshold` 参数：
- 镜头太多（误检多）→ 增大阈值（如 0.6、0.7）
- 镜头太少（漏检多）→ 减小阈值（如 0.3、0.4）

### Q: 如何过滤掉极短的闪镜？
A: 增大 `min_shot_frames` 参数，比如设置为 20 或 30，帧数少于该值的镜头会被过滤掉。

### Q: 前端预览图不显示怎么办？
A: 请确保 ComfyUI 能正常访问临时目录。预览图保存在 ComfyUI 的 temp 目录下。

## 致谢

核心算法来自 [Shot_Detection](https://github.com/PhucNguyenLamp/Shot_Detection) 仓库，
基于 TransNetV2 模型改进。

## 许可证

遵循原仓库许可证。
