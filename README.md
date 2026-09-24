<div align="center">

# 🏸 BadCoach-AI · 羽球教练 AI

### 单双打自适应的羽毛球比赛视频分析系统

[![GitHub](https://img.shields.io/badge/GitHub-optomao--BadCoach--AI-181717?style=flat-square&logo=github)](https://github.com/optomao/BadCoach-AI)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/Web-React%20%2B%20Vite-61DAFB?style=flat-square&logo=react&logoColor=111827)](web/frontend/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](web/api/)
[![License](https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=flat-square)](LICENSE)

**本地优先的羽毛球视频分析工具：自适应单双打球员追踪、姿态检测、羽毛球轨迹、球场坐标映射，生成可复盘的标注视频与统计数据。**

[English](README_EN.md) · [快速开始](#-快速开始) · [Web Demo](#-web-demo) · [效果展示](#-效果展示) · [路线图](#-路线图)

</div>

---

## ✨ 项目简介

**BadCoach-AI**（羽球教练 AI）面向羽毛球训练、比赛复盘和计算机视觉研究场景。输入一段比赛视频，完成一次球场四点标定后，系统会自适应识别单打或双打模式，将画面中的检测结果转换为视频叠加层、球场轨迹、运动统计和结构化数据。

系统支持单打（A/B 两位球员）和双打（A/B/C/D 四位球员）两种模式，通过前 30 帧暖场自动检测或手动选择模式。每位球员拥有独立的统计面板，显示移动距离、速度、回合数等实时数据。

> 基于 [AI-YuJian-AI](https://github.com/lzylovec/AI-YuJian-AI)（羽见 AI）二次开发，原项目由 [lzylovec](https://github.com/lzylovec) 创建。

## 🎬 效果展示

<div align="center">

![BadCoach-AI 分析效果预览](assets/demo.gif)

*完整演示视频：[assets/demo.mp4](assets/demo.mp4)*

</div>

| 球员位置热力图 | 球员位置散点图 |
| :---: | :---: |
| ![球员位置热力图](assets/match_heatmap.png) | ![球员位置散点图](assets/match_scatter.png) |

![球场标定示例](assets/label_court_example.png)

### 视频面板布局

统计面板靠画面右侧单列纵向排列，不遮挡球场区域：

- **单打**：A 球员（上侧）+ B 球员（下侧），两块面板
- **双打**：A/B 球员（上侧）+ C/D 球员（下侧），四块面板自动缩放

## 🧭 两种使用方式

| 入口 | 适合场景 | 启动方式 |
| :--- | :--- | :--- |
| **Python CLI** | 快速跑单个视频、调试参数、批处理脚本 | `python main.py --video-path ...` |
| **本地 Web Demo** | 上传视频、裁剪片段、可视化标定、查看历史任务 | FastAPI `8000` + Vite `5173` |

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- FFmpeg，并加入系统 `PATH`
- 推荐 NVIDIA GPU；CPU 可以运行，但视频分析会明显变慢
- 默认依赖安装 CPU 版 PyTorch 和 ONNX Runtime

### 2. 安装分析依赖

```bash
git clone https://github.com/optomao/BadCoach-AI.git
cd BadCoach-AI

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 准备模型

模型权重不随仓库提交。请将以下文件放置到 `weights/` 目录：

```text
weights/
├── yolo11s-ball.pt                                    # 羽毛球检测（必需）
├── yolo11n-pose.pt                                    # YOLO Pose 备用模型
├── yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx        # RTMPose lightweight 检测器
├── yolox_m_8xb8-300e_humanart-c2c7a14a.onnx           # RTMPose balanced 检测器
├── rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx  # RTMPose lightweight
├── rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx   # RTMPose balanced/performance
├── rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.onnx   # RTMO lightweight
├── rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.onnx  # RTMO balanced
└── rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.onnx  # RTMO performance
```

> 如果缺少 ONNX 文件，`rtmlib` 会按需在线下载。首次运行时可能需要等待下载。

### 4. 运行 CLI

```bash
# 使用默认配置分析示例视频（RTMPose balanced + 自动单双打识别）
python main.py --video-path videos/demo.mp4

# 指定对局模式
python main.py --video-path videos/demo.mp4 --match-type doubles

# 选择姿态模型
python main.py --video-path videos/demo.mp4 --pose-family rtmpose --pose-mode balanced
python main.py --video-path videos/demo.mp4 --pose-family rtmo --pose-mode lightweight
python main.py --video-path videos/demo.mp4 --pose-family yolo-pose --yolo-pose-model yolo11n-pose.pt

# 生成英文可视化文字
python main.py --video-path videos/demo.mp4 --language en
```

首次运行时：

1. 如果没有传入 `--template-path`，程序会弹出文件选择框，请选择一张球场清晰可见的模板帧。
2. 在标定窗口依次点击四个球场角点：**左上 → 右上 → 右下 → 左下**。
3. 标定结果会缓存到 `results/<视频名>/court_annotations.txt`，后续运行会自动复用。

## 🖥️ Web Demo

Web Demo 是一个本地运行的可视化工作台：视频只写入本机的 `storage/`，分析结果写入本机的 `results/`，不依赖云端服务。

### 启动后端

```bash
pip install -r requirements.txt -r web-requirements.txt
python -m web.api.run
```

后端地址：<http://127.0.0.1:8000>

### 启动前端

另开一个终端，并激活同一个虚拟环境：

```bash
cd web/frontend
npm install
npm run dev
```

前端地址：<http://127.0.0.1:5173>

Web Demo 支持：

- 上传比赛视频并预览
- 按起止时间截取分析片段
- 选择对局模式：**自动识别 / 单打 / 双打**
- 选择姿态模型：RTMPose / RTMO / YOLO Pose
- 在浏览器中点击四个球场角点完成标定
- 实时滚动日志面板，显示分析 FPS 和进度百分比
- 查看排队、运行、完成和失败状态
- 预览带标注视频、热力图和散点图
- 下载分析产物、查看历史任务并删除本地任务

## 🧠 功能特性

### 核心改进（相对于原项目）

- **单双打自适应追踪**：前 30 帧暖场自动检测每侧最大球员数，≥2 人/侧 → 双打。支持手动覆盖（auto / singles / doubles）。
- **A/B/C/D 槽位模型**：单打 = A（上侧）+ B（下侧），双打 = A/B（上侧）+ C/D（下侧）。每槽位独立配色、独立统计、独立记录。
- **最近邻槽位匹配**：同侧多目标按距离匹配到槽位，避免球员互相遮挡时的目标逃逸。
- **幽灵保持机制**：球员短暂丢失（≤0.5 秒）时沿用最后位置，速度置 0，避免面板跳变。
- **RTMPose 默认检测器**：修正 YOLOX 检测器输入尺寸（yolox_tiny=416, yolox_m=640），补齐全部 RTMPose/RTMO 模型权重。
- **阿里巴巴普惠体**：所有中文可视化（视频面板、热力图、散点图）统一使用阿里巴巴普惠体，消除乱码。
- **面板布局修复**：统计面板靠画面右侧单列纵向排列，双打 4 块面板自动缩放不重叠，回合数移至左下角。
- **实时性能监控**：Web Demo 前端显示滚动日志、实时 FPS 和进度百分比。
- **性能基准工具**：提供 `benchmark_pose.py` 和 `benchmark_compare.py` 用于逐阶段性能分析。

### 视觉分析

- **球员姿态检测**：支持 RTMPose（默认）、RTMO 和 Ultralytics YOLO Pose。
- **羽毛球检测**：使用 YOLO 模型定位羽毛球，并绘制跨帧轨迹。
- **球场坐标映射**：通过四点透视变换，将图像坐标映射到标准羽毛球场坐标。
- **球员追踪**：区分上下半场球员，单打 2 人 / 双打 4 人独立追踪。
- **回合识别**：根据连续球场视图识别回合区间并标记回合编号。

### 结果与分析

- **运动统计**：每位球员独立的移动距离、瞬时速度、平均速度、最大速度和回合数。
- **位置图表**：生成每位球员的位置热力图和散点图，图例移至球场图左上角避免与统计面板重叠。
- **视频叠加层**：可开关姿态 ROI、骨架、球员轨迹、球场轨迹、羽毛球轨迹和统计面板。
- **结构化导出**：输出 `metadata.json`（含 `match` 段）、`session_summary.json` 和逐帧 `detections.jsonl`（schema 1.1，含 `match_mode` 和 A/B/C/D 槽位键）。
- **中英文可视化**：通过 `--language zh/en` 切换 CLI 输出图中的文字。

## 🏗️ 处理流程

```text
比赛视频
   │
   ├── 暖场自动检测（前 30 帧）→ 判定单打/双打
   ├── 球员姿态检测（RTMPose / RTMO / YOLO Pose）
   ├── 羽毛球检测（YOLO）
   └── 球场四点标定
          │
          ▼
   透视变换：图像坐标 → 标准球场坐标
          │
          ▼
   槽位匹配（最近邻）→ A/B（单打）或 A/B/C/D（双打）
          │
          ▼
   球员追踪、回合识别、速度与距离统计
          │
          ├── 带标注 MP4（右侧面板，阿里巴巴普惠体）
          ├── 热力图 / 散点图（图例在左上角）
          └── JSON / JSONL 结构化数据（schema 1.1）
```

## ⚙️ 常用参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--video-path` | 输入视频路径（必填） | — |
| `--match-type` | 对局模式：`auto` / `singles` / `doubles` | `auto` |
| `--output-dir` | 输出目录 | `results/<视频文件名>` |
| `--ball-model` | 羽毛球检测模型路径 | `weights/yolo11s-ball.pt` |
| `--pose-family` | `rtmpose` / `rtmo` / `yolo-pose` | `rtmpose` |
| `--pose-mode` | `lightweight` / `balanced` / `performance` | `balanced` |
| `--yolo-pose-model` | YOLO Pose 模型路径或模型名 | `yolo11n-pose.pt` |
| `--template-path` | 球场模板图像路径 | 文件选择框 |
| `--display` | 是否显示 OpenCV 预览窗口 | `true` |
| `--skeletons` | 是否绘制人体骨架 | `true` |
| `--player-trajectories` | 是否绘制球员轨迹 | `true` |
| `--court-trajectory` | 是否绘制球场轨迹 | `true` |
| `--shuttlecock-trajectory` | 是否绘制羽毛球轨迹 | `true` |
| `--player-stats` | 是否显示球员统计面板 | `true` |
| `--visualize-positions` | 是否生成热力图和散点图 | `true` |
| `--audio` | 是否保留原视频音频 | `true` |
| `--language` | 可视化语言：`zh` / `en` | `zh` |
| `--save-images` | 是否保存逐帧图像 | `false` |
| `--performance-stats` | 是否打印性能统计 | `false` |

## 📦 输出结果

CLI 默认输出到 `results/<视频文件名>/`：

```text
results/<video_name>/
├── metadata.json                  # 视频、模型、标定、对局模式和输出元信息
├── detections.jsonl               # 逐帧检测记录（schema 1.1，A/B/C/D 槽位）
├── detect_<video_name>.mp4        # 带骨架、轨迹和统计面板的视频
├── court_annotations.txt          # CLI 四点标定缓存
└── position_visualizations/
    ├── heatmaps/                  # 每位球员的位置热力图
    └── scatter_plots/             # 每位球员的位置散点图
```

### 性能基准测试

```bash
# 单阶段性能分析
python benchmark_pose.py

# 多模式对比（RTMPose / RTMO / YOLO11n-pose）
python benchmark_compare.py
```

## 🧩 项目结构

```text
BadCoach-AI/
├── main.py                       # CLI 入口
├── benchmark_pose.py             # 单阶段性能基准测试
├── benchmark_compare.py          # 多姿态模式对比基准测试
├── badminton_analysis/           # 核心视频分析管线
│   ├── court/                    # 球场标定与坐标映射
│   ├── data/                     # JSON / JSONL 持久化
│   ├── detection/                # 姿态与羽毛球检测（RTMPose / RTMO / YOLO Pose）
│   ├── media/                    # 视频与音频处理
│   ├── tracking/                 # 球员追踪（自适应 A/B/C/D 槽位）
│   └── visualization/            # 视频叠加层与图表（阿里巴巴普惠体）
│       └── fonts/                # 字体文件
├── web/
│   ├── api/                     # FastAPI 任务服务
│   └── frontend/                # React + Vite 前端
├── assets/                      # 演示 GIF、视频和示例图
├── templates/                   # 球场模板图
├── videos/                      # 示例输入视频
├── weights/                     # 模型权重（.gitignore 忽略）
├── requirements.txt             # 分析依赖
└── web-requirements.txt         # Web 后端依赖
```

## 📊 性能参考

CPU 模式下（1280x720 视频，RTMPose balanced）：

| 方案 | 单帧耗时 | FPS | 检测人数 |
| :--- | :--- | :--- | :--- |
| RTMPose balanced | ~554ms | 1.8 | 10 |
| RTMPose lightweight | ~265ms | 3.8 | 12 |
| RTMO balanced | ~178ms | 5.6 | 1 |
| RTMO lightweight | ~87ms | 11.6 | 1 |
| YOLO11n-pose | ~69ms | 14.6 | — |

> GPU 加速（onnxruntime-gpu + CUDA）预计可获 10-20x 加速。

## 🔮 路线图

- [x] 羽毛球比赛视频逐帧分析
- [x] RTMPose / RTMO / YOLO Pose 多姿态模型
- [x] YOLO 羽毛球检测模型接入
- [x] 手动球场标定与坐标映射
- [x] 球员移动轨迹、速度、距离和回合统计
- [x] 热力图、散点图和结构化数据导出
- [x] 本地 Web Demo：上传、标定、排队分析和结果下载
- [x] **单双打自适应追踪（A/B/C/D 槽位）**
- [x] **对局模式手动选择（auto / singles / doubles）**
- [x] **阿里巴巴普惠体中文可视化**
- [x] **实时 FPS 与进度监控**
- [x] **性能基准测试工具**
- [ ] 更稳定的击球点识别
- [ ] 更精准的羽毛球检测模型
- [ ] 更完整的技术动作统计
- [ ] 自动球场关键点检测
- [ ] 批量视频分析工作流
- [ ] GPU 加速支持

## 🛠️ 技术栈

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Ultralytics%20YOLO-111F68?style=flat-square" alt="Ultralytics YOLO" />
  <img src="https://img.shields.io/badge/ONNX%20Runtime-005CED?style=flat-square" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/FFmpeg-007808?style=flat-square&logo=ffmpeg&logoColor=white" alt="FFmpeg" />
</p>

## 🙏 致谢

- [AI-YuJian-AI](https://github.com/lzylovec/AI-YuJian-AI)（羽见 AI）：本项目基于此开源项目二次开发，原项目由 [lzylovec](https://github.com/lzylovec) 创建
- [TrackNetV2](https://github.com/wywyWang/TrackNetV2)：羽毛球数据集相关工作
- [RTMPose](https://github.com/open-mmlab/mmpose)：人体姿态估计
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)：目标检测生态
- [阿里巴巴普惠体](https://fonts.alibabagroup.com/)：中文可视化字体

## 📄 许可证

项目代码采用 [Apache License 2.0](LICENSE)。模型权重不随仓库提交，请在下载和分发时遵循各自上游项目的许可证与归属要求。

---

<div align="center">

如果这个项目对你有帮助，欢迎 ⭐ Star 支持一下。

**Made with ❤️ by [optomao](https://github.com/optomao)**

</div>
