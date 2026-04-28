# 电动车入梯检测与预警系统（YOLOv11 + PySide6）

这是一个可演示的完整项目，包含：
- YOLOv11 训练脚本（基于你的基础模型继续训练）
- 电动车入梯实时检测与报警模块（声响 + 截图 + 日志）
- PySide6/Qt 预警平台（可接摄像头或视频文件）

## 1. 项目结构

```text
pythonProject4/
├─ dataset/
│  ├─ images/train|val|test
│  └─ labels/train|val|test
├─ weights/                 # 放基础模型和训练后的best.pt
├─ outputs/                 # 报警截图和日志
├─ src/
│  ├─ config.py
│  └─ detect_engine.py
├─ tools/
│  └─ prepare_dataset.py
├─ train.py                 # 训练入口
├─ demo_detect.py           # 命令行演示入口
├─ app.py                   # PySide6界面入口
└─ requirements.txt
```

## 2. 数据集准备（重点）

请将你的本地图片和标注整理成 YOLO 标准格式：

```text
dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ labels/
   ├─ train/
   ├─ val/
   └─ test/
```

标签文件 `.txt` 每行格式：

```text
class x_center y_center width height
```

说明：
- 你只做“电动车”一个类别时，`class` 固定为 `0`
- 坐标全部是 `0~1` 的归一化值

生成数据集配置：

```bash
venv\Scripts\python tools\prepare_dataset.py --class-name 电动车
```

会生成 `dataset/elevator_ebike.yaml`。

如果你的原始数据是“图片和同名txt标注混放在一个目录”，可直接一键导入并切分：

```bash
venv\Scripts\python tools\import_flat_dataset.py --source 你的原始数据目录 --clear
```

然后再执行：

```bash
venv\Scripts\python tools\prepare_dataset.py --class-name 电动车
```

## 3. 安装依赖

```bash
venv\Scripts\python -m pip install -r requirements.txt
```

## 4. 训练模型

如果你有 YOLOv11 基础模型（比如 `yolo11n.pt`），可直接训练：

```bash
venv\Scripts\python train.py --weights yolo11n.pt --data dataset/elevator_ebike.yaml --epochs 100 --imgsz 640 --batch 8 --device 0
```

如果你是 CPU 环境（报错里显示 `torch.cuda.is_available(): False`），请改用：

```bash
venv\Scripts\python train.py --weights yolo11n.pt --data dataset/elevator_ebike.yaml --epochs 100 --imgsz 640 --batch 8 --device cpu
```

训练前会自动输出正样本统计与小数据告警。可用参数：
- `--min-positive-images 30`：设置最小正样本阈值（默认30）
- `--strict-data-check`：正样本不足时直接中止训练

训练完成后，最佳权重通常在：

```text
runs/elevator_ebike/yolo11_train/weights/best.pt
```

把它复制到：

```text
weights/best.pt
```

## 5. 命令行演示（快速验证）

```bash
venv\Scripts\python demo_detect.py --model weights/best.pt --source 0 --target-class 电动车 --conf 0.35
```

- `--source 0` 表示默认摄像头
- 也可以传视频路径，如 `--source demo.mp4`

## 6. PySide6/Qt 预警平台

启动图形平台：

```bash
venv\Scripts\python app.py
```

功能：
- 选择模型 `.pt`
- 选择视频源（摄像头或视频）
- 支持单张图片检测（即点即测）
- 实时显示检测画面
- 检测到电动车时：
  - 声响报警（Windows Beep）
  - 自动保存报警截图到 `outputs/alarm_images/`
  - 写入事件日志 `outputs/alarm_events.csv`

图片检测方式：
- 在界面里点击 `选择图片并检测`
- 选择单张图片后会直接显示检测结果，并在日志区输出是否报警

## 7. 演示建议流程

1. 准备并标注数据集（正样本：有电动车入梯；负样本：无电动车）
2. 训练并得到 `best.pt`
3. 用 `demo_detect.py` 快速验证识别能力
4. 启动 `app.py` 在平台中演示实时预警

## 8. 为什么看不到 best.pt

如果你没有看到 `best.pt`，通常是以下原因：
- 训练没有真正启动（命令未执行/环境不对）
- 训练中断或报错（显存不足、数据集路径错误、标签格式错误）
- 训练尚未结束（`best.pt` 在训练完成后写入）
- 训练输出目录不是默认路径
- 数据集结构不完整（例如缺少 `dataset/images/val`）

可先在项目根目录执行自动收集命令：

```bash
venv\Scripts\python tools\collect_best.py
```

它会从 `runs/` 下自动查找最新的 `best.pt` 并复制到 `weights/best.pt`。

## 9. 提升精度建议

- 增加不同光照、角度、遮挡、多人共乘的样本
- 加入纯电梯空场景负样本，降低误报
- 重点标注小目标、边缘目标
- 调整 `--imgsz` 到 `960` 或更高（显存允许时）
- 调整阈值（平台默认 35%）
