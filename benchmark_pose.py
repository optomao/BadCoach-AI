"""性能基准测试：定位 RTMPose + YOLO 球检测的处理瓶颈"""
import time
import cv2
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试视频
VIDEO = "videos/demo.mp4"

def benchmark_stage(name, func, frame, iterations=10):
    """测试单个阶段性能"""
    # 预热
    func(frame)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = func(frame)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    fps = 1.0 / avg if avg > 0 else 0
    print(f"  {name:30s}  平均: {avg*1000:8.1f}ms  FPS: {fps:6.1f}  (min={min(times)*1000:.1f}ms max={max(times)*1000:.1f}ms)")
    return avg, result

def main():
    print("=" * 80)
    print("羽毛球分析系统 - 性能基准测试")
    print("=" * 80)

    # 1. 读取视频信息
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n视频: {VIDEO}")
    print(f"  分辨率: {w}x{h}  FPS: {fps}  总帧数: {total}")
    print(f"  时长: {total/fps:.1f}s")

    # 取中间帧做测试
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("无法读取测试帧!")
        return

    print(f"\n测试帧: {frame.shape[1]}x{frame.shape[0]}")

    # 2. 测试 RTMPose (balanced 模式)
    print("\n" + "-" * 60)
    print("阶段 1: RTMPose 姿态检测 (balanced 模式)")
    print("-" * 60)
    from badminton_analysis.detection.rtmpose import RTMPoseProcessor
    t0 = time.perf_counter()
    rtmpose = RTMPoseProcessor(mode='balanced', device='auto')
    t1 = time.perf_counter()
    print(f"  模型加载耗时: {t1-t0:.2f}s")

    avg_pose, (kp, scores) = benchmark_stage("RTMPose 推理", rtmpose.process_frame, frame, iterations=5)
    n_persons = kp.shape[0] if kp is not None else 0
    print(f"  检测到 {n_persons} 人")

    # 3. 测试 YOLO 球检测
    print("\n" + "-" * 60)
    print("阶段 2: YOLO 羽毛球检测 (yolo11s-ball.pt)")
    print("-" * 60)
    from ultralytics import YOLO
    ball_path = "weights/yolo11s-ball.pt"
    if os.path.exists(ball_path):
        t0 = time.perf_counter()
        ball_model = YOLO(ball_path)
        t1 = time.perf_counter()
        print(f"  模型加载耗时: {t1-t0:.2f}s")

        def ball_detect(f):
            return ball_model(f, verbose=False, conf=0.3)
        avg_ball, _ = benchmark_stage("YOLO 球检测", ball_detect, frame, iterations=5)
    else:
        print("  球模型文件不存在，跳过")
        avg_ball = 0

    # 4. 测试帧 resize
    print("\n" + "-" * 60)
    print("阶段 3: 图像预处理 (resize 到 640)")
    print("-" * 60)
    def resize_frame(f):
        h, w = f.shape[:2]
        scale = min(640 / w, 640 / h)
        return cv2.resize(f, (int(w * scale), int(h * scale)))
    avg_resize, _ = benchmark_stage("cv2.resize", resize_frame, frame)

    # 5. 测试完整流水线 (模拟一帧全流程)
    print("\n" + "-" * 60)
    print("阶段 4: 完整单帧流水线模拟")
    print("-" * 60)

    # 模拟完整流程
    def full_pipeline(f):
        # 1. resize
        resized = resize_frame(f)
        # 2. 姿态检测
        kp, sc = rtmpose.process_frame(f)
        # 3. 球检测 (如果有模型)
        if os.path.exists(ball_path):
            ball_model(f, verbose=False, conf=0.3)
        return kp

    avg_full, _ = benchmark_stage("完整流水线", full_pipeline, frame, iterations=5)

    # 6. 测试视频写入
    print("\n" + "-" * 60)
    print("阶段 5: 视频写入 (VideoWriter)")
    print("-" * 60)
    tmp_path = "_bench_out.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(tmp_path, fourcc, 30, (w, h))
    def write_frame(f):
        writer.write(f)
    avg_write, _ = benchmark_stage("VideoWriter.write", write_frame, frame, iterations=20)
    writer.release()
    os.remove(tmp_path)

    # 7. 总结
    print("\n" + "=" * 80)
    print("性能瓶颈分析总结")
    print("=" * 80)
    stages = [
        ("RTMPose 姿态推理", avg_pose),
        ("YOLO 球检测", avg_ball),
        ("图像 resize", avg_resize),
        ("视频写入", avg_write),
    ]
    total_avg = sum(t for _, t in stages) + avg_full - avg_pose - avg_ball - avg_resize  # 去重
    # 简单汇总
    print(f"\n各阶段耗时占比 (单帧总耗时 ~{avg_full*1000:.1f}ms, 理论 FPS={1/avg_full:.1f}):")
    for name, t in stages:
        pct = (t / avg_full * 100) if avg_full > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {name:30s}  {t*1000:8.1f}ms  ({pct:5.1f}%)  {bar}")

    print(f"\n  完整单帧耗时: {avg_full*1000:.1f}ms")
    print(f"  理论处理 FPS: {1/avg_full:.1f}")
    print(f"  实时率 (相对 {fps}fps 视频): {fps * avg_full * 100:.1f}%")
    print(f"  处理 1 秒视频需要: {avg_full * fps:.1f} 秒")

    # 瓶颈排序
    sorted_stages = sorted(stages, key=lambda x: -x[1])
    print(f"\n最大瓶颈: {sorted_stages[0][0]} ({sorted_stages[0][1]*1000:.1f}ms)")
    print(f"次大瓶颈: {sorted_stages[1][0]} ({sorted_stages[1][1]*1000:.1f}ms)")

    print("\n加速建议:")
    if sorted_stages[0][0].startswith("RTMPose"):
        print("  1. 使用 GPU 加速 (onnxruntime-gpu + CUDA) → 预计 10-20x 加速")
        print("  2. 换 RTMPose lightweight 模式 (yolox_tiny + rtmpose-s) → 预计 2-3x 加速")
        print("  3. 帧跳跃策略 (每 2-3 帧处理一次) → 预计 2-3x 加速")
        print("  4. 换 YOLO11n-pose 单阶段模型 → 预计 3-5x 加速 (但精度降低)")
    elif sorted_stages[0][0].startswith("YOLO"):
        print("  1. 使用 GPU 加速 YOLO 推理")
        print("  2. 降低 YOLO 输入分辨率")
        print("  3. 球检测帧跳跃 (不是每帧都检测球)")

if __name__ == "__main__":
    main()
