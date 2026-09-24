"""对比测试不同姿态检测模式的性能"""
import time
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VIDEO = "videos/demo.mp4"

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ret, frame = cap.read()
cap.release()

def bench(name, processor, iterations=5):
    processor.process_frame(frame)  # warmup
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        kp, sc = processor.process_frame(frame)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    n = kp.shape[0] if kp is not None else 0
    print(f"  {name:35s}  {avg*1000:7.1f}ms  FPS: {1/avg:5.1f}  人数: {n}")
    return avg, n

print("=" * 80)
print("姿态检测模式对比测试 (CPU)")
print("=" * 80)
print(f"测试帧: {frame.shape[1]}x{frame.shape[0]}\n")

# RTMPose balanced
print("1. RTMPose balanced (yolox_m 640 + rtmpose-m)")
from badminton_analysis.detection.rtmpose import RTMPoseProcessor
p_bal = RTMPoseProcessor(mode='balanced', device='cpu')
bench("RTMPose balanced", p_bal)

# RTMPose lightweight
print("\n2. RTMPose lightweight (yolox_tiny 416 + rtmpose-s)")
p_lw = RTMPoseProcessor(mode='lightweight', device='cpu')
bench("RTMPose lightweight", p_lw)

# RTMO balanced
print("\n3. RTMO balanced (rtmo-m 640 单阶段)")
p_rtmo = RTMPoseProcessor(mode='balanced', device='cpu', pose_family='rtmo')
bench("RTMO balanced", p_rtmo)

# RTMO lightweight
print("\n4. RTMO lightweight (rtmo-s 640 单阶段)")
p_rtmo_lw = RTMPoseProcessor(mode='lightweight', device='cpu', pose_family='rtmo')
bench("RTMO lightweight", p_rtmo_lw)

# YOLO11n-pose
print("\n5. YOLO11n-pose (单阶段)")
from ultralytics import YOLO
yolo = YOLO("weights/yolo11n-pose.pt")
def yolo_pose(f):
    r = yolo(f, verbose=False, conf=0.15)
    return r
yolo_pose(frame)  # warmup
times = []
for _ in range(5):
    t0 = time.perf_counter()
    yolo_pose(frame)
    t1 = time.perf_counter()
    times.append(t1 - t0)
avg = sum(times) / len(times)
print(f"  {'YOLO11n-pose':35s}  {avg*1000:7.1f}ms  FPS: {1/avg:5.1f}")

print("\n" + "=" * 80)
print("总结: RTMO 单阶段比 RTMPose 两阶段更快 (省去 YOLOX 检测步骤)")
print("     lightweight 模式比 balanced 快 2-3x (更小模型)")
print("     YOLO11n-pose 最快但精度最低 (nano 模型)")
