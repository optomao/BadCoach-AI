from collections import deque

import numpy as np

from ..court.mapper import CourtMapper


# 槽位颜色方案(BGR)
PLAYER_COLORS = {
    "A": (0, 255, 255),    # 青
    "B": (0, 200, 0),      # 绿
    "C": (255, 0, 255),    # 品红
    "D": (0, 130, 255),    # 橙
}

# 各模式下的槽位布局:上半区/下半区各含哪些槽位
SIDE_SLOTS_BY_MODE = {
    "singles": {"upper": ["A"], "lower": ["B"]},
    "doubles": {"upper": ["A", "B"], "lower": ["C", "D"]},
}


class PlayerTracker:
    """
    球员追踪系统(单双打自适应)。

    通过预热期自动识别单打(2人:A/B)或双打(4人:A/B 上半区、C/D 下半区),
    使用最近邻匹配保持同侧多球员身份稳定,按槽位维护位置/历史/统计,
    每个球场帧写一条结构化检测记录。
    """

    def __init__(self, corners, threshold=680, history_size=50, detection_writer=None,
                 fps=30, match_type="auto", warmup_frames=30):
        self.threshold = threshold
        self.fps = fps
        self.detection_writer = detection_writer
        self.max_frame_distance = 8.0 / self.fps
        self.match_type = match_type
        self.warmup_remaining = warmup_frames

        # 运行时确定的模式,初始默认 singles,预热后锁定
        self.match_mode = "singles"
        # 非 auto 模式直接锁定
        if match_type in ("singles", "doubles"):
            self.match_mode = match_type
            self.warmup_remaining = 0

        self.history_size = history_size
        self.side_slots: dict[str, list[str]] = {}
        self.active_slots: list[str] = []
        self.side_of_slot: dict[str, str] = {}
        # 每槽位最近一次质心,用于最近邻匹配
        self.last_positions: dict[str, tuple | None] = {}
        # 每槽位连续未检测帧数,用于"幽灵"保持(短暂丢失时保留最后位置)
        self.frames_since_detection: dict[str, int] = {}
        # 保持帧数阈值:超过此帧数未检测到才清空槽位
        self.hold_frames = max(10, int(fps * 0.5))

        # 预热期统计:每侧出现的最大检测人数
        self._warmup_max_upper = 0
        self._warmup_max_lower = 0

        self._init_slot_structures()

        self.court_mapper = CourtMapper(corners)

    # ---- 初始化槽位结构 ----

    def _init_slot_structures(self):
        """根据 match_mode 初始化 side_slots 及各按槽位键控的数据结构。"""
        self.side_slots = {
            side: list(slots) for side, slots in SIDE_SLOTS_BY_MODE[self.match_mode].items()
        }
        self.active_slots = [s for slots in self.side_slots.values() for s in slots]
        self.side_of_slot = {slot: side for side, slots in self.side_slots.items() for slot in slots}

        self.players = {slot: None for slot in self.active_slots}
        self.history = {slot: deque(maxlen=self.history_size) for slot in self.active_slots}
        self.court_history = {slot: deque(maxlen=self.history_size) for slot in self.active_slots}
        self.match_stats = {
            slot: {"total_distance": 0, "max_speed": 0, "total_frames": 0}
            for slot in self.active_slots
        }
        self.rally_stats = {
            slot: {"total_distance": 0, "max_speed": 0, "total_frames": 0}
            for slot in self.active_slots
        }
        self.current_speed = {slot: 0 for slot in self.active_slots}
        self.last_positions = {slot: None for slot in self.active_slots}
        self.frames_since_detection = {slot: self.hold_frames for slot in self.active_slots}

    def get_active_slots(self) -> list[str]:
        return list(self.active_slots)

    def _empty_player_record(self):
        return {
            "image": None,
            "court": None,
            "speed": None,
            "hands": {
                "left": None,
                "right": None,
            },
        }

    def _initialize_player_record(self):
        return {slot: self._empty_player_record() for slot in self.active_slots}

    def _point_or_none(self, point, zero_is_none=False):
        if point is None:
            return None
        try:
            x, y = point[0], point[1]
        except (TypeError, IndexError):
            return None
        if x is None or y is None:
            return None
        if zero_is_none and float(x) == 0.0 and float(y) == 0.0:
            return None
        return [float(x), float(y)]

    # ---- 检测记录写入 ----

    def write_detection_record(self, frame_index, players_record, ball_image_position, detect_frame_count):
        if self.detection_writer is None:
            return

        record = {
            "schema_version": "1.1",
            "frame": int(frame_index),
            "time_sec": round(frame_index / self.fps, 6) if self.fps else None,
            "detect_frame": int(detect_frame_count),
            "players": players_record,
            "match_mode": self.match_mode,
            "shuttlecock": {
                "image": self._point_or_none(ball_image_position, zero_is_none=True),
            },
        }
        self.detection_writer.write(record)

    # ---- 主更新流程 ----

    def update(self, frame_index, centroids, ball_image_position, left_hand_positions, right_hand_positions, detect_frame_count):
        # 预热期内尝试判定模式
        if self.warmup_remaining > 0:
            self._update_warmup(centroids)

        players_record = self._initialize_player_record()

        # 按 threshold 分侧
        upper_centroids = []
        lower_centroids = []
        for centroid in centroids:
            if centroid[1] < self.threshold:
                upper_centroids.append(centroid)
            else:
                lower_centroids.append(centroid)

        # 逐侧匹配到槽位
        matched_slots = set()
        side_detections = {"upper": upper_centroids, "lower": lower_centroids}
        for side, dets in side_detections.items():
            assignments = self._match_side_detections(side, dets)
            for slot, centroid in assignments.items():
                left_hand = left_hand_positions.get(centroid[1])
                right_hand = right_hand_positions.get(centroid[1])
                try:
                    self._update_player_position(slot, centroid, left_hand, right_hand, players_record)
                    matched_slots.add(slot)
                    self.frames_since_detection[slot] = 0
                except Exception as exc:
                    print(f"Error processing player position: {exc}")
                    import traceback
                    traceback.print_exc()

        # 幽灵保持:未匹配到的槽位若在 hold_frames 内,沿用最后位置
        for slot in self.active_slots:
            if slot in matched_slots:
                continue
            self.frames_since_detection[slot] += 1
            if self.frames_since_detection[slot] <= self.hold_frames:
                last = self.last_positions[slot]
                if last is not None:
                    # 沿用最后已知位置,但标记速度为0
                    players_record[slot]["image"] = self._point_or_none(last)
                    court_pos = self.court_mapper.image_to_court(last)
                    players_record[slot]["court"] = self._point_or_none(court_pos)
                    players_record[slot]["speed"] = 0.0
                    self.current_speed[slot] = 0.0

        # 累计在场帧数(仅保持期内算在场)
        for slot in self.active_slots:
            if players_record[slot]["image"] is not None:
                self.match_stats[slot]["total_frames"] += 1
                self.rally_stats[slot]["total_frames"] += 1

        self.write_detection_record(frame_index, players_record, ball_image_position, detect_frame_count)
        return self.players

    def _update_warmup(self, centroids):
        """预热期统计各侧最大检测人数,任一侧≥2 则锁定双打。"""
        upper_count = sum(1 for c in centroids if c[1] < self.threshold)
        lower_count = len(centroids) - upper_count
        self._warmup_max_upper = max(self._warmup_max_upper, upper_count)
        self._warmup_max_lower = max(self._warmup_max_lower, lower_count)

        if self._warmup_max_upper >= 2 or self._warmup_max_lower >= 2:
            self._lock_mode("doubles")
            return

        self.warmup_remaining -= 1
        if self.warmup_remaining <= 0:
            self._lock_mode("singles")

    def _lock_mode(self, mode):
        """锁定比赛模式并重建槽位结构。"""
        if self.match_mode == mode and not self.warmup_remaining:
            return
        self.match_mode = mode
        self.warmup_remaining = 0
        self._init_slot_structures()

    def _match_side_detections(self, side, detections) -> dict:
        """
        贪心最近邻:将该侧检测点分配到该侧的槽位。
        返回 {slot: centroid}。每个槽位最多匹配一个检测,每个检测最多匹配一个槽位。
        """
        slots = self.side_slots.get(side, [])
        if not slots or not detections:
            return {}

        # 构建所有(槽位, 检测)距离对,按距离升序贪心匹配
        pairs = []
        for slot in slots:
            last = self.last_positions[slot]
            for idx, det in enumerate(detections):
                if last is None:
                    # 无历史位置时给一个中性权重,用检测间相对位置后续排序
                    dist = 0.0
                else:
                    dist = float(np.hypot(det[0] - last[0], det[1] - last[1]))
                pairs.append((dist, slot, idx))

        pairs.sort(key=lambda p: p[0])
        used_slots = set()
        used_dets = set()
        assignments: dict[str, tuple] = {}

        # 有历史的槽位优先匹配
        for _dist, slot, idx in pairs:
            if slot in used_slots or idx in used_dets:
                continue
            assignments[slot] = detections[idx]
            used_slots.add(slot)
            used_dets.add(idx)

        # 若该侧仍有未分配的检测点且仍有空槽(预热期或无历史),按检测顺序补充
        if len(assignments) < len(slots):
            free_slots = [s for s in slots if s not in used_slots]
            free_dets = [d for i, d in enumerate(detections) if i not in used_dets]
            for slot, det in zip(free_slots, free_dets):
                assignments[slot] = det

        return assignments

    def _update_player_position(self, slot, centroid, left_hand_pos, right_hand_pos, players_record):
        self.players[slot] = centroid
        self.last_positions[slot] = centroid
        self.history[slot].append(centroid)

        court_position = self.court_mapper.image_to_court(centroid)
        self.court_history[slot].append(court_position)

        player_record = players_record[slot]
        player_record["image"] = self._point_or_none(centroid)
        player_record["court"] = self._point_or_none(court_position)
        player_record["speed"] = float(self.current_speed[slot])
        if left_hand_pos:
            player_record["hands"]["left"] = self._point_or_none(left_hand_pos)
        if right_hand_pos:
            player_record["hands"]["right"] = self._point_or_none(right_hand_pos)

    def _update_rally_and_match_stats(self, slot, distance, speed):
        capped_speed = round(min(speed, 8.0), 2)

        self.rally_stats[slot]["total_distance"] += distance
        self.rally_stats[slot]["max_speed"] = max(self.rally_stats[slot]["max_speed"], capped_speed)
        self.current_speed[slot] = capped_speed

        self.match_stats[slot]["total_distance"] += distance
        self.match_stats[slot]["max_speed"] = max(self.match_stats[slot]["max_speed"], capped_speed)
        self.current_speed[slot] = capped_speed

    def start_new_rally(self):
        for slot in self.active_slots:
            self.rally_stats[slot]["total_distance"] = 0
            self.rally_stats[slot]["max_speed"] = 0
            self.rally_stats[slot]["total_frames"] = 0

    def get_player_movement_stats(self):
        stats = {}
        for slot in self.active_slots:
            history = [pos for pos in list(self.court_history[slot]) if pos is not None]
            slot_stats = {
                "current_speed": 0,
                "rally_avg_speed": 0,
                "rally_max_speed": 0,
                "rally_distance": 0,
                "match_avg_speed": 0,
                "match_max_speed": 0,
                "match_distance": 0,
                "position_count": len(history),
            }

            if len(history) < 2:
                stats[slot] = slot_stats
                continue

            current_time = len(history) - 1
            window_start = max(0, current_time - int(self.fps / 2))
            half_second_total_distance = 0
            valid_frames = 0
            actual_time_span = 0
            sample_interval = 5

            if current_time - window_start < sample_interval:
                sample_points = [window_start, current_time]
            else:
                sample_points = list(range(window_start, current_time + 1, sample_interval))
                if current_time not in sample_points:
                    sample_points.append(current_time)

            for i in range(len(sample_points) - 1):
                idx1 = sample_points[i]
                idx2 = sample_points[i + 1]
                p1 = np.array(history[idx1])
                p2 = np.array(history[idx2])
                distance = np.linalg.norm(p2 - p1)
                time_span = (idx2 - idx1) / self.fps
                max_possible_distance = self.max_frame_distance * (idx2 - idx1)

                if distance > 0.05 and distance < max_possible_distance:
                    half_second_total_distance += distance
                    valid_frames += 1
                    actual_time_span += time_span

            current_speed = 0
            if valid_frames > 0 and actual_time_span > 0:
                current_speed = half_second_total_distance / actual_time_span
                self._update_rally_and_match_stats(slot, half_second_total_distance, current_speed)

            current_speed = min(current_speed, 8.0)

            rally_distance = self.rally_stats[slot]["total_distance"]
            rally_max_speed = self.rally_stats[slot]["max_speed"]
            rally_frames = self.rally_stats[slot]["total_frames"]
            if rally_frames > 1 and self.fps > 0:
                rally_time = rally_frames / self.fps
                rally_avg_speed = rally_distance / rally_time if rally_time > 0 else 0
            else:
                rally_avg_speed = 0

            match_distance = self.match_stats[slot]["total_distance"]
            match_max_speed = self.match_stats[slot]["max_speed"]
            match_frames = self.match_stats[slot]["total_frames"]
            if match_frames > 1 and self.fps > 0:
                match_time = match_frames / self.fps
                match_avg_speed = match_distance / match_time if match_time > 0 else 0
            else:
                match_avg_speed = 0

            slot_stats["current_speed"] = round(current_speed, 2)
            slot_stats["rally_avg_speed"] = round(rally_avg_speed, 2)
            slot_stats["rally_max_speed"] = round(rally_max_speed, 2)
            slot_stats["rally_distance"] = round(rally_distance, 2)
            slot_stats["match_avg_speed"] = round(match_avg_speed, 2)
            slot_stats["match_max_speed"] = round(match_max_speed, 2)
            slot_stats["match_distance"] = round(match_distance, 2)
            stats[slot] = slot_stats

        return stats

    def get_player_trajectories(self):
        return {slot: list(history) for slot, history in self.history.items()}

    def close(self):
        if self.detection_writer is not None:
            self.detection_writer.close()
