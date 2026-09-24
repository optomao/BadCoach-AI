import time

import cv2
import numpy as np


class PlayerPoseVisualizer:
    """Detect, filter, and draw player pose keypoints."""

    def __init__(
        self,
        rtmpose_processor=None,
        show_skeletons=True,
        show_player_trajectories=True,
        show_performance_stats=False,
        court_filter_margin=0.75,
    ):
        if rtmpose_processor is None:
            raise RuntimeError("A pose processor instance must be provided to PlayerPoseVisualizer.")
        self.rtmpose_processor = rtmpose_processor
        self.show_skeletons = show_skeletons
        self.show_player_trajectories = show_player_trajectories
        self.show_performance_stats = show_performance_stats
        self.current_pose_data = None
        self.court_mapper = None
        self.court_filter_margin = court_filter_margin

        self.skeleton_connections = [
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
        ]

    def detect_players(self, roi, x1, y1, court_mapper=None):
        centroids = []
        point_left_hands = {}
        point_right_hands = {}

        t0 = time.time()
        keypoints_all, _confidence_scores = self.rtmpose_processor.process_frame(roi)
        if self.show_performance_stats:
            inference_name = getattr(self.rtmpose_processor, "inference_name", "Pose")
            print(f"{inference_name} inference took {time.time() - t0:.2f} sec")

        if keypoints_all is None:
            self.current_pose_data = None
            return centroids, point_left_hands, point_right_hands

        persons = self._normalize_people(keypoints_all)
        filtered_people = []
        active_court_mapper = court_mapper or self.court_mapper

        for kp in persons:
            kp_arr = np.asarray(kp)
            if kp_arr.ndim != 2 or kp_arr.shape[0] < 17 or kp_arr.shape[1] < 2:
                continue

            # 统计有效关键点数量(坐标>1 视为有效)
            valid_mask = (kp_arr[:, 0] > 1) & (kp_arr[:, 1] > 1)
            valid_count = int(valid_mask.sum())
            # 至少需要 5 个有效关键点才认为是有效人体
            if valid_count < 5:
                continue

            # 降级质心计算: 双脚 → 双髋 → 所有有效关键点均值
            # 远离镜头的球员(上半区)脚踝常检测不到,用髋部或均值兜底
            lf = kp_arr[15]
            rf = kp_arr[16]
            lh_hip = kp_arr[11]
            rh_hip = kp_arr[12]

            feet_valid = (lf[0] > 1 and lf[1] > 1 and rf[0] > 1 and rf[1] > 1)
            hips_valid = (lh_hip[0] > 1 and lh_hip[1] > 1 and rh_hip[0] > 1 and rh_hip[1] > 1)

            if feet_valid:
                mid_x = (float(lf[0]) + float(rf[0])) / 2
                mid_y = (float(lf[1]) + float(rf[1])) / 2 + 10
            elif hips_valid:
                # 髋部中心,向下偏移补偿脚部位置
                mid_x = (float(lh_hip[0]) + float(rh_hip[0])) / 2
                mid_y = (float(lh_hip[1]) + float(rh_hip[1])) / 2 + 30
            else:
                # 所有有效关键点均值
                valid_pts = kp_arr[valid_mask]
                mid_x = float(valid_pts[:, 0].mean())
                mid_y = float(valid_pts[:, 1].mean())

            mid_point = (mid_x + x1, mid_y + y1)

            if not self._is_on_court(mid_point, active_court_mapper):
                continue

            filtered_people.append(kp_arr)
            centroids.append(mid_point)

            lh = kp_arr[9]
            rh = kp_arr[10]
            if lh[0] > 1 and lh[1] > 1:
                point_left_hands[mid_point[1]] = (int(lh[0] + x1), int(lh[1] + y1))
            if rh[0] > 1 and rh[1] > 1:
                point_right_hands[mid_point[1]] = (int(rh[0] + x1), int(rh[1] + y1))

        if filtered_people:
            self.current_pose_data = {
                "keypoints": np.asarray(filtered_people),
                "offset_x": x1,
                "offset_y": y1,
            }
        else:
            self.current_pose_data = None

        return centroids, point_left_hands, point_right_hands

    def _normalize_people(self, keypoints):
        if isinstance(keypoints, np.ndarray):
            if keypoints.ndim == 2:
                return [keypoints]
            if keypoints.ndim == 3:
                return [keypoints[i] for i in range(keypoints.shape[0])]
            return []
        if isinstance(keypoints, (list, tuple)):
            return list(keypoints)
        return []

    def _is_on_court(self, image_point, court_mapper):
        if court_mapper is None:
            return True
        court_position = court_mapper.image_to_court(image_point)
        if court_position is None or len(court_position) < 2:
            return False
        x, y = float(court_position[0]), float(court_position[1])
        margin = self.court_filter_margin
        return -margin <= x <= 6.1 + margin and -margin <= y <= 13.4 + margin

    def draw_players(self, frame, player_tracker, cached_movement_stats, stats_visualizer=None, rally_count=0):
        if self.show_skeletons and self.current_pose_data is not None:
            t0 = time.time()
            self._draw_skeleton_on_frame(
                frame,
                self.current_pose_data["keypoints"],
                self.current_pose_data["offset_x"],
                self.current_pose_data["offset_y"],
            )
            if self.show_performance_stats:
                print(f"Drawing skeleton took {time.time() - t0:.2f} sec")

        # 各槽位颜色(BGR)
        slot_colors = {
            'A': (0, 255, 255),
            'B': (0, 200, 0),
            'C': (255, 0, 255),
            'D': (0, 130, 255),
        }

        t0 = time.time()
        active_slots = player_tracker.get_active_slots() if hasattr(player_tracker, 'get_active_slots') else ['upper', 'lower']
        for slot in active_slots:
            if player_tracker.players.get(slot) is None:
                continue

            color = slot_colors.get(slot, (0, 255, 255) if slot in ('upper', 'A') else (255, 0, 255))
            cv2.circle(frame, tuple(map(int, player_tracker.players[slot])), 5, color, -1, cv2.LINE_AA)

            if self.show_player_trajectories:
                history = list(player_tracker.history[slot])
                for i, pos in enumerate(history):
                    if pos is None:
                        continue
                    radius = int(2 + (i / len(history)) * 3) if history else 2
                    cv2.circle(frame, tuple(map(int, pos)), radius, color, -1, cv2.LINE_AA)

        if self.show_performance_stats:
            print(f"Drawing players and trajectories took {time.time() - t0:.2f} sec")

        if stats_visualizer is not None:
            t0 = time.time()
            stats_visualizer.draw_player_stats(
                frame, cached_movement_stats, rally_count,
                active_slots=active_slots,
            )
            if self.show_performance_stats:
                print(f"Drawing player stats took {time.time() - t0:.2f} sec")

    def _draw_skeleton_on_frame(self, frame, keypoints, offset_x, offset_y):
        for person in self._normalize_people(keypoints):
            person_arr = np.asarray(person)
            if person_arr.ndim != 2 or person_arr.shape[1] < 2:
                continue

            keypoint_count = person_arr.shape[0]
            for a, b in self.skeleton_connections:
                if a >= keypoint_count or b >= keypoint_count:
                    continue
                x1, y1 = float(person_arr[a, 0]), float(person_arr[a, 1])
                x2, y2 = float(person_arr[b, 0]), float(person_arr[b, 1])
                if x1 > 1 and y1 > 1 and x2 > 1 and y2 > 1:
                    pt1 = (int(x1 + offset_x), int(y1 + offset_y))
                    pt2 = (int(x2 + offset_x), int(y2 + offset_y))
                    cv2.line(frame, pt1, pt2, (255, 191, 0), 2, cv2.LINE_AA)

            for i in range(keypoint_count):
                x_raw, y_raw = float(person_arr[i, 0]), float(person_arr[i, 1])
                if x_raw > 1 and y_raw > 1:
                    cv2.circle(frame, (int(x_raw + offset_x), int(y_raw + offset_y)), 3, (255, 128, 0), -1, cv2.LINE_AA)

    def get_current_pose_data(self):
        return self.current_pose_data
