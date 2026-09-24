import cv2
import json
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict
import matplotlib.font_manager as fm

plt.style.use('default')

# 设置中文字体 - 使用阿里巴巴普惠体 (Alibaba PuHuiTi)
def _load_chinese_font():
    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(module_dir, '..', '..', '..'))
    workspace_root = os.path.abspath(os.path.join(package_root, '..'))
    candidates = [
        # 阿里巴巴普惠体 (Alibaba PuHuiTi) - 首选
        os.path.join(module_dir, 'fonts', 'AlibabaPuHuiTi-3-35-Thin.ttf'),
        "C:/Users/io/AI-YuJian-AI/badminton_analysis/visualization/fonts/AlibabaPuHuiTi-3-35-Thin.ttf",
        # 后备字体
        "C:/Windows/Fonts/Source Han Serif SC Heavy (TrueType).ttf",
        os.path.join(module_dir, 'simhei.ttf'),
        os.path.join(package_root, 'simhei.ttf'),
        os.path.join(workspace_root, 'simhei.ttf'),
        os.path.join(os.getcwd(), 'simhei.ttf'),
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]

    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                font_name = fm.FontProperties(fname=font_path).get_name()
            except Exception:
                font_name = 'serif'
            plt.rcParams['font.family'] = ['serif']
            plt.rcParams['font.serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return fm.FontProperties(fname=font_path)

    plt.rcParams['axes.unicode_minus'] = False
    return None


chinese_font = _load_chinese_font()

class PlayerPositionVisualizer:
    """
    Player Position Visualization Class
    """
    
    def __init__(self, detections_path, output_dir=None, court_width=6.1, court_length=13.4, fps=30):
        """
        Initialize the player position visualizer
        Args:
            detections_path: Path to detections.jsonl containing player position data
            output_dir: Output directory, defaults to visualizations subdirectory in the detection file's directory
            court_width: Badminton court width in meters (default: 6.1m)
            court_length: Badminton court length in meters (default: 13.4m)
        """
        self.detections_path = detections_path
        self.court_width = court_width
        self.court_length = court_length
        
        # Set output directory
        if output_dir is None:
            detections_dir = os.path.dirname(os.path.abspath(detections_path))
            self.output_dir = os.path.join(detections_dir, 'position_visualizations')
        else:
            self.output_dir = output_dir
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'scatter_plots'), exist_ok=True)
        
        # 运动统计参数
        self.fps = fps  # 视频帧率，用于计算速度
        self.movement_stats = defaultdict(dict)
        self.MAX_SPEED = 8.0  # 人类最大速度限制(m/s)
        self.MIN_MOVEMENT = 0.05  # 最小移动距离(m)，低于此值视为噪声
        self.MAX_FRAME_DISTANCE = 8.0 / self.fps  # 单帧最大移动距离(m)，基于最大速度和帧率计算
        
        # Load data
        self.df = self._load_data()
    
        # Court image parameters
        self.img_width = 610  # Image width (pixels)
        self.img_height = 1340  # Image height (pixels)
        
        # Heatmap grid parameters
        self.heatmap_grid_size = (30, 60)  # Grid size (width grid count, length grid count)
        
        self.upper_color = '#d94b4b'
        self.lower_color = '#1f76b4'

        # 按槽位的颜色与标签(单打 A/B,双打 A/B/C/D;兼容旧 upper/lower)
        self.player_colors = {
            'A': '#d94b4b', 'B': '#2ca02c', 'C': '#1f76b4', 'D': '#ff7f0e',
            'upper': '#d94b4b', 'lower': '#1f76b4',
        }
        self.player_labels = {
            'A': 'A球员', 'B': 'B球员', 'C': 'C球员', 'D': 'D球员',
            'upper': '上场球员', 'lower': '下场球员',
        }

        self.court_line_color = '#33443d'
        
    def _calculate_movement_stats(self, slot_dfs, rally_segments, frames):
        """计算每个回合每位球员的统计数据（平均速度，最大速度，总移动距离）"""
        self.movement_stats = defaultdict(dict)
        frame_times = frames / self.fps  # 将帧数转换为时间(秒)

        # 计算整场比赛的统计数据(按槽位)
        for slot, slot_df in slot_dfs.items():
            all_slot = slot_df[slot_df['valid_coords']]
            if not all_slot.empty:
                self.movement_stats['match'][slot] = self._calculate_player_stats(
                    all_slot[['court_x', 'court_y']].values,
                    frame_times[all_slot.index].values
                )

        # 对每个回合分别计算(按槽位)
        for rally_id, (start_idx, end_idx) in enumerate(rally_segments, 1):
            rally_times = frame_times[start_idx:end_idx].values
            for slot, slot_df in slot_dfs.items():
                slot_rally = slot_df[(slot_df['rally_id'] == rally_id) & (slot_df['valid_coords'])]
                positions = slot_rally[['court_x', 'court_y']].values
                if len(positions) > 1:
                    self.movement_stats[rally_id][slot] = self._calculate_player_stats(positions, rally_times)
    
    def _calculate_player_stats(self, positions, times):
        """计算单个球员的运动统计数据"""
        # 初始化统计数据
        stats = {
            'total_distance': 0.0,
            'max_speed': 0.0,
            'avg_speed': 0.0,
            'total_frames': len(positions)  # 总帧数
        }
        
        # 如果数据点少于2个，无法计算统计信息
        if len(positions) < 2:
            return stats
        
        # 计算总距离和最大速度
        total_valid_distance = 0.0
        max_speed = 0.0
        
        # 采样间隔，每5帧采样一次
        sample_interval = 5
        current_time = len(positions) - 1
        
        # 确保至少有一个采样点
        if current_time < sample_interval:
            sample_points = [0, current_time]
        else:
            # 创建采样点列表
            sample_points = list(range(0, current_time + 1, sample_interval))
            # 确保最后一个点被包含
            if current_time not in sample_points:
                sample_points.append(current_time)
        
        # 计算采样点之间的距离
        for i in range(len(sample_points) - 1):
            idx1 = sample_points[i]
            idx2 = sample_points[i + 1]
            
            p1 = positions[idx1]
            p2 = positions[idx2]
            
            # 计算欧几里得距离(米)
            dist = np.sqrt(((p2 - p1)**2).sum())
            
            # 计算时间差(秒)
            time_diff = times[idx2] - times[idx1] if idx2 < len(times) else (idx2 - idx1) / self.fps
            
            # 基于时间间隔调整最大允许距离
            max_possible_distance = self.MAX_FRAME_DISTANCE * (idx2 - idx1)
            
            # 过滤微小移动和异常值
            if dist > self.MIN_MOVEMENT and dist < max_possible_distance:
                # 累加有效距离
                total_valid_distance += dist
                
                # 计算速度并更新最大速度
                if time_diff > 0:
                    speed = dist / time_diff
                    speed = min(speed, self.MAX_SPEED)  # 限制最大速度
                    max_speed = max(max_speed, speed)
        
        # 更新统计数据
        stats['total_distance'] = round(total_valid_distance, 2)
        stats['max_speed'] = round(max_speed, 2)
        
        # 计算平均速度 - 使用总距离除以总时间，考虑球员静止的时间
        total_time = times[-1] - times[0] if len(times) > 1 else stats['total_frames'] / self.fps
        if total_time > 0:
            stats['avg_speed'] = round(total_valid_distance / total_time, 2)
        
        return stats
    
    def _load_data(self):
        """Load detections.jsonl and convert to required format(按槽位动态识别)"""
        try:
            # 第一遍:收集每帧的各槽位球场坐标,同时发现所有槽位键
            # slot_rows[slot] = list of (frame_index, court_x, court_y)
            slot_rows: dict[str, list] = defaultdict(list)
            frame_list: list[int] = []
            # 兼容旧数据:upper→A, lower→B
            legacy_map = {"upper": "A", "lower": "B"}

            with open(self.detections_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    frame = record.get("frame")
                    frame_list.append(frame)
                    players = record.get("players", {}) or {}
                    for raw_slot, pdata in players.items():
                        if not isinstance(pdata, dict):
                            continue
                        court = pdata.get("court") or [None, None]
                        slot = legacy_map.get(raw_slot, raw_slot)
                        slot_rows[slot].append((frame, court[0], court[1]))

            if not frame_list:
                print("Error: No frames found in detections file")
                return pd.DataFrame()

            discovered_slots = list(slot_rows.keys())
            print(f"\nDiscovered player slots: {discovered_slots}")

            # 构建基础 DataFrame(Frame + 每槽位坐标列)
            base = pd.DataFrame({"Frame": frame_list})

            # 帧序列用于 rally 切分(基于全帧列表)
            frames = base['Frame'].astype(int).tolist()
            gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            rally_breaks = [i+1 for i, gap in enumerate(gaps) if gap > 100]
            rally_segments = []
            start_idx = 0
            for break_idx in rally_breaks:
                rally_segments.append((start_idx, break_idx))
                start_idx = break_idx
            if start_idx < len(frames):
                rally_segments.append((start_idx, len(frames)))
            rally_segments = [(start, end) for start, end in rally_segments if end - start >= 150]
            print(f"Detected {len(rally_segments)} valid rallies")

            # 为每个槽位构造独立 DataFrame
            slot_dfs: dict[str, pd.DataFrame] = {}
            for slot, rows in slot_rows.items():
                sdf = pd.DataFrame(rows, columns=["Frame", "court_x", "court_y"])
                sdf = sdf.drop_duplicates(subset=["Frame"], keep="last")
                # 与全帧基表对齐(缺失补 None)
                sdf = base.merge(sdf, on="Frame", how="left")
                sdf['valid_coords'] = (sdf['court_x'].notna()) & (sdf['court_y'].notna()) & \
                                      (sdf['court_x'] >= 0) & (sdf['court_y'] >= 0)
                sdf['player_position'] = slot
                sdf['rally_id'] = 0
                for rally_id, (start, end) in enumerate(rally_segments, 1):
                    mask = (sdf.index >= start) & (sdf.index < end)
                    sdf.loc[mask, 'rally_id'] = rally_id
                slot_dfs[slot] = sdf

            # 计算每个回合的移动统计数据
            self._calculate_movement_stats(slot_dfs, rally_segments, base['Frame'])

            # 合并所有槽位数据
            combined_df = pd.concat(slot_dfs.values(), ignore_index=True)
            combined_df = combined_df.dropna(subset=['court_x', 'court_y'])
            combined_df = combined_df[combined_df['rally_id'] > 0]
            combined_df = combined_df[combined_df['valid_coords'] == True]
            combined_df = combined_df.drop('valid_coords', axis=1)

            print(f"\nData conversion complete, {len(combined_df)} records total")
            return combined_df

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()  # Return empty DataFrame
            
    def _create_court_image(self):
        """Create court background image"""
        # Create blank white image
        self.court_img = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 255
        
        # Calculate court area dimensions
        padding = 50  # Border padding in pixels
        court_img_width = self.img_width - 2 * padding
        court_img_height = self.img_height - 2 * padding
        
        # Court corners (in image coordinates)
        top_left = (padding, padding)
        top_right = (padding + court_img_width, padding)
        bottom_right = (padding + court_img_width, padding + court_img_height)
        bottom_left = (padding, padding + court_img_height)
        
        # Draw court outline
        cv2.line(self.court_img, top_left, top_right, (0, 0, 0), 2)
        cv2.line(self.court_img, top_right, bottom_right, (0, 0, 0), 2)
        cv2.line(self.court_img, bottom_right, bottom_left, (0, 0, 0), 2)
        cv2.line(self.court_img, bottom_left, top_left, (0, 0, 0), 2)
        
        # Draw net
        net_y = padding + court_img_height // 2
        cv2.line(self.court_img, (padding, net_y), (padding + court_img_width, net_y), (0, 0, 0), 1)
        
        # Draw center line
        center_x = padding + court_img_width // 2
        cv2.line(self.court_img, (center_x, padding), (center_x, padding + court_img_height), (0, 0, 0), 1)
        
        # Draw service courts
        upper_service_line_y = padding + int(court_img_height * 0.25)
        lower_service_line_y = padding + int(court_img_height * 0.75)
        
        cv2.line(self.court_img, (padding, upper_service_line_y), 
                 (padding + court_img_width, upper_service_line_y), (0, 0, 0), 1)
        cv2.line(self.court_img, (padding, lower_service_line_y),
                 (padding + court_img_width, lower_service_line_y), (0, 0, 0), 1)
        
        # Draw service line edges
        cv2.line(self.court_img,
                 (int(padding + court_img_width * 0.25), padding),
                 (int(padding + court_img_width * 0.25), upper_service_line_y), 
                 (0, 0, 0), 1)
        cv2.line(self.court_img,
                 (int(padding + court_img_width * 0.75), padding),
                 (int(padding + court_img_width * 0.75), upper_service_line_y), 
                 (0, 0, 0), 1)
        cv2.line(self.court_img,
                 (int(padding + court_img_width * 0.25), lower_service_line_y),
                 (int(padding + court_img_width * 0.25), padding + court_img_height), 
                 (0, 0, 0), 1)
        cv2.line(self.court_img,
                 (int(padding + court_img_width * 0.75), lower_service_line_y),
                 (int(padding + court_img_width * 0.75), padding + court_img_height), 
                 (0, 0, 0), 1)
                
        return self.court_img
        
    def _draw_court(self, ax=None):
        """在matplotlib图形上绘制标准羽毛球场地"""
        axis = ax if ax is not None else plt.gca()
        
        # 关键：反转Y轴以匹配真实球场方向（0在顶部，13.4在底部）
        axis.invert_yaxis()
        
        # 标准羽毛球场地尺寸（米）
        doubles_width = self.court_width  # 双打场地宽度 (6.10米)
        court_length = self.court_length  # 场地长度 (13.40米)
        single_width = 0.46   # 单打线距离双打线的距离
        service_line = 1.98   # 发球线到网的距离
        back_service = 0.76   # 后发球线到底线的距离
        
        # 绘制场地外框（双打场地外框）
        court_rect = plt.Rectangle((0, 0), doubles_width, court_length, 
                                 fill=False, color=self.court_line_color, linewidth=4)
        axis.add_patch(court_rect)
        
        # 绘制单打线
        axis.plot([single_width, single_width], [0, court_length], self.court_line_color, linewidth=4)
        axis.plot([doubles_width - single_width, doubles_width - single_width], 
                 [0, court_length], self.court_line_color, linewidth=4)
        
        # 绘制网线（中间在y=场地长度/2）
        axis.axhline(y=court_length/2, color=self.court_line_color, linestyle='--', linewidth=4)
        
        # 绘制中线（只画到发球线）
        axis.plot([doubles_width/2, doubles_width/2], [0, court_length/2-service_line], self.court_line_color, linewidth=4)  # 上半场
        axis.plot([doubles_width/2, doubles_width/2], [court_length/2+service_line, court_length], self.court_line_color, linewidth=4)  # 下半场
        
        # 绘制发球线
        # 前发球线（距网1.98米）
        axis.axhline(y=court_length/2-service_line, color=self.court_line_color, linestyle='-', linewidth=4)
        axis.axhline(y=court_length/2+service_line, color=self.court_line_color, linestyle='-', linewidth=4)
        
        # 后发球线（距底线0.76米）
        axis.axhline(y=back_service, color=self.court_line_color, linestyle='-', linewidth=4)
        axis.axhline(y=court_length-back_service, color=self.court_line_color, linestyle='-', linewidth=4)
        
        # 设置显示范围（带边距）
        axis.set_xlim(-0.5, doubles_width + 0.5)
        axis.set_ylim(court_length + 0.5, -0.5)  # 注意：Y轴范围是反向的
        
    def _court_to_image_coords(self, court_x, court_y):
        """Convert court coordinates to image coordinates"""
        img_x = int(court_x / self.court_width * self.img_width)
        img_y = int(court_y / self.court_length * self.img_height)
        return img_x, img_y
        
    def _generate_rally_visualizations(self):
        """Generate visualizations for each rally"""
        if self.df.empty:
            print("No data to visualize")
            return

        # Get all rally IDs
        rally_ids = self.df['rally_id'].unique()

        for rally_id in rally_ids:
            if pd.isna(rally_id):
                continue

            print(f"Processing visualizations for rally {rally_id}...")

            rally_df = self.df[self.df['rally_id'] == rally_id]
            # Generate heatmap
            self._generate_heatmap(rally_df, f"rally_{int(rally_id)}_heatmap.png")
            # Generate scatter plot
            self._generate_scatter_plot(rally_df, f"rally_{int(rally_id)}_scatter.png")

        print("All rally visualizations generated")

    def _generate_match_visualizations(self):
        """Generate visualizations for the entire match"""
        if self.df.empty:
            print("No data to visualize")
            return

        print("Generating match-wide visualizations...")
        # Generate heatmap
        self._generate_heatmap(self.df, "match_heatmap.png")
        # Generate scatter plot
        self._generate_scatter_plot(self.df, "match_scatter.png")
        print("Match-wide visualizations generated successfully")
            
    def _generate_heatmap(self, source_df, filename):
        """Generate heatmap(按槽位遍历)"""
        fig = plt.figure(figsize=(14, 8), facecolor='#fbfcfa')
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.08)
        ax = fig.add_subplot(grid[0, 0])
        stats_ax = fig.add_subplot(grid[0, 1])
        ax.set_facecolor('#ffffff')
        stats_ax.set_facecolor('#fbfcfa')

        # Create court background
        self._draw_court(ax)

        # 按槽位绘制热力图
        legend_handles = []
        for slot in source_df['player_position'].unique():
            slot_df = source_df[source_df['player_position'] == slot]
            if slot_df.empty:
                continue
            color = self.player_colors.get(slot, self.upper_color)
            light = '#ffe1dd' if slot in ('A', 'upper') else '#dbeafe'
            cmap = LinearSegmentedColormap.from_list(f"{slot}_cmap", [(1, 1, 1, 0), light, color])
            sns.kdeplot(
                x=slot_df['court_x'],
                y=slot_df['court_y'],
                cmap=cmap,
                fill=True,
                alpha=0.6,
                levels=10,
                thresh=0.03,
                bw_adjust=0.9,
                ax=ax,
            )
            legend_handles.append(mpatches.Patch(color=color, label=self.player_labels.get(slot, slot)))

        # 图例放左侧球场图上,避免与右侧统计文本重叠
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc='upper left',
                bbox_to_anchor=(0.01, 0.99),
                fontsize=11,
                framealpha=0.9,
                facecolor='#ffffff',
                edgecolor='#d7dfdb',
                prop=chinese_font,
            )

        # 添加统计信息
        rally_id = None
        if 'rally_id' in source_df.columns:
            rally_ids = source_df['rally_id'].unique()
            if len(rally_ids) == 1 and rally_ids[0] != 0:
                rally_id = int(rally_ids[0])

        if rally_id and rally_id in self.movement_stats:
            self._add_stats_to_plot(rally_id, stats_ax)
        else:
            self._add_stats_to_plot(None, stats_ax)

        ax.set_xlim(0, self.court_width)
        ax.set_ylim(self.court_length, 0)
        ax.set_title('球员位置热力图', color='#101614', fontsize=18, fontweight='bold', pad=14, fontproperties=chinese_font)
        ax.set_xlabel('场地宽度 (米)', color='#33443d', fontproperties=chinese_font)
        ax.set_ylabel('场地长度 (米)', color='#33443d', fontproperties=chinese_font)
        ax.tick_params(colors='#33443d')
        ax.set_aspect('equal', adjustable='box')

        save_path = os.path.join(self.output_dir, 'heatmaps', filename)
        fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print(f"热力图已保存至: {save_path}")
            
    # 注意：这个方法被新的_calculate_player_stats(self, positions, times)方法替代
    # 保留此方法是为了兼容性，但不再使用
            
    def _add_stats_to_plot(self, rally_id=None, ax=None):
        """
        在图表上添加统计信息(按槽位遍历)
        Args:
            rally_id: 回合ID，如果为None则显示所有回合的总统计信息
        """
        if not self.movement_stats:
            return
        if rally_id is not None and rally_id not in self.movement_stats:
            return

        # 统计各槽位在各回合中出现的键,确定要显示哪些槽位
        all_slots: list[str] = []
        for stats in self.movement_stats.values():
            for slot in stats.keys():
                if slot not in all_slots:
                    all_slots.append(slot)

        if rally_id is not None:
            stats = self.movement_stats[rally_id]
            info_text = f"回合 {rally_id} 统计:\n\n"
            for slot in all_slots:
                if slot not in stats:
                    continue
                s = stats[slot]
                label = self.player_labels.get(slot, slot)
                info_text += f"{label}:\n"
                info_text += f"  平均速度: {s['avg_speed']:.2f} 米/秒\n"
                info_text += f"  最大速度: {s['max_speed']:.2f} 米/秒\n"
                info_text += f"  移动距离: {s['total_distance']:.2f} 米\n"
        else:
            info_text = f"比赛统计\n\n"
            for slot in all_slots:
                distances = []
                speeds = []
                avg_speeds = []
                for rally_stats in self.movement_stats.values():
                    if slot in rally_stats:
                        distances.append(rally_stats[slot]['total_distance'])
                        if rally_stats[slot].get('max_speed', 0) > 0:
                            speeds.append(rally_stats[slot]['max_speed'])
                        if rally_stats[slot].get('avg_speed', 0) > 0:
                            avg_speeds.append(rally_stats[slot]['avg_speed'])
                label = self.player_labels.get(slot, slot)
                info_text += f"{label}:\n"
                if distances:
                    total_distance = sum(distances)
                    info_text += f"  总移动距离: {total_distance:.2f} 米\n"
                    info_text += f"  平均每回合距离: {total_distance/len(distances):.2f} 米\n"
                if avg_speeds:
                    info_text += f"  平均速度: {sum(avg_speeds)/len(avg_speeds):.2f} 米/秒\n"
                if speeds:
                    info_text += f"  最大速度: {max(speeds):.2f} 米/秒\n"

        if ax is not None:
            ax.axis('off')
            ax.text(
                0.04, 0.96, "移动统计", transform=ax.transAxes,
                fontsize=20, weight='bold', color='#101614', fontproperties=chinese_font,
            )
            ax.text(
                0.04, 0.88, info_text, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='#ffffff', alpha=0.96, boxstyle='round,pad=0.9', edgecolor='#d7dfdb'),
                fontsize=12, linespacing=1.4, color='#101614', fontproperties=chinese_font,
            )
            return

        plt.text(0.98, 0.5, info_text,
                horizontalalignment='right',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='#ffffff', alpha=0.88, boxstyle='round,pad=0.7', edgecolor='#d7dfdb'),
                fontsize=14, weight='bold', color='#101614', fontproperties=chinese_font)
    
    def _generate_scatter_plot(self, source_df, filename):
        """Generate scatter plot(按槽位遍历)"""
        fig = plt.figure(figsize=(14, 8), facecolor='#fbfcfa')
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.08)
        ax = fig.add_subplot(grid[0, 0])
        stats_ax = fig.add_subplot(grid[0, 1])
        ax.set_facecolor('#ffffff')
        stats_ax.set_facecolor('#fbfcfa')

        # Create court background
        self._draw_court(ax)

        # 按槽位绘制散点
        markers = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D', 'upper': 'o', 'lower': '^'}
        scatter_handles = []
        for slot in source_df['player_position'].unique():
            slot_df = source_df[source_df['player_position'] == slot]
            if slot_df.empty:
                continue
            color = self.player_colors.get(slot, self.upper_color)
            marker = markers.get(slot, 'o')
            label = self.player_labels.get(slot, slot)
            scatter = ax.scatter(
                slot_df['court_x'], slot_df['court_y'],
                alpha=0.58, s=24, marker=marker, color=color, label=label,
            )
            scatter_handles.append(scatter)

        # 图例放左侧球场图上,避免与右侧统计文本重叠
        if scatter_handles:
            ax.legend(
                handles=scatter_handles,
                loc='upper left',
                bbox_to_anchor=(0.01, 0.99),
                fontsize=11,
                framealpha=0.9,
                facecolor='#ffffff',
                edgecolor='#d7dfdb',
                prop=chinese_font,
            )

        # 添加统计信息
        rally_id = None
        if 'rally_id' in source_df.columns:
            rally_ids = source_df['rally_id'].unique()
            if len(rally_ids) == 1 and rally_ids[0] != 0:
                rally_id = int(rally_ids[0])

        if rally_id and rally_id in self.movement_stats:
            self._add_stats_to_plot(rally_id, stats_ax)
        else:
            self._add_stats_to_plot(None, stats_ax)

        ax.set_xlim(0, self.court_width)
        ax.set_ylim(self.court_length, 0)
        ax.set_title('球员位置散点图', color='#101614', fontsize=18, fontweight='bold', pad=14, fontproperties=chinese_font)
        ax.set_xlabel('场地宽度 (米)', color='#33443d', fontproperties=chinese_font)
        ax.set_ylabel('场地长度 (米)', color='#33443d', fontproperties=chinese_font)
        ax.tick_params(colors='#33443d')
        ax.set_aspect('equal', adjustable='box')

        save_path = os.path.join(self.output_dir, 'scatter_plots', filename)
        fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print(f"散点图已保存至: {save_path}")
    
    def visualize(self):
        """Execute visualization processing"""
        if self.df.empty:
            print("No data to visualize")
            return False
            
        try:
            # Generate visualizations for each rally
            self._generate_rally_visualizations()
            
            # Generate visualizations for the entire match
            self._generate_match_visualizations()
            
            return True
        except Exception as e:
            print(f"可视化过程中出错: {e}")
            return False
            
            
def analyze_player_positions(detections_path, output_dir=None, fps=30, include_summary=False):
    """
    Analyze player position data and generate visualizations
    
    Args:
        detections_path: Path to detections.jsonl containing player position data
        output_dir: Output directory, defaults to visualizations subdirectory in the detection file's directory
        
    Returns:
        bool: Whether processing was successful
    """
    print(f"\n分析球员位置数据: {detections_path}")
    
    # Create visualizer
    visualizer = PlayerPositionVisualizer(detections_path, output_dir, fps=fps)
    
    # Execute visualization
    success = visualizer.visualize()
    
    if success:
        print(f"球员位置分析完成，可视化结果已保存至: {visualizer.output_dir}")
    else:
        print("球员位置分析失败")
        
    if not include_summary:
        return success

    artifacts = {
        "match_heatmap": os.path.join(visualizer.output_dir, "heatmaps", "match_heatmap.png"),
        "match_scatter": os.path.join(visualizer.output_dir, "scatter_plots", "match_scatter.png"),
    }
    artifacts = {key: value for key, value in artifacts.items() if os.path.exists(value)}
    return {
        "success": success,
        "output_dir": visualizer.output_dir,
        "movement_stats": dict(visualizer.movement_stats),
        "artifacts": artifacts,
    }


# 测试代码：允许直接运行该文件来测试可视化效果
if __name__ == "__main__":
    import sys
    from tkinter import Tk, filedialog
    
    # Use file dialog to select detection file
    print("请选择 detections.jsonl 文件...")
    
    try:
        # Create hidden tkinter root window (for file dialog only)
        root = Tk()
        root.withdraw()
        
        # Set default directory
        default_dir = "results"
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()
        
        # Open file selection dialog
        file_path = filedialog.askopenfilename(
            title="选择球员位置检测文件",
            filetypes=[("JSONL文件", "*.jsonl"), ("所有文件", "*.*")],
            initialdir=default_dir
        )
        
        # 如果用户取消选择，则退出
        if not file_path:
            print("未选择文件，退出程序")
            sys.exit(0)
            
        # 调用分析函数
        success = analyze_player_positions(file_path)
        
        if success:
            print("\n可视化测试成功完成")
        else:
            print("\n可视化测试失败")
            
    except Exception as e:
        print(f"\n测试错误: {e}")
        
    finally:
        try:
            # 关闭tkinter窗口
            root.destroy()
        except:
            pass
        
