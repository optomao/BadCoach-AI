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

# Set global plotting style
plt.style.use('default')

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
        
        # Movement statistics parameters
        self.fps = fps  # Video frame rate, used for speed calculation
        self.movement_stats = defaultdict(dict)
        self.MAX_SPEED = 8.0  # Human maximum speed limit (m/s)
        self.MIN_MOVEMENT = 0.05  # Minimum movement distance (m), below this value is considered noise
        self.MAX_FRAME_DISTANCE = 8.0 / self.fps  # Maximum frame-to-frame distance (m), based on max speed and fps
        
        # Load data
        self.df = self._load_data()
    
        # Court image parameters
        self.img_width = 610  # Image width (pixels)
        self.img_height = 1340  # Image height (pixels)
        
        # Heatmap grid parameters
        self.heatmap_grid_size = (30, 60)  # Grid size (width grid count, length grid count)
        
        self.upper_color = '#d94b4b'
        self.lower_color = '#1f76b4'

        # Per-slot colors and labels (singles A/B, doubles A/B/C/D; compatible with legacy upper/lower)
        self.player_colors = {
            'A': '#d94b4b', 'B': '#2ca02c', 'C': '#1f76b4', 'D': '#ff7f0e',
            'upper': '#d94b4b', 'lower': '#1f76b4',
        }
        self.player_labels = {
            'A': 'Player A', 'B': 'Player B', 'C': 'Player C', 'D': 'Player D',
            'upper': 'Upper Player', 'lower': 'Lower Player',
        }

        self.court_line_color = '#33443d'
        
    def _calculate_movement_stats(self, slot_dfs, rally_segments, frames):
        """Calculate per-rally per-player statistics (average speed, maximum speed, total distance)"""
        self.movement_stats = defaultdict(dict)
        frame_times = frames / self.fps  # Convert frame numbers to time (seconds)

        # Calculate match-wide statistics (per slot)
        for slot, slot_df in slot_dfs.items():
            all_slot = slot_df[slot_df['valid_coords']]
            if not all_slot.empty:
                self.movement_stats['match'][slot] = self._calculate_player_stats(
                    all_slot[['court_x', 'court_y']].values,
                    frame_times[all_slot.index].values
                )

        # Calculate per rally (per slot)
        for rally_id, (start_idx, end_idx) in enumerate(rally_segments, 1):
            rally_times = frame_times[start_idx:end_idx].values
            for slot, slot_df in slot_dfs.items():
                slot_rally = slot_df[(slot_df['rally_id'] == rally_id) & (slot_df['valid_coords'])]
                positions = slot_rally[['court_x', 'court_y']].values
                if len(positions) > 1:
                    self.movement_stats[rally_id][slot] = self._calculate_player_stats(positions, rally_times)
    
    def _calculate_player_stats(self, positions, times):
        """Calculate movement statistics for a single player"""
        # Initialize statistics
        stats = {
            'total_distance': 0.0,
            'max_speed': 0.0,
            'avg_speed': 0.0,
            'total_frames': len(positions)  # Total frames
        }
        
        # If less than 2 data points, cannot calculate statistics
        if len(positions) < 2:
            return stats
        
        # Calculate total distance and maximum speed
        total_valid_distance = 0.0
        max_speed = 0.0
        
        # Sampling interval, sample every 5 frames
        sample_interval = 5
        current_time = len(positions) - 1
        
        # Ensure at least one sampling point
        if current_time < sample_interval:
            sample_points = [0, current_time]
        else:
            # Create list of sampling points
            sample_points = list(range(0, current_time + 1, sample_interval))
            # Ensure the last point is included
            if current_time not in sample_points:
                sample_points.append(current_time)
        
        # Calculate distance between sampling points
        for i in range(len(sample_points) - 1):
            idx1 = sample_points[i]
            idx2 = sample_points[i + 1]
            
            p1 = positions[idx1]
            p2 = positions[idx2]
            
            # Calculate Euclidean distance (meters)
            dist = np.sqrt(((p2 - p1)**2).sum())
            
            # Calculate time difference (seconds)
            time_diff = times[idx2] - times[idx1] if idx2 < len(times) else (idx2 - idx1) / self.fps
            
            # Adjust maximum allowed distance based on time interval
            max_possible_distance = self.MAX_FRAME_DISTANCE * (idx2 - idx1)
            
            # Filter small movements and anomalies
            if dist > self.MIN_MOVEMENT and dist < max_possible_distance:
                # Accumulate valid distance
                total_valid_distance += dist
                
                # Calculate speed and update maximum speed
                if time_diff > 0:
                    speed = dist / time_diff
                    speed = min(speed, self.MAX_SPEED)  # Limit maximum speed
                    max_speed = max(max_speed, speed)
        
        # Update statistics
        stats['total_distance'] = round(total_valid_distance, 2)
        stats['max_speed'] = round(max_speed, 2)
        
        # Calculate average speed - use total distance divided by total time, considering stationary time
        total_time = times[-1] - times[0] if len(times) > 1 else stats['total_frames'] / self.fps
        if total_time > 0:
            stats['avg_speed'] = round(total_valid_distance / total_time, 2)
        
        return stats
    
    def _load_data(self):
        """Load detections.jsonl and convert to required format (dynamic slot detection)"""
        try:
            # First pass: collect per-frame per-slot court coordinates, discover all slot keys
            # slot_rows[slot] = list of (frame_index, court_x, court_y)
            slot_rows: dict[str, list] = defaultdict(list)
            frame_list: list[int] = []
            # Compatible with legacy data: upper→A, lower→B
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

            # Build base DataFrame (Frame + per-slot coordinate columns)
            base = pd.DataFrame({"Frame": frame_list})

            # Frame sequence for rally segmentation (based on full frame list)
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

            # Build independent DataFrame per slot
            slot_dfs: dict[str, pd.DataFrame] = {}
            for slot, rows in slot_rows.items():
                sdf = pd.DataFrame(rows, columns=["Frame", "court_x", "court_y"])
                sdf = sdf.drop_duplicates(subset=["Frame"], keep="last")
                # Align with full-frame base table (fill missing with None)
                sdf = base.merge(sdf, on="Frame", how="left")
                sdf['valid_coords'] = (sdf['court_x'].notna()) & (sdf['court_y'].notna()) & \
                                      (sdf['court_x'] >= 0) & (sdf['court_y'] >= 0)
                sdf['player_position'] = slot
                sdf['rally_id'] = 0
                for rally_id, (start, end) in enumerate(rally_segments, 1):
                    mask = (sdf.index >= start) & (sdf.index < end)
                    sdf.loc[mask, 'rally_id'] = rally_id
                slot_dfs[slot] = sdf

            # Calculate movement statistics per rally
            self._calculate_movement_stats(slot_dfs, rally_segments, base['Frame'])

            # Combine all slot data
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
        """Draw standard badminton court on matplotlib figure"""
        axis = ax if ax is not None else plt.gca()
        
        # Key: invert Y axis to match real court orientation (0 at top, 13.4 at bottom)
        axis.invert_yaxis()
        
        # Standard badminton court dimensions (meters)
        doubles_width = self.court_width  # Doubles court width (6.10m)
        court_length = self.court_length  # Court length (13.40m)
        single_width = 0.46   # Singles line distance from doubles line
        service_line = 1.98   # Service line to net distance
        back_service = 0.76   # Back service line to baseline distance
        
        # Draw court outline (doubles court outline)
        court_rect = plt.Rectangle((0, 0), doubles_width, court_length, 
                                 fill=False, color=self.court_line_color, linewidth=4)
        axis.add_patch(court_rect)
        
        # Draw singles lines
        axis.plot([single_width, single_width], [0, court_length], self.court_line_color, linewidth=4)
        axis.plot([doubles_width - single_width, doubles_width - single_width], 
                 [0, court_length], self.court_line_color, linewidth=4)
        
        # Draw net line (middle at y=court_length/2)
        axis.axhline(y=court_length/2, color=self.court_line_color, linestyle='--', linewidth=4)
        
        # Draw center line (only to service line)
        axis.plot([doubles_width/2, doubles_width/2], [0, court_length/2-service_line], self.court_line_color, linewidth=4)  # Upper half
        axis.plot([doubles_width/2, doubles_width/2], [court_length/2+service_line, court_length], self.court_line_color, linewidth=4)  # Lower half
        
        # Draw service lines
        # Front service line (1.98m from net)
        axis.axhline(y=court_length/2-service_line, color=self.court_line_color, linestyle='-', linewidth=4)
        axis.axhline(y=court_length/2+service_line, color=self.court_line_color, linestyle='-', linewidth=4)
        
        # Back service line (0.76m from baseline)
        axis.axhline(y=back_service, color=self.court_line_color, linestyle='-', linewidth=4)
        axis.axhline(y=court_length-back_service, color=self.court_line_color, linestyle='-', linewidth=4)
        
        # Set display range (with padding)
        axis.set_xlim(-0.5, doubles_width + 0.5)
        axis.set_ylim(court_length + 0.5, -0.5)  # Note: Y axis range is inverted
        
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
        """Generate heatmap (iterate per slot)"""
        fig = plt.figure(figsize=(14, 8), facecolor='#fbfcfa')
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.08)
        ax = fig.add_subplot(grid[0, 0])
        stats_ax = fig.add_subplot(grid[0, 1])
        ax.set_facecolor('#ffffff')
        stats_ax.set_facecolor('#fbfcfa')

        # Create court background
        self._draw_court(ax)

        # Draw heatmap per slot
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

        # Place legend on the left court plot to avoid overlap with stats text
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc='upper left',
                bbox_to_anchor=(0.01, 0.99),
                fontsize=11,
                framealpha=0.9,
                facecolor='#ffffff',
                edgecolor='#d7dfdb',
            )

        # Add statistics
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
        ax.set_title('Player Position Heatmap', color='#101614', fontsize=18, fontweight='bold', pad=14)
        ax.set_xlabel('Court Width (m)', color='#33443d')
        ax.set_ylabel('Court Length (m)', color='#33443d')
        ax.tick_params(colors='#33443d')
        ax.set_aspect('equal', adjustable='box')

        save_path = os.path.join(self.output_dir, 'heatmaps', filename)
        fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print(f"Heatmap saved to: {save_path}")
            
    # Note: this method is replaced by the new _calculate_player_stats(self, positions, times) method
    # Kept for compatibility but no longer used
            
    def _add_stats_to_plot(self, rally_id=None, ax=None):
        """
        Add statistics information to the plot (iterate per slot)
        Args:
            rally_id: Rally ID, if None then display overall statistics for all rallies
        """
        if not self.movement_stats:
            return
        if rally_id is not None and rally_id not in self.movement_stats:
            return

        # Collect all slots appearing in any rally stats, determine which slots to display
        all_slots: list[str] = []
        for stats in self.movement_stats.values():
            for slot in stats.keys():
                if slot not in all_slots:
                    all_slots.append(slot)

        if rally_id is not None:
            stats = self.movement_stats[rally_id]
            info_text = f"Rally {rally_id} Statistics:\n\n"
            for slot in all_slots:
                if slot not in stats:
                    continue
                s = stats[slot]
                label = self.player_labels.get(slot, slot)
                info_text += f"{label}:\n"
                info_text += f"  Average Speed: {s['avg_speed']:.2f} m/s\n"
                info_text += f"  Maximum Speed: {s['max_speed']:.2f} m/s\n"
                info_text += f"  Distance Moved: {s['total_distance']:.2f} m\n"
        else:
            info_text = f"Match Statistics\n\n"
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
                    info_text += f"  Total Distance: {total_distance:.2f} m\n"
                    info_text += f"  Average Distance per Rally: {total_distance/len(distances):.2f} m\n"
                if avg_speeds:
                    info_text += f"  Average Speed: {sum(avg_speeds)/len(avg_speeds):.2f} m/s\n"
                if speeds:
                    info_text += f"  Maximum Speed: {max(speeds):.2f} m/s\n"

        if ax is not None:
            ax.axis('off')
            ax.text(
                0.04, 0.96, "Movement Statistics", transform=ax.transAxes,
                fontsize=20, weight='bold', color='#101614',
            )
            ax.text(
                0.04, 0.88, info_text, transform=ax.transAxes, va='top', ha='left',
                bbox=dict(facecolor='#ffffff', alpha=0.96, boxstyle='round,pad=0.9', edgecolor='#d7dfdb'),
                fontsize=12, linespacing=1.4, color='#101614',
            )
            return

        plt.text(0.98, 0.5, info_text,
                horizontalalignment='right',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='#ffffff', alpha=0.88, boxstyle='round,pad=0.7', edgecolor='#d7dfdb'),
                fontsize=14, weight='bold', color='#101614')
    
    def _generate_scatter_plot(self, source_df, filename):
        """Generate scatter plot (iterate per slot)"""
        fig = plt.figure(figsize=(14, 8), facecolor='#fbfcfa')
        grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 0.75], wspace=0.08)
        ax = fig.add_subplot(grid[0, 0])
        stats_ax = fig.add_subplot(grid[0, 1])
        ax.set_facecolor('#ffffff')
        stats_ax.set_facecolor('#fbfcfa')

        # Create court background
        self._draw_court(ax)

        # Draw scatter per slot
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

        # Place legend on the left court plot to avoid overlap with stats text
        if scatter_handles:
            ax.legend(
                handles=scatter_handles,
                loc='upper left',
                bbox_to_anchor=(0.01, 0.99),
                fontsize=11,
                framealpha=0.9,
                facecolor='#ffffff',
                edgecolor='#d7dfdb',
            )

        # Add statistics
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
        ax.set_title('Player Position Scatter Plot', color='#101614', fontsize=18, fontweight='bold', pad=14)
        ax.set_xlabel('Court Width (m)', color='#33443d')
        ax.set_ylabel('Court Length (m)', color='#33443d')
        ax.tick_params(colors='#33443d')
        ax.set_aspect('equal', adjustable='box')

        save_path = os.path.join(self.output_dir, 'scatter_plots', filename)
        fig.savefig(save_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        print(f"Scatter plot saved to: {save_path}")
    
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
            print(f"Error during visualization: {e}")
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
    print(f"\nAnalyzing player position data: {detections_path}")
    
    # Create visualizer
    visualizer = PlayerPositionVisualizer(detections_path, output_dir, fps=fps)
    
    # Execute visualization
    success = visualizer.visualize()
    
    if success:
        print(f"Player position analysis complete, visualizations saved to: {visualizer.output_dir}")
    else:
        print("Player position analysis failed")
        
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


# Test code: allow running this file directly to test visualization
if __name__ == "__main__":
    import sys
    from tkinter import Tk, filedialog
    
    # Use file dialog to select detection file
    print("Please select detections.jsonl file...")
    
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
            title="Select Player Position Detection File",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*.*")],
            initialdir=default_dir
        )
        
        # If user cancels selection, exit
        if not file_path:
            print("No file selected, exiting program")
            sys.exit(0)
            
        # Call analysis function
        success = analyze_player_positions(file_path)
        
        if success:
            print("\nVisualization test completed successfully")
        else:
            print("\nVisualization test failed")
            
    except Exception as e:
        print(f"\nTest error: {e}")
        
    finally:
        try:
            # Close tkinter window
            root.destroy()
        except:
            pass
