
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import cv2
import os
from pathlib import Path

class JIGSAWS3DVideoGeneratorEnhanced:
    """
    Enhanced JIGSAWS 3D video generator with combined left/right views
    and camera-oriented perspectives for better surgical visualization
    """

    def __init__(self, base_path, subject_id="B001", task="Suturing"):
        """Initialize the enhanced 3D video generator"""
        self.base_path = Path(base_path)
        self.subject_id = subject_id
        self.task = task

        # File paths
        self.kinematics_file = self.base_path / "kinematics" / "AllGestures" / f"{task}_{subject_id}.txt"
        self.transcriptions_file = self.base_path / "transcriptions" / f"{task}_{subject_id}.txt"

        # Data containers
        self.kinematics_data = None
        self.transcriptions_data = None

        # Video parameters
        self.fps = 30  # JIGSAWS standard frame rate
        self.frame_count = 0

        # 3D plot parameters
        self.trail_length = 150  # Number of previous points to show as trail

        # Task-specific configurations
        self.task_configs = {
            'Suturing': {
                'master_view': (20, 45),  # Camera-like angle for master console
                'slave_view': (15, 0),    # Surgical site view (more top-down)
                'colors': {'left': '#FF4444', 'right': '#4444FF'},  # Red/Blue
                'description': 'Suturing Task: Precise needle work through tissue'
            },
            'Needle_Passing': {
                'master_view': (25, 30),  # Slightly different angle
                'slave_view': (25, 30),    # More top-down for needle passing
                'colors': {'left': '#FF6B35', 'right': '#2E8B57'},  # Orange/Green
                'description': 'Needle Passing: Threading through metal hoops'
            },
            'Knot_Tying': {
                'master_view': (20, 60),
                'slave_view': (20, 0),
                'colors': {'left': '#9B59B6', 'right': '#E67E22'},  # Purple/Orange
                'description': 'Knot Tying: Creating secure surgical knots'
            }
        }

    def load_data(self):
        """Load all JIGSAWS data files"""
        print(f"Loading JIGSAWS data files for {self.task}...")

        # Load kinematics data
        try:
            if self.kinematics_file.exists():
                self.kinematics_data = np.loadtxt(self.kinematics_file)
                self.frame_count = len(self.kinematics_data)
                print(f"[OK] Loaded kinematics: {self.kinematics_data.shape}")
            else:
                print(f"[ERROR] Kinematics file not found: {self.kinematics_file}")
                return False
        except Exception as e:
            print(f"[ERROR] Error loading kinematics: {e}")
            return False

        # Load transcriptions with G1/G2 format parsing
        try:
            if self.transcriptions_file.exists():
                transcriptions = []
                with open(self.transcriptions_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue

                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                # Parse frame numbers
                                start_frame = int(parts[0])
                                end_frame = int(parts[1])

                                # Parse gesture ID (handle G1, G2, etc. format)
                                gesture_str = parts[2]
                                if gesture_str.startswith('G'):
                                    gesture_id = int(gesture_str[1:])  # Remove 'G' and convert to int
                                else:
                                    gesture_id = int(gesture_str)  # Direct integer

                                transcriptions.append([start_frame, end_frame, gesture_id])

                            except ValueError:
                                continue  # Skip invalid lines silently

                if transcriptions:
                    self.transcriptions_data = np.array(transcriptions)
                    print(f"[OK] Loaded transcriptions: {len(transcriptions)} gestures")
                else:
                    self.transcriptions_data = None
                    print("[WARNING] No valid transcription data found")
            else:
                print(f"[WARNING] Transcriptions file not found: {self.transcriptions_file}")
                self.transcriptions_data = None
        except Exception as e:
            print(f"[WARNING] Error loading transcriptions: {e}")
            self.transcriptions_data = None

        return self.kinematics_data is not None

    def extract_manipulator_data(self, manipulator_type):
        """Extract specific manipulator data"""
        if self.kinematics_data is None:
            return None

        # Define column indices based on JIGSAWS format
        if manipulator_type == "master_left":
            pos_cols = [0, 1, 2]
            vel_cols = [12, 13, 14]
            gripper_col = 18
        elif manipulator_type == "master_right":
            pos_cols = [19, 20, 21]
            vel_cols = [31, 32, 33]
            gripper_col = 37
        elif manipulator_type == "slave_right":
            pos_cols = [38, 39, 40]
            vel_cols = [50, 51, 52]
            gripper_col = 56
        elif manipulator_type == "slave_left":
            pos_cols = [57, 58, 59]
            vel_cols = [69, 70, 71]
            gripper_col = 75
        else:
            return None

        return {
            'positions': self.kinematics_data[:, pos_cols],
            'velocities': self.kinematics_data[:, vel_cols],
            'gripper': self.kinematics_data[:, gripper_col],
            'time': np.arange(len(self.kinematics_data)) / self.fps
        }

    def get_best_video_writer(self, output_filename):
        """Get the best available video writer and correct filename"""
        available_writers = list(animation.writers.list())

        # Priority order: ffmpeg > pillow
        if 'ffmpeg' in available_writers:
            try:
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=self.fps, metadata=dict(artist='JIGSAWS Enhanced Visualizer'), bitrate=1800)
                print("Using ffmpeg writer")
                return writer, output_filename  # Keep original .mp4 extension
            except Exception:
                pass

        if 'pillow' in available_writers:
            try:
                Writer = animation.writers['pillow']
                writer = Writer(fps=self.fps, metadata=dict(artist='JIGSAWS Enhanced Visualizer'))
                # Change extension to .gif for pillow writer
                gif_filename = output_filename.replace('.mp4', '.gif')
                print("Using pillow writer (GIF output)")
                return writer, gif_filename
            except Exception:
                pass

        print("[ERROR] No suitable video writers available")
        return None, output_filename

    def create_combined_trajectory_video(self, manipulator_group, output_filename):
        """Create combined 3D trajectory video for left+right manipulators"""
        print(f"Creating combined 3D video for {manipulator_group}...")

        # Extract data for both left and right
        if manipulator_group == "master":
            left_data = self.extract_manipulator_data("master_left")
            right_data = self.extract_manipulator_data("master_right")
            view_angle = self.task_configs[self.task]['master_view']
            title_suffix = "Master Console (Surgeon Hands)"
        elif manipulator_group == "slave":
            left_data = self.extract_manipulator_data("slave_left")
            right_data = self.extract_manipulator_data("slave_right")
            view_angle = self.task_configs[self.task]['slave_view']
            title_suffix = "Slave Tools (Surgical Site)"
        else:
            print(f"[ERROR] Invalid manipulator group: {manipulator_group}")
            return False

        if left_data is None or right_data is None:
            print(f"[ERROR] Failed to extract {manipulator_group} data")
            return False

        # Set up the figure and 3D axes
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Get task configuration
        colors = self.task_configs[self.task]['colors']

        # Combine positions for scaling
        all_positions = np.vstack([left_data['positions'], right_data['positions']])
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        z_min, z_max = all_positions[:, 2].min(), all_positions[:, 2].max()

        # Add padding for better visualization
        padding = 0.15
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        ax.set_xlim([x_min - padding * x_range, x_max + padding * x_range])
        ax.set_ylim([y_min - padding * y_range, y_max + padding * y_range])
        ax.set_zlim([z_min - padding * z_range, z_max + padding * z_range])

        # Set labels and title
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Y Position (mm)', fontsize=12)
        ax.set_zlabel('Z Position (mm)', fontsize=12)

        # Enhanced title with task information
        ax.set_title(f'JIGSAWS {self.task} - {title_suffix}\nSubject: {self.subject_id} | {self.task_configs[self.task]["description"]}', 
                    fontsize=14, fontweight='bold')

        # Set camera-like view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        # Initialize plot elements for LEFT manipulator
        left_trail, = ax.plot([], [], [], '-', color=colors['left'], alpha=0.6, 
                             linewidth=3, label='Left Tool Trail')
        left_current, = ax.plot([], [], [], 'o', color=colors['left'], 
                               markersize=12, markeredgecolor='white', markeredgewidth=2, 
                               label='Left Tool Current')
        left_start, = ax.plot([left_data['positions'][0, 0]], 
                             [left_data['positions'][0, 1]], 
                             [left_data['positions'][0, 2]], 
                             's', color=colors['left'], markersize=15, 
                             markeredgecolor='black', markeredgewidth=2,
                             alpha=0.8, label='Left Start')

        # Initialize plot elements for RIGHT manipulator
        right_trail, = ax.plot([], [], [], '-', color=colors['right'], alpha=0.6, 
                              linewidth=3, label='Right Tool Trail')
        right_current, = ax.plot([], [], [], 'o', color=colors['right'], 
                                markersize=12, markeredgecolor='white', markeredgewidth=2,
                                label='Right Tool Current')
        right_start, = ax.plot([right_data['positions'][0, 0]], 
                              [right_data['positions'][0, 1]], 
                              [right_data['positions'][0, 2]], 
                              's', color=colors['right'], markersize=15,
                              markeredgecolor='black', markeredgewidth=2,
                              alpha=0.8, label='Right Start')

        # Information display panels
        time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                             verticalalignment='top', 
                             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

        left_info = ax.text2D(0.02, 0.85, '', transform=ax.transAxes, fontsize=10,
                             verticalalignment='top',
                             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['left'], alpha=0.7))

        right_info = ax.text2D(0.52, 0.85, '', transform=ax.transAxes, fontsize=10,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['right'], alpha=0.7))

        gesture_text = ax.text2D(0.02, 0.70, '', transform=ax.transAxes, fontsize=11,
                                verticalalignment='top',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        # Enhanced legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # Gesture dictionary (comprehensive for all tasks)
        gestures = {
            1: "Reaching for needle",
            2: "Positioning needle/tool", 
            3: "Pushing needle through tissue",
            4: "Transferring needle L<->R",
            5: "Moving to center",
            6: "Pulling suture",
            8: "Orienting needle",
            9: "Tightening suture",
            10: "Loosening suture",
            11: "Dropping/completing",
            12: "Knot tying motion",
            13: "Knot securing",
            14: "Knot tightening", 
            15: "Final knot adjustment"
        }

        def get_current_gesture(frame_idx):
            """Get current gesture for given frame"""
            if self.transcriptions_data is None:
                return "No gesture data"
            for start, end, gesture_id in self.transcriptions_data:
                if start <= frame_idx <= end:
                    return gestures.get(gesture_id, f"Gesture {gesture_id}")
            return "Transition"

        def animate(frame):
            """Enhanced animation function for combined view"""
            # Calculate trail indices
            trail_start = max(0, frame - self.trail_length)
            trail_end = frame + 1

            # Update LEFT manipulator
            if trail_end > trail_start:
                left_positions = left_data['positions']
                trail_x = left_positions[trail_start:trail_end, 0]
                trail_y = left_positions[trail_start:trail_end, 1]
                trail_z = left_positions[trail_start:trail_end, 2]
                left_trail.set_data_3d(trail_x, trail_y, trail_z)

            left_current.set_data_3d([left_data['positions'][frame, 0]], 
                                    [left_data['positions'][frame, 1]], 
                                    [left_data['positions'][frame, 2]])

            # Update RIGHT manipulator
            if trail_end > trail_start:
                right_positions = right_data['positions']
                trail_x = right_positions[trail_start:trail_end, 0]
                trail_y = right_positions[trail_start:trail_end, 1]
                trail_z = right_positions[trail_start:trail_end, 2]
                right_trail.set_data_3d(trail_x, trail_y, trail_z)

            right_current.set_data_3d([right_data['positions'][frame, 0]], 
                                     [right_data['positions'][frame, 1]], 
                                     [right_data['positions'][frame, 2]])

            # Update information displays
            current_time = left_data['time'][frame]
            left_vel = np.linalg.norm(left_data['velocities'][frame])
            right_vel = np.linalg.norm(right_data['velocities'][frame])
            left_gripper = left_data['gripper'][frame]
            right_gripper = right_data['gripper'][frame]
            current_gesture = get_current_gesture(frame)

            # Time and frame info
            time_text.set_text(f'Time: {current_time:.2f}s | Frame: {frame}/{len(left_data["positions"])-1}\n'
                              f'Progress: {(frame/len(left_data["positions"])*100):.1f}%')

            # Left tool info
            left_info.set_text(f'LEFT TOOL\nVel: {left_vel:.1f} mm/s\nGripper: {left_gripper:.1f}°')

            # Right tool info  
            right_info.set_text(f'RIGHT TOOL\nVel: {right_vel:.1f} mm/s\nGripper: {right_gripper:.1f}°')

            # Current gesture
            gesture_text.set_text(f'Current Gesture:\n{current_gesture}')

            return (left_trail, left_current, right_trail, right_current, 
                   time_text, left_info, right_info, gesture_text)

        # Create animation
        print(f"Generating combined animation with {len(left_data['positions'])} frames...")
        anim = animation.FuncAnimation(fig, animate, frames=len(left_data['positions']), 
                                     interval=1000/self.fps, blit=False, repeat=False)

        # Get the best available writer and correct filename
        writer, corrected_filename = self.get_best_video_writer(output_filename)
        if writer is None:
            print("[ERROR] No video writer available")
            plt.close(fig)
            return False

        # Save video
        print(f"Saving combined video: {corrected_filename}")

        try:
            anim.save(corrected_filename, writer=writer)
            print(f"[OK] Combined video saved successfully: {corrected_filename}")
            return True

        except Exception as e:
            print(f"[ERROR] Error saving video: {e}")
            return False
        finally:
            plt.close(fig)

    def create_all_videos(self, output_dir="jigsaws_3d_videos_enhanced"):
        """Create enhanced combined videos for master and slave manipulators"""
        if not self.load_data():
            return False

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\n[VIDEO] Creating enhanced videos for {self.task} task...")

        success_count = 0
        created_files = []

        # Create master combined video (left + right hands at console)
        master_output = output_path / f"{self.task}_{self.subject_id}_master_combined.mp4"
        if self.create_combined_trajectory_video("master", str(master_output)):
            success_count += 1
            # Check what file was actually created
            gif_file = master_output.with_suffix('.gif')
            if gif_file.exists():
                created_files.append(str(gif_file))
            elif master_output.exists():
                created_files.append(str(master_output))

        # Create slave combined video (left + right tools in surgical site)
        slave_output = output_path / f"{self.task}_{self.subject_id}_slave_combined.mp4"
        if self.create_combined_trajectory_video("slave", str(slave_output)):
            success_count += 1
            # Check what file was actually created
            gif_file = slave_output.with_suffix('.gif')
            if gif_file.exists():
                created_files.append(str(gif_file))
            elif slave_output.exists():
                created_files.append(str(slave_output))

        print(f"\n[OK] Successfully created {success_count}/2 enhanced videos")
        print(f"[FOLDER] Videos saved in: {output_path.absolute()}")

        if created_files:
            print("\n[FILES] Enhanced files created:")
            for file_path in created_files:
                print(f"  [VIDEO] {Path(file_path).name}")

        # Create summary report (with fixed encoding)
        self.create_video_summary(output_path, created_files)

        return success_count > 0

    def create_video_summary(self, output_path, created_files):
        """Create a summary report of generated videos - FIXED VERSION"""
        summary = f"""# JIGSAWS Enhanced 3D Visualization Summary

## Task: {self.task}
## Subject: {self.subject_id} 
## Generated: {len(created_files)} videos
## Duration: {self.frame_count/self.fps:.1f} seconds ({self.frame_count} frames)

## Enhanced Features:
[OK] Combined left+right manipulators in single view
[OK] Camera-oriented viewing angles for realistic perspective  
[OK] Color-coded trajectories (Red/Blue or task-specific)
[OK] Real-time information overlays
[OK] Gesture labeling with timeline
[OK] Professional presentation quality

## Files Created:
"""
        for file_path in created_files:
            filename = Path(file_path).name
            if 'master' in filename:
                summary += f"[HANDS] {filename} - Surgeon hand movements at console\n"
            else:
                summary += f"[TOOLS] {filename} - Robot tool movements in surgical site\n"

        summary += f"""
## Video Specifications:
- Frame Rate: {self.fps} fps
- Trail Length: {self.trail_length} points
- View Angles: Master {self.task_configs[self.task]['master_view']}, Slave {self.task_configs[self.task]['slave_view']}
- Colors: {self.task_configs[self.task]['colors']}

## Usage Instructions:
1. Preview videos to verify 3D trajectories
2. Use in presentation alongside camera footage
3. Recommended layout: [Camera Left] [Camera Right]
                       [Master Combined] [Slave Combined]
4. Perfect synchronization maintained at {self.fps} fps

## Task Description: 
{self.task_configs[self.task]['description']}
"""

        # FIXED: Use UTF-8 encoding to handle any Unicode characters
        try:
            with open(output_path / "video_summary.txt", 'w', encoding='utf-8') as f:
                f.write(summary)
            print("[OK] Created video summary report")
        except Exception as e:
            print(f"[WARNING] Could not create summary file: {e}")
            # Continue without summary file - videos are still created successfully

def main():
    """Main function for enhanced JIGSAWS video generation"""

    print("[SYSTEM] JIGSAWS Enhanced 3D Video Generator")
    print("=" * 60)
    print("Features:")
    print("[OK] Combined left+right views")  
    print("[OK] Camera-oriented angles")
    print("[OK] Multi-task support") 
    print("[OK] Enhanced visualizations")
    print("=" * 60)

    # Configuration - MODIFY THESE PATHS AND SETTINGS
    base_path = r"jigsaw_datasets"

    # Task selection - CHANGE THIS FOR DIFFERENT TASKS
    task = "Needle_Passing"  # Options: "Suturing", "Needle_Passing", "Knot_Tying"
    subject_id = "I005"

    # Construct full path for selected task
    full_path = Path(base_path) / task

    print(f"[TARGET] Selected Task: {task}")
    print(f"[SUBJECT] Subject: {subject_id}")
    print(f"[PATH] Data Path: {full_path}")

    # Create the enhanced generator
    generator = JIGSAWS3DVideoGeneratorEnhanced(full_path, subject_id, task)

    # Check if required files exist
    print("\n[CHECK] Checking for required files...")
    if not generator.kinematics_file.exists():
        print(f"[ERROR] Kinematics file missing: {generator.kinematics_file}")
        print("\n[INFO] Available tasks in your data folder:")
        base_path_obj = Path(base_path)
        if base_path_obj.exists():
            for task_dir in base_path_obj.iterdir():
                if task_dir.is_dir():
                    print(f"   [FOLDER] {task_dir.name}")
        print("\nPlease update the 'task' variable to match your available data.")
        return

    print(f"[OK] Found kinematics file: {generator.kinematics_file.name}")

    # Generate enhanced videos
    print("\n[START] Generating enhanced 3D videos...")
    success = generator.create_all_videos()

    if success:
        print("\n[SUCCESS] Enhanced 3D videos created!")
        print("\n[INFO] What you got:")
        print("[HANDS] Master Combined: Surgeon hand movements (both hands)")
        print("[TOOLS] Slave Combined: Robot tool movements (both tools)")
        print("\n[NEXT] Next steps:")
        print("1. Preview your enhanced videos")
        print("2. Use 2x2 layout: [Camera L] [Camera R]")
        print("                  [Master]   [Slave]")
        print("3. Perfect sync maintained for presentation!")
    else:
        print("\n[ERROR] Video generation failed. Check error messages above.")

if __name__ == "__main__":
    main()
