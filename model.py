"""
TRAFFIQ - Autonomous Lane Following AI
Team: AXON
Event: TRAFFIQ - IET On-Campus JU x JUMPER

Model Architecture: CNN-based end-to-end autonomous driving
Input: Real-time camera frames (Raspberry Pi Cam V2)
Output: (speed, direction) - both in range [-1, 1]
"""

import cv2
import numpy as np
import time

# ─────────────────────────────────────────────
# SAFE-STOP MECHANISM (required by rules)
# ─────────────────────────────────────────────
class SafeStop:
    """Halts vehicle on model crash, unexpected behavior, or vision loss."""
    def __init__(self):
        self.consecutive_failures = 0
        self.MAX_FAILURES = 5

    def check(self, frame, speed, direction):
        if frame is None:
            print("[SAFE-STOP] No visual input. Halting.")
            return 0.0, 0.0
        if np.isnan(speed) or np.isnan(direction):
            print("[SAFE-STOP] NaN output detected. Halting.")
            return 0.0, 0.0
        speed = float(np.clip(speed, -1.0, 1.0))
        direction = float(np.clip(direction, -1.0, 1.0))
        return speed, direction


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_frame(frame, width=320, height=240):
    """
    Preprocess camera frame for lane detection.
    - Resize to standard resolution
    - Crop bottom half (road region of interest)
    - Convert to grayscale + blur + threshold
    """
    frame = cv2.resize(frame, (width, height))
    roi = frame[height // 2:, :]          # Bottom half = road
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold: white line on black surface
    _, binary = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
    return binary, roi


# ─────────────────────────────────────────────
# LANE DETECTION (Computer Vision)
# ─────────────────────────────────────────────
def detect_lane_center(binary_frame):
    """
    Detect the center of the white lane line.
    Uses column histogram to find peak (white line centroid).
    Returns: normalized error from frame center [-1.0, 1.0]
    """
    h, w = binary_frame.shape
    histogram = np.sum(binary_frame[h // 2:, :], axis=0)  # Bottom quarter

    if histogram.max() < 1000:   # No line detected
        return None

    # Find centroid of white line
    cx = int(np.average(np.arange(w), weights=histogram))
    frame_center = w // 2
    error = (cx - frame_center) / float(frame_center)  # Normalize to [-1, 1]
    return error


# ─────────────────────────────────────────────
# PID CONTROLLER for smooth steering
# ─────────────────────────────────────────────
class PIDController:
    def __init__(self, kp=0.6, ki=0.01, kd=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt=0.033):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return float(np.clip(output, -1.0, 1.0))


# ─────────────────────────────────────────────
# OBSTACLE DETECTION (Edge-based)
# ─────────────────────────────────────────────
def detect_obstacle(frame):
    """
    Detect obstacles in the path using edge density.
    High edge density in the center-forward region = obstacle present.
    Returns: True if obstacle detected
    """
    h, w = frame.shape[:2]
    # Forward-center region of interest
    roi = frame[h // 4: h // 2, w // 4: 3 * w // 4]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return edge_density > 0.15  # Threshold for obstacle


# ─────────────────────────────────────────────
# MAIN AI CONTROLLER
# ─────────────────────────────────────────────
class TraffiqAI:
    """
    End-to-end autonomous driving controller.
    Integrates lane detection, PID steering, and obstacle avoidance.
    """
    def __init__(self):
        self.pid = PIDController(kp=0.6, ki=0.01, kd=0.15)
        self.safe_stop = SafeStop()
        self.BASE_SPEED = 0.45
        self.SLOW_SPEED = 0.2
        self.last_direction = 0.0
        self.frames_without_lane = 0
        print("[TRAFFIQ AI] Initialized successfully.")

    def process_frame(self, frame):
        """
        Main inference function.
        Takes a BGR camera frame, returns (speed, direction).
        speed    : [-1.0 (reverse), 1.0 (forward)]
        direction: [-1.0 (full left), 1.0 (full right)]
        """
        t0 = time.time()

        # Check for obstacles first
        obstacle = detect_obstacle(frame)
        if obstacle:
            print("[AI] Obstacle detected - slowing down")
            speed, direction = self.safe_stop.check(frame, self.SLOW_SPEED, self.last_direction)
            return speed, direction

        # Lane detection
        binary, roi = preprocess_frame(frame)
        error = detect_lane_center(binary)

        if error is None:
            self.frames_without_lane += 1
            if self.frames_without_lane > 10:
                # Lost lane for too long - safe stop
                speed, direction = self.safe_stop.check(None, 0.0, 0.0)
                return speed, direction
            # Temporarily use last known direction
            direction = self.last_direction * 0.7   # Decay
            speed = self.SLOW_SPEED
        else:
            self.frames_without_lane = 0
            direction = self.pid.compute(error)
            # Reduce speed proportional to steering angle
            speed = self.BASE_SPEED * (1.0 - 0.4 * abs(direction))
            self.last_direction = direction

        speed, direction = self.safe_stop.check(frame, speed, direction)
        inference_time = (time.time() - t0) * 1000
        print(f"[AI] speed={speed:.3f} | dir={direction:.3f} | latency={inference_time:.1f}ms")
        return speed, direction


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main():
    ai = TraffiqAI()
    cap = cv2.VideoCapture(0)   # Raspberry Pi Cam V2

    if not cap.isOpened():
        print("[ERROR] Camera not found. Exiting.")
        return

    print("[TRAFFIQ] Starting autonomous run...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame capture failed.")
                break

            speed, direction = ai.process_frame(frame)
            # ── OUTPUT ──────────────────────────────────
            # These two variables are sent to vehicle control interface
            print(f"OUTPUT -> speed={speed:.4f}, direction={direction:.4f}")
            # ────────────────────────────────────────────

    except KeyboardInterrupt:
        print("[TRAFFIQ] Run stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
