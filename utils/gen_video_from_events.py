import tonic
import load_dvs_lips as dvs

import numpy as np
import cv2

def gen_video_from_events(events, output_path, frame_size=tonic.datasets.DVSLip.sensor_size[:2], target_time=1.0, fade_time=0.2, fps=60):
    # Create from events in form of (x, y, p, t)
    # Get max timestamp
    events = events[0]
    print(f"First event: {events[0]}")
    max_time = events[-1][3]
    new_representation = []
    for event in events:
        temp = float(event[3]) / float(max_time) * float(target_time)
        new_representation.append([event[0], event[1], event[2], temp])
    events = np.array(new_representation)
    max_time = events[-1][3]
    print(f"Max time: {max_time}")
    print(f"Last event: {events[-1]}")

    num_frames = int(target_time * fps)
    frame_duration = target_time / num_frames

    print(f"Generating {num_frames} frames at {fps} FPS with {fade_time}s fade time.")

    H, W = frame_size
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    t0 = 0

    for i in range(num_frames):
        t1 = t0 + frame_duration

        recent_events = events[(events[:, 3] >= t0 - fade_time) & (events[:, 3] < t1)]
        frame = np.zeros((H, W, 3), dtype=np.float32)

        for x, y, p, t in recent_events:
            dt = t1 - t
            if dt < fade_time:
                intensity = (fade_time - dt) / fade_time

                x = int(x)
                y = int(y)

                if p > 0:
                    frame[y, x, 2] = 255 * intensity
                else:
                    frame[y, x, 0] = 255 * intensity

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        video_writer.write(frame)

        t0 = t1

    video_writer.release()
    print(f"Video saved to {output_path}")

def play_in_window(video_path, scale=2.0):
    print(f"Playing {video_path}")
    window_name = "Event Video Playback"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame.")
        cap.release()
        return

    h, w = frame.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    cv2.resizeWindow(window_name, new_w, new_h)
    print(f"Resized to {new_w}x{new_h}")

    cap.release()

    while True:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()

if __name__ == "__main__":
    ds = dvs.get_dataset(train=True)
    # get random sample
    sample = ds[np.random.randint(len(ds))]
    gen_video_from_events(sample, "test_vid.mp4", target_time=2.0, fps=60, fade_time=0.2)
    play_in_window("test_vid.mp4", scale=4.0)
