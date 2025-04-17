import os
import cv2
import numpy as np
import logging
import time
from logging.handlers import RotatingFileHandler
from datetime import datetime
from collections import deque
import threading
import json
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

### Helper FUNCTION Call###

from helper import (
    loading_animation,
    measure_execution_time,
    setup_logger,
    create_results_folder,
)

#### Helper Functions End ######


from ultralytics import YOLO

# YOLO model initialization
YOLOmodel = YOLO("yolov8n.pt")

# Class IDs for vehicles: 2 (car), 5 (bus), 7 (truck)
FRONT_VEHICLE_CLASS_ID = [2, 5, 7]
YOLO_class_names = {2: "Car", 5: "Bus", 7: "Truck"}


# Function to calculate optical flow and return mean magnitude
def calculate_optical_flow_parallel(args):
    prev_gray, curr_gray, frame_count, fps, pixel_to_meter_ratio = args
    try:
        # Optical Flow Calculation
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        threshold = 1.0
        magnitude_thresholded = np.where(magnitude > threshold, magnitude, 0)
        mean_magnitude = np.mean(magnitude_thresholded)

        # Distance Calculation
        current_distance = mean_magnitude * pixel_to_meter_ratio

        # Speed Calculation
        travel_time = 1 / fps
        speed_meters_per_second = current_distance / travel_time
        speed_km_per_hour = speed_meters_per_second * 3.6

        return {
            "frame_number": frame_count,
            "distance_pixels": mean_magnitude,
            "distance_meters": current_distance,
            "speed_km_per_hour": speed_km_per_hour,
        }
    except Exception as e:
        return {
            "frame_number": frame_count,
            "distance_pixels": None,
            "distance_meters": None,
            "speed_km_per_hour": None,
        }


def convert_results_to_serializable(results):
    """
    Convert results list to a JSON-serializable format by converting NumPy types to standard Python types.
    """
    serializable_results = []
    for result in results:
        serializable_results.append(
            {
                "frame_number": int(result["frame_number"]),
                "distance_pixels": (
                    float(result["distance_pixels"])
                    if result["distance_pixels"] is not None
                    else None
                ),
                "distance_meters": (
                    float(result["distance_meters"])
                    if result["distance_meters"] is not None
                    else None
                ),
                "speed_km_per_hour": (
                    float(result["speed_km_per_hour"])
                    if result["speed_km_per_hour"] is not None
                    else None
                ),
            }
        )
    return serializable_results


def convert_traffic_density_to_serializable(data):
    serializable_data = []
    for item in data:
        serializable_data.append({
            "frame_number": int(item["frame_number"]),
            "num_vehicles": int(item["num_vehicles"]),
            "traffic_density": str(item["traffic_density"])
        })
    return serializable_data


# Function to process frames in parallel
# @measure_execution_time
# def process_frames_parallel(frame_pairs, fps, pixel_to_meter_ratio):
#     args = [
#         (prev_gray, curr_gray, frame_count, fps, pixel_to_meter_ratio)
#         for (prev_gray, curr_gray, frame_count) in frame_pairs
#     ]
#     with ProcessPoolExecutor() as executor:
#         print("process_frames_parallel - Starting parallel execution.")
#         results = list(executor.map(calculate_optical_flow_parallel, args))
#     return results


def process_frames_parallel(frame_pairs, fps, pixel_to_meter_ratio):
    try:
        print(
            f"process_frames_parallel - Received {len(frame_pairs)} frame pairs for processing."
        )
        logging.info(
            f"process_frames_parallel - Received {len(frame_pairs)} frame pairs for processing."
        )
        args = [
            (prev_gray, curr_gray, frame_count, fps, pixel_to_meter_ratio)
            for (prev_gray, curr_gray, frame_count) in frame_pairs
        ]

        # Debug prints to check the 'args' structure
        print("process_frames_parallel - First 5 'args':")
        logging.info("process_frames_parallel - First 5 'args':")
        for i in range(min(5, len(args))):
            print(args[i])
            logging.info((args[i]))

        with ProcessPoolExecutor(max_workers=10) as executor:
            # Debug print before mapping
            print("process_frames_parallel - Starting parallel execution.")
            logging.info("process_frames_parallel - Starting parallel execution.")
            # results = list(executor.map(calculate_optical_flow_parallel, args))
            results = list(
                tqdm(
                    executor.map(calculate_optical_flow_parallel, args),
                    total=len(frame_pairs),
                )
            )
            # Debug print after mapping
            print("process_frames_parallel - Parallel execution completed.")
            logging.info("process_frames_parallel - Parallel execution completed.")

        # Debug print to check the 'results' structure
        print("process_frames_parallel - First 5 'results':")
        logging.info("process_frames_parallel - First 5 'results':")
        for i in range(min(5, len(args))):
            print(args[i])
            logging.info(results[i])
        return results
    except Exception as e:
        print(f"process_frames_parallel - An error occurred: {e}")
        logging.error(f"process_frames_parallel - An error occurred: {e}")
        return []


def classify_traffic_density(num_vehicles):
    if num_vehicles <= 3:
        return "Low"
    elif num_vehicles <= 6:
        return "Medium"
    else:
        return "Heavy"


@measure_execution_time
def Traffic_density_optical_flow_test(video_path, run_folder, logger=None):
    try:
        if logger == None:
            logger = setup_logger(run_folder, "Traffic_density&optical_flow.log")
        stop_event = threading.Event()  # Event to control animation
        loading_thread = threading.Thread(target=loading_animation, args=(stop_event,))
        loading_thread.start()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Unable to open video file: {video_path}")
            return

        frame_pairs = []
        previous_frame_gray = None
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        pixel_to_meter_ratio =  0.287
        frame_count = 1
        traffic_data = []
        print("Frame Processing started:  \n ")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.info("Video processing completed.")
                break

            print("Processing For Frame Number :", frame_count)  # debug Print
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous_frame_gray is not None:
                frame_pairs.append((previous_frame_gray, frame_gray, frame_count))
            # Traffic Density Calculation
            results = YOLOmodel(frame)
            boxes = results[0].boxes.data.cpu().numpy()
            vehicles = [
                box
                for box in boxes
                if len(box) > 5 and int(box[5]) in FRONT_VEHICLE_CLASS_ID
            ]
            traffic_density = classify_traffic_density(len(vehicles))
            print(f"Traffic Density: {traffic_density}")
            traffic_data.append(
                {
                    "frame_number": frame_count,
                    "num_vehicles": len(vehicles),
                    "traffic_density": traffic_density,
                }
            )
            previous_frame_gray = frame_gray
            frame_count += 1
        cap.release()
        print("All Frames Processed")
        # Process all frame pairs in parallel
        print(
            "Started calculating Speed and Distance using Optical Flow Test on dashcam video .\n"
        )
        logging.info(
            "Started calculating Speed and Distance using Optical Flow Test on dashcam video ."
        )
        stop_event.set()
        results = process_frames_parallel(frame_pairs, fps, pixel_to_meter_ratio)
        print("Calculations Done.")
        # Summarize Results
        total_distance_pixels = sum(
            r["distance_pixels"] for r in results if r["distance_pixels"] is not None
        )
        total_distance_meters = total_distance_pixels * pixel_to_meter_ratio
        average_speed = np.mean(
            [
                r["speed_km_per_hour"]
                for r in results
                if r["speed_km_per_hour"] is not None
            ]
        )
        serializable_traffic_data = convert_traffic_density_to_serializable(traffic_data)
        serializable_results = convert_results_to_serializable(results)
        summary = {
            "traffic_density_data": serializable_traffic_data,
            "optical_flow_data": serializable_results,
            "total_distance_km": float(total_distance_meters / 1000),
            "average_speed_km_per_hour": float(average_speed),
        }
        total_distance_km = total_distance_meters / 1000
        logging.info(f"Total distance traveled: {total_distance_km:.2f} km")
        logging.info(f"Average speed: {average_speed:.2f} km/h")
        print(f"Total distance traveled: {total_distance_km:.2f} km")
        print(f"Average speed: {average_speed:.2f} km/h")
        # Save Results to JSON
        json_path = os.path.join(
            run_folder, "optical_flow_and_Traffic_Density_results.json"
        )
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"Results saved to {json_path}")
        logging.info(f"Results saved to {json_path}")

    except Exception as e:
        stop_event.set()
        logging.error(f"Error during video processing: {e}")
        print(f"Error during video processing: {e}")
        return


# # # Test the script
if __name__ == "__main__":
    video_path = "cut_sample_dashcam.avi"
    # Create the results folder
    run_folder = create_results_folder()
    Traffic_density_optical_flow_test(video_path, run_folder)
