import os

os.environ["ROBOFLOW_API_KEY"] = "dQyGKn5NaUVsYp7jKCXk"
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
import supervision as sv
from inference import get_model
from ultralytics import YOLO
import ollama

### Helper FUNCTION ###
from helper import (
    loading_animation,
    measure_execution_time,
    setup_logger,
    create_results_folder,
)


#### Helper Functions End ######


# Initialize the pre-trained YOLOv8 and Roboflow lane detection model
model = get_model(model_id="lane-detection-segmentation-edyqp/7")
YOLOmodel = YOLO("yolov8n.pt")

# Class IDs for vehicles: 2 (car), 5 (bus), 7 (truck)
FRONT_VEHICLE_CLASS_ID = [2, 5, 7]  # Class IDs for car, bus, truck in YOLOv8.
YOLO_class_names = {2: "Car", 5: "Bus", 7: "Truck"}


@measure_execution_time
def ollama_process(model, img_path, prompt):
    try:
        # Use the resized image for processing
        with open(img_path, "rb") as file:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [file.read()],
                    },
                ],
            )

        output = response["message"]["content"]
        return output
    except:
        return None


# New ROI function
def set_dynamic_roi(frame):
    try:
        height, width = frame.shape[:2]
        roi_top = int(height * 0.6)
        roi_bottom = height
        roi = frame[roi_top:roi_bottom, :]
        return roi, roi_top, roi_bottom
    except Exception as e:
        logging.error(f"Error while setting Region Of Interest: {e}")
        return None, None, None


# New function to process each frame for lane detection
def process_frame_for_lane_detection(frame):
    try:
        # Perform inference
        results = model.infer(frame)[0]

        # Convert inference results to detections
        detections = sv.Detections.from_inference(results)
        lane_lines = []

        if detections is None:
            logging.warning("No detections found in the current frame.")
            return [], 0, frame

        # Validate detections.mask, class_id, and xyxy
        if (
            detections.mask is not None
            and detections.class_id is not None
            and detections.xyxy is not None
        ):
            for i, mask in enumerate(detections.mask):
                class_id = detections.class_id[i]
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                lane_lines.append({"coords": (x1, y1, x2, y2), "class_id": class_id})

            # Annotate the image with our inference results
            label_annotator = sv.LabelAnnotator()
            mask_annotator = sv.MaskAnnotator()
            frame = mask_annotator.annotate(scene=frame, detections=detections)
            frame = label_annotator.annotate(scene=frame, detections=detections)
        else:
            logging.warning(
                "Incomplete detection data: mask, class_id, or xyxy is None."
            )

        # Set dynamic ROI
        roi_top = set_dynamic_roi(frame)[1]
        return lane_lines, roi_top, frame

    except cv2.error as e:
        logging.error(f"OpenCV error during lane detection processing: {e}")
        print(f"OpenCV error during lane detection processing: {e}")
        return [], 0, frame
    except AttributeError as e:
        logging.error(f"Attribute error during lane detection processing: {e}")
        print(f"Attribute error during lane detection processing: {e}")
        return [], 0, frame
    except Exception as e:
        logging.error(f"Unexpected error during lane detection processing: {e}")
        print(f"Unexpected error during lane detection processing: {e}")
        return [], 0, frame


def detect_lane_discipline_violation(frame, lane_lines, roi_top):
    try:
        height, width = frame.shape[:2]
        # Adjust car position based on dashcam offset (assuming offset is 10% towards the left)
        car_position = int(width * 0.6)

        for line_data in lane_lines:
            x1, y1, x2, y2 = line_data["coords"]
            class_id = line_data["class_id"]

            # Calculate the slope of the lane line
            if x2 != x1:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
            else:
                slope = float("inf")  # Infinite slope for vertical lines

            # Extrapolate the lane line towards the bottom of the ROI
            y_bottom = int(height * 0.8)  # Bottom of the ROI
            x_bottom = (
                int(x1 + (y_bottom - y1) / slope) if slope != float("inf") else x1
            )

            # Check if the extrapolated line crosses the car's position near the bottom
            if x1 < car_position < x2 or x2 < car_position < x1:
                if (
                    abs(x_bottom - car_position) < width * 0.1
                ):  # Allow some tolerance (10% of frame width)
                    return True, class_id

        return False, None
    except Exception as e:
        logging.error(f"Error during Lane discipline detection: {e}")
        return False, None


def convert_to_serializable(data):
    """
    Recursively convert NumPy data types to Python native types for JSON serialization.
    Args:
        data: The data to be converted.
    Returns:
        A JSON-serializable version of the data.
    """
    if isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def classify_subcategories(
    active_violation,
    frame_count,
    traffic_density_data,
    optical_flow_data,
    violation_frames_dir,
):
    """
    Classifies lane discipline violations into subcategories.
    """
    sub_violations = active_violation.get("sub_violations", {})
    tracking_data = active_violation.setdefault(
        "tracking_data", {}
    )  # For intermediate states

    # Get data for the current frame
    traffic_density_framedata = traffic_density_data.get(frame_count, {})
    if traffic_density_framedata:
        traffic_density = traffic_density_framedata.get("traffic_density")
    else:
        traffic_density = "Low"
    optical_flow = optical_flow_data.get(frame_count, {})
    if optical_flow:
        current_speed = optical_flow.get("speed_km_per_hour", 0)
    else:
        current_speed = 0

    # Safe/Unsafe Lane Change
    if active_violation["class_id"] in [0, 2]:  # Dashed lines only
        if "unsafe_lane_change" not in sub_violations and traffic_density == "Heavy":
            sub_violations["unsafe_lane_change"] = {
                "start_frame": active_violation["start_frame"],
                "start_frame_path": active_violation["start_frame_path"],
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"frame_{frame_count}.jpg"
                ),
            }
        elif "safe_lane_change" not in sub_violations and traffic_density in [
            "Medium",
            "Low",
        ]:
            sub_violations["safe_lane_change"] = {
                "start_frame": active_violation["start_frame"],
                "start_frame_path": active_violation["start_frame_path"],
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"frame_{frame_count}.jpg"
                ),
            }

    # crossing on a Solid Line
    if active_violation["class_id"] in [1, 3]:  # Solid lines
        tracking_data.setdefault("solid_line_overtaking", 0)
        tracking_data["solid_line_overtaking"] += 1

        if (
            tracking_data["solid_line_overtaking"] >= 2
            and "solid_line_overtaking" not in sub_violations
        ):
            sub_violations["solid_line_overtaking"] = {
                "start_frame": active_violation["start_frame"],
                "start_frame_path": active_violation["start_frame_path"],
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"frame_{frame_count}.jpg"
                ),
            }

    # Driving in the Middle of a Dashed Road
    if active_violation["class_id"] in [0, 2]:  # Dashed lines
        tracking_data.setdefault("dash_line_middle", 0)
        tracking_data["dash_line_middle"] += 1

        if (
            tracking_data["dash_line_middle"] >= 2
            and "dash_line_middle" not in sub_violations
        ):
            sub_violations["dash_line_middle"] = {
                "start_frame": active_violation["start_frame"],
                "start_frame_path": active_violation["start_frame_path"],
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"frame_{frame_count}.jpg"
                ),
            }

    # Weaving
    weaving_threshold = 100 if current_speed < 30 else 50
    tracking_data.setdefault(
        "weaving", {"lane_changes": [], "threshold": weaving_threshold}
    )

    # Track lane change frames
    tracking_data["weaving"]["lane_changes"].append(frame_count)

    # Remove consecutive frames for weaving check
    non_consecutive_changes = [
        f
        for i, f in enumerate(tracking_data["weaving"]["lane_changes"])
        if i == 0 or f - tracking_data["weaving"]["lane_changes"][i - 1] > 1
    ]
    if len(non_consecutive_changes) >= 3 and "weaving" not in sub_violations:
        sub_violations["weaving"] = {
            "start_frame": non_consecutive_changes[0],
            "start_frame_path": active_violation["start_frame_path"],
            "end_frame": non_consecutive_changes[-1],
            "end_frame_path": os.path.join(
                violation_frames_dir, f"frame_{frame_count}.jpg"
            ),
        }
    print("Tracking Data:", tracking_data)
    print("Sub Violations:", sub_violations)
    return sub_violations


def detect_overtaking(
    frame,
    previous_vehicles,
    frame_count,
    violation_frames_dir,
    annotated_frame,
    overtaking_state,
):
    """
    Detects overtaking based on YOLO-detected vehicles in front with better filtering and cooldown logic.
    """
    if overtaking_state is None:
        overtaking_state = {
            "active": False,
            "cooldown": 0,
            "last_frame": None,
            "last_box": None,
            "end_frame": None,
        }

    # Extract current state
    overtaking_active = overtaking_state.get("active", False)
    cooldown_counter = overtaking_state.get("cooldown", 0)

    # Handle cooldown: decrement and finalize event
    if overtaking_active and cooldown_counter > 0:
        overtaking_state["cooldown"] -= 1
        overtaking_state["end_frame"] = frame_count
        overtaking_state["end_frame_path"] = os.path.join(
            violation_frames_dir, f"overtaking_frame_{frame_count}.jpg"
        )
        return False, overtaking_state

    # Run YOLO inference to detect vehicles
    results = YOLOmodel(frame)
    boxes = results[0].boxes.data.cpu().numpy()

    # Filter vehicles in front using predefined class IDs
    vehicles = [
        {
            "class_id": int(box[5]),
            "box": box[:4],  # (x1, y1, x2, y2)
            "confidence": box[4],
        }
        for box in boxes
        if len(box) > 5 and int(box[5]) in FRONT_VEHICLE_CLASS_ID
    ]

    overtaking_detected = False
    for vehicle in vehicles:
        box = vehicle["box"]
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Compare with previous vehicles and active event
        for prev_vehicle in previous_vehicles:
            prev_x1, prev_y1, prev_x2, prev_y2 = map(int, prev_vehicle["box"])
            prev_width = prev_x2 - prev_x1
            prev_height = prev_y2 - prev_y1
            prev_area = prev_width * prev_height

            # Check if the detected vehicle is significantly closer
            if (
                area > prev_area and y2 > prev_y2
            ):  # Larger box and lower bottom y-coordinate
                # Additional filter: Ensure spatial separation (e.g., 10% of frame height)
                if overtaking_state.get("last_box") is not None:
                    _, _, last_x2, last_y2 = map(int, overtaking_state["last_box"])
                    if (
                        abs(y2 - last_y2) < frame.shape[0] * 0.1
                    ):  # Skip if not moving significantly
                        continue

                # Check time gap: Ensure at least 30 frames since last detection
                if overtaking_state.get("last_frame") is not None:
                    if frame_count - overtaking_state["last_frame"] < 30:
                        continue

                overtaking_detected = True
                break

        if overtaking_detected:
            # Save the violation frame
            overtaking_frame_path = os.path.join(
                violation_frames_dir, f"overtaking_frame_{frame_count}.jpg"
            )
            cv2.imwrite(overtaking_frame_path, annotated_frame)

            # Update overtaking state
            overtaking_state.update(
                {
                    "active": True,
                    "cooldown": 100,  # Set cooldown period
                    "start_frame": frame_count,
                    "start_frame_path": overtaking_frame_path,
                    "class_id": vehicle["class_id"],
                    "box": box,
                    "last_frame": frame_count,
                    "last_box": box,
                }
            )
            return overtaking_detected, overtaking_state

    # Finalize active event if no new detections
    if overtaking_active:
        overtaking_state.update(
            {
                "active": False,
                "cooldown": 0,
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"overtaking_frame_{frame_count}.jpg"
                ),
            }
        )

    return False, overtaking_state


def classify_subcategories_overtaking(
    active_violation,
    frame_count,
    traffic_density_data,
    optical_flow_data,
    violation_frames_dir,
    lane_violations,
):
    """
    Classifies overtaking violations into subcategories: Safe, Unsafe Overtaking, and Overtaking on a Solid Line.
    """
    if isinstance(active_violation, list):
        active_violation = active_violation[0] if active_violation else {}

    if not isinstance(active_violation, dict):
        raise ValueError(
            "active_violation must be a dictionary or a list containing a dictionary."
        )

    sub_violations = active_violation.get("sub_violations", {})
    tracking_data = active_violation.setdefault("tracking_data", {})

    # Get data for the current frame
    traffic_density_framedata = traffic_density_data.get(frame_count, {})
    traffic_density = traffic_density_framedata.get("traffic_density", "Low Density")
    optical_flow = optical_flow_data.get(frame_count, {})
    current_speed = optical_flow.get("speed_km_per_hour", 0)

    start_frame = active_violation.get("start_frame", frame_count)
    start_frame_path = active_violation.get("start_frame_path", "")

    prompt = """
        Analyze the given image to assess overtaking safety from the perspective of the vehicle with the dashcam (not any other vehicles on the road).
        Classify each category as follows:

        Road Size: Narrow, One-lane, or Two-lane.
        Weather Condition: Sunny and Clear, Cloudy and Clear, Cloudy and Raining Lightly, Cloudy and Raining Heavily, Foggy But Visible, Foggy and Limited Visibility, Night and Clear, or Night and Foggy.
        Visibility: Clear Visibility or Limited Visibility.

        Format: 
        Road Size: [Classification]
        Weather Condition: [Classification]
        Visibility: [Classification]
    """

    # Extract road size, weather, and visibility using LLM processing
    if start_frame_path:
        llm_response = ollama_process(
            model="llava-llama3", img_path=start_frame_path, prompt=prompt
        )
        road_data = dict(
            line.split(": ") for line in llm_response.split("\n") if ": " in line
        )
        road_size = road_data.get("Road Size", "Two-lane")
        weather_condition = road_data.get("Weather Condition", "Sunny and Clear")
        visibility = road_data.get("Visibility", "Clear Visibility")
    else:
        road_size = "Two-lane"
        weather_condition = "Sunny and Clear"
        visibility = "Clear Visibility"

    # Unsafe or Safe Overtaking conditions
    unsafe_conditions = {
        "weather": {
            "Cloudy and Raining Heavily",
            "Night and Foggy",
        },
        "traffic_density": {"High Density"},
        "speed": lambda speed: speed > 61,
    }

    for value, condition in [
        (road_size, "road_size"),
        (weather_condition, "weather"),
        (traffic_density, "traffic_density"),
        (visibility, "visibility"),
        (current_speed, "speed"),
    ]:
        if callable(unsafe_conditions.get(condition)):
            if unsafe_conditions[condition](value):
                sub_violations["Unsafe_overtaking"] = {
                    "start_frame": start_frame,
                    "start_frame_path": start_frame_path,
                    "end_frame": frame_count,
                    "end_frame_path": os.path.join(
                        violation_frames_dir, f"frame_{frame_count}.jpg"
                    ),
                }
                tracking_data["overtaking"] = {
                    "condition": condition,
                    "value": value,
                }
                break
        elif value in unsafe_conditions.get(condition, {}):
            sub_violations["Unsafe_overtaking"] = {
                "start_frame": start_frame,
                "start_frame_path": start_frame_path,
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"frame_{frame_count}.jpg"
                ),
            }
            tracking_data["overtaking"] = {
                "condition": condition,
                "value": value,
            }
            break

    if "Unsafe_overtaking" not in sub_violations:
        sub_violations["Safe_for_Overtaking"] = {
            "start_frame": start_frame,
            "start_frame_path": start_frame_path,
            "end_frame": frame_count,
            "end_frame_path": os.path.join(
                violation_frames_dir, f"frame_{frame_count}.jpg"
            ),
        }

    # Overtaking on a Solid Line supersedes other categories
    overtaking_frames = range(active_violation.get("start_frame", 0), frame_count + 1)
    for frame in overtaking_frames:
        lane_violation = lane_violations.get(frame, {})
        if "solid_line_overtaking" in lane_violation.get("sub_violations", {}):
            sub_violations.clear()  # Clear other subcategories
            sub_violations["Overtaking_on_a_Solid_Line"] = {
                "start_frame": max(
                    active_violation.get("start_frame", 0),
                    lane_violation["sub_violations"]["solid_line_overtaking"][
                        "start_frame"
                    ],
                ),
                "start_frame_path": lane_violation["sub_violations"][
                    "solid_line_overtaking"
                ]["start_frame_path"],
                "end_frame": frame_count,
                "end_frame_path": os.path.join(
                    violation_frames_dir, f"frame_{frame_count}.jpg"
                ),
            }
            tracking_data["solid_line_overtaking"] = {
                "frame": frame,
                "details": lane_violation.get("sub_violations", {}).get(
                    "solid_line_overtaking"
                ),
            }
            break

    return sub_violations


@measure_execution_time
def Lane_discipline_only(video_path, json_file, run_folder, logger=None):
    try:
        # Initialize folders and logger
        if logger == None:
            logger = setup_logger(run_folder, "Lane_disciplineandOvertaking.log")

        stop_event = threading.Event()  # Event to control animation
        loading_thread = threading.Thread(target=loading_animation, args=(stop_event,))
        loading_thread.start()

        # Load JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        traffic_density_data = {
            d["frame_number"]: d for d in data.get("traffic_density_data", [])
        }
        optical_flow_data = {
            d["frame_number"]: d for d in data.get("optical_flow_data", [])
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Unable to open video file: {video_path}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = 1

        # Track violations
        lane_discipline_violations = []
        overtaking_violations = []
        active_violations = {}

        # Previous vehicles for overtaking detection
        previous_vehicles = []

        # Initialize overtaking state
        overtaking_state = {
            "active": False,
            "cooldown": 0,
            "last_frame": None,
            "last_box": None,  # Initialize last_box
        }

        # Create a folder for saving violation frames
        violation_frames_dir = os.path.join(run_folder, "violation_frames")
        os.makedirs(violation_frames_dir, exist_ok=True)
        output_video_path = os.path.join(run_folder, "lane_line_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        print("Frame Processing started:  \n Detected Fps:", fps)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.info("Video processing completed.")
                break

            print("Processing For Frame Number:", frame_count)

            # Lane detection
            lane_lines, roi_top, annotated_frame = process_frame_for_lane_detection(
                frame
            )

            # Lane discipline violation detection
            violation_detected, class_id = detect_lane_discipline_violation(
                frame, lane_lines, roi_top
            )

            # Handle lane discipline violations
            if violation_detected:
                print(
                    f"Lane Discipline Violation detected at Frame {frame_count} with Class ID: {class_id}"
                )
                logging.info(
                    f"Lane Discipline Violation detected at Frame {frame_count} with Class ID: {class_id}"
                )

                # Save the frame corresponding to the violation
                violation_frame_path = os.path.join(
                    violation_frames_dir, f"lane_violation_frame_{frame_count}.jpg"
                )
                cv2.imwrite(violation_frame_path, annotated_frame)

                # Track active lane discipline violation
                if frame_count not in active_violations:
                    active_violations[frame_count] = {
                        "start_frame": frame_count,
                        "start_frame_path": violation_frame_path,
                        "class_id": class_id,
                        "cooldown_counter": 100,
                        "sub_violations": {},
                    }

            # Check and classify subcategories for active lane violations
            for start_frame in list(active_violations.keys()):
                violation = active_violations[start_frame]
                violation["cooldown_counter"] -= 1

                sub_violations = classify_subcategories(
                    violation,
                    frame_count,
                    traffic_density_data,
                    optical_flow_data,
                    violation_frames_dir,
                )

                if isinstance(sub_violations, dict):  # Ensure it's a valid dictionary
                    violation["sub_violations"].update(sub_violations)

                if violation["cooldown_counter"] <= 0:
                    # Finalize and save lane discipline violation
                    violation["end_frame"] = frame_count
                    violation["end_frame_path"] = os.path.join(
                        violation_frames_dir, f"lane_violation_frame_{frame_count}.jpg"
                    )
                    lane_discipline_violations.append(violation)
                    del active_violations[start_frame]

            # Handle overtaking detection
            overtaking_detected, overtaking_state = detect_overtaking(
                frame,
                previous_vehicles,
                frame_count,
                violation_frames_dir,
                annotated_frame,
                overtaking_state,
            )

            if overtaking_detected:
                print(f"Overtaking detected at Frame {frame_count}")
                logging.info(f"Overtaking detected at Frame {frame_count}")

                # Save overtaking violation when detected
                overtaking_violation = {
                    "parameter": "overtaking",
                    "start_frame": overtaking_state["start_frame"],
                    "start_frame_path": overtaking_state["start_frame_path"],
                    "end_frame": None,  # End frame will be updated when cooldown ends
                    "end_frame_path": None,
                    "class_id": overtaking_state["class_id"],
                    "sub_violations": {},  # Placeholder for future subcategories
                    "box": overtaking_state["box"],
                    "tracking_data": {},  # Initialize tracking data for overtaking
                }
                # Classify overtaking subcategories
                sub_violations = classify_subcategories_overtaking(
                    overtaking_violation,  # Pass the specific violation dictionary
                    frame_count,
                    traffic_density_data,
                    optical_flow_data,
                    violation_frames_dir,
                    active_violations,  # Pass active lane violations
                )

                # Update sub_violations in the overtaking_violation
                print(sub_violations)
                overtaking_violation["sub_violations"].update(sub_violations)
                print(overtaking_violation["sub_violations"])  # Debug print
                # Append the overtaking violation to the list
                overtaking_violations.append(overtaking_violation)

            # Finalize overtaking violation when cooldown ends
            if not overtaking_state.get("active") and overtaking_state.get("end_frame"):
                overtaking_violations[-1].update(
                    {
                        "end_frame": overtaking_state["end_frame"],
                        "end_frame_path": overtaking_state["end_frame_path"],
                    }
                )

            # Update previous vehicles for overtaking detection
            results = YOLOmodel(frame)
            previous_vehicles = [
                {
                    "class_id": int(box[5]),
                    "box": box[:4],  # (x1, y1, x2, y2)
                    "confidence": box[4],
                }
                for box in results[0].boxes.data.cpu().numpy()
                if len(box) > 5 and int(box[5]) in FRONT_VEHICLE_CLASS_ID
            ]
            out.write(annotated_frame)
            frame_count += 1

        # Combine results into a unified JSON structure
        output = {
            "lane_discipline_violations": lane_discipline_violations,
            "overtaking_violations": overtaking_violations,
        }

        # Save violations to a JSON file
        violations_log_path = os.path.join(run_folder, "combined_violations.json")
        serializable_data = convert_to_serializable(output)
        with open(violations_log_path, "w") as f:
            json.dump(serializable_data, f, indent=4)

        stop_event.set()
        print(f"Combined violations saved to {violations_log_path}")
        logging.info(f"Combined violations saved to {violations_log_path}")
        return fps
    except Exception as e:
        stop_event.set()
        logging.error(f"Error during video processing: {e}")
        print(f"Error during video processing: {e}")


# Test the script
if __name__ == "__main__":
    video_path = "/home/ubuntu/ml/scripts/tmp/cut_sample_dashcam.avi"
    json_path = "/home/ubuntu/ml/scripts/results/Run7/optical_flow_and_Traffic_Density_results.json"
    run_folder = create_results_folder()
    Lane_discipline_only(video_path, json_path, run_folder)
