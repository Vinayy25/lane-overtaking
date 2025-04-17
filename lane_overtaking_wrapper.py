import os

from density_speed_module import *
from lane_discipline_and_overtaking_analysis import Lane_discipline_only
from helper import *

#from utils.mongo_utils import *

"""
# Function to check road condition
def check_road_condition(start_frame):
    try:
        road_condition = ""
        # hit flask post api to get road condition
        url = "http://127.0.0.1:5000/check_road_condition"
        file_name = start_frame.split("/")[-1]
        with open(start_frame, "rb") as image_file:
            files = {"frame": (file_name, image_file, "image/jpeg")}
            response = requests.post(url, files=files)

        if response.status_code == 200:
            response_json = response.json()
            road_condition = response_json["road_condition"]
            print(f"Road condition is : {road_condition}")
            return road_condition
        else:
            print("Error in fetching road condition : ", response.status_code)
            return road_condition

    except Exception as e:
        print(f"Error in fetching road condition: {e}")
        return road_condition
"""

@measure_execution_time
def dashcam_lane_and_overtaking_violation(
    video_path,
    feed_type,
    cctv_id,
    trip_id,
    car_number,
    logger=None,
    run_folder=None,
    violations_collection=None,
):
    """
    Extracts lane and overtaking violations from a video and generates a JSON file with violation details.

    Args:
        video_path (str): Path to the video file.
        feed_type (str): Type of video feed (e.g., "dashcam", "cctv").
        trip_id (str): Unique identifier for the trip.
        car_number (str): License plate number of the car.
        logger (logging.Logger, optional): Logger object for logging messages. Defaults to None.
        run_folder (str, optional): Directory where results are stored. Defaults to None.
    """

    # Create the results folder
    if run_folder is None:
        run_folder = create_results_folder()
    if logger is None:
        logger = setup_logger(run_folder, "laneandOD.log")

    logger.info("Starting Lane and Overtaking Violations on dashcam video.")
    print("Starting Lane and Overtaking Violations on dashcam video.")

    # Run traffic density and optical flow analysis
    Traffic_density_optical_flow_test(video_path, run_folder)
    print("Traffic Density and Optical Flow calculations done.")

    # Load JSON outputs
    OFTD_json_file_path = os.path.join(
        run_folder, "optical_flow_and_Traffic_Density_results.json"
    )

    # Run the Lane_discipline_only script with the JSON file and run_folder
    fps = Lane_discipline_only(video_path, OFTD_json_file_path, run_folder)
    recorded_violations_json_path = os.path.join(run_folder, "combined_violations.json")

    if not os.path.exists(recorded_violations_json_path):
        logger.error(f"combined_violations.json not found at {recorded_violations_json_path}")
        print(f"Error: combined_violations.json not found at {recorded_violations_json_path}")
        return

    with open(OFTD_json_file_path, "r") as f1, open(
        recorded_violations_json_path, "r"
    ) as f2:
        oft_data = json.load(f1)
        violation_data = json.load(f2)

    optical_flow_data = {
        item["frame_number"]: item for item in oft_data["optical_flow_data"]
    }
    traffic_density_data = {
        item["frame_number"]: item for item in oft_data["traffic_density_data"]
    }
    timestamp = datetime.now()
    violations = []

    # Process lane discipline violations
    for violation in violation_data.get("lane_discipline_violations", []):
        start_frame = violation["start_frame"]
        end_frame = violation["end_frame"]
        sub_violations = violation["sub_violations"]

        # Determine the parameter and sub-parameter based on the hierarchy and conditions
        if "solid_line_overtaking" in sub_violations:
            parameter = "lane_indiscipline"
            sub_parameter = "solid_line_crossing"
        elif "unsafe_lane_change" in sub_violations:
            parameter = "lane_indiscipline"
            sub_parameter = "unsafe_lane_change"
        elif "weaving" in sub_violations and "dash_line_middle" in sub_violations:
            parameter = "lane_indiscipline"
            sub_parameter = "dash_line_middle"
        else:  # Ignore other conditions
            continue

        speed = round(
            optical_flow_data.get(start_frame, {}).get("speed_km_per_hour", 0)
        )  # Default to 0 if not found
        TD = traffic_density_data.get(start_frame, {}).get(
            "traffic_density", "low"
        )  # Default to low if not found
        road_condition = "Good"  # check_road_condition(violation["start_frame_path"])

        violations.append(
            {
                "violations": {"parameter": parameter, "sub-parameter": sub_parameter},
                "start_frame_number": start_frame,
                "end_frame_number": end_frame,
                "start_frame": violation["start_frame_path"],
                "end_frame": violation["end_frame_path"],
                "start_frame_time": get_video_timestamp(fps, start_frame),
                "end_frame_time": get_video_timestamp(fps, end_frame),
                "frame_path": "results/Run36/violation_frames",
                "raw_video_path": video_path,
                "trip_datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "trip_id": trip_id,
                "feed_type": feed_type,
                "cctv_id": "001",
                "car_number": car_number,
                "speed_ml": speed,
                "road_condition": road_condition,
                "traffic_density": TD,
                "processed_datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    # Process overtaking violations
    overtaking_violations = []
    overtaking_data = violation_data.get("overtaking_violations", [])

    for idx, violation in enumerate(overtaking_data):
        # Extract start and end frames
        start_frame = violation["start_frame"]
        end_frame = (
            overtaking_data[idx + 1]["start_frame"] - 1
            if idx + 1 < len(overtaking_data)
            else violation["start_frame"]
        )

        # Default sub-parameter from overtaking violation
        sub_parameter = list(violation["sub_violations"].keys())[
            0
        ]  # Extract key dynamically

        # Check for solid line crossing overlap
        for lane_violation in violation_data.get("lane_discipline_violations", []):
            lane_start = lane_violation["start_frame"]
            lane_end = lane_violation["end_frame"]

            # Check if lane violation overlaps with overtaking violation
            if lane_start >= start_frame and lane_start <= end_frame:
                if "solid_line_overtaking" in lane_violation["sub_violations"]:
                    # Update sub-parameter to Overtaking_on_a_Solid_Line
                    sub_parameter = "Overtaking_on_a_Solid_Line"
                    break

        # Get speed and traffic density
        speed = round(
            optical_flow_data.get(start_frame, {}).get("speed_km_per_hour", 0)
        )
        TD = traffic_density_data.get(start_frame, {}).get("traffic_density", "low")

        # Append the processed overtaking violation
        overtaking_violations.append(
            {
                "violations": {
                    "parameter": "overtaking",
                    "sub-parameter": sub_parameter,
                },
                "start_frame_number": start_frame,
                "end_frame_number": end_frame,
                "start_frame": violation["start_frame_path"],
                "end_frame": violation["end_frame_path"]
                or f"results/Run36/violation_frames/frame_{end_frame}.jpg",
                "start_frame_time": get_video_timestamp(fps, start_frame),
                "end_frame_time": get_video_timestamp(fps, end_frame),
                "frame_path": "results/Run36/violation_frames",
                "raw_video_path": video_path,
                "trip_datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "trip_id": trip_id,
                "feed_type": feed_type,
                "cctv_id": "001",
                "car_number": car_number,
                "speed_ml": speed,
                "road_condition": "Good",  # check_road_condition(violation["start_frame_path"]),
                "traffic_density": TD,
                "processed_datetime": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    # Combine lane and overtaking violations, sorted by start_frame_number
    final_violations = sorted(
        violations + overtaking_violations, key=lambda v: v["start_frame_number"]
    )

    # for violation_data in final_violations["violations"]:
    #     # Save JSON data to MongoDB
    #     try:
    #         # Insert into db using insert_document function
    #         result = insert_violation_document(violations_collection, violation_data)
    #         logger.info(f"Lane Task - Saved JSON data to MongoDB: {result}")
    #     except Exception as e:
    #         logger.error(f"Lane Task - Failed to insert data: {e}")

    # Save to JSON
    output_path = os.path.join(run_folder, "final_violations.json")
    with open(output_path, "w") as output_file:
        json.dump({"violations": final_violations}, output_file, indent=4)

    print(f"Violations JSON saved to {output_path}")
    return final_violations


if __name__ == "__main__":
    from density_speed_module import Traffic_density_optical_flow_test
    from helper import create_results_folder

    video_paths = ["new_test/test.mp4"]
    feed_type = "dashcam"
    trip_id = "001"
    cctv_id = "NA"
    car_number = "KA"
    for video_path in video_paths:
        dashcam_lane_and_overtaking_violation(
            video_path, feed_type, cctv_id, trip_id, car_number
        )
