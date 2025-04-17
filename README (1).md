# Lane Discipline, Overtaking, Density, and Speed Analysis

## Description

This project analyzes dashcam video footage for lane discipline violations, unsafe overtaking maneuvers, traffic density, and vehicle speed. It uses a combination of lane detection, object detection, and optical flow techniques to identify and classify these events. The results are provided in a JSON format, including detailed information about each violation and the overall traffic conditions.

## Key Components

### Lane Discipline and Overtaking Analysis

*   **Lane Detection:** Uses a Roboflow lane detection model to identify lane boundaries.
*   **Object Detection:** Uses a YOLOv8 model to detect vehicles and their positions.
*   **Lane Discipline Violations:** Detects and classifies violations such as crossing solid lines and unsafe lane changes.
*   **Overtaking Analysis:** Detects and classifies overtaking events based on vehicle positions and lane markings.

### Traffic Density and Optical Flow

*   **Traffic Density:** Uses a YOLOv8 model to count vehicles and estimate traffic density.
*   **Optical Flow:** Calculates optical flow to estimate vehicle speeds.

## Files

| File                                             | Content   | Status   | Comments                                                                                                                                                                                            |
| ------------------------------------------------ | --------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `lane_discipline_and_overtaking_analysis.py` | not\_empty | coded    | Contains functions for lane detection, lane discipline violation detection, overtaking analysis, and result classification.                                                                         |
| `lane_overtaking_wrapper.py`                 | not\_empty | coded    | Wrapper script that integrates lane analysis, overtaking analysis, and traffic density/optical flow calculations.                                                                               |
| `density_speed_module.py`                       | not\_empty | coded    | Contains functions for calculating traffic density and vehicle speed using optical flow.                                                                                                         |
| `helper.py`                                     | not\_empty | coded    | Contains helper functions for loading animations, measuring execution time, setting up loggers, and creating result folders.                                                                      |
| `requirements.txt`                             | not\_empty | coded    | Lists required packages, including OpenCV, NumPy, and Ultralytics YOLOv8.                                                                                                                         |

## Setup

1.  **Install Ollama:**
    ```bash
    curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh
    ollama pull llava-llama3
    ```
2.  **Install Miniconda:** Download and install Miniconda.
### Download Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
```

### Install Miniconda:
```bash
bash miniconda.sh
```
Follow the prompts to complete the installation. You may need to restart your terminal or source the `bashrc` file to activate conda.

```bash
source ~/.bashrc
```
3.  **Create a Conda Environment:** Create and activate a new conda environment with Python 3.10.
4.  **Clone the Repository:** Clone the repository containing the project files.
5.  **Install Required Packages:** Install the packages listed in `requirements.txt`.

## Usage

1.  **Prepare the Dashcam Footage:** Ensure your dashcam footage is saved as a video file.
2.  **Run the Script:** Execute the `lane_overtaking_wrapper.py` script by providing the path to your video file.

**Sample Code:**

```python
video_path = 'path_to_your_dashcam_video.mp4'
dashcam_lane_and_overtaking_violation(video_path)
```


### Output
The script produces the following output:

- Video with Annotated Lane Lines: Saved in the results/RunX/ folder.-
- Violation Frames: Saved as images in results/RunX/violation_frames/.
- optical_flow_and_Traffic_Density_results JSON Metadata: Detailing estimated vehicle speed and Traffic density of each frame .
- combined_violations JSON Metadata: Detailing each violation with start and end frame numbers, and violation type.
- Final Violations JSON Metadata: Detailing each violation with all details, and sub violation type.

### Example Final_Violations JSON Output Structure
```json

{
    "violations": [
        {
            "violations": {
                "parameter": "lane_indiscipline",
                "sub-parameter": "solid_line_crossing"
            },
            "start_frame_number": 59,
            "end_frame_number": 110,
            "start_frame": "results/Run9/violation_frames/lane_violation_frame_59.jpg",
            "end_frame": "results/Run9/violation_frames/lane_violation_frame_110.jpg",
            "start_frame_time": "00:00:02",
            "end_frame_time": "00:00:04",
            "frame_path": "results/Run36/violation_frames",
            "raw_video_path": "new_test/MOVI0017.mp4",
            "trip_datetime": "2025-01-22 14:15:49",
            "trip_id": "001",
            "feed_type": "dashcam",
            "cctv_id": "001",
            "car_number": "KA",
            "speed_ml": 83,
            "road_condition": "Good",
            "traffic_density": "Low",
            "processed_datetime": "2025-01-22 14:15:49"
        },
    ]
}

```
### Logging
The script logs information, errors, and processing times for debugging and tracking. Logs are saved in the laneandOD.log, Lane_disciplineandOvertaking.log and Traffic_density&optical_flow.log files within each run's folder.


 

 
 
