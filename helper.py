import os
import time
import logging
from logging.handlers import RotatingFileHandler
import threading


def loading_animation(stop_event):
    animation = "|/-\\"
    i = 0
    while not stop_event.is_set():
        print(animation[i % len(animation)], end="\r")
        i += 1
        time.sleep(0.1)


def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Time Taken: {execution_time} seconds")
        logging.info(f"Time Taken: {execution_time} seconds")
        return result

    return wrapper


def setup_logger(run_folder, log_file_name):  # Modified to accept log file name
    try:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        if logger.hasHandlers():
            logger.handlers.clear()

        log_file = os.path.join(run_folder, log_file_name)
        handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
        )
        logger.addHandler(handler)

        logger.info(
            "Logger initialized."
        )  # Updated from `logging.info` to `logger.info`
        print("Logger initialized.")
        return logger
    except Exception as e:
        print(f"Error Setting Up Logger: {e}")
        return None


def create_results_folder():
    try:
        results_folder = "results"
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)

        existing_runs = sorted(
            [
                int(d[3:])
                for d in os.listdir(results_folder)
                if d.startswith("Run") and d[3:].isdigit()
            ],
            reverse=True,
        )
        next_run_number = existing_runs[0] + 1 if existing_runs else 1
        run_folder = os.path.join(results_folder, f"Run{next_run_number}")
        os.makedirs(run_folder, exist_ok=True)
        logging.info(f"Results folder created: {run_folder}")
        print(f"Results folder created: {run_folder}")
        return run_folder
    except Exception as e:
        logging.error(f"Error while Creating Results Folder: {e}")
        return ""


def get_video_timestamp(fps, frame_number):
    """
    Calculate the timestamp of a video given the FPS and frame number.

    Args:
        fps (int): Frames per second of the video.
        frame_number (int): The current frame number.

    Returns:
        str: Timestamp in the format hh:mm:ss.
    """
    # Calculate total seconds
    total_seconds = frame_number / fps

    # Convert to hours, minutes, and seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Format as hh:mm:ss
    timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"
    return timestamp
