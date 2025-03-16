import functools
import gc
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import fastf1
import numpy as np
import pandas as pd
import psutil
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telemetry_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("telemetry_extractor")
logging.getLogger("fastf1").setLevel(logging.WARNING)
logging.getLogger("fastf1").propagate = False

# Enable caching
fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"

# Setup optimized HTTP session with connection pooling and retry logic
session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
HEADERS = {"User-Agent": f"FastF1/"}

# Global cache for session objects to prevent reloading
SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}
TELEMETRY_CACHE = {}

# Determine optimal number of workers based on CPU cores
CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_MAX_WORKERS = min(CPU_COUNT * 2, 16)  # Use more workers but cap at reasonable limit
DEFAULT_PROCESS_WORKERS = max(CPU_COUNT - 1, 1)  # Leave one core free for system


class TelemetryExtractor:
    """Optimized class to handle extraction of F1 telemetry data."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
    ):
        """Initialize the TelemetryExtractor."""
        self.year = year
        self.events = events or ["Australian Grand Prix"]
        self.sessions = sessions or ["Race"]

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False, load_only: str = None
    ) -> fastf1.core.Session:
        """Get a cached session object to prevent reloading with selective data loading."""
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)

            # Selective loading based on what's needed
            if load_only == 'laps':
                f1session.load(telemetry=False, weather=False, messages=False, laps=True)
            elif load_only == 'telemetry':
                f1session.load(telemetry=True, weather=False, messages=False, laps=False)
            else:
                f1session.load(telemetry=load_telemetry, weather=False, messages=False)

            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        """Get list of driver codes for a given event and session."""
        try:
            f1session = self.get_session(event, session, load_only='laps')
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Get drivers available for a given event and session."""
        try:
            f1session = self.get_session(event, session, load_only='laps')
            laps = f1session.laps
            team_colors = utils.team_colors(self.year)
            laps["color"] = laps["Team"].map(team_colors)

            unique_drivers = laps["Driver"].unique()

            drivers = [
                {
                    "driver": driver,
                    "team": laps[laps.Driver == driver].Team.iloc[0],
                }
                for driver in unique_drivers
            ]

            return {"drivers": drivers}
        except Exception as e:
            logger.error(f"Error getting drivers for {event} {session}: {str(e)}")
            return {"drivers": []}

    def laps_data(
        self, event: Union[str, int], session: str, driver: str, f1session=None
    ) -> Dict[str, List]:
        """Get lap data for a specific driver in a session."""
        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_only='laps')

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()  # Create a copy here
            driver_laps["LapTime"] = driver_laps["LapTime"].apply(
                lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
            )
            driver_laps = driver_laps[driver_laps.LapTime.notnull()]

            return {
                "time": driver_laps["LapTime"].tolist(),
                "lap": driver_laps["LapNumber"].tolist(),
                "compound": driver_laps["Compound"].tolist(),
            }
        except Exception as e:
            logger.error(
                f"Error getting lap data for {driver} in {event} {session}: {str(e)}"
            )
            return {"time": [], "lap": [], "compound": []}

    @functools.lru_cache(maxsize=128)
    def accCalc(
        self, telemetry_key: str, Nax: int, Nay: int, Naz: int
    ) -> pd.DataFrame:
        """Calculate acceleration components from telemetry data - optimized with vectorization."""
        # Retrieve telemetry from cache using the key
        if telemetry_key not in TELEMETRY_CACHE:
            logger.error(f"Telemetry key {telemetry_key} not found in cache")
            return None

        telemetry = TELEMETRY_CACHE[telemetry_key].copy()

        # Convert to numpy arrays for faster processing
        time_array = telemetry["Time"].to_numpy() / np.timedelta64(1, "s")
        speed_array = telemetry["Speed"].to_numpy() / 3.6  # km/h to m/s
        x_array = telemetry["X"].to_numpy()
        y_array = telemetry["Y"].to_numpy()
        z_array = telemetry["Z"].to_numpy()
        dist_array = telemetry["Distance"].to_numpy()

        # Calculate x-acceleration
        dtime = np.gradient(time_array)
        ax = np.gradient(speed_array) / dtime

        # Clean up outliers using vectorized operations
        ax_mask = ax > 25
        if np.any(ax_mask):
            ax[ax_mask] = np.roll(ax, 1)[ax_mask]

        # Smooth x-acceleration
        ax_smooth = np.convolve(ax, np.ones((Nax,)) / Nax, mode="same")

        # Calculate gradients
        dx = np.gradient(x_array)
        dy = np.gradient(y_array)
        dz = np.gradient(z_array)

        # Calculate theta (angle in xy-plane)
        theta = np.arctan2(dy, dx + np.finfo(float).eps)
        theta[0] = theta[1]  # Fix first element
        theta_noDiscont = np.unwrap(theta)

        # Calculate distance and curvature
        ds = np.gradient(dist_array)
        dtheta = np.gradient(theta_noDiscont)

        # Clean up outliers in dtheta
        dtheta_mask = np.abs(dtheta) > 0.5
        if np.any(dtheta_mask):
            dtheta[dtheta_mask] = np.roll(dtheta, 1)[dtheta_mask]

        # Calculate curvature and lateral acceleration
        C = dtheta / (ds + 0.0001)  # To avoid division by 0
        ay = np.square(speed_array) * C

        # Remove extreme values
        ay_mask = np.abs(ay) > 150
        ay[ay_mask] = 0

        # Smooth y-acceleration
        ay_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")

        # Calculate z-acceleration
        z_theta = np.arctan2(dz, dx + np.finfo(float).eps)
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)

        z_dtheta = np.gradient(z_theta_noDiscont)

        # Clean up outliers in z_dtheta
        z_dtheta_mask = np.abs(z_dtheta) > 0.5
        if np.any(z_dtheta_mask):
            z_dtheta[z_dtheta_mask] = np.roll(z_dtheta, 1)[z_dtheta_mask]

        # Calculate z-curvature and vertical acceleration
        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(speed_array) * z_C

        # Remove extreme values
        az_mask = np.abs(az) > 150
        az[az_mask] = 0

        # Smooth z-acceleration
        az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")

        # Add acceleration columns to telemetry
        telemetry["Ax"] = ax_smooth
        telemetry["Ay"] = ay_smooth
        telemetry["Az"] = az_smooth

        return telemetry

    def process_lap(
        self,
        event: str,
        session: str,
        driver: str,
        lap_number: int,
        driver_dir: str,
        f1session=None,
        driver_laps=None,
    ) -> Dict:
        """Process a single lap for a driver and return data for batch writing."""
        file_path = f"{driver_dir}/{lap_number}_tel.json"

        # Skip if file already exists
        if os.path.exists(file_path):
            return {"exists": True, "file_path": file_path}

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            if driver_laps is None:
                laps = f1session.laps
                driver_laps = laps.pick_drivers(driver)
                driver_laps["LapTime"] = driver_laps["LapTime"].apply(
                    lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
                )

            # Get the telemetry for lap_number
            selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

            if selected_lap.empty:
                logger.warning(
                    f"No data for {driver} lap {lap_number} in {event} {session}"
                )
                return None

            telemetry = selected_lap.get_telemetry()

            # Create a unique key for this telemetry and store in cache
            telemetry_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}-telemetry"
            TELEMETRY_CACHE[telemetry_key] = telemetry

            # Process telemetry with optimized accCalc
            acc_tel = self.accCalc(telemetry_key, 3, 9, 9)

            # Clean up cache after processing
            del TELEMETRY_CACHE[telemetry_key]

            acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

            # Create a unique data key for this telemetry
            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"

            # Convert DRS and Brake to binary values
            acc_tel["DRS"] = acc_tel["DRS"].apply(
                lambda x: 1 if x in [10, 12, 14] else 0
            )
            acc_tel["Brake"] = acc_tel["Brake"].apply(lambda x: 1 if x == True else 0)

            telemetry_data = {
                "tel": {
                    "time": acc_tel["Time"].tolist(),
                    "rpm": acc_tel["RPM"].tolist(),
                    "speed": acc_tel["Speed"].tolist(),
                    "gear": acc_tel["nGear"].tolist(),
                    "throttle": acc_tel["Throttle"].tolist(),
                    "brake": acc_tel["Brake"].tolist(),
                    "drs": acc_tel["DRS"].tolist(),
                    "distance": acc_tel["Distance"].tolist(),
                    "rel_distance": acc_tel["RelativeDistance"].tolist(),
                    "acc_x": acc_tel["Ax"].tolist(),
                    "acc_y": acc_tel["Ay"].tolist(),
                    "acc_z": acc_tel["Az"].tolist(),
                    "x": acc_tel["X"].tolist(),
                    "y": acc_tel["Y"].tolist(),
                    "z": acc_tel["Z"].tolist(),
                    "dataKey": data_key,
                }
            }

            return {"file_path": file_path, "data": telemetry_data}
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return None

    def get_circuit_info(self, event: str, session: str) -> Optional[Dict[str, List]]:
        """Get circuit corner information."""
        cache_key = f"{self.year}-{event}-{session}"

        if cache_key in CIRCUIT_INFO_CACHE:
            return CIRCUIT_INFO_CACHE[cache_key]

        try:
            f1session = self.get_session(event, session, load_only='laps')
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]

            # Try to get corner data from fastf1 first
            try:
                circuit_info = f1session.get_circuit_info().corners
                corner_info = {
                    "CornerNumber": circuit_info["Number"].tolist(),
                    "X": circuit_info["X"].tolist(),
                    "Y": circuit_info["Y"].tolist(),
                    "Angle": circuit_info["Angle"].tolist(),
                    "Distance": circuit_info["Distance"].tolist(),
                }
                CIRCUIT_INFO_CACHE[cache_key] = corner_info
                return corner_info
            except (AttributeError, KeyError):
                # Fall back to API method if fastf1 method fails
                circuit_info = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_info = {
                        "CornerNumber": circuit_info["Number"].tolist(),
                        "X": circuit_info["X"].tolist(),
                        "Y": circuit_info["Y"].tolist(),
                        "Angle": circuit_info["Angle"].tolist(),
                        "Distance": (circuit_info["Distance"] / 10).tolist(),
                    }
                    CIRCUIT_INFO_CACHE[cache_key] = corner_info
                    return corner_info

            logger.warning(f"Could not get corner data for {event} {session}")
            return None
        except Exception as e:
            logger.error(f"Error getting circuit info for {event} {session}: {str(e)}")
            return None

    def _get_circuit_info_from_api(self, circuit_key: int) -> Optional[pd.DataFrame]:
        """Get circuit information from the MultiViewer API."""
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            # Use the optimized session with connection pooling
            response = session.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None

            data = response.json()
            rows = []
            for entry in data["corners"]:
                rows.append(
                    (
                        float(entry.get("trackPosition", {}).get("x", 0.0)),
                        float(entry.get("trackPosition", {}).get("y", 0.0)),
                        int(entry.get("number", 0)),
                        str(entry.get("letter", "")),
                        float(entry.get("angle", 0.0)),
                        float(entry.get("length", 0.0)),
                    )
                )

            return pd.DataFrame(
                rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a single driver with optimized batch file writing."""
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            # Save lap times
            laptimes = self.laps_data(event, session, driver, f1session)
            with open(f"{driver_dir}/laptimes.json", "w") as json_file:
                json.dump(laptimes, json_file)

            # Get driver laps
            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()  # Create a copy here
            driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
            driver_laps["LapTime"] = driver_laps["LapTime"].apply(
                lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
            )
            lap_numbers = driver_laps["LapNumber"].tolist()

            # Process laps in parallel and collect results for batch writing
            lap_data_results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        self.process_lap,
                        event,
                        session,
                        driver,
                        lap_number,
                        driver_dir,
                        f1session,
                        driver_laps,
                    )
                    for lap_number in lap_numbers
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result and not result.get("exists", False):
                        lap_data_results.append(result)

            # Batch write files
            for result in lap_data_results:
                with open(result["file_path"], "w") as json_file:
                    json.dump(result["data"], json_file)

            # Clear any per-driver caches if needed
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str, max_workers: int = None) -> None:
        """Process a single event and session, extracting all telemetry data."""
        logger.info(f"Processing {event} - {session}")

        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS

        # Create base directory for this event/session
        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            # Load session data once
            f1session = self.get_session(event, session, load_telemetry=True)

            # Save drivers information
            drivers_info = self.session_drivers(event, session)
            with open(f"{base_dir}/drivers.json", "w") as json_file:
                json.dump(drivers_info, json_file)

            # Save circuit corner information
            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                with open(f"{base_dir}/corners.json", "w") as json_file:
                    json.dump(corner_info, json_file)

            # Get driver list
            drivers = self.session_drivers_list(event, session)

            # Process drivers in parallel with optimized worker count
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self.process_driver, event, session, driver, base_dir, f1session
                    )
                    for driver in drivers
                ]

                for future in as_completed(futures):
                    future.result()  # Just to catch any exceptions

            # Check memory usage after processing each session
            check_memory_usage()

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {str(e)}")

    def process_all_data(self, max_workers: int = None) -> None:
        """Process all configured events and sessions, with optimized parallelization."""
        logger.info(f"Starting optimized telemetry extraction for {self.year} season")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        if max_workers is None:
            max_workers = DEFAULT_MAX_WORKERS

        start_time = time.time()

        # Use a smaller number of workers for the outer loop to avoid overwhelming the system
        outer_workers = max(2, max_workers // 2)

        # Process each event and session in parallel
        with ThreadPoolExecutor(max_workers=outer_workers) as executor:
            futures = []
            for event in self.events:
                for session in self.sessions:
                    # Pass the max_workers parameter to the inner function
                    inner_workers = max(2, max_workers // len(self.sessions))
                    futures.append(
                        executor.submit(self.process_event_session, event, session, inner_workers)
                    )

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")
                # Check memory usage after each task completes
                check_memory_usage()

        elapsed_time = time.time() - start_time
        logger.info(f"Telemetry extraction completed in {elapsed_time:.2f} seconds")


def check_memory_usage(threshold_percent=80):
    """
    Check if memory usage exceeds threshold and clear caches if needed.

    Args:
        threshold_percent: Memory usage percentage threshold

    Returns:
        True if memory was cleared, False otherwise
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logger.info(
        f"Current memory usage: {memory_percent:.2f}% ({memory_info.rss / 1024 / 1024:.2f} MB)"
    )

    if memory_percent > threshold_percent:
        logger.warning(
            f"Memory usage exceeds {threshold_percent}% threshold, clearing caches"
        )
        # Clear the session cache
        SESSION_CACHE.clear()
        CIRCUIT_INFO_CACHE.clear()
        TELEMETRY_CACHE.clear()

        # Force garbage collection
        gc.collect()

        # Log new memory usage
        new_memory_percent = psutil.Process(os.getpid()).memory_percent()
        logger.info(
            f"New memory usage after clearing caches: {new_memory_percent:.2f}%"
        )
        return True

    return False


def is_data_available(year, events, sessions):
    """
    Check if data is available for the specified year, events, and sessions.

    Args:
        year: The F1 season year
        events: List of event names to check
        sessions: List of session names to check

    Returns:
        bool: True if data is available, False otherwise
    """
    try:
        # Try to load the first event and session as a test
        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event = events[0]
        session = sessions[0]

        logger.info(f"Checking data availability for {year} {event} {session}...")

        # Try to get the session without loading telemetry
        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        # Check if we have lap data
        if f1session.laps.empty:
            logger.info(f"No lap data available yet for {year} {event} {session}")
            return False

        # Check if we have at least one driver
        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        logger.info(f"Data is available for {year} {event} {session}")
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point for the script."""
    try:
        # Create extractor
        extractor = TelemetryExtractor()

        # Use more workers on GitHub Actions or based on CPU count
        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 16 if is_github_actions else DEFAULT_MAX_WORKERS

        # Wait for data to be available
        wait_time = 30  # seconds between checks
        max_attempts = 720  # 12 hours max wait time (720 * 60 seconds)
        attempt = 0

        logger.info(f"Starting to wait for {extractor.year} season data...")

        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                logger.info(
                    f"Data is available for {extractor.year} season. Starting extraction..."
                )
                extractor.process_all_data(max_workers=max_workers)
                break
            else:
                attempt += 1
                logger.info(
                    f"Data not yet available. Waiting {wait_time} seconds before retry ({attempt}/{max_attempts})..."
                )
                time.sleep(wait_time)

                # Check memory usage and clear if needed
                check_memory_usage()

        if attempt >= max_attempts:
            logger.error(
                f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
            )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    # Uncomment the following line to run with profiling
    # import cProfile
    # cProfile.run('main()', 'telQ_profile.stats')
    main()
