import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union

import fastf1
import numpy as np
import pandas as pd
import requests

import utils

logging.getLogger("fastf1").setLevel(logging.WARNING)
# Prevent propagation to avoid duplicate logs
logging.getLogger("fastf1").propagate = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telemetry_extraction.log"), logging.StreamHandler()],
)
logger = logging.getLogger("telemetry_extractor")

fastf1.Cache.enable_cache("cache")

DEFAULT_YEAR = 2025
PROTO = "https"
HOST = "api.multiviewer.app"
HEADERS = {"User-Agent": f"FastF1/"}


class TelemetryExtractor:
    """Class to handle extraction of F1 telemetry data."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
    ):
        """
        Initialize the TelemetryExtractor.

        Args:
            year: The F1 season year
            events: List of events to process (e.g., 'Australian Grand Prix')
            sessions: List of sessions to process (e.g., 'Qualifying', 'Race')
        """
        self.year = year
        self.events = events or [
            # 'Bahrain Grand Prix',
            # 'Saudi Arabian Grand Prix',
            "Australian Grand Prix",
            # 'Azerbaijan Grand Prix',
            # 'Miami Grand Prix',
            # "Emilia Romagna Grand Prix",
            # 'Monaco Grand Prix',
            # 'Spanish Grand Prix',
            # 'Canadian Grand Prix',
            # 'Austrian Grand Prix',
            # 'British Grand Prix',
            # 'Hungarian Grand Prix',
            # 'Belgian Grand Prix',
            # 'Dutch Grand Prix',
            # 'Italian Grand Prix',
            # 'Singapore Grand Prix',
            # 'United States Grand Prix',
            # 'Mexico City Grand Prix',
            # 'São Paulo Grand Prix',
            # 'Las Vegas Grand Prix',
            # "Qatar Grand Prix",
            # 'Abu Dhabi Grand Prix',
            # 'Chinese Grand Prix',
        ]
        self.sessions = sessions or [
            "Qualifying",
        ]
        # Cache for loaded sessions to avoid reloading
        self._session_cache = {}
        # Cache for circuit info
        self._circuit_info_cache = {}

    def _get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """
        Get a cached session or load it if not available.

        Args:
            event: Event name or number
            session: Session name
            load_telemetry: Whether to load telemetry data

        Returns:
            Loaded FastF1 session
        """
        cache_key = f"{self.year}_{event}_{session}_{load_telemetry}"

        if cache_key not in self._session_cache:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=False, messages=False)
            self._session_cache[cache_key] = f1session

        return self._session_cache[cache_key]

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Get drivers available for a given event and session.

        Args:
            event: Event name or number
            session: Session name

        Returns:
            Dictionary with driver information
        """
        try:
            f1session = self._get_session(event, session)
            laps = f1session.laps
            team_colors = utils.team_colors(self.year)

            # Use vectorized operations
            laps_with_color = laps.copy()
            laps_with_color["color"] = laps_with_color["Team"].map(team_colors)

            # Get unique drivers with their teams efficiently
            driver_team_df = laps_with_color[["Driver", "Team"]].drop_duplicates()

            drivers = [
                {
                    "driver": row["Driver"],
                    "team": row["Team"],
                }
                for _, row in driver_team_df.iterrows()
            ]

            return {"drivers": drivers}
        except Exception as e:
            logger.error(f"Error getting drivers for {event} {session}: {str(e)}")
            return {"drivers": []}

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        """
        Get list of driver codes for a given event and session.

        Args:
            event: Event name or number
            session: Session name

        Returns:
            List of driver codes
        """
        try:
            f1session = self._get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def laps_data(
        self, event: Union[str, int], session: str, driver: str, f1session=None
    ) -> Dict[str, List]:
        """
        Get lap data for a specific driver in a session.

        Args:
            event: Event name or number
            session: Session name
            driver: Driver code
            f1session: Optional pre-loaded session

        Returns:
            Dictionary with lap times, numbers, and compounds
        """
        try:
            if f1session is None:
                f1session = self._get_session(event, session)

            laps = f1session.laps
            driver_laps = laps.pick_driver(driver)

            # Convert to numpy for faster processing
            lap_times = pd.to_numeric(
                driver_laps["LapTime"].apply(
                    lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
                )
            )

            # Filter out null lap times
            valid_indices = ~pd.isna(lap_times)
            lap_times = lap_times[valid_indices]
            lap_numbers = driver_laps["LapNumber"][valid_indices].tolist()
            compounds = driver_laps["Compound"][valid_indices].tolist()

            return {
                "time": lap_times.tolist(),
                "lap": lap_numbers,
                "compound": compounds,
            }
        except Exception as e:
            logger.error(
                f"Error getting lap data for {driver} in {event} {session}: {str(e)}"
            )
            return {"time": [], "lap": [], "compound": []}

    def accCalc(
        self, telemetry: pd.DataFrame, Nax: int, Nay: int, Naz: int
    ) -> pd.DataFrame:
        """
        Calculate acceleration components from telemetry data.

        Args:
            telemetry: Telemetry DataFrame
            Nax: Smoothing window for x-acceleration
            Nay: Smoothing window for y-acceleration
            Naz: Smoothing window for z-acceleration

        Returns:
            Telemetry DataFrame with added acceleration columns
        """
        # Convert to numpy arrays for faster operations
        vx = telemetry["Speed"].to_numpy() / 3.6
        time_float = telemetry["Time"].astype("timedelta64[ns]").astype(np.int64) / 1e9
        dtime = np.gradient(time_float)
        ax = np.gradient(vx) / dtime

        # Vectorized operation instead of loop
        ax_mask = ax > 25
        if np.any(ax_mask):
            # Create a mask of indices to fix
            indices = np.where(ax_mask)[0]
            # For each index, replace with previous value
            for i in indices:
                if i > 0:  # Ensure we don't go out of bounds
                    ax[i] = ax[i - 1]

        # Use more efficient convolution
        ax_smooth = np.convolve(ax, np.ones(Nax) / Nax, mode="same")

        # Extract coordinates as numpy arrays
        x = telemetry["X"].to_numpy()
        y = telemetry["Y"].to_numpy()
        z = telemetry["Z"].to_numpy()

        # Calculate gradients
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)

        # Calculate theta with epsilon to avoid division by zero
        eps = np.finfo(float).eps
        theta = np.arctan2(dy, dx + eps)
        theta[0] = theta[1]  # Fix first value
        theta_noDiscont = np.unwrap(theta)

        dist = telemetry["Distance"].to_numpy()
        ds = np.gradient(dist)
        dtheta = np.gradient(theta_noDiscont)

        # Vectorized operation for dtheta correction
        dtheta_mask = np.abs(dtheta) > 0.5
        if np.any(dtheta_mask):
            indices = np.where(dtheta_mask)[0]
            for i in indices:
                if i > 0:
                    dtheta[i] = dtheta[i - 1]

        # Calculate curvature with small constant to avoid division by zero
        C = dtheta / (ds + 0.0001)

        # Calculate lateral acceleration
        ay = np.square(vx) * C

        # Vectorized masking for ay
        ay[np.abs(ay) > 150] = 0
        ay_smooth = np.convolve(ay, np.ones(Nay) / Nay, mode="same")

        # Calculate z-acceleration (similar process)
        z_theta = np.arctan2(dz, dx + eps)
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)

        z_dtheta = np.gradient(z_theta_noDiscont)

        # Vectorized operation for z_dtheta correction
        z_dtheta_mask = np.abs(z_dtheta) > 0.5
        if np.any(z_dtheta_mask):
            indices = np.where(z_dtheta_mask)[0]
            for i in indices:
                if i > 0:
                    z_dtheta[i] = z_dtheta[i - 1]

        # Calculate z-curvature and vertical acceleration
        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(vx) * z_C

        # Vectorized masking for az
        az[np.abs(az) > 150] = 0
        az_smooth = np.convolve(az, np.ones(Naz) / Naz, mode="same")

        # Add acceleration columns to telemetry
        telemetry["Ax"] = ax_smooth
        telemetry["Ay"] = ay_smooth
        telemetry["Az"] = az_smooth

        return telemetry

    def telemetry_data(
        self,
        event: Union[str, int],
        session: str,
        driver: str,
        lap_number: int,
        f1session=None,
    ) -> Dict[str, Dict[str, List]]:
        """
        Get detailed telemetry data for a specific lap.

        Args:
            event: Event name or number
            session: Session name
            driver: Driver code
            lap_number: Lap number to extract data for
            f1session: Optional pre-loaded session

        Returns:
            Dictionary with telemetry data
        """
        try:
            if f1session is None:
                f1session = self._get_session(event, session, load_telemetry=True)

            laps = f1session.laps
            driver_laps = laps.pick_driver(driver)

            # Convert lap times to seconds
            driver_laps["LapTime"] = pd.to_numeric(
                driver_laps["LapTime"].apply(
                    lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
                )
            )

            # Get the telemetry for lap_number
            selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

            telemetry = selected_lap.get_telemetry()
            acc_tel = self.accCalc(telemetry, 3, 9, 9)

            # Convert Time to seconds efficiently
            acc_tel["Time"] = (
                acc_tel["Time"].astype("timedelta64[ns]").astype(np.int64) / 1e9
            )

            # Create a unique data key for this telemetry
            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"

            # Convert DRS and Brake to binary values efficiently
            acc_tel["DRS"] = np.where(np.isin(acc_tel["DRS"], [10, 12, 14]), 1, 0)
            acc_tel["Brake"] = np.where(acc_tel["Brake"] == True, 1, 0)

            return {
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
        except Exception as e:
            logger.error(
                f"Error getting telemetry for {driver} lap {lap_number} in {event} {session}: {str(e)}"
            )
            return {"tel": {}}

    def get_circuit_info(self, event: str, session: str) -> Optional[Dict[str, List]]:
        """
        Get circuit corner information.

        Args:
            event: Event name
            session: Session name

        Returns:
            Dictionary with corner data
        """
         cache_key = f"{self.year}_{event}_{session}_circuit"

        if cache_key in self._circuit_info_cache:
            return self._circuit_info_cache[cache_key]

        try:
            f1session = self._get_session(event, session)

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
                self._circuit_info_cache[cache_key] = corner_info
                return corner_info
            except (AttributeError, KeyError) as e:
                logger.warning(f"Error getting circuit info from FastF1: {e}")

                # Fall back to API method if fastf1 method fails
                circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]
                circuit_info = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_info = {
                        "CornerNumber": circuit_info["Number"].tolist(),
                        "X": circuit_info["X"].tolist(),
                        "Y": circuit_info["Y"].tolist(),
                        "Angle": circuit_info["Angle"].tolist(),
                        "Distance": (circuit_info["Distance"] / 10).tolist(),
                    }
                    self._circuit_info_cache[cache_key] = corner_info
                    return corner_info

            logger.warning(f"Could not get corner data for {event} {session}")
            return None
        except Exception as e:
            logger.error(f"Error getting circuit info for {event} {session}: {str(e)}")
            return None

    def _get_circuit_info_from_api(self, circuit_key: int) -> Optional[pd.DataFrame]:
        """
        Get circuit information from the MultiViewer API.

        Args:
            circuit_key: The unique circuit key

        Returns:
            DataFrame with circuit information
        """
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None

            data = response.json()

            # Process the data more efficiently
            rows = []
            for entry in data["corners"]:
                rows.append({
                    "X": float(entry.get("trackPosition", {}).get("x", 0.0)),
                    "Y": float(entry.get("trackPosition", {}).get("y", 0.0)),
                    "Number": int(entry.get("number", 0)),
                    "Letter": str(entry.get("letter", "")),
                    "Angle": float(entry.get("angle", 0.0)),
                    "Distance": float(entry.get("length", 0.0)),
                })

            return pd.DataFrame(rows)
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None




    def _process_driver_laps(self, event: str, session: str, driver: str, f1session) -> None:
        """
        Process and save all laps for a specific driver.

        Args:
            event: Event name
            session: Session name
            driver: Driver code
            f1session: Pre-loaded session
        """
        logger.info(f"Processing driver: {driver} for {event} - {session}")

        # Create driver directory
        driver_dir = f"{event}/{session}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        # Save lap times
        laptimes = self.laps_data(event, session, driver, f1session)
        with open(f"{driver_dir}/laptimes.json", "w") as json_file:
            json.dump(laptimes, json_file)

        # Get telemetry for each lap
        try:
            laps = f1session.laps
            driver_laps = laps.pick_driver(driver)
            driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
            lap_numbers = driver_laps["LapNumber"].tolist()

            # Load telemetry session once for all laps
            tel_session = self._get_session(event, session, load_telemetry=True)

            # Process laps in parallel for better performance
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for lap_number in lap_numbers:
                    futures.append(
                        executor.submit(
                            self._process_single_lap,
                            event,
                            session,
                            driver,
                            lap_number,
                            tel_session,
                            driver_dir
                        )
                    )

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in lap processing task: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing laps for {driver}: {str(e)}")

    def _process_single_lap(self, event, session, driver, lap_number, f1session, driver_dir):
        """Process and save telemetry for a single lap."""
        try:
            telemetry = self.telemetry_data(event, session, driver, lap_number, f1session)
            if telemetry["tel"]:
                file_path = f"{driver_dir}/{lap_number}_tel.json"
                with open(file_path, "w") as json_file:
                    json.dump(telemetry, json_file)
                logger.debug(f"Saved telemetry for {driver} lap {lap_number}")
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str) -> None:
        """
        Process a single event and session, extracting all telemetry data.

        Args:
            event: Event name
            session: Session name
        """
        logger.info(f"Processing {event} - {session}")

        try:
            # Create base directory for this event/session
            base_dir = f"{event}/{session}"
            os.makedirs(base_dir, exist_ok=True)

            # Load session once for all operations
            f1session = self._get_session(event, session)

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

            # Process each driver in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for driver in drivers:
                    futures.append(
                        executor.submit(
                            self._process_driver_laps,
                            event,
                            session,
                            driver,
                            f1session
                        )
                    )

                # Wait for all tasks to complete
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Error in driver processing task: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {str(e)}")

    def process_all_data(self, max_workers: int = 2) -> None:
        """
        Process all configured events and sessions, with optional parallelization.

        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        logger.info(f"Starting telemetry extraction for {self.year} season")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        # Process each event and session in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for event in self.events:
                for session in self.sessions:
                    futures.append(
                        executor.submit(self.process_event_session, event, session)
                    )

            # Wait for all tasks to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        logger.info("Telemetry extraction completed")




def main():
    """Main entry point for the script."""

    # Create extractor and process data
    extractor = TelemetryExtractor()
    extractor.process_all_data(max_workers=2)


if __name__ == "__main__":
    main()
