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
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=True, weather=False, messages=False)

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
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=True, weather=False, messages=False)
            laps = f1session.laps
            return list(laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def laps_data(
        self, event: Union[str, int], session: str, driver: str
    ) -> Dict[str, List]:
        """
        Get lap data for a specific driver in a session.

        Args:
            event: Event name or number
            session: Session name
            driver: Driver code

        Returns:
            Dictionary with lap times, numbers, and compounds
        """
        try:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=False, weather=False, messages=False)
            laps = f1session.laps

            driver_laps = laps.pick_driver(driver)
            driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()
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
        # Convert speed from km/h to m/s
        vx = telemetry["Speed"] / 3.6
        time_float = telemetry["Time"] / np.timedelta64(1, "s")
        dtime = np.gradient(time_float)
        ax = np.gradient(vx) / dtime

        # Clean up outliers
        for i in np.arange(1, len(ax) - 1).astype(int):
            if ax[i] > 25:
                ax[i] = ax[i - 1]

        # Smooth x-acceleration
        ax_smooth = np.convolve(ax, np.ones((Nax,)) / Nax, mode="same")

        # Get position data
        x = telemetry["X"]
        y = telemetry["Y"]
        z = telemetry["Z"]

        # Calculate gradients
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)

        # Calculate theta (angle in xy-plane)
        theta = np.arctan2(dy, (dx + np.finfo(float).eps))
        theta[0] = theta[1]
        theta_noDiscont = np.unwrap(theta)

        # Calculate distance and curvature
        dist = telemetry["Distance"]
        ds = np.gradient(dist)
        dtheta = np.gradient(theta_noDiscont)

        # Clean up outliers
        for i in np.arange(1, len(dtheta) - 1).astype(int):
            if abs(dtheta[i]) > 0.5:
                dtheta[i] = dtheta[i - 1]

        # Calculate curvature and lateral acceleration
        C = dtheta / (ds + 0.0001)  # To avoid division by 0
        ay = np.square(vx) * C

        # Remove extreme values
        indexProblems = np.abs(ay) > 150
        ay[indexProblems] = 0

        # Smooth y-acceleration
        ay_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")

        # Calculate z-acceleration (similar process)
        z_theta = np.arctan2(dz, (dx + np.finfo(float).eps))
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)

        z_dtheta = np.gradient(z_theta_noDiscont)

        # Clean up outliers
        for i in np.arange(1, len(z_dtheta) - 1).astype(int):
            if abs(z_dtheta[i]) > 0.5:
                z_dtheta[i] = z_dtheta[i - 1]

        # Calculate z-curvature and vertical acceleration
        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(vx) * z_C

        # Remove extreme values
        indexProblems = np.abs(az) > 150
        az[indexProblems] = 0

        # Smooth z-acceleration
        az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")

        # Add acceleration columns to telemetry
        telemetry["Ax"] = ax_smooth
        telemetry["Ay"] = ay_smooth
        telemetry["Az"] = az_smooth

        return telemetry

    def telemetry_data(
        self, event: Union[str, int], session: str, driver: str, lap_number: int
    ) -> Dict[str, Dict[str, List]]:
        """
        Get detailed telemetry data for a specific lap.

        Args:
            event: Event name or number
            session: Session name
            driver: Driver code
            lap_number: Lap number to extract data for

        Returns:
            Dictionary with telemetry data
        """
        try:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=True, weather=False, messages=False)
            laps = f1session.laps

            driver_laps = laps.pick_driver(driver)
            driver_laps["LapTime"] = driver_laps["LapTime"].dt.total_seconds()

            # Get the telemetry for lap_number
            selected_lap = driver_laps[driver_laps.LapNumber == lap_number]
            telemetry = selected_lap.get_telemetry()
            acc_tel = self.accCalc(telemetry, 3, 9, 9)
            acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

            # Create a unique data key for this telemetry
            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"

            # Convert DRS and Brake to binary values
            acc_tel["DRS"] = acc_tel["DRS"].apply(
                lambda x: 1 if x in [10, 12, 14] else 0
            )
            acc_tel["Brake"] = acc_tel["Brake"].apply(lambda x: 1 if x == True else 0)

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
        try:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load()
            circuit_key = f1session.session_info['Meeting']['Circuit']['Key']

            # Try to get corner data from fastf1 first
            try:
                circuit_info = f1session.get_circuit_info().corners
                corner_info = {
                    "CornerNumber": circuit_info['Number'].tolist(),
                    "X": circuit_info['X'].tolist(),
                    "Y": circuit_info['Y'].tolist(),
                    "Angle": circuit_info['Angle'].tolist(),
                    "Distance": circuit_info['Distance'].tolist(),
                }
                return corner_info
            except (AttributeError, KeyError):
                # Fall back to API method if fastf1 method fails
                circuit_info = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_info = {
                        "CornerNumber": circuit_info['Number'].tolist(),
                        "X": circuit_info['X'].tolist(),
                        "Y": circuit_info['Y'].tolist(),
                        "Angle": circuit_info['Angle'].tolist(),
                        "Distance": (circuit_info['Distance']/10).tolist(),
                    }
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
            rows = []
            for entry in data['corners']:
                rows.append((
                    float(entry.get('trackPosition', {}).get('x', 0.0)),
                    float(entry.get('trackPosition', {}).get('y', 0.0)),
                    int(entry.get('number', 0)),
                    str(entry.get('letter', "")),
                    float(entry.get('angle', 0.0)),
                    float(entry.get('length', 0.0))
                ))

            return pd.DataFrame(
                rows,
                columns=['X', 'Y', 'Number', 'Letter', 'Angle', 'Distance']
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None

    def process_event_session(self, event: str, session: str) -> None:
        """
        Process a single event and session, extracting all telemetry data.

        Args:
            event: Event name
            session: Session name
        """
        logger.info(f"Processing {event} - {session}")

        # Create base directory for this event/session
        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        # Save drivers information
        drivers_info = self.session_drivers(event, session)
        with open(f"{base_dir}/drivers.json", "w") as json_file:
            json.dump(drivers_info, json_file)

        # Save circuit corner information
        corner_info = self.get_circuit_info(event, session)
        if corner_info:
            with open(f"{base_dir}/corners.json", "w") as json_file:
                json.dump(corner_info, json_file)

        # Process each driver
        drivers = self.session_drivers_list(event, session)
        for driver in drivers:
            logger.info(f"Processing driver: {driver}")

            # Create driver directory
            driver_dir = f"{base_dir}/{driver}"
            os.makedirs(driver_dir, exist_ok=True)

            # Save lap times
            laptimes = self.laps_data(event, session, driver)
            with open(f"{driver_dir}/laptimes.json", "w") as json_file:
                json.dump(laptimes, json_file)

            # Get telemetry for each lap
            try:
                f1session = fastf1.get_session(self.year, event, session)
                f1session.load(telemetry=False, weather=False, messages=False)
                laps = f1session.laps
                driver_laps = laps.pick_driver(driver)
                driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
                lap_numbers = driver_laps["LapNumber"].tolist()

                # Process each lap with error handling
                for lap_number in lap_numbers:
                    try:
                        telemetry = self.telemetry_data(event, session, driver, lap_number)
                        if telemetry["tel"]:
                            file_path = f"{driver_dir}/{lap_number}_tel.json"
                            with open(file_path, "w") as json_file:
                                json.dump(telemetry, json_file)
                    except Exception as e:
                        logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Error processing laps for {driver}: {str(e)}")


    def process_all_data(self, max_workers: int = 4) -> None:
        """
        Process all configured events and sessions, with optional parallelization.

        Args:
            max_workers: Maximum number of worker threads for parallel processing
        """
        logger.info(f"Starting telemetry extraction for {self.year} season")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        # Process each event and session
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for event in self.events:
                for session in self.sessions:
                    futures.append(
                        executor.submit(self.process_event_session, event, session)
                    )

            # Wait for all tasks to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        logger.info("Telemetry extraction completed")



def main():
    """Main entry point for the script."""


    # Create extractor and process data
    extractor = TelemetryExtractor( )
    extractor.process_all_data(max_workers=4)


if __name__ == "__main__":
    main()
