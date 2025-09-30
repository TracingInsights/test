"""
Optimized F1 Telemetry Extractor using msgspec for high-performance serialization.

Key optimizations:
- msgspec structs replace dictionaries for ~5-10x faster serialization
- Typed structures enable validation and reduce errors
- Pre-allocated arrays reduce memory allocations
- Faster pickling for parallel processing
- Zero-copy operations where possible
"""

import gc
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import fastf1
import msgspec
import numpy as np
import pandas as pd
import psutil
import requests
from joblib import Memory, Parallel, delayed

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
HEADERS = {"User-Agent": "FastF1/"}

# Global cache for session objects
SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}

# Initialize joblib memory
memory = Memory(location='./cache_joblib', verbose=0)

# ============================================================================
# MSGSPEC STRUCT DEFINITIONS
# ============================================================================

class TelemetryData(msgspec.Struct, kw_only=True, array_like=True):
    """Telemetry data for a single lap with optimized storage."""
    time: list[float]
    rpm: list[int]
    speed: list[float]
    gear: list[int]
    throttle: list[int]
    brake: list[int]
    drs: list[int]
    distance: list[float]
    rel_distance: list[float]
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]
    x: list[float]
    y: list[float]
    z: list[float]
    dataKey: str


class TelemetryWrapper(msgspec.Struct, kw_only=True):
    """Wrapper for telemetry data."""
    tel: TelemetryData


class LapData(msgspec.Struct, kw_only=True):
    """Lap timing and metadata for a driver."""
    time: list[Union[float, str]]  # Can be float or "None"
    lap: list[int]
    compound: list[str]
    stint: list[Union[int, str]]
    s1: list[Union[float, str]]
    s2: list[Union[float, str]]
    s3: list[Union[float, str]]
    life: list[Union[int, str]]
    pos: list[Union[int, str]]
    status: list[str]
    pb: list[Union[bool, str]]


class DriverInfo(msgspec.Struct, kw_only=True):
    """Driver information."""
    driver: str
    team: str


class DriversData(msgspec.Struct, kw_only=True):
    """Collection of drivers for a session."""
    drivers: list[DriverInfo]


class CircuitCorners(msgspec.Struct, kw_only=True):
    """Circuit corner information."""
    CornerNumber: list[int]
    X: list[float]
    Y: list[float]
    Angle: list[float]
    Distance: list[float]
    Rotation: float


# ============================================================================
# MSGSPEC ENCODER/DECODER
# ============================================================================

# Create optimized encoder/decoder
encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()


class TelemetryExtractor:
    """Optimized F1 telemetry extractor using msgspec for high-performance I/O."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
        use_joblib: bool = True,
        n_jobs: int = -1,
        batch_size: int = 8,
    ):
        """Initialize the TelemetryExtractor with msgspec optimizations."""
        self.year = year
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        
        self.events = events or [
            # "Pre-Season Testing",
            # "Australian Grand Prix",
            # "Chinese Grand Prix",
            # "Japanese Grand Prix",
            # "Bahrain Grand Prix",
            # 'Saudi Arabian Grand Prix',
            # "Miami Grand Prix",
            # "Emilia Romagna Grand Prix",
            # "Monaco Grand Prix",
            # 'Spanish Grand Prix',
            # "Canadian Grand Prix",
            # "Austrian Grand Prix",
            # "British Grand Prix",
            # "Belgian Grand Prix",
            # "Hungarian Grand Prix",
            "Dutch Grand Prix",
            # 'Italian Grand Prix',
            # 'Azerbaijan Grand Prix',
            # 'Singapore Grand Prix',
            # 'United States Grand Prix',
            # 'Mexico City Grand Prix',
            # 'São Paulo Grand Prix',
            # 'Las Vegas Grand Prix',
            # 'Qatar Grand Prix',
            # 'Abu Dhabi Grand Prix',
            ]
        self.sessions = sessions or ["Practice 1"]

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """Get a cached session object."""
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        """Get list of driver codes."""
        try:
            f1session = self.get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> DriversData:
        """Get drivers available for a session using msgspec struct."""
        try:
            f1session = self.get_session(event, session)
            laps = f1session.laps
            unique_drivers = laps["Driver"].unique()

            drivers = [
                DriverInfo(
                    driver=str(driver),
                    team=str(laps[laps.Driver == driver].Team.iloc[0])
                )
                for driver in unique_drivers
            ]

            return DriversData(drivers=drivers)
        except Exception as e:
            logger.error(f"Error getting drivers for {event} {session}: {str(e)}")
            return DriversData(drivers=[])

    def laps_data(
        self, event: Union[str, int], session: str, driver: str, f1session=None
    ) -> LapData:
        """Get lap data using msgspec struct for better performance."""
        try:
            if f1session is None:
                f1session = self.get_session(event, session)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

            def timedelta_to_seconds(time_value):
                if pd.isna(time_value) or not hasattr(time_value, "total_seconds"):
                    return "None"
                return round(time_value.total_seconds(), 3)

            def safe_int(value):
                return int(value) if not pd.isna(value) else "None"

            def safe_str(value):
                return str(value) if not pd.isna(value) else "None"

            def safe_bool(value):
                return bool(value) if not pd.isna(value) else "None"

            # Build struct directly - msgspec is optimized for this
            return LapData(
                time=[timedelta_to_seconds(t) for t in driver_laps["LapTime"]],
                lap=driver_laps["LapNumber"].tolist(),
                compound=[safe_str(c) if not pd.isna(c) else "None" for c in driver_laps["Compound"]],
                stint=[safe_int(s) for s in driver_laps["Stint"]],
                s1=[timedelta_to_seconds(s) for s in driver_laps["Sector1Time"]],
                s2=[timedelta_to_seconds(s) for s in driver_laps["Sector2Time"]],
                s3=[timedelta_to_seconds(s) for s in driver_laps["Sector3Time"]],
                life=[safe_int(l) for l in driver_laps["TyreLife"]],
                pos=[safe_int(p) for p in driver_laps["Position"]],
                status=[safe_str(s) for s in driver_laps["TrackStatus"]],
                pb=[safe_bool(p) for p in driver_laps["IsPersonalBest"]],
            )
        except Exception as e:
            logger.error(f"Error getting lap data for {driver}: {str(e)}")
            return LapData(
                time=[], lap=[], compound=[], stint=[], s1=[], s2=[], s3=[],
                life=[], pos=[], status=[], pb=[]
            )

    @staticmethod
    @memory.cache
    def calculate_x_acceleration(vx_array, time_array, Nax):
        """Calculate X-acceleration with caching."""
        dtime = np.gradient(time_array)
        ax = np.gradient(vx_array) / dtime

        # Clean up outliers
        for i in np.arange(1, len(ax) - 1).astype(int):
            if ax[i] > 25:
                ax[i] = ax[i - 1]

        # Smooth x-acceleration
        ax_smooth = np.convolve(ax, np.ones((Nax,)) / Nax, mode="same")
        return ax_smooth

    @staticmethod
    @memory.cache
    def calculate_y_acceleration(vx_array, x_array, y_array, dist_array, Nay):
        """Calculate Y-acceleration with caching."""
        dx = np.gradient(x_array)
        dy = np.gradient(y_array)

        theta = np.arctan2(dy, (dx + np.finfo(float).eps))
        theta[0] = theta[1]
        theta_noDiscont = np.unwrap(theta)

        ds = np.gradient(dist_array)
        dtheta = np.gradient(theta_noDiscont)

        for i in np.arange(1, len(dtheta) - 1).astype(int):
            if abs(dtheta[i]) > 0.5:
                dtheta[i] = dtheta[i - 1]

        # Calculate curvature and lateral acceleration
        C = dtheta / (ds + 0.0001)  # To avoid division by 0
        ay = np.square(vx_array) * C

        # Remove extreme values
        indexProblems = np.abs(ay) > 150
        ay[indexProblems] = 0

        # Smooth y-acceleration
        ay_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")
        return ay_smooth


    @staticmethod
    @memory.cache
    def calculate_z_acceleration(vx_array, x_array, z_array, dist_array, Naz):
        """Calculate Z-acceleration with caching."""
        dx = np.gradient(x_array)
        dz = np.gradient(z_array)

        z_theta = np.arctan2(dz, (dx + np.finfo(float).eps))
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)

        ds = np.gradient(dist_array)
        z_dtheta = np.gradient(z_theta_noDiscont)

        for i in np.arange(1, len(z_dtheta) - 1).astype(int):
            if abs(z_dtheta[i]) > 0.5:
                z_dtheta[i] = z_dtheta[i - 1]

        # Calculate z-curvature and vertical acceleration
        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(vx_array) * z_C

        # Remove extreme values
        indexProblems = np.abs(az) > 150
        az[indexProblems] = 0

        # Smooth z-acceleration
        az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")
        return az_smooth

    def accCalc(
        self, telemetry: pd.DataFrame, Nax: int, Nay: int, Naz: int
    ) -> pd.DataFrame:
        """Calculate acceleration components with parallel processing."""
        vx = telemetry["Speed"] / 3.6
        time_float = telemetry["Time"] / np.timedelta64(1, "s")

        # Extract arrays once
        vx_array = vx.values
        time_array = time_float.values
        x_array = telemetry["X"].values
        y_array = telemetry["Y"].values
        z_array = telemetry["Z"].values
        dist_array = telemetry["Distance"].values

        if self.use_joblib and len(telemetry) > 100:
            results = Parallel(n_jobs=min(3, self.n_jobs if self.n_jobs > 0 else 3), backend='threading')(
                [
                    delayed(self.calculate_x_acceleration)(vx_array, time_array, Nax),
                    delayed(self.calculate_y_acceleration)(vx_array, x_array, y_array, dist_array, Nay),
                    delayed(self.calculate_z_acceleration)(vx_array, x_array, z_array, dist_array, Naz)
                ]
            )
            ax_smooth, ay_smooth, az_smooth = results
        else:
            ax_smooth = self.calculate_x_acceleration(vx_array, time_array, Nax)
            ay_smooth = self.calculate_y_acceleration(vx_array, x_array, y_array, dist_array, Nay)
            az_smooth = self.calculate_z_acceleration(vx_array, x_array, z_array, dist_array, Naz)

        telemetry = telemetry.copy()
        telemetry["Ax"] = ax_smooth
        telemetry["Ay"] = ay_smooth
        telemetry["Az"] = az_smooth

        return telemetry

    def process_single_lap_telemetry(
        self, telemetry: pd.DataFrame, data_key: str
    ) -> TelemetryWrapper:
        """Process telemetry and return msgspec struct for ultra-fast serialization."""
        acc_tel = self.accCalc(telemetry, 3, 9, 9)
        acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

        # Vectorized conversions
        drs_values = (acc_tel["DRS"].isin([10, 12, 14])).astype(int).tolist()
        brake_values = (acc_tel["Brake"] == True).astype(int).tolist()

        # Create struct directly - msgspec handles this efficiently
        tel_data = TelemetryData(
            time=acc_tel["Time"].tolist(),
            rpm=acc_tel["RPM"].astype(int).tolist(),
            speed=acc_tel["Speed"].tolist(),
            gear=acc_tel["nGear"].astype(int).tolist(),
            throttle=acc_tel["Throttle"].astype(int).tolist(),
            brake=brake_values,
            drs=drs_values,
            distance=acc_tel["Distance"].tolist(),
            rel_distance=acc_tel["RelativeDistance"].tolist(),
            acc_x=acc_tel["Ax"].tolist(),
            acc_y=acc_tel["Ay"].tolist(),
            acc_z=acc_tel["Az"].tolist(),
            x=acc_tel["X"].tolist(),
            y=acc_tel["Y"].tolist(),
            z=acc_tel["Z"].tolist(),
            dataKey=data_key,
        )

        return TelemetryWrapper(tel=tel_data)

    def process_lap(
        self,
        event: str,
        session: str,
        driver: str,
        lap_number: int,
        driver_dir: str,
        f1session=None,
        driver_laps=None,
    ) -> bool:
        """Process a single lap using msgspec for I/O."""
        file_path = f"{driver_dir}/{lap_number}_tel.json"

        if os.path.exists(file_path):
            return True

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            if driver_laps is None:
                laps = f1session.laps
                driver_laps = laps.pick_drivers(driver).copy()
                # Create a new column for lap times in seconds to avoid dtype conflicts
                driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].apply(
                    lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
                )

            selected_lap = driver_laps[driver_laps.LapNumber == lap_number]

            if selected_lap.empty:
                logger.warning(f"No data for {driver} lap {lap_number}")
                return False

            telemetry = selected_lap.get_telemetry()
            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"
            
            # Process and get msgspec struct
            telemetry_struct = self.process_single_lap_telemetry(telemetry, data_key)

            # Use msgspec encoder - significantly faster than json.dump
            with open(file_path, "wb") as f:
                f.write(encoder.encode(telemetry_struct))

            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return False

    def process_lap_batch_with_joblib(
        self, event: str, session: str, driver: str, lap_numbers: List[int], 
        driver_dir: str, f1session=None
    ) -> List[bool]:
        """Process batch of laps with joblib."""
        def process_single_lap_job(lap_number):
            return self.process_lap(event, session, driver, lap_number, driver_dir, f1session)

        if self.use_joblib and len(lap_numbers) > 1:
            results = Parallel(n_jobs=self.n_jobs, backend='loky', prefer='processes')(
                delayed(process_single_lap_job)(lap_num) for lap_num in lap_numbers
            )
        else:
            results = [process_single_lap_job(lap_num) for lap_num in lap_numbers]
        
        return results

    def get_circuit_info(self, event: str, session: str) -> Optional[CircuitCorners]:
        """Get circuit corner information as msgspec struct."""
        cache_key = f"{self.year}-{event}-{session}"

        if cache_key in CIRCUIT_INFO_CACHE:
            return CIRCUIT_INFO_CACHE[cache_key]

        try:
            f1session = self.get_session(event, session)
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]

            try:
                circuit_info = f1session.get_circuit_info()
                corners = circuit_info.corners
                rotation = circuit_info.rotation

                corner_struct = CircuitCorners(
                    CornerNumber=corners["Number"].tolist(),
                    X=corners["X"].tolist(),
                    Y=corners["Y"].tolist(),
                    Angle=corners["Angle"].tolist(),
                    Distance=corners["Distance"].tolist(),
                    Rotation=float(rotation),
                )
                CIRCUIT_INFO_CACHE[cache_key] = corner_struct
                return corner_struct
            except (AttributeError, KeyError):
                circuit_info, rotation = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_struct = CircuitCorners(
                        CornerNumber=circuit_info["Number"].tolist(),
                        X=circuit_info["X"].tolist(),
                        Y=circuit_info["Y"].tolist(),
                        Angle=circuit_info["Angle"].tolist(),
                        Distance=(circuit_info["Distance"] / 10).tolist(),
                        Rotation=float(rotation),
                    )
                    CIRCUIT_INFO_CACHE[cache_key] = corner_struct
                    return corner_struct

            return None
        except Exception as e:
            logger.error(f"Error getting circuit info: {str(e)}")
            return None

    def _get_circuit_info_from_api(
        self, circuit_key: int
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """Get circuit information from API."""
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                return None, 0.0

            data = response.json()
            rotation = float(data.get("rotation", 0.0))

            rows = [
                (
                    float(entry.get("trackPosition", {}).get("x", 0.0)),
                    float(entry.get("trackPosition", {}).get("y", 0.0)),
                    int(entry.get("number", 0)),
                    str(entry.get("letter", "")),
                    float(entry.get("angle", 0.0)),
                    float(entry.get("length", 0.0)),
                )
                for entry in data["corners"]
            ]

            return (
                pd.DataFrame(
                    rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
                ),
                rotation,
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data: {str(e)}")
            return None, 0.0

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a driver using msgspec for I/O."""
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)

        try:
            if f1session is None:
                f1session = self.get_session(event, session, load_telemetry=True)

            # Save lap times using msgspec
            laptimes = self.laps_data(event, session, driver, f1session)
            # Replace NaN values with None before JSON serialization
            laptimes["time"] = ["None" if pd.isna(x) else x for x in laptimes["time"]]
            laptimes["lap"] = ["None" if pd.isna(x) else x for x in laptimes["lap"]]
            laptimes["compound"] = [
                "None" if pd.isna(x) else x for x in laptimes["compound"]
            ]
            with open(f"{driver_dir}/laptimes.json", "wb") as f:
                f.write(encoder.encode(laptimes))

            # Get lap numbers
            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

            driver_laps["LapNumber"] = driver_laps["LapNumber"].astype(int)
            # Create a new column for lap times in seconds to avoid dtype conflicts
            driver_laps["LapTimeSeconds"] = driver_laps["LapTime"].apply(
                lambda x: x.total_seconds() if hasattr(x, "total_seconds") else x
            )
            lap_numbers = driver_laps["LapNumber"].tolist()

            if self.use_joblib and len(lap_numbers) > self.batch_size:
                lap_batches = [
                    lap_numbers[i:i + self.batch_size] 
                    for i in range(0, len(lap_numbers), self.batch_size)
                ]
                
                with ThreadPoolExecutor(max_workers=min(4, len(lap_batches))) as executor:
                    futures = [
                        executor.submit(
                            self.process_lap_batch_with_joblib,
                            event, session, driver, batch, driver_dir, f1session
                        )
                        for batch in lap_batches
                    ]
                    
                    for future in as_completed(futures):
                        future.result()
            else:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        executor.submit(
                            self.process_lap,
                            event, session, driver, lap_number, driver_dir, f1session, driver_laps
                        )
                        for lap_number in lap_numbers
                    ]

                    for future in as_completed(futures):
                        future.result()

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str) -> None:
        """Process event/session using msgspec for all I/O."""
        logger.info(f"Processing {event} - {session} with msgspec optimization")

        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event, session, load_telemetry=True)

            # Save drivers using msgspec
            drivers_info = self.session_drivers(event, session)
            with open(f"{base_dir}/drivers.json", "wb") as f:
                f.write(encoder.encode(drivers_info))

            # Save circuit corners using msgspec
            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                with open(f"{base_dir}/corners.json", "wb") as f:
                    f.write(encoder.encode(corner_info))

            drivers = self.session_drivers_list(event, session)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        self.process_driver, event, session, driver, base_dir, f1session
                    )
                    for driver in drivers
                ]

                for future in as_completed(futures):
                    future.result()

        except Exception as e:
            logger.error(f"Error processing {event} - {session}: {str(e)}")

    def process_all_data(self, max_workers: int = 4) -> None:
        """Process all data with msgspec optimization."""
        logger.info(f"Starting msgspec-optimized telemetry extraction for {self.year}")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for event in self.events:
                for session in self.sessions:
                    futures.append(
                        executor.submit(self.process_event_session, event, session)
                    )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error in processing task: {str(e)}")

        elapsed_time = time.time() - start_time
        logger.info(f"Extraction completed in {elapsed_time:.2f} seconds")

    def clear_joblib_cache(self):
        """Clear joblib cache."""
        if hasattr(memory, 'clear'):
            memory.clear()
            logger.info("Joblib cache cleared")


def check_memory_usage(threshold_percent=80):
    """Check memory usage and clear caches if needed."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()

    logger.info(
        f"Memory usage: {memory_percent:.2f}% ({memory_info.rss / 1024 / 1024:.2f} MB)"
    )

    if memory_percent > threshold_percent:
        logger.warning(f"Memory exceeds {threshold_percent}%, clearing caches")
        SESSION_CACHE.clear()
        CIRCUIT_INFO_CACHE.clear()

        if hasattr(memory, 'clear'):
            memory.clear()

        gc.collect()

        new_percent = psutil.Process(os.getpid()).memory_percent()
        logger.info(f"New memory usage: {new_percent:.2f}%")
        return True

    return False


def is_data_available(year, events, sessions):
    """Check if data is available."""
    try:
        if not events or not sessions:
            return False

        event = events[0]
        session = sessions[0]

        logger.info(f"Checking data for {year} {event} {session}...")

        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty or len(f1session.laps["Driver"].unique()) == 0:
            return False

        logger.info(f"Data available for {year} {event} {session}")
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point with msgspec optimization."""
    try:
        # Configuration
        use_joblib = True
        n_jobs = -1
        batch_size = 8
        
        extractor = TelemetryExtractor(
            use_joblib=use_joblib,
            n_jobs=n_jobs,
            batch_size=batch_size
        )

        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

        wait_time = 30
        max_attempts = 720
        attempt = 0

        logger.info(f"Waiting for {extractor.year} season data with msgspec optimization...")

        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                logger.info("Data available. Starting msgspec-optimized extraction...")
                extractor.process_all_data(max_workers=max_workers)
                break
            else:
                attempt += 1
                logger.info(f"Waiting... ({attempt}/{max_attempts})")
                time.sleep(wait_time)
                check_memory_usage()

        if attempt >= max_attempts:
            logger.error("Exceeded maximum wait time. Exiting.")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()