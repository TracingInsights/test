import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Union

import fastf1
import msgspec
import numpy as np
import pandas as pd
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
HEADERS = {"User-Agent": f"FastF1/"}

# Global cache for session objects to prevent reloading
SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}

# Initialize joblib memory for persistent caching
memory = Memory(location='./cache_joblib', verbose=0)


# ============================================================================
# MSGSPEC STRUCTS - Type-safe, high-performance data structures
# ============================================================================

class TelemetryData(msgspec.Struct, kw_only=True, array_like=True):
    """Optimized telemetry data structure with msgspec."""
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


class LapData(msgspec.Struct, kw_only=True, array_like=True):
    """Optimized lap data structure."""
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
    """Driver information structure."""
    driver: str
    team: str


class DriversData(msgspec.Struct, kw_only=True):
    """Wrapper for drivers data."""
    drivers: list[DriverInfo]


class CircuitInfo(msgspec.Struct, kw_only=True, array_like=True):
    """Circuit corner information."""
    CornerNumber: list[int]
    X: list[float]
    Y: list[float]
    Angle: list[float]
    Distance: list[float]
    Rotation: float


# ============================================================================
# MSGSPEC ENCODERS/DECODERS - Ultra-fast serialization
# ============================================================================

# Use MessagePack for binary encoding (faster than JSON)
# For compatibility, you can also use JSONEncoder
encoder = msgspec.msgpack.Encoder()
decoder_telemetry = msgspec.msgpack.Decoder(TelemetryWrapper)
decoder_lap = msgspec.msgpack.Decoder(LapData)
decoder_drivers = msgspec.msgpack.Decoder(DriversData)
decoder_circuit = msgspec.msgpack.Decoder(CircuitInfo)

# JSON encoders/decoders as fallback (still faster than stdlib json)
json_encoder = msgspec.json.Encoder()
json_decoder_telemetry = msgspec.json.Decoder(TelemetryWrapper)
json_decoder_lap = msgspec.json.Decoder(LapData)
json_decoder_drivers = msgspec.json.Decoder(DriversData)
json_decoder_circuit = msgspec.json.Decoder(CircuitInfo)


# ============================================================================
# OPTIMIZED FILE I/O FUNCTIONS
# ============================================================================

def write_msgpack(data: msgspec.Struct, filepath: str) -> None:
    """Write msgspec struct to file using MessagePack format."""
    with open(filepath, "wb") as f:
        f.write(encoder.encode(data))


def write_json(data: msgspec.Struct, filepath: str) -> None:
    """Write msgspec struct to file using JSON format (for readability)."""
    with open(filepath, "wb") as f:
        f.write(json_encoder.encode(data))


def read_msgpack(filepath: str, decoder) -> msgspec.Struct:
    """Read msgpack file and decode to struct."""
    with open(filepath, "rb") as f:
        return decoder.decode(f.read())


def read_json(filepath: str, decoder) -> msgspec.Struct:
    """Read JSON file and decode to struct."""
    with open(filepath, "rb") as f:
        return decoder.decode(f.read())


# ============================================================================
# TELEMETRY EXTRACTOR CLASS
# ============================================================================

class TelemetryExtractor:
    """Optimized class to handle extraction of F1 telemetry data with msgspec."""

    def __init__(
        self,
        year: int = DEFAULT_YEAR,
        events: List[str] = None,
        sessions: List[str] = None,
        use_joblib: bool = True,
        n_jobs: int = -1,
        batch_size: int = 8,
        use_msgpack: bool = True,  # New option for binary format
    ):
        """Initialize the TelemetryExtractor."""
        self.year = year
        self.use_joblib = use_joblib
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.use_msgpack = use_msgpack  # Use MessagePack for better performance
        
        # File extensions
        self.ext = ".msgpack" if use_msgpack else ".json"
        self.write_func = write_msgpack if use_msgpack else write_json
        
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
            # 'Abu Dhabi Grand Prix',        ]
        self.sessions = sessions or ["Practice 1"]

    def get_session(
        self, event: Union[str, int], session: str, load_telemetry: bool = False
    ) -> fastf1.core.Session:
        """Get a cached session object to prevent reloading."""
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    def session_drivers_list(self, event: Union[str, int], session: str) -> List[str]:
        """Get list of driver codes for a given event and session."""
        try:
            f1session = self.get_session(event, session)
            return list(f1session.laps["Driver"].unique())
        except Exception as e:
            logger.error(f"Error getting driver list for {event} {session}: {str(e)}")
            return []

    def session_drivers(
        self, event: Union[str, int], session: str
    ) -> DriversData:
        """Get drivers available for a given event and session."""
        try:
            f1session = self.get_session(event, session)
            laps = f1session.laps
            team_colors = utils.team_colors(self.year)
            laps["color"] = laps["Team"].map(team_colors)

            unique_drivers = laps["Driver"].unique()

            drivers = [
                DriverInfo(
                    driver=driver,
                    team=laps[laps.Driver == driver].Team.iloc[0],
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
        """Get lap data for a specific driver in a session."""
        try:
            if f1session is None:
                f1session = self.get_session(event, session)

            laps = f1session.laps
            driver_laps = laps.pick_drivers(driver).copy()

            # Helper function to convert timedelta to seconds or "None"
            def timedelta_to_seconds(time_value):
                if pd.isna(time_value) or not hasattr(time_value, "total_seconds"):
                    return "None"
                return round(time_value.total_seconds(), 3)

            # Helper for nullable int values
            def nullable_int(value):
                return "None" if pd.isna(value) else int(value)
            
            # Helper for nullable bool values
            def nullable_bool(value):
                return "None" if pd.isna(value) else bool(value)
            
            # Helper for nullable string values
            def nullable_str(value):
                return "None" if pd.isna(value) else str(value)

            # Convert lap times to seconds
            lap_times = [timedelta_to_seconds(lt) for lt in driver_laps["LapTime"]]
            sector1_times = [timedelta_to_seconds(s1) for s1 in driver_laps["Sector1Time"]]
            sector2_times = [timedelta_to_seconds(s2) for s2 in driver_laps["Sector2Time"]]
            sector3_times = [timedelta_to_seconds(s3) for s3 in driver_laps["Sector3Time"]]

            # Handle compounds (nullable string)
            compounds = [nullable_str(c) if pd.isna(c) else c for c in driver_laps["Compound"]]

            # Create LapData struct
            return LapData(
                time=lap_times,
                lap=driver_laps["LapNumber"].tolist(),
                compound=compounds,
                stint=[nullable_int(s) for s in driver_laps["Stint"]],
                s1=sector1_times,
                s2=sector2_times,
                s3=sector3_times,
                life=[nullable_int(life) for life in driver_laps["TyreLife"]],
                pos=[nullable_int(pos) for pos in driver_laps["Position"]],
                status=[nullable_str(status) for status in driver_laps["TrackStatus"]],
                pb=[nullable_bool(pb) for pb in driver_laps["IsPersonalBest"]],
            )
        except Exception as e:
            logger.error(
                f"Error getting lap data for {driver} in {event} {session}: {str(e)}"
            )
            return LapData(
                time=[],
                lap=[],
                compound=[],
                stint=[],
                s1=[],
                s2=[],
                s3=[],
                life=[],
                pos=[],
                status=[],
                pb=[],
            )

    @staticmethod
    @memory.cache
    def calculate_x_acceleration(vx_array, time_array, Nax):
        """Calculate and smooth X-acceleration component using joblib caching."""
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
        """Calculate and smooth Y-acceleration component using joblib caching."""
        dx = np.gradient(x_array)
        dy = np.gradient(y_array)

        theta = np.arctan2(dy, (dx + np.finfo(float).eps))
        theta[0] = theta[1]
        theta_noDiscont = np.unwrap(theta)

        ds = np.gradient(dist_array)
        dtheta = np.gradient(theta_noDiscont)

        # Clean up outliers
        for i in np.arange(1, len(dtheta) - 1).astype(int):
            if abs(dtheta[i]) > 0.5:
                dtheta[i] = dtheta[i - 1]

        C = dtheta / (ds + 0.0001)
        ay = np.square(vx_array) * C

        indexProblems = np.abs(ay) > 150
        ay[indexProblems] = 0

        ay_smooth = np.convolve(ay, np.ones((Nay,)) / Nay, mode="same")
        return ay_smooth

    @staticmethod
    @memory.cache
    def calculate_z_acceleration(vx_array, x_array, z_array, dist_array, Naz):
        """Calculate and smooth Z-acceleration component using joblib caching."""
        dx = np.gradient(x_array)
        dz = np.gradient(z_array)

        z_theta = np.arctan2(dz, (dx + np.finfo(float).eps))
        z_theta[0] = z_theta[1]
        z_theta_noDiscont = np.unwrap(z_theta)

        ds = np.gradient(dist_array)
        z_dtheta = np.gradient(z_theta_noDiscont)

        # Clean up outliers
        for i in np.arange(1, len(z_dtheta) - 1).astype(int):
            if abs(z_dtheta[i]) > 0.5:
                z_dtheta[i] = z_dtheta[i - 1]

        z_C = z_dtheta / (ds + 0.0001)
        az = np.square(vx_array) * z_C

        indexProblems = np.abs(az) > 150
        az[indexProblems] = 0

        az_smooth = np.convolve(az, np.ones((Naz,)) / Naz, mode="same")
        return az_smooth

    def accCalc(
        self, telemetry: pd.DataFrame, Nax: int, Nay: int, Naz: int
    ) -> pd.DataFrame:
        """Calculate acceleration components from telemetry data with joblib parallelization."""
        vx = telemetry["Speed"] / 3.6
        time_float = telemetry["Time"] / np.timedelta64(1, "s")

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

    def process_single_lap_telemetry(self, telemetry: pd.DataFrame, data_key: str) -> TelemetryWrapper:
        """Process telemetry for a single lap and return msgspec struct."""
        acc_tel = self.accCalc(telemetry, 3, 9, 9)
        acc_tel["Time"] = acc_tel["Time"].dt.total_seconds()

        # Convert DRS and Brake to binary values
        acc_tel["DRS"] = acc_tel["DRS"].apply(lambda x: 1 if x in [10, 12, 14] else 0)
        acc_tel["Brake"] = acc_tel["Brake"].apply(lambda x: 1 if x == True else 0)

        # Convert numpy arrays to lists for msgspec
        # Use native Python types for better msgspec performance
        tel_data = TelemetryData(
            time=[float(x) for x in acc_tel["Time"].values],
            rpm=[int(x) for x in acc_tel["RPM"].values],
            speed=[float(x) for x in acc_tel["Speed"].values],
            gear=[int(x) for x in acc_tel["nGear"].values],
            throttle=[int(x) for x in acc_tel["Throttle"].values],
            brake=[int(x) for x in acc_tel["Brake"].values],
            drs=[int(x) for x in acc_tel["DRS"].values],
            distance=[float(x) for x in acc_tel["Distance"].values],
            rel_distance=[float(x) for x in acc_tel["RelativeDistance"].values],
            acc_x=[float(x) for x in acc_tel["Ax"].values],
            acc_y=[float(x) for x in acc_tel["Ay"].values],
            acc_z=[float(x) for x in acc_tel["Az"].values],
            x=[float(x) for x in acc_tel["X"].values],
            y=[float(x) for x in acc_tel["Y"].values],
            z=[float(x) for x in acc_tel["Z"].values],
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
        """Process a single lap for a driver."""        
        file_path = f"{driver_dir}/{lap_number}_tel{self.ext}"

        # Skip if file already exists
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
                logger.warning(
                    f"No data for {driver} lap {lap_number} in {event} {session}"
                )
                return False

            telemetry = selected_lap.get_telemetry()
            
            data_key = f"{self.year}-{event}-{session}-{driver}-{lap_number}"
            
            # Process telemetry and get msgspec struct
            telemetry_struct = self.process_single_lap_telemetry(telemetry, data_key)

            # Write using msgspec (much faster than JSON)
            self.write_func(telemetry_struct, file_path)

            return True
        except Exception as e:
            logger.error(f"Error processing lap {lap_number} for {driver}: {str(e)}")
            return False

    def process_lap_batch_with_joblib(
        self, event: str, session: str, driver: str, lap_numbers: List[int], 
        driver_dir: str, f1session=None
    ) -> List[bool]:
        """Process a batch of laps using joblib for CPU-intensive work."""
        
        def process_single_lap_job(lap_number):
            return self.process_lap(event, session, driver, lap_number, driver_dir, f1session)

        if self.use_joblib and len(lap_numbers) > 1:
            results = Parallel(n_jobs=self.n_jobs, backend='loky', prefer='processes')(
                delayed(process_single_lap_job)(lap_num) for lap_num in lap_numbers
            )
        else:
            results = [process_single_lap_job(lap_num) for lap_num in lap_numbers]
        
        return results

    def get_circuit_info(self, event: str, session: str) -> Optional[CircuitInfo]:
        """Get circuit corner information."""
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

                corner_struct = CircuitInfo(
                    CornerNumber=corners["Number"].tolist(),
                    X=corners["X"].tolist(),
                    Y=corners["Y"].tolist(),
                    Angle=corners["Angle"].tolist(),
                    Distance=corners["Distance"].tolist(),
                    Rotation=rotation,
                )
                CIRCUIT_INFO_CACHE[cache_key] = corner_struct
                return corner_struct
            except (AttributeError, KeyError):
                circuit_info, rotation = self._get_circuit_info_from_api(circuit_key)
                if circuit_info is not None:
                    corner_struct = CircuitInfo(
                        CornerNumber=circuit_info["Number"].tolist(),
                        X=circuit_info["X"].tolist(),
                        Y=circuit_info["Y"].tolist(),
                        Angle=circuit_info["Angle"].tolist(),
                        Distance=(circuit_info["Distance"] / 10).tolist(),
                        Rotation=rotation,
                    )
                    CIRCUIT_INFO_CACHE[cache_key] = corner_struct
                    return corner_struct

            logger.warning(f"Could not get corner data for {event} {session}")
            return None
        except Exception as e:
            logger.error(f"Error getting circuit info for {event} {session}: {str(e)}")
            return None

    def _get_circuit_info_from_api(
        self, circuit_key: int
    ) -> Tuple[Optional[pd.DataFrame], float]:
        """Get circuit information from the MultiViewer API."""
        url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                logger.debug(f"[{response.status_code}] {response.content.decode()}")
                return None, 0.0

            data = response.json()
            rotation = float(data.get("rotation", 0.0))

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

            return (
                pd.DataFrame(
                    rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"]
                ),
                rotation,
            )
        except Exception as e:
            logger.error(f"Error fetching circuit data from API: {str(e)}")
            return None, 0.0

    def process_driver(
        self, event: str, session: str, driver: str, base_dir: str, f1session=None
    ) -> None:
        """Process all laps for a single driver with optimized joblib batching."""
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
            self.write_func(laptimes, f"{driver_dir}/laptimes{self.ext}")

            # Get driver laps
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
                        future.result()

        except Exception as e:
            logger.error(f"Error processing driver {driver}: {str(e)}")

    def process_event_session(self, event: str, session: str) -> None:
        """Process a single event and session, extracting all telemetry data."""
        logger.info(f"Processing {event} - {session} with msgspec ({self.ext[1:]} format)")

        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)

        try:
            f1session = self.get_session(event, session, load_telemetry=True)

            # Save drivers information using msgspec
            drivers_info = self.session_drivers(event, session)
            self.write_func(drivers_info, f"{base_dir}/drivers{self.ext}")

            # Save circuit corner information using msgspec
            corner_info = self.get_circuit_info(event, session)
            if corner_info:
                self.write_func(corner_info, f"{base_dir}/corners{self.ext}")

            drivers = self.session_drivers_list(event, session)

            # Process drivers in parallel
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
        """Process all configured events and sessions, with parallelization."""
        logger.info(f"Starting msgspec-optimized telemetry extraction for {self.year} season")
        logger.info(f"Format: {'MessagePack (binary)' if self.use_msgpack else 'JSON'}")
        logger.info(f"Events: {self.events}")
        logger.info(f"Sessions: {self.sessions}")
        
        if self.use_joblib:
            logger.info(f"Joblib settings: n_jobs={self.n_jobs}, batch_size={self.batch_size}")

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
        logger.info(f"Telemetry extraction completed in {elapsed_time:.2f} seconds")

    def clear_joblib_cache(self):
        """Clear the joblib memory cache."""
        if hasattr(memory, 'clear'):
            memory.clear()
            logger.info("Joblib cache cleared")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

import gc
import psutil

def check_memory_usage(threshold_percent=80):
    """Check if memory usage exceeds threshold and clear caches if needed."""
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
        SESSION_CACHE.clear()
        CIRCUIT_INFO_CACHE.clear()

        if hasattr(memory, 'clear'):
            memory.clear()
            logger.info("Joblib cache cleared")

        gc.collect()

        new_memory_percent = psutil.Process(os.getpid()).memory_percent()
        logger.info(
            f"New memory usage after clearing caches: {new_memory_percent:.2f}%"
        )
        return True

    return False


def is_data_available(year, events, sessions):
    """Check if data is available for the specified year, events, and sessions."""
    try:
        if not events or not sessions:
            logger.warning("No events or sessions specified to check")
            return False

        event = events[0]
        session = sessions[0]

        logger.info(f"Checking data availability for {year} {event} {session}...")

        f1session = fastf1.get_session(year, event, session)
        f1session.load(telemetry=False, weather=False, messages=False)

        if f1session.laps.empty:
            logger.info(f"No lap data available yet for {year} {event} {session}")
            return False

        if len(f1session.laps["Driver"].unique()) == 0:
            logger.info(f"No driver data available yet for {year} {event} {session}")
            return False

        logger.info(f"Data is available for {year} {event} {session}")
        return True

    except Exception as e:
        logger.info(f"Data not yet available: {str(e)}")
        return False


def main():
    """Main entry point for the script with msgspec optimization."""
    try:
        # Configuration options
        use_joblib = True  # Set to False to disable joblib optimizations
        n_jobs = -1  # -1 uses all available cores
        batch_size = 8  # Number of laps per batch for joblib processing
        use_msgpack = True  # True for binary MessagePack, False for JSON
        
        # Create extractor with msgspec + joblib options
        extractor = TelemetryExtractor(
            use_joblib=use_joblib,
            n_jobs=n_jobs,
            batch_size=batch_size,
            use_msgpack=use_msgpack,
        )

        # Use more workers on GitHub Actions
        is_github_actions = os.environ.get("GITHUB_ACTIONS") == "true"
        max_workers = 12 if is_github_actions else 8

        # Wait for data to be available
        wait_time = 30  # seconds between checks
        max_attempts = 720  # 12 hours max wait time
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

                check_memory_usage()

        if attempt >= max_attempts:
            logger.error(
                f"Exceeded maximum wait time ({max_attempts * wait_time / 3600} hours). Exiting."
            )

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()