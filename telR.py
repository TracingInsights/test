import json
import logging
import os
import time
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union

import fastf1
import numpy as np
import pandas as pd
import requests
from joblib import Memory

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

SESSION_CACHE = {}
CIRCUIT_INFO_CACHE = {}
memory = Memory(location='./cache_joblib', verbose=0)

class TelemetryExtractor:
    def __init__(self, year=DEFAULT_YEAR, events=None, sessions=None):
        self.year = year
        self.events = events or ['Abu Dhabi Grand Prix']
        self.sessions = sessions or ["Race"]

    def get_session(self, event, session, load_telemetry=False):
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key not in SESSION_CACHE:
            f1session = fastf1.get_session(self.year, event, session)
            f1session.load(telemetry=load_telemetry, weather=True, messages=True)
            SESSION_CACHE[cache_key] = f1session
        return SESSION_CACHE[cache_key]

    @staticmethod
    @memory.cache
    def calc_acc_full(vx, t, x, y, z, dist):
        dt = np.gradient(t); dt[dt == 0] = 1e-6
        ax = np.gradient(vx) / dt
        for i in range(1, len(ax)-1):
            if ax[i] > 25: ax[i] = ax[i-1]
        ax_s = np.convolve(ax, np.ones((3,)) / 3, mode="same")

        dx, dy = np.gradient(x), np.gradient(y)
        ty = np.arctan2(dy, dx + 2.220446049250313e-16); ty[0] = ty[1]; ty_u = np.unwrap(ty)
        ds, dty = np.gradient(dist), np.gradient(ty_u)
        for i in range(1, len(dty)-1):
            if abs(dty[i]) > 0.5: dty[i] = dty[i-1]
        ay_s = np.convolve(np.square(vx) * (dty / (ds + 0.0001)), np.ones((9,)) / 9, mode="same")
        ay_s[np.abs(ay_s) > 150] = 0

        dz = np.gradient(z)
        tz = np.arctan2(dz, dx + 2.220446049250313e-16); tz[0] = tz[1]; tz_u = np.unwrap(tz)
        dtz = np.gradient(tz_u)
        for i in range(1, len(dtz)-1):
            if abs(dtz[i]) > 0.5: dtz[i] = dtz[i-1]
        az_s = np.convolve(np.square(vx) * (dtz / (ds + 0.0001)), np.ones((9,)) / 9, mode="same")
        az_s[np.abs(az_s) > 150] = 0
        return ax_s, ay_s, az_s

    def process_lap(self, event, session, driver, lap, driver_dir):
        ln = int(lap.LapNumber)
        file_path = f"{driver_dir}/{ln}_tel.json"
        if os.path.exists(file_path): return True
        try:
            tel = lap.get_telemetry()
            vx, t = tel["Speed"].values / 3.6, tel["Time"].dt.total_seconds().values
            x, y, z, dist = tel["X"].values, tel["Y"].values, tel["Z"].values, tel["Distance"].values
            ax, ay, az = self.calc_acc_full(vx, t, x, y, z, dist)
            
            data = {
                "tel": {
                    "time": t.tolist(), "rpm": tel["RPM"].tolist(), "speed": tel["Speed"].tolist(),
                    "gear": tel["nGear"].tolist(), "throttle": tel["Throttle"].tolist(),
                    "brake": tel["Brake"].astype(int).tolist(),
                    "drs": np.isin(tel["DRS"].values, [10, 12, 14]).astype(int).tolist(),
                    "distance": dist.tolist(), "rel_distance": tel["RelativeDistance"].tolist(),
                    "acc_x": ax.tolist(), "acc_y": ay.tolist(), "acc_z": az.tolist(),
                    "x": x.tolist(), "y": y.tolist(), "z": z.tolist(),
                    "dataKey": f"{self.year}-{event}-{session}-{driver}-{ln}",
                }
            }
            with open(file_path, "w") as f: json.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error lap {ln}: {e}")
            return False

    def process_driver(self, event, session, driver, base_dir, f1session):
        driver_dir = f"{base_dir}/{driver}"
        os.makedirs(driver_dir, exist_ok=True)
        try:
            laps = f1session.laps.pick_drivers(driver).copy()
            def to_secs(s): return ["None" if pd.isna(x) else round(x.total_seconds(), 3) for x in s]
            def to_int(s): return ["None" if pd.isna(x) else int(x) for x in s]
            lt = {
                "time": to_secs(laps["LapTime"]), "lap": to_int(laps["LapNumber"]),
                "compound": laps["Compound"].fillna("None").tolist(), "stint": to_int(laps["Stint"]),
                "s1": to_secs(laps["Sector1Time"]), "s2": to_secs(laps["Sector2Time"]), "s3": to_secs(laps["Sector3Time"]),
                "life": to_int(laps["TyreLife"]), "pos": to_int(laps["Position"]),
                "status": [str(x) if not pd.isna(x) else "None" for x in laps["TrackStatus"]],
                "pb": [bool(x) if not pd.isna(x) else "None" for x in laps["IsPersonalBest"]],
            }
            with open(f"{driver_dir}/laptimes.json", "w") as f: json.dump(lt, f)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self.process_lap, event, session, driver, laps.iloc[i], driver_dir) for i in range(len(laps))]
                for f in as_completed(futures): f.result()
        except Exception as e: logger.error(f"Error driver {driver}: {e}")

    def process_event_session(self, event, session):
        logger.info(f"Processing {event} - {session}")
        base_dir = f"{event}/{session}"
        os.makedirs(base_dir, exist_ok=True)
        try:
            f1session = self.get_session(event, session, True)
            laps = f1session.laps
            team_map = laps.groupby("Driver")["Team"].first().to_dict()
            with open(f"{base_dir}/drivers.json", "w") as f:
                json.dump({"drivers": [{"driver": d, "team": team_map[d]} for d in laps["Driver"].unique()]}, f)
            ci = self.get_circuit_info(event, session)
            if ci:
                with open(f"{base_dir}/corners.json", "w") as f: json.dump(ci, f)
            
            drivers = list(laps["Driver"].unique())
            with ThreadPoolExecutor(max_workers=2) as executor:
                for d in drivers: executor.submit(self.process_driver, event, session, d, base_dir, f1session)
        except Exception as e: logger.error(f"Error session: {e}")

    def get_circuit_info(self, event, session):
        cache_key = f"{self.year}-{event}-{session}"
        if cache_key in CIRCUIT_INFO_CACHE: return CIRCUIT_INFO_CACHE[cache_key]
        try:
            f1session = self.get_session(event, session)
            circuit_key = f1session.session_info["Meeting"]["Circuit"]["Key"]
            try:
                ci = f1session.get_circuit_info()
                res = {
                    "CornerNumber": ci.corners["Number"].tolist(), "X": ci.corners["X"].tolist(),
                    "Y": ci.corners["Y"].tolist(), "Angle": ci.corners["Angle"].tolist(),
                    "Distance": ci.corners["Distance"].tolist(), "Rotation": ci.rotation,
                }
                CIRCUIT_INFO_CACHE[cache_key] = res
                return res
            except:
                url = f"{PROTO}://{HOST}/api/v1/circuits/{circuit_key}/{self.year}"
                resp = requests.get(url, headers=HEADERS)
                if resp.status_code == 200:
                    data = resp.json(); rot = float(data.get("rotation", 0.0)); rows = []
                    for entry in data["corners"]:
                        rows.append((float(entry.get("trackPosition", {}).get("x", 0.0)),
                                     float(entry.get("trackPosition", {}).get("y", 0.0)),
                                     int(entry.get("number", 0)), str(entry.get("letter", "")),
                                     float(entry.get("angle", 0.0)), float(entry.get("length", 0.0))))
                    df = pd.DataFrame(rows, columns=["X", "Y", "Number", "Letter", "Angle", "Distance"])
                    res = {
                        "CornerNumber": df["Number"].tolist(), "X": df["X"].tolist(),
                        "Y": df["Y"].tolist(), "Angle": df["Angle"].tolist(),
                        "Distance": (df["Distance"] / 10).tolist(), "Rotation": rot,
                    }
                    CIRCUIT_INFO_CACHE[cache_key] = res
                    return res
            return None
        except: return None

    def process_all_data(self):
        start_time = time.time()
        for e in self.events:
            for s in self.sessions: self.process_event_session(e, s)
        logger.info(f"Completed in {time.time() - start_time:.2f} seconds")

def check_memory_usage(threshold_percent=80):
    process = psutil.Process(os.getpid())
    if process.memory_percent() > threshold_percent:
        SESSION_CACHE.clear(); CIRCUIT_INFO_CACHE.clear(); gc.collect()

def is_data_available(year, events, sessions):
    try:
        f1session = fastf1.get_session(year, events[0], sessions[0])
        f1session.load(telemetry=False, weather=False, messages=False)
        return not f1session.laps.empty
    except: return False

def main():
    try:
        extractor = TelemetryExtractor()
        wait_time, max_attempts = 30, 720
        attempt = 0
        while attempt < max_attempts:
            if is_data_available(extractor.year, extractor.events, extractor.sessions):
                extractor.process_all_data()
                break
            else:
                attempt += 1
                time.sleep(wait_time); check_memory_usage()
    except Exception as e:
        logger.error(f"Main error: {e}")
        raise

if __name__ == "__main__":
    main()
