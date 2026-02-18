"""
TrainClear Prediction Generation
Combines Amtrak API + ML model to generate real-time predictions

Change Log:
- Added directional awareness (is_moving_toward) to filter trains heading away
- Added station safety interlock to prevent false APPROACHING/BLOCKING at stations
- Added 0-MPH / yard sanity check for stopped trains not at stations
- Added redundant update suppression (change detection) to avoid unnecessary updates
- Refined explanations for all statuses to be human-readable
"""

import pickle
import json
import sys
import os
import math
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from feature_engineering import FeatureEngine
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('TrainClear.predictions')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Compass heading -> degrees mapping (clockwise from north)
HEADING_TO_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}

# Default yard/switching delay when a train is stopped and not at a station
YARD_SWITCHING_DELAY_MINUTES = 15
# Stalled train delay (stopped far from a yard)
STALLED_TRAIN_DELAY_MINUTES = 60

# Brooklyn Yard coordinates for distance-based switching vs stalled logic
BROOKLYN_YARD_LAT = 45.4896
BROOKLYN_YARD_LON = -122.6437
YARD_SWITCHING_RADIUS_KM = 0.5

# Amtrak proximity detection tiers
AMTRAK_DETECTION_RADIUS_MILES = 20.0
AMTRAK_BLOCKING_THRESHOLD_MILES = 0.15
AMTRAK_APPROACHING_THRESHOLD_MILES = 2.0
AMTRAK_MAX_ETA_TO_SURFACE_MINUTES = 45  # Don't surface INCOMING if ETA > 45 min
PREDEPARTURE_RADIUS_MILES = 5.0  # Only show predeparture if train is within 5mi

# API health monitoring and caching
AMTRAK_CACHE = {
    'data': None,
    'timestamp': None,
    'cache_duration_minutes': 5
}

API_HEALTH = {
    'last_success': None,
    'last_failure': None,
    'consecutive_failures': 0
}

# Retry configuration (exponential backoff)
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

# Last Known Good TTL (don't show ghost trains during long outages)
LAST_KNOWN_GOOD_TTL_MINUTES = 30


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    """Return great-circle distance in kilometres between two GPS points."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def is_moving_toward(train_lat, train_lon, train_heading, crossing_lat, crossing_lon):
    """
    Determine whether a train is moving toward or away from a crossing.

    Computes the bearing from the train's position to the crossing and compares
    it with the train's reported compass heading.  If the angular difference
    exceeds 90 degrees, the train is considered to be moving *away*.

    Args:
        train_lat, train_lon: Train's current GPS coordinates.
        train_heading: Compass heading string (e.g. 'N', 'NE', 'SSW').
        crossing_lat, crossing_lon: Crossing GPS coordinates.

    Returns:
        True if the train appears to be heading toward the crossing (or heading
        is unknown), False otherwise.
    """
    # If heading is unknown/empty, conservatively assume moving toward
    if not train_heading or train_heading.upper() == 'UNKNOWN':
        logger.debug("Heading unknown; assuming train is moving toward crossing")
        return True

    heading_deg = HEADING_TO_DEGREES.get(train_heading.upper())
    if heading_deg is None:
        # Unrecognised heading string -- try to parse as a numeric degree value
        try:
            heading_deg = float(train_heading)
        except (ValueError, TypeError):
            logger.debug(f"Unrecognised heading '{train_heading}'; assuming toward")
            return True

    # Calculate bearing from train to crossing using the forward-azimuth formula
    lat1 = math.radians(train_lat)
    lat2 = math.radians(crossing_lat)
    delta_lon = math.radians(crossing_lon - train_lon)

    x = math.sin(delta_lon) * math.cos(lat2)
    y = (math.cos(lat1) * math.sin(lat2)
         - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    bearing = math.degrees(math.atan2(x, y)) % 360

    # Angular difference, wrapped to [0, 180]
    diff = abs(heading_deg - bearing) % 360
    if diff > 180:
        diff = 360 - diff

    toward = diff <= 90
    logger.debug(
        f"Heading {train_heading} ({heading_deg} deg) vs bearing-to-crossing "
        f"{bearing:.1f} deg -> diff {diff:.1f} deg -> {'TOWARD' if toward else 'AWAY'}"
    )
    return toward


def get_current_station_status(train):
    """
    Inspect the *stations* array inside an Amtraker train object to determine
    whether the train is currently stopped at a station.

    The Amtraker v3 API includes a ``stations`` list on each train.  Each entry
    has (among others) a ``status`` field which is ``"Station"`` when the train
    is physically at that stop.

    Returns:
        (is_at_station: bool, station_code: str | None, station_name: str | None)
    """
    stations = train.get('stations', [])
    if not stations:
        return False, None, None

    for station in stations:
        if station.get('status') == 'Station':
            code = station.get('code', station.get('stationCode', ''))
            name = station.get('stationName', station.get('name', code))
            logger.debug(f"Train is at station: {name} ({code})")
            return True, code, name

    return False, None, None


def _request_with_retry(url, timeout=10, max_retries=MAX_RETRIES):
    """
    HTTP GET with exponential backoff retry for transient failures.
    Returns (response, error_msg) -- response is None on total failure.
    """
    backoff = INITIAL_BACKOFF_SECONDS
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=timeout)

            if response.status_code == 200:
                return response, None

            if response.status_code in RETRYABLE_STATUS_CODES:
                last_error = f"HTTP {response.status_code} on attempt {attempt}/{max_retries}"
                if attempt < max_retries:
                    logger.warning(f"   {last_error}, retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER
                continue
            else:
                # Non-retryable error (4xx, etc.)
                return response, f"HTTP {response.status_code} (non-retryable)"

        except requests.exceptions.Timeout:
            last_error = f"Timeout on attempt {attempt}/{max_retries}"
            if attempt < max_retries:
                logger.warning(f"   {last_error}, retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

        except requests.exceptions.RequestException as e:
            last_error = f"Connection error on attempt {attempt}/{max_retries}: {e}"
            if attempt < max_retries:
                logger.warning(f"   {last_error}, retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

    return None, f"All {max_retries} retries exhausted. Last error: {last_error}"


def check_amtrak_api_health():
    """
    Test Amtrak API connectivity with retry logic.
    Returns (is_healthy: bool, error_message: str or None)
    """
    response, error_msg = _request_with_retry(
        "https://api-v3.amtraker.com/v3/trains",
        timeout=5,
        max_retries=2  # Lighter retry for health check
    )

    if response is not None and response.status_code == 200:
        API_HEALTH['last_success'] = datetime.now()
        API_HEALTH['consecutive_failures'] = 0
        return True, None
    else:
        API_HEALTH['last_failure'] = datetime.now()
        API_HEALTH['consecutive_failures'] += 1
        return False, error_msg or "Unknown error"


def calculate_amtrak_duration(speed_mph):
    """
    Estimate crossing blockage time based on Amtrak train speed.

    Typical Amtrak passenger train: 8-12 cars, ~1,000 feet long.
    Blockage includes gate-down lead time (~30 s) and gate-up lag (~30 s).
    """
    train_length_miles = 1000 / 5280  # ~0.19 miles
    if speed_mph <= 0:
        speed_mph = 10  # conservative fallback for stopped trains
    crossing_time = (train_length_miles / (speed_mph / 60)) + 1  # +1 min gate buffer
    return round(max(crossing_time, 1.5), 1)  # minimum 1.5 min


# ---------------------------------------------------------------------------
# Prediction Generator
# ---------------------------------------------------------------------------

class PredictionGenerator:
    """Generate real-time train crossing predictions"""

    def __init__(self):
        self.feature_engine = FeatureEngine()

        # Load trained models
        self.load_models()

        # Track the last prediction state for change-detection (in-memory)
        self._previous_predictions = {}

        # Distance history: {(crossing_id, train_num): [dist_prev, dist_curr]}
        # Used to detect trains moving away even when heading is ambiguous.
        self._distance_history = {}

        # Crossing configuration - VERIFIED Universal Ground Truth values
        self.crossings_config = {
            'se-11th-ave': {
                'lat': 45.5037,
                'lon': -122.6547,
                'fra_id': '759733U',
                'name': 'SE 11th Avenue'
            },
            'se-division-12th': {
                'lat': 45.5033,
                'lon': -122.6538,
                'fra_id': '759735H',        # VERIFIED: was 083313H (private spur)
                'name': 'SE Division & 12th'
            },
            'se-division-7th': {
                'lat': 45.5054,
                'lon': -122.6574,
                'fra_id': '759730Y',        # FRA labels as "SE8TH AVE" but is really 7th
                'name': 'SE Division & 7th'
            },
            # REMOVED: 'se-8th-division' (083310N) - private/industrial spur,
            # not in FRA public blocked crossing reports
            'se-salmon-12th': {
                'lat': 45.514392,
                'lon': -122.664830,
                'fra_id': '754552X',        # VERIFIED: was 754559V
                'name': 'SE Salmon & 12th'
            },
            'se-main-12th': {
                'lat': 45.514392,
                'lon': -122.664830,
                'fra_id': '754553E',        # VERIFIED: was 083312B (private spur)
                'name': 'SE Main & 12th'
            },
            'se-hawthorne-12th': {
                'lat': 45.512250,
                'lon': -122.664950,
                'fra_id': '754558N',
                'name': 'SE Hawthorne & 12th'
            },
            'se-madison-12th': {
                'lat': 45.512984,
                'lon': -122.664833,
                'fra_id': '754554L',
                'name': 'SE Madison & 12th'
            },
            'se-yamhill-12th': {
                'lat': 45.518666,
                'lon': -122.664803,
                'fra_id': '754550J',
                'name': 'SE Yamhill & 12th'
            },
            'se-taylor-12th': {
                'lat': 45.516000,
                'lon': -122.664800,
                'fra_id': '754551R',
                'name': 'SE Taylor & 12th'
            },
            'se-washington-12th': {
                'lat': 45.518666,
                'lon': -122.664803,
                'fra_id': '754543Y',
                'name': 'SE Washington & 12th'
            },
            # NW Portland Crossings - high-volume crossings from raw FRA data
            'nw-17th-ave': {
                'lat': 45.536963,
                'lon': -122.688756,
                'fra_id': '101880L',
                'name': 'NW 17th Ave'
            },
            'nw-naito-pkwy': {
                'lat': 45.529906,
                'lon': -122.677634,
                'fra_id': '810142C',
                'name': 'NW Naito Pkwy'
            },
            'nw-9th-ave': {
                'lat': 45.532361,
                'lon': -122.680371,
                'fra_id': '838534K',
                'name': 'NW 9th Ave'
            },
            'nw-15th-ave-s': {
                'lat': 45.532889,
                'lon': -122.686573,
                'name': 'NW 15th Ave (S)'
            },
            'nw-15th-ave': {
                'lat': 45.536084,
                'lon': -122.686078,
                'fra_id': '838536Y',
                'name': 'NW 15th Ave'
            },
            'van-houten-pl': {
                'lat': 45.577211,
                'lon': -122.736405,
                'fra_id': '808374S',
                'name': 'Van Houten Pl'
            },
        }

    def load_models(self):
        """Load trained ML models"""
        logger.info("Loading trained models...")

        try:
            with open('python/ml/models/train_classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)

            with open('python/ml/models/duration_regressor.pkl', 'rb') as f:
                self.duration_regressor = pickle.load(f)

            with open('python/ml/models/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)

            # Sanity check: verify classifier has expected class structure
            assert hasattr(self.classifier, 'classes_'), "Classifier missing classes_ attribute"
            logger.info(f"Classifier classes: {self.classifier.classes_}")
            # Expected: [0, 1] where 0=No Train, 1=Train Present
            # predict_proba(X)[0][1] returns P(train_blocked)

            logger.info("Models loaded successfully")
        except FileNotFoundError:
            logger.error("Models not found. Run train_model.py first!")
            raise

    # ------------------------------------------------------------------ #
    #  Amtrak data fetching                                               #
    # ------------------------------------------------------------------ #

    def fetch_amtrak_trains(self):
        """
        Fetch Amtrak train positions with health check and caching.
        Returns cached data if API fails and cache is still valid (<5 min old).
        """
        logger.info("Fetching Amtrak data...")

        # Check if we have valid cached data
        cache_available = False
        if AMTRAK_CACHE['data'] is not None and AMTRAK_CACHE['timestamp'] is not None:
            cache_age_minutes = (datetime.now() - AMTRAK_CACHE['timestamp']).total_seconds() / 60
            if cache_age_minutes < AMTRAK_CACHE['cache_duration_minutes']:
                cache_available = True
                logger.debug(f"Cache available ({cache_age_minutes:.1f} min old)")

        # Health check
        is_healthy, error_msg = check_amtrak_api_health()

        if not is_healthy:
            logger.warning(f"Amtrak API health check failed: {error_msg}")
            logger.warning(f"Consecutive failures: {API_HEALTH['consecutive_failures']}")

            # Use cached data if available
            if cache_available:
                cache_age = (datetime.now() - AMTRAK_CACHE['timestamp']).total_seconds() / 60
                logger.info(f"Using cached data from {cache_age:.1f} minutes ago")
                return AMTRAK_CACHE['data']
            else:
                logger.warning("No valid cache, falling back to ML-only predictions")
                return []

        # API is healthy, fetch fresh data with retry
        response, error_msg = _request_with_retry(
            'https://api-v3.amtraker.com/v3/trains',
            timeout=10,
            max_retries=MAX_RETRIES
        )

        if response is not None and response.status_code == 200:
            try:
                data = response.json()

                # Flatten into list of all trains
                trains = []
                for train_num, train_list in data.items():
                    trains.extend(train_list)

                # Update cache
                AMTRAK_CACHE['data'] = trains
                AMTRAK_CACHE['timestamp'] = datetime.now()

                logger.info(f"Fetched {len(trains)} active Amtrak trains (cache updated)")
                return trains
            except Exception as e:
                logger.error(f"Error parsing Amtrak response: {e}")
        else:
            logger.warning(f"Amtrak API fetch failed: {error_msg}")

        # Fall back to cache
        if cache_available:
            cache_age = (datetime.now() - AMTRAK_CACHE['timestamp']).total_seconds() / 60
            logger.info(f"Using cached data from {cache_age:.1f} minutes ago")
            return AMTRAK_CACHE['data']

        logger.warning("No valid cache, falling back to ML-only predictions")
        return []

    # ------------------------------------------------------------------ #
    #  Amtrak proximity check (with directional, station & yard logic)    #
    # ------------------------------------------------------------------ #

    def check_amtrak_proximity(self, crossing, trains):
        """
        Check if any Amtrak train is near this crossing.

        Applies the following data-integrity checks before returning a result:
          0. GPS staleness filter   - trains with lastValTS > 15 min are skipped.
          1. Directional awareness  - trains heading *away* are skipped.
          2. Station safety interlock - trains at a station are downgraded to
             NEARBY_STATION so the app does not falsely show APPROACHING or
             BLOCKING while the train is still dwelling at the platform.
          3. 0-MPH / yard sanity check - a stopped train that is NOT at a
             station is classified as SWITCHING (near yard) or STALLED.

        Tier system (10-mile detection radius):
          PREDEPARTURE  - trainState == 'Predeparture', within 5 mi
          BLOCKING      - distance < 0.15 mi
          APPROACHING   - 0.15 to 2.0 mi, moving toward, speed > 1 mph
          INCOMING      - 2.0 to 10.0 mi, moving toward, ETA <= 45 min
          (skip)        - beyond 10.0 mi

        Returns: (is_near, train_info) or (False, None)
        """
        for train in trains:
            try:
                train_lat = train.get('lat')
                train_lon = train.get('lon')

                if train_lat is None or train_lon is None:
                    continue

                distance_km = _haversine_km(train_lat, train_lon, crossing['lat'], crossing['lon'])
                distance_miles = distance_km * 0.621371

                speed = train.get('velocity', 0)
                heading = train.get('heading', 'UNKNOWN')
                train_num = train.get('trainNum', 'UNKNOWN')
                route_name = train.get('routeName', 'Unknown Route')

                # Staleness filter: skip trains with GPS data older than 15 min
                last_val_ts = train.get('lastValTS')
                if last_val_ts:
                    try:
                        from datetime import timezone
                        last_val_dt = datetime.fromisoformat(last_val_ts.replace('Z', '+00:00'))
                        age_minutes = (datetime.now(timezone.utc) - last_val_dt).total_seconds() / 60
                        if age_minutes > 15:
                            logger.info(f"   Train #{train_num}: GPS data is {age_minutes:.0f} min stale — skipping")
                            continue
                    except (ValueError, TypeError):
                        pass  # If we can't parse, don't skip

                # ---------------------------------------------------------
                # (1) Directional Awareness
                # For trains beyond the blocking threshold but within the
                # detection radius, skip if heading away from the crossing.
                # Trains within the blocking threshold are always counted
                # regardless of heading because they are physically blocking.
                # ---------------------------------------------------------
                is_predeparture = train.get('trainState') == 'Predeparture'
                if AMTRAK_BLOCKING_THRESHOLD_MILES <= distance_miles < AMTRAK_DETECTION_RADIUS_MILES:
                    # Skip directional check for predeparture trains (heading is not meaningful)
                    if not is_predeparture and not is_moving_toward(train_lat, train_lon, heading,
                                            crossing['lat'], crossing['lon']):
                        logger.info(
                            f"   Train #{train_num} is {distance_miles:.2f} mi away "
                            f"but heading {heading} (AWAY from crossing) -- skipping"
                        )
                        continue

                # ---------------------------------------------------------
                # (1b) Distance-trend check (2 consecutive polls)
                # Even if heading looks ambiguous, if the distance to the
                # crossing has been *increasing* for 2 consecutive polls
                # the train is moving away → skip.
                # ---------------------------------------------------------
                history_key = (crossing.get('fra_id', crossing.get('name', '')), train_num)
                prev_distances = self._distance_history.get(history_key, [])
                prev_distances.append(distance_miles)
                # Keep only the last 3 samples
                if len(prev_distances) > 3:
                    prev_distances = prev_distances[-3:]
                self._distance_history[history_key] = prev_distances

                if (not is_predeparture
                        and len(prev_distances) >= 3
                        and prev_distances[-1] > prev_distances[-2] > prev_distances[-3]
                        and distance_miles >= AMTRAK_BLOCKING_THRESHOLD_MILES):
                    logger.info(
                        f"   Train #{train_num} distance increasing over "
                        f"last 3 polls ({prev_distances}) -- moving away, skipping"
                    )
                    continue

                # ---------------------------------------------------------
                # (2) Station Safety Interlock
                # Check the Amtrak 'stations' array for a status of
                # "Station" (train physically at a platform).
                # ---------------------------------------------------------
                is_at_station, station_code, station_name = get_current_station_status(train)

                # ---------------------------------------------------------
                # (3) 0-MPH / Yard Sanity Check
                # Treat any speed < 1 mph as effectively stopped.  The
                # Amtrak API (or ML model) may report tiny velocities like
                # 0.3 mph which are functionally zero.
                # ---------------------------------------------------------
                effectively_stopped = (speed is not None and speed < 1)

                # Base crossing time for the physical blockage
                crossing_time = calculate_amtrak_duration(speed if speed > 0 else 40)

                # ---- Tiered status determination ----

                # Union Station codes that overlap some crossings --
                # these are the ONLY station where BLOCKING is still
                # possible even when the train reports "at station".
                UNION_STATION_CODES = {'PDX', 'PDL', 'PTLD'}

                # --- TIER 0: PREDEPARTURE ---
                if is_predeparture and distance_miles < PREDEPARTURE_RADIUS_MILES:
                    status = 'PREDEPARTURE'
                    confidence = 'MEDIUM'
                    duration = 15  # conservative estimate for departure + transit time
                    explanation = (
                        f"Amtrak #{train_num} ({route_name}) preparing to depart. "
                        f"Crossing may be affected in ~15 min."
                    )
                    logger.info(
                        f"   Predeparture: #{train_num} ({route_name}) "
                        f"{distance_miles:.2f} mi away, trainState=Predeparture"
                    )

                # --- TIER 1: BLOCKING (distance < 0.15 miles) ---
                elif distance_miles < AMTRAK_BLOCKING_THRESHOLD_MILES:
                    if is_at_station and station_code not in UNION_STATION_CODES:
                        status = 'NEARBY_STATION'
                        confidence = 'MEDIUM'
                        station_label = station_name or station_code or 'a station'
                        duration = crossing_time + 2
                        explanation = (
                            f"Amtrak #{train_num} at {station_label} "
                            f"— not blocking road crossing"
                        )
                        logger.info(
                            f"   Station interlock: #{train_num} at "
                            f"{station_label} (<{AMTRAK_BLOCKING_THRESHOLD_MILES} mi) -- NOT blocking"
                        )
                    else:
                        status = 'BLOCKING'
                        confidence = 'HIGH'
                        duration = crossing_time
                        explanation = (
                            f"{route_name} #{train_num} is at the crossing now"
                        )

                # --- TIER 2: APPROACHING (0.15 to 2.0 miles, moving toward, speed > 1mph) ---
                elif distance_miles < AMTRAK_APPROACHING_THRESHOLD_MILES:
                    if is_at_station:
                        status = 'NEARBY_STATION'
                        confidence = 'MEDIUM'
                        station_label = station_name or station_code or 'a station'
                        eta = (distance_miles / 30 * 60) + 2  # assume 30mph departure
                        duration = round(eta + crossing_time, 1)
                        explanation = (
                            f"Amtrak #{train_num} at {station_label}, awaiting departure. "
                            f"May affect crossing in ~{eta:.0f} min"
                        )
                        logger.info(
                            f"   Station interlock: #{train_num} at {station_label} "
                            f"({distance_miles:.2f} mi) -- NEARBY_STATION"
                        )
                    elif effectively_stopped:
                        # Stopped but NOT at a station -> switching or stalled
                        dist_to_yard_km = _haversine_km(
                            train_lat, train_lon,
                            BROOKLYN_YARD_LAT, BROOKLYN_YARD_LON,
                        )
                        if dist_to_yard_km < YARD_SWITCHING_RADIUS_KM:
                            status = 'SWITCHING'
                            confidence = 'LOW'
                            duration = YARD_SWITCHING_DELAY_MINUTES
                            explanation = (
                                f"Train #{train_num} stopped near Brooklyn Yard "
                                f"({dist_to_yard_km:.1f} km) — likely switching "
                                f"(~{YARD_SWITCHING_DELAY_MINUTES} min)"
                            )
                            logger.info(
                                f"   Switching: #{train_num} stopped "
                                f"{dist_to_yard_km:.1f} km from yard, applying "
                                f"{YARD_SWITCHING_DELAY_MINUTES}-min estimate"
                            )
                        else:
                            status = 'STALLED'
                            confidence = 'LOW'
                            duration = STALLED_TRAIN_DELAY_MINUTES
                            explanation = (
                                f"Train #{train_num} stalled "
                                f"{distance_miles:.2f} mi from crossing "
                                f"(~{STALLED_TRAIN_DELAY_MINUTES} min estimate)"
                            )
                            logger.info(
                                f"   Stalled: #{train_num} stopped "
                                f"{dist_to_yard_km:.1f} km from yard, applying "
                                f"{STALLED_TRAIN_DELAY_MINUTES}-min estimate"
                            )
                    else:
                        status = 'APPROACHING'
                        confidence = 'HIGH'
                        eta = (distance_miles / speed * 60) if speed > 0 else 5
                        duration = round(eta + crossing_time, 1)
                        explanation = (
                            f"{route_name} #{train_num} approaching at {speed} mph "
                            f"— ETA ~{eta:.0f} min ({distance_miles:.1f} mi away)"
                        )

                # --- TIER 3: INCOMING (2.0 to 10.0 miles) ---
                elif distance_miles < AMTRAK_DETECTION_RADIUS_MILES:
                    # Skip stopped trains 2-10 miles away — not actionable
                    if effectively_stopped:
                        logger.info(
                            f"   Train #{train_num} stopped {distance_miles:.1f} mi away -- too far to be actionable, skipping"
                        )
                        continue

                    eta = (distance_miles / speed * 60) if speed > 0 else None
                    if eta is None or eta > AMTRAK_MAX_ETA_TO_SURFACE_MINUTES:
                        logger.info(
                            f"   Train #{train_num} {distance_miles:.1f} mi away, "
                            f"ETA {'N/A' if eta is None else f'{eta:.0f} min'} -- too uncertain, skipping"
                        )
                        continue

                    if eta <= 15:
                        confidence = 'MEDIUM'
                    else:
                        confidence = 'LOW'

                    status = 'INCOMING'
                    duration = round(eta + crossing_time, 1)
                    explanation = (
                        f"{route_name} #{train_num} detected {distance_miles:.0f} mi away "
                        f"at {speed} mph — ETA ~{eta:.0f} min"
                    )

                # --- TIER 4: DEFAULT (beyond detection radius) ---
                else:
                    continue  # Train is too far away

                now = datetime.now()
                return True, {
                    'source': 'AMTRAK_API',
                    'data_source': 'AMTRAK_API',
                    'train_number': train_num,
                    'route_name': route_name,
                    'distance_miles': round(distance_miles, 2),
                    'speed_mph': speed,
                    'heading': heading,
                    'status': status,
                    'confidence': confidence,
                    'duration_minutes': round(duration, 1),
                    'predicted_clear_time': (
                        now + timedelta(minutes=duration)
                    ).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'train_type': 'passenger',
                    'explanation': explanation,
                }

            except Exception as e:
                logger.warning(f"Error processing train: {e}", exc_info=True)
                continue

        return False, None

    # ------------------------------------------------------------------ #
    #  Propagation to connected crossings                                 #
    # ------------------------------------------------------------------ #

    def propagate_train_detection(self, detected_crossing, train_info):
        """
        Propagate train detection to connected crossings.

        When a train is detected at one crossing, predict when it will
        reach connected crossings based on distance and speed.
        """
        propagated_predictions = {}

        if not hasattr(self, 'crossing_chains') or not self.crossing_chains:
            return propagated_predictions

        # Find which chain this crossing belongs to
        for chain_name, chain_info in self.crossing_chains.items():
            if detected_crossing not in chain_info['crossings']:
                continue

            # Find position in chain
            crossing_index = chain_info['crossings'].index(detected_crossing)

            # Get train speed (default 30 mph if unknown)
            train_speed = train_info.get('speed_mph', 30)
            if train_speed < 5:
                train_speed = 15  # Assume slow-moving if stopped

            # Propagate to downstream crossings
            cumulative_distance = 0
            cumulative_time = 0

            for i in range(crossing_index + 1, len(chain_info['crossings'])):
                downstream_crossing = chain_info['crossings'][i]
                previous_crossing = chain_info['crossings'][i - 1]

                # Get distance to next crossing
                distance = chain_info['distances'].get(previous_crossing, 0)
                cumulative_distance += distance

                # Calculate ETA (distance / speed * 60 = minutes)
                eta_minutes = (cumulative_distance / train_speed) * 60
                cumulative_time = eta_minutes

                # Determine confidence based on distance
                if cumulative_distance < 0.2:  # Within 0.2 miles
                    confidence = 'HIGH'
                elif cumulative_distance < 0.5:
                    confidence = 'MEDIUM'
                else:
                    confidence = 'LOW'

                now = datetime.now()
                prop_duration = round(eta_minutes + train_info.get('duration_minutes', 2.5), 1)

                route = train_info.get('route_name', 'Amtrak')
                t_num = train_info.get('train_number', '?')

                propagated_predictions[downstream_crossing] = {
                    'source': 'PROPAGATED',
                    'data_source': 'AMTRAK_API',
                    'from_crossing': detected_crossing,
                    'confidence': confidence,
                    'eta_minutes': round(eta_minutes, 1),
                    'distance_miles': round(cumulative_distance, 2),
                    'duration_minutes': prop_duration,
                    'predicted_clear_time': (
                        now + timedelta(minutes=prop_duration)
                    ).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'train_info': train_info,
                    'train_type': 'passenger',
                    'status': 'APPROACHING',
                    'explanation': (
                        f"{route} #{t_num} headed this way from "
                        f"{detected_crossing} -- ETA ~{eta_minutes:.0f} min"
                    ),
                }

        return propagated_predictions

    # ------------------------------------------------------------------ #
    #  ML predictions                                                     #
    # ------------------------------------------------------------------ #

    def generate_ml_prediction(self, crossing_id, timestamp=None):
        """
        Generate ML-based prediction for a crossing.
        """
        if timestamp is None:
            timestamp = datetime.now(ZoneInfo('America/Los_Angeles'))

        # Extract features
        features = self.feature_engine.extract_all_features(
            timestamp=timestamp,
            crossing_id=crossing_id,
            crossings_config=self.crossings_config,
            train_speed=None,
            historical_data=None
        )

        # Encode categorical features
        crossing_encoded = self.label_encoders['crossing_id'].transform([crossing_id])[0]
        train_type_encoded = self.label_encoders['train_type'].transform(['none'])[0]
        speed_encoded = self.label_encoders['train_speed_category'].transform(['unknown'])[0]

        # Load model metadata to get exact feature list
        if not hasattr(self, 'required_features'):
            with open('python/ml/models/model_metadata.json', 'r') as f:
                metadata = json.load(f)
                self.required_features = metadata['feature_columns']

        # Build feature dict using ONLY features the model expects
        feature_dict = {}

        for feat in self.required_features:
            if feat == 'crossing_id_encoded':
                feature_dict[feat] = crossing_encoded
            elif feat == 'train_type_encoded':
                feature_dict[feat] = train_type_encoded
            elif feat == 'train_speed_category_encoded':
                feature_dict[feat] = speed_encoded
            elif feat in features:
                feature_dict[feat] = features[feat]
            else:
                feature_dict[feat] = 0  # Default for any missing features

        # Convert to DataFrame with correct column order
        feature_df = pd.DataFrame([feature_dict])
        feature_df = feature_df[self.required_features]

        # Predict
        # P(train_blocked) — class index 1 (classes_ = [0=No Train, 1=Train Present])
        probability = self.classifier.predict_proba(feature_df)[0][1]
        prediction = self.classifier.predict(feature_df)[0]

        if prediction == 1:
            duration = self.duration_regressor.predict(feature_df)[0]

            # Treat ML-predicted duration < 1 as effectively minimal
            # (the model may output very small values for uncertain predictions)
            if duration < 1:
                logger.debug(
                    f"ML predicted duration {duration:.2f} min (< 1); "
                    f"treating as minimal blockage"
                )

            # Determine base confidence from probability
            if probability > 0.8:
                confidence = 'HIGH'
            elif probability > 0.6:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'

            train_type = 'freight'  # ML only runs for freight; Amtrak handles passenger
            train_label = 'freight train'
            explanation_notes = []

            # Brooklyn Yard proximity adjustment
            is_yard_zone = bool(features.get('is_within_yard_zone'))
            train_stopped = (features.get('current_train_speed') or 0) < 1
            if is_yard_zone:
                if duration < 30:
                    duration *= 2  # switching operations take longer
                    explanation_notes.append(
                        "extended estimate (Brooklyn Yard switching zone)"
                    )
                if train_stopped:
                    confidence = 'LOW'  # only unpredictable when actually stopped
            elif duration > 45:
                confidence = 'MEDIUM'  # long delays are harder to predict

            # Rain adjustment: increase duration by 15% when raining
            if features.get('is_raining'):
                duration *= 1.15
                explanation_notes.append("rain detected (expect delays)")

            duration = round(duration, 1)
            now = datetime.now(ZoneInfo('America/Los_Angeles'))

            # Build explanation with final adjusted duration
            clear_time = now + timedelta(minutes=duration)
            clear_time_str = clear_time.strftime('%I:%M %p').lstrip('0')
            explanation_parts = [
                f"{probability:.0%} chance of {train_label}; "
                f"estimated ~{duration:.0f} min blockage "
                f"(clear by ~{clear_time_str})"
            ]
            explanation_parts.extend(explanation_notes)

            # ML status based on probability — no directional data available
            if probability > 0.8:
                ml_status = 'BLOCKING'
            elif probability > 0.6:
                ml_status = 'APPROACHING'
            else:
                ml_status = 'DETECTED'  # low-confidence freight signal

            return {
                'source': 'ML_MODEL',
                'data_source': 'ML_MODEL',
                'status': ml_status,
                'probability': round(probability, 2),
                'confidence': confidence,
                'duration_minutes': duration,
                'predicted_clear_time': clear_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'train_type': train_type,
                'train_number': None,
                'route_name': None,
                'distance_miles': None,
                'speed_mph': None,
                'heading': None,
                'is_freight_peak': features['is_freight_peak'],
                'is_passenger_peak': features.get('is_passenger_peak', 0),
                'is_within_yard_zone': is_yard_zone,
                'explanation': '; '.join(explanation_parts),
            }
        else:
            return None

    # ------------------------------------------------------------------ #
    #  Change detection (redundant update suppression)                    #
    # ------------------------------------------------------------------ #

    def _load_previous_predictions(self, output_file):
        """
        Load the most recent predictions from disk so we can compare against
        them for change-detection.  Returns a dict keyed by crossing_id, or
        an empty dict if the file is missing / invalid.
        """
        try:
            with open(output_file, 'r') as f:
                prev = json.load(f)
                logger.debug(f"Loaded previous predictions from {output_file}")
                return prev
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"No usable previous predictions ({e})")
            return {}

    def _is_redundant_update(self, crossing_id, new_pred, previous_predictions):
        """
        Check whether the new prediction is materially identical to the
        previous one for this crossing.

        A redundant update means the status and duration haven't meaningfully
        changed, so we can carry forward the old prediction with only the
        timestamp refreshed.  This avoids unnecessary downstream work (e.g.
        push notifications, Firebase writes).

        Returns True if the update can be suppressed.
        """
        prev = previous_predictions.get(crossing_id)
        if prev is None:
            return False  # No previous state -- always write

        prev_status = prev.get('status')
        new_status = new_pred.get('status')
        prev_duration = prev.get('duration_minutes')
        new_duration = new_pred.get('duration_minutes')

        # CLEAR -> CLEAR: suppress everything except a timestamp refresh
        if prev_status == 'CLEAR' and new_status == 'CLEAR':
            logger.debug(f"   {crossing_id}: CLEAR->CLEAR (timestamp-only update)")
            return True

        # Same status AND same duration (within 0.5 min tolerance) -> suppress
        if (prev_status == new_status
                and prev_duration is not None
                and new_duration is not None):
            if abs(prev_duration - new_duration) < 0.5:
                logger.debug(
                    f"   {crossing_id}: {prev_status}->{new_status}, "
                    f"duration delta < 0.5 min -- suppressing redundant update"
                )
                return True

        return False

    # ------------------------------------------------------------------ #
    #  Last Known Good preservation (Firestore-backed)                    #
    # ------------------------------------------------------------------ #

    def _get_firestore_client(self):
        """Lazy-init Firebase client (reuses upload_to_firebase.init_firebase)."""
        if not hasattr(self, '_firestore_db') or self._firestore_db is None:
            try:
                from upload_to_firebase import init_firebase
                self._firestore_db = init_firebase()
            except Exception as e:
                logger.warning(f"Could not init Firebase for LKG: {e}")
                self._firestore_db = None
        return self._firestore_db

    def _save_last_known_good(self, predictions):
        """Save predictions to Firestore _last_known_good collection."""
        db = self._get_firestore_client()
        if db is None:
            return
        try:
            now_iso = datetime.now().isoformat()
            for cid, pred in predictions.items():
                doc = dict(pred)
                doc['lkg_saved_at'] = now_iso
                db.collection('_last_known_good').document(cid).set(doc)
            logger.info(f"Saved last known good for {len(predictions)} crossings")
        except Exception as e:
            logger.warning(f"Failed to save last known good: {e}")

    def _load_last_known_good(self):
        """
        Load last known good predictions from Firestore.
        Applies 30-minute TTL to prevent ghost trains during long outages.
        """
        db = self._get_firestore_client()
        if db is None:
            return {}
        try:
            preds = {}
            now = datetime.now()
            docs = db.collection('_last_known_good').stream()
            for doc in docs:
                data = doc.to_dict()
                # Enforce TTL: skip entries older than 30 minutes
                saved_at_str = data.get('lkg_saved_at')
                if saved_at_str:
                    try:
                        saved_at = datetime.fromisoformat(saved_at_str)
                        age_minutes = (now - saved_at).total_seconds() / 60
                        if age_minutes > LAST_KNOWN_GOOD_TTL_MINUTES:
                            logger.debug(f"   LKG for {doc.id} expired ({age_minutes:.0f} min old)")
                            continue
                    except (ValueError, TypeError):
                        pass  # If we can't parse, include it anyway
                preds[doc.id] = data
            logger.info(f"Loaded {len(preds)} last known good predictions (TTL {LAST_KNOWN_GOOD_TTL_MINUTES} min)")
            return preds
        except Exception as e:
            logger.warning(f"Failed to load last known good: {e}")
            return {}

    # ------------------------------------------------------------------ #
    #  Hysteresis / stability buffer (Firestore-backed)                  #
    # ------------------------------------------------------------------ #

    REQUIRED_CONSECUTIVE_POLLS = 2
    STABILITY_WINDOW_SECONDS = 30

    def _load_hysteresis_state(self):
        """Load hysteresis state from Firestore (persists across ephemeral runs)."""
        db = self._get_firestore_client()
        if db is None:
            return {}
        try:
            state = {}
            docs = db.collection('_hysteresis').stream()
            for doc in docs:
                state[doc.id] = doc.to_dict()
            logger.debug(f"Loaded hysteresis state for {len(state)} crossings")
            return state
        except Exception as e:
            logger.warning(f"Could not load hysteresis state: {e}")
            return {}

    def _save_hysteresis_state(self, state):
        """Persist hysteresis state to Firestore."""
        db = self._get_firestore_client()
        if db is None:
            return
        try:
            for cid, entry in state.items():
                db.collection('_hysteresis').document(cid).set(entry)
        except Exception as e:
            logger.warning(f"Failed to save hysteresis state: {e}")

    def _apply_hysteresis(self, crossing_id, new_status, hysteresis_state):
        """
        Apply stability buffer to prevent status flickering.

        A crossing only transitions to a new status if the same new status
        has been observed for REQUIRED_CONSECUTIVE_POLLS consecutive polls.
        This prevents GPS noise from briefly flipping BLOCKING->CLEAR->BLOCKING.

        Returns the stabilized status (may differ from new_status).
        """
        now_iso = datetime.now().isoformat()

        entry = hysteresis_state.get(crossing_id, {
            'confirmed_status': None,
            'pending_status': None,
            'pending_count': 0,
            'pending_since': None,
        })

        confirmed = entry.get('confirmed_status')

        if confirmed is None:
            # First time seeing this crossing -- accept immediately
            hysteresis_state[crossing_id] = {
                'confirmed_status': new_status,
                'pending_status': None,
                'pending_count': 0,
                'pending_since': None,
            }
            return new_status

        if new_status == confirmed:
            # Status unchanged -- reset any pending transition
            entry['pending_status'] = None
            entry['pending_count'] = 0
            entry['pending_since'] = None
            hysteresis_state[crossing_id] = entry
            return confirmed

        # Status differs from confirmed -- track pending transition
        if new_status == entry.get('pending_status'):
            entry['pending_count'] = entry.get('pending_count', 0) + 1
        else:
            # Different pending status -- restart counter
            entry['pending_status'] = new_status
            entry['pending_count'] = 1
            entry['pending_since'] = now_iso

        # Check if we've reached the threshold
        if entry['pending_count'] >= self.REQUIRED_CONSECUTIVE_POLLS:
            logger.info(
                f"   {crossing_id}: Hysteresis transition "
                f"{confirmed} -> {new_status} "
                f"(confirmed after {entry['pending_count']} polls)"
            )
            entry['confirmed_status'] = new_status
            entry['pending_status'] = None
            entry['pending_count'] = 0
            entry['pending_since'] = None
            hysteresis_state[crossing_id] = entry
            return new_status

        # Not yet confirmed -- hold the previous status
        logger.info(
            f"   {crossing_id}: Hysteresis holding at {confirmed} "
            f"(pending {new_status}, count={entry['pending_count']}/{self.REQUIRED_CONSECUTIVE_POLLS})"
        )
        hysteresis_state[crossing_id] = entry
        return confirmed

    # ------------------------------------------------------------------ #
    #  Main orchestration                                                 #
    # ------------------------------------------------------------------ #

    def generate_predictions_for_all_crossings(self):
        """
        Generate predictions for all crossings.

        Priority:
        1. Check Amtrak API (passenger trains)
        2. Propagate to connected crossings
        3. Run ML model (freight predictions)

        Change-detection is applied at every step so that identical
        predictions are carried forward with only a refreshed timestamp.
        Hysteresis buffer prevents single-poll status flickers.
        """
        print("\n" + "="*60)
        print("[*]GENERATING PREDICTIONS")
        print("="*60)

        # Load previous predictions for change-detection
        output_file = 'python/ml/data/current_predictions.json'
        previous_predictions = self._load_previous_predictions(output_file)

        # Load hysteresis state (persisted in Firestore across runs)
        hysteresis_state = self._load_hysteresis_state()

        # Fetch Amtrak data
        amtrak_trains = self.fetch_amtrak_trains()

        # Get current timestamp in Portland local time
        now = datetime.now(ZoneInfo('America/Los_Angeles'))

        predictions = {}
        detections = {}  # Store Amtrak detections for propagation

        # ------- First pass: Detect Amtrak trains -------
        for crossing_id, crossing in self.crossings_config.items():
            print(f"\n[>]{crossing['name']} ({crossing_id})")

            # Check Amtrak proximity (Tier 1)
            is_amtrak_near, amtrak_info = self.check_amtrak_proximity(crossing, amtrak_trains)

            if is_amtrak_near:
                print(f"   [TRAIN]Amtrak detected: {amtrak_info['status']}")
                new_pred = {
                    'crossing_id': crossing_id,
                    'crossing_name': crossing['name'],
                    'timestamp': now.isoformat(),
                    **amtrak_info
                }

                # Hysteresis: stabilize status before storing
                raw_status = new_pred.get('status', 'CLEAR')
                stable_status = self._apply_hysteresis(crossing_id, raw_status, hysteresis_state)
                if stable_status != raw_status:
                    logger.info(f"{crossing_id}: hysteresis held at {stable_status}, raw was {raw_status}")
                    new_pred['status'] = stable_status

                # CLEAR status must never carry duration or predicted_clear_time
                if new_pred['status'] == 'CLEAR':
                    new_pred['duration_minutes'] = None
                    new_pred['predicted_clear_time'] = None

                # Change-detection: skip redundant updates
                if self._is_redundant_update(crossing_id, new_pred, previous_predictions):
                    prev = previous_predictions[crossing_id]
                    prev['timestamp'] = now.isoformat()
                    predictions[crossing_id] = prev
                    logger.info(f"   {crossing_id}: redundant update suppressed (timestamp refreshed)")
                else:
                    predictions[crossing_id] = new_pred

                detections[crossing_id] = amtrak_info

        # ------- Second pass: Propagate detections to connected crossings -------
        for detected_crossing, train_info in detections.items():
            propagated = self.propagate_train_detection(detected_crossing, train_info)

            for downstream_crossing, prop_info in propagated.items():
                # Only propagate if we don't already have a detection
                if downstream_crossing not in predictions:
                    print(f"\n   [>>]Propagating to {self.crossings_config[downstream_crossing]['name']}")
                    print(f"      ETA: {prop_info['eta_minutes']:.1f} min")

                    new_pred = {
                        'crossing_id': downstream_crossing,
                        'crossing_name': self.crossings_config[downstream_crossing]['name'],
                        'timestamp': now.isoformat(),
                        **prop_info
                    }

                    # Hysteresis: stabilize propagated status
                    raw_status = new_pred.get('status', 'CLEAR')
                    stable_status = self._apply_hysteresis(downstream_crossing, raw_status, hysteresis_state)
                    if stable_status != raw_status:
                        logger.info(f"{downstream_crossing}: hysteresis held at {stable_status}, raw was {raw_status}")
                        new_pred['status'] = stable_status

                    # CLEAR status must never carry duration or predicted_clear_time
                    if new_pred['status'] == 'CLEAR':
                        new_pred['duration_minutes'] = None
                        new_pred['predicted_clear_time'] = None

                    if self._is_redundant_update(downstream_crossing, new_pred, previous_predictions):
                        prev = previous_predictions[downstream_crossing]
                        prev['timestamp'] = now.isoformat()
                        predictions[downstream_crossing] = prev
                    else:
                        predictions[downstream_crossing] = new_pred

        # ------- Third pass: ML predictions for crossings without detections -------
        for crossing_id, crossing in self.crossings_config.items():
            if crossing_id in predictions:
                continue  # Skip if already have Amtrak or propagated prediction

            print(f"\n[>]{crossing['name']} ({crossing_id})")

            # Run ML model (Tier 2)
            ml_prediction = self.generate_ml_prediction(crossing_id, now)

            if ml_prediction and ml_prediction['probability'] > 0.5:
                prob = ml_prediction['probability']
                # Status already set by generate_ml_prediction():
                #   >0.8 = BLOCKING, >0.6 = APPROACHING, 0.5-0.6 = DETECTED

                print(f"   [ML]ML prediction: {prob:.0%} probability")
                new_pred = {
                    'crossing_id': crossing_id,
                    'crossing_name': crossing['name'],
                    'timestamp': now.isoformat(),
                    **ml_prediction
                }

                # Hysteresis: stabilize ML-predicted status
                raw_status = new_pred.get('status', 'CLEAR')
                stable_status = self._apply_hysteresis(crossing_id, raw_status, hysteresis_state)
                if stable_status != raw_status:
                    logger.info(f"{crossing_id}: hysteresis held at {stable_status}, raw was {raw_status}")
                    new_pred['status'] = stable_status

                # CLEAR status must never carry duration or predicted_clear_time
                if new_pred['status'] == 'CLEAR':
                    new_pred['duration_minutes'] = None
                    new_pred['predicted_clear_time'] = None

                if self._is_redundant_update(crossing_id, new_pred, previous_predictions):
                    prev = previous_predictions[crossing_id]
                    prev['timestamp'] = now.isoformat()
                    predictions[crossing_id] = prev
                else:
                    predictions[crossing_id] = new_pred
            else:
                prob = ml_prediction['probability'] if ml_prediction else 0.0
                new_pred = {
                    'crossing_id': crossing_id,
                    'crossing_name': crossing['name'],
                    'timestamp': now.isoformat(),
                    'source': 'ML_MODEL',
                    'data_source': 'ML_MODEL',
                    'probability': prob,
                    'confidence': 'LOW',
                    'status': 'CLEAR',
                    'train_number': None,
                    'route_name': None,
                    'distance_miles': None,
                    'speed_mph': None,
                    'heading': None,
                    'train_type': None,
                    'predicted_clear_time': None,
                    'duration_minutes': None,
                    'explanation': (
                        'No train activity detected or predicted at this time'
                    ),
                }

                # Hysteresis: stabilize CLEAR status (prevents BLOCKING->CLEAR flicker)
                raw_status = 'CLEAR'
                stable_status = self._apply_hysteresis(crossing_id, raw_status, hysteresis_state)
                if stable_status != raw_status:
                    logger.info(f"{crossing_id}: hysteresis held at {stable_status}, raw was {raw_status}")
                    new_pred['status'] = stable_status
                    new_pred['explanation'] = 'Train may still be present'

                # CLEAR status must never carry duration or predicted_clear_time
                if new_pred['status'] == 'CLEAR':
                    new_pred['duration_minutes'] = None
                    new_pred['predicted_clear_time'] = None

                # CLEAR -> CLEAR suppression
                if self._is_redundant_update(crossing_id, new_pred, previous_predictions):
                    prev = previous_predictions[crossing_id]
                    prev['timestamp'] = now.isoformat()
                    predictions[crossing_id] = prev
                    logger.info(f"   {crossing_id}: CLEAR->CLEAR (timestamp refreshed)")
                else:
                    print(f"   [OK]Clear (low probability)")
                    predictions[crossing_id] = new_pred

        # ------- Last Known Good preservation -------
        if API_HEALTH['consecutive_failures'] == 0:
            # API was healthy this run -- save as last known good
            self._save_last_known_good(predictions)
        elif API_HEALTH['consecutive_failures'] >= 2:
            # API is degraded -- restore LKG for crossings that went CLEAR
            # only because the API is down (prevents clearing valid blockages)
            lkg = self._load_last_known_good()
            for cid, lkg_pred in lkg.items():
                current = predictions.get(cid)
                if (current
                        and current.get('status') == 'CLEAR'
                        and current.get('data_source') == 'ML_MODEL'
                        and lkg_pred.get('status') not in ('CLEAR', None)):
                    lkg_pred['explanation'] = (
                        lkg_pred.get('explanation', '') +
                        ' (API unavailable; using last known state)'
                    )
                    lkg_pred['confidence'] = 'LOW'
                    lkg_pred['timestamp'] = now.isoformat()
                    predictions[cid] = lkg_pred
                    logger.info(f"   {cid}: Restored LKG ({lkg_pred['status']})")

        # Persist hysteresis state for next run
        self._save_hysteresis_state(hysteresis_state)

        # Store current state for in-memory comparison on future runs
        self._previous_predictions = predictions
        return predictions

    def save_predictions(self, predictions, output_file='python/ml/data/current_predictions.json'):
        """Save predictions to JSON file"""
        logger.info(f"Saving predictions to {output_file}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info("Predictions saved")


def main():
    """Main prediction generation entry point."""

    print("="*60)
    print("[TRAIN]TrainClear Prediction Generator")
    print("="*60)

    # Initialize generator
    generator = PredictionGenerator()

    # Generate predictions
    predictions = generator.generate_predictions_for_all_crossings()

    # Save to file
    generator.save_predictions(predictions)

    # Display summary
    print("\n" + "="*60)
    print("[*]PREDICTION SUMMARY")
    print("="*60)

    for crossing_id, pred in predictions.items():
        if pred.get('status') == 'CLEAR':
            print(f"[OK]{pred['crossing_name']}: CLEAR")
            explanation = pred.get('explanation', '')
            if explanation:
                print(f"   {explanation}")
        else:
            source = pred.get('data_source', pred.get('source', 'UNKNOWN'))
            duration = pred.get('duration_minutes', 0)
            confidence = pred.get('confidence', 'LOW')
            clear_time = pred.get('predicted_clear_time', '')
            explanation = pred.get('explanation', '')
            print(f"[TRAIN]{pred['crossing_name']}: {pred.get('status', 'UNKNOWN')} "
                  f"-- {duration:.1f} min ({source}, confidence: {confidence})")
            if clear_time:
                print(f"   Clear by: {clear_time}")
            if explanation:
                print(f"   {explanation}")

    # API health summary
    print("\n=== Amtrak API Health Summary ===")
    if API_HEALTH['last_success']:
        print(f"Last successful API call: {API_HEALTH['last_success'].strftime('%I:%M %p')}")
    if API_HEALTH['last_failure']:
        print(f"Last API failure: {API_HEALTH['last_failure'].strftime('%I:%M %p')}")
    print(f"Consecutive failures: {API_HEALTH['consecutive_failures']}")

    if API_HEALTH['consecutive_failures'] >= 3:
        print("WARNING: Multiple consecutive API failures detected!")
        print("Consider checking Amtraker status or switching to Railway.app hosting")

    print("\nPrediction generation complete!")
    print("Run this script every 10-30 minutes to keep predictions fresh")


if __name__ == '__main__':
    main()
