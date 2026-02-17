"""
TrainClear ML Feature Engineering
Implements all features from our ML model specification
"""

from datetime import datetime, date
import logging
import math
import time

# --------------------------------------------------------------------------- #
# Federal holiday calendar (US)                                                #
# Fixed-date and floating holidays through 2030.  For floating holidays       #
# (MLK Day, Presidents' Day, etc.) we pre-compute the exact dates per year.   #
# --------------------------------------------------------------------------- #

def _nth_weekday(year, month, weekday, n):
    """Return the n-th occurrence of *weekday* (0=Mon) in *month*."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return date(year, month, 1 + offset + 7 * (n - 1))


def _last_weekday(year, month, weekday):
    """Return the last occurrence of *weekday* in *month*."""
    d = date(year, month, 28)
    while True:
        try:
            nxt = d.replace(day=d.day + 7)
        except ValueError:
            break
        if nxt.month != month:
            break
        d = nxt
    # walk back to the desired weekday
    while d.weekday() != weekday:
        d = d.replace(day=d.day - 1)
    return d


def _federal_holidays(year):
    """Return a set of ``date`` objects for all US federal holidays in *year*."""
    holidays = {
        date(year, 1, 1),               # New Year's Day
        _nth_weekday(year, 1, 0, 3),    # MLK Day (3rd Monday in Jan)
        _nth_weekday(year, 2, 0, 3),    # Presidents' Day (3rd Mon in Feb)
        _last_weekday(year, 5, 0),      # Memorial Day (last Mon in May)
        date(year, 6, 19),              # Juneteenth
        date(year, 7, 4),               # Independence Day
        _nth_weekday(year, 9, 0, 1),    # Labor Day (1st Mon in Sep)
        _nth_weekday(year, 10, 0, 2),   # Columbus Day (2nd Mon in Oct)
        date(year, 11, 11),             # Veterans Day
        _nth_weekday(year, 11, 3, 4),   # Thanksgiving (4th Thu in Nov)
        date(year, 12, 25),             # Christmas Day
    }
    return holidays


# Pre-compute a cache so we don't rebuild every call
_HOLIDAY_CACHE = {}


def is_federal_holiday(dt):
    """Return True if *dt* (date or datetime) falls on a US federal holiday."""
    d = dt.date() if isinstance(dt, datetime) else dt
    yr = d.year
    if yr not in _HOLIDAY_CACHE:
        _HOLIDAY_CACHE[yr] = _federal_holidays(yr)
    return d in _HOLIDAY_CACHE[yr]

# --------------------------------------------------------------------------- #
# Weather signal (Open-Meteo API, free, no auth)                               #
# --------------------------------------------------------------------------- #

_weather_logger = logging.getLogger('TrainClear.weather')

PORTLAND_LAT = 45.5152
PORTLAND_LON = -122.6784

# In-memory cache with 60-minute TTL
_WEATHER_CACHE = {
    'data': None,
    'fetched_at': 0.0,
    'ttl_seconds': 3600,
}

# WMO weather codes indicating rain/drizzle/showers
_RAIN_WEATHER_CODES = set(range(51, 68)) | set(range(80, 83)) | {95, 96, 97, 98, 99}


def fetch_current_weather():
    """
    Fetch current weather for Portland from Open-Meteo (free, no API key).

    Returns dict with is_raining (0/1), precipitation_mm, temperature_c.
    Gracefully degrades on any error (returns defaults).
    Uses 60-minute in-memory cache and 3-second timeout.
    """
    now = time.time()

    # Return cached data if fresh
    if (_WEATHER_CACHE['data'] is not None
            and now - _WEATHER_CACHE['fetched_at'] < _WEATHER_CACHE['ttl_seconds']):
        return _WEATHER_CACHE['data']

    default = {'is_raining': 0, 'precipitation_mm': 0.0, 'temperature_c': 10.0}

    try:
        import requests
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={PORTLAND_LAT}&longitude={PORTLAND_LON}"
            f"&current=precipitation,rain,weather_code,temperature_2m"
            f"&timezone=America/Los_Angeles"
        )
        resp = requests.get(url, timeout=3)
        if resp.status_code != 200:
            _weather_logger.warning(f"Open-Meteo returned {resp.status_code}")
            return default

        current = resp.json().get('current', {})
        precip = current.get('precipitation', 0.0)
        rain = current.get('rain', 0.0)
        wmo_code = current.get('weather_code', 0)
        temp = current.get('temperature_2m', 10.0)

        is_raining = int(
            precip > 0.1
            or rain > 0.1
            or wmo_code in _RAIN_WEATHER_CODES
        )

        result = {
            'is_raining': is_raining,
            'precipitation_mm': round(precip, 2),
            'temperature_c': round(temp, 1),
        }

        _WEATHER_CACHE['data'] = result
        _WEATHER_CACHE['fetched_at'] = now
        _weather_logger.info(f"Weather fetched: raining={is_raining}, precip={precip}mm, temp={temp}C")
        return result

    except Exception as e:
        _weather_logger.warning(f"Weather fetch failed (using defaults): {e}")
        return default


class FeatureEngine:
    """Extract features for ML model training"""
    
    def __init__(self):
        # Brooklyn Yard location
        self.brooklyn_yard = {
            'lat': 45.4896,
            'lon': -122.6437,
            'mp': 768.1
        }
    
    def extract_temporal_features(self, timestamp):
        """
        Extract time-based features
        Returns dict of temporal features
        """
        dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        
        hour = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        
        # Rush hour detection
        is_morning_rush = (7 <= hour <= 9)
        is_evening_rush = (16 <= hour <= 18)
        is_rush_hour = is_morning_rush or is_evening_rush
        
        # 🆕 GEMINI: Freight vs passenger peak hours
        is_freight_peak = (20 <= hour or hour < 6)  # 8pm-6am
        is_passenger_peak = is_morning_rush or is_evening_rush
        
        # 🆕 GEMINI: Brooklyn Yard switching peaks
        is_peak_switching_hour = (
            (22 <= hour or hour < 4) or  # 10pm-4am (nocturnal)
            (13 <= hour <= 16)            # 1pm-4pm (afternoon surge)
        )
        
        # Weekday detection
        is_weekday = day_of_week < 5
        is_weekend = not is_weekday
        
        # Federal holiday detection
        is_holiday = is_federal_holiday(dt)
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekday': int(is_weekday),
            'is_weekend': int(is_weekend),
            'is_rush_hour': int(is_rush_hour),
            'is_morning_rush': int(is_morning_rush),
            'is_evening_rush': int(is_evening_rush),
            'is_holiday': int(is_holiday),
            'is_freight_peak': int(is_freight_peak),
            'is_passenger_peak': int(is_passenger_peak),
            'is_peak_switching_hour': int(is_peak_switching_hour),
            'week_of_year': dt.isocalendar()[1],
            'month': dt.month,
        }
    
    def extract_location_features(self, crossing_id, crossings_config):
        """
        Extract crossing-specific features
        """
        crossing = crossings_config.get(crossing_id)
        if not crossing:
            raise ValueError(f"Unknown crossing: {crossing_id}")
        
        # Calculate distance to Brooklyn Yard
        distance_km = self.calculate_distance(
            crossing['lat'], crossing['lon'],
            self.brooklyn_yard['lat'], self.brooklyn_yard['lon']
        )
        
        # Yard proximity features (0.5 km = actual yard throat only)
        is_within_yard_zone = distance_km < 0.5
        is_near_yard = distance_km < 1.5
        
        # FRA daily train counts (from Gemini data)
        train_counts = {
            '759733U': 24,  # SE 11th - 22-26 trains/day
            '759735H': 24,  # SE 12th - same corridor as 11th
            '759730Y': 4,   # SE Division & 7th - 2-4 trains/day
            '754552X': 12,  # SE Salmon
            '754553E': 12,  # SE Main
            '754558N': 12,  # SE Hawthorne
            '754554L': 12,  # SE Madison
            '754550J': 12,  # SE Yamhill
            '754551R': 12,  # SE Taylor
            '754543Y': 12,  # SE Washington & 12th - same corridor as Salmon/Taylor
            # REMOVED: 083310N (private/industrial spur)
            # NW Portland crossings
            '101880L': 16,  # NW 17th Ave - high volume NW corridor
            '810142C': 16,  # NW Naito Pkwy - waterfront corridor
            '838534K': 16,  # NW 9th Ave - NW industrial district
            '838536Y': 16,  # NW 15th Ave - NW corridor
            '808374S': 8,   # Van Houten Pl - less frequent, further out
        }

        daily_train_count = train_counts.get(crossing.get('fra_id'), 12)

        # Freight ratio (from Gemini: SE 12th = 62.5% freight)
        freight_ratios = {
            '759733U': 0.625,
            '759735H': 0.625,  # Same corridor as 11th
            # REMOVED: 083310N (private/industrial spur)
            # NW Portland crossings - predominantly freight
            '101880L': 0.75,   # NW 17th Ave - heavy freight corridor
            '810142C': 0.60,   # NW Naito Pkwy - mixed traffic
            '838534K': 0.70,   # NW 9th Ave - industrial/freight
            '838536Y': 0.75,   # NW 15th Ave - freight corridor
            '808374S': 0.80,   # Van Houten Pl - primarily freight
            # Others assume 50/50
        }
        
        freight_ratio = freight_ratios.get(crossing.get('fra_id'), 0.5)
        
        return {
            'crossing_id': crossing_id,
            'distance_to_brooklyn_yard_km': distance_km,
            'is_within_yard_zone': int(is_within_yard_zone),
            'is_near_yard': int(is_near_yard),
            'daily_train_count': daily_train_count,
            'freight_ratio': freight_ratio,
        }
    
    def extract_speed_features(self, speed_mph, distance_to_yard_km):
        """
        🆕 GEMINI: Speed-based train type inference
        """
        if speed_mph is None:
            return {
                'current_train_speed': None,
                'train_speed_category': 'unknown',
                'is_stopped': 0,
                'is_switching_speed': 0,
                'adjusted_speed_mph': None,
            }
        
        # Speed categories
        if speed_mph < 5:
            category = 'stopped'
            is_stopped = 1
            is_switching_speed = 1
        elif speed_mph < 15:
            category = 'slow'
            is_stopped = 0
            is_switching_speed = 1
        elif speed_mph < 30:
            category = 'medium'
            is_stopped = 0
            is_switching_speed = 0
        else:
            category = 'fast'
            is_stopped = 0
            is_switching_speed = 0
        
        # 🆕 GEMINI: Dynamic speed decay near yard
        if distance_to_yard_km < 2.0:
            speed_decay_factor = 1 - (distance_to_yard_km / 2.0) * 0.5
            adjusted_speed = speed_mph * speed_decay_factor
        else:
            adjusted_speed = speed_mph
        
        return {
            'current_train_speed': speed_mph,
            'train_speed_category': category,
            'is_stopped': is_stopped,
            'is_switching_speed': is_switching_speed,
            'adjusted_speed_mph': adjusted_speed,
        }
    
    def predict_train_type_from_speed(self, speed_mph):
        """
        🆕 GEMINI: Infer train type from speed
        """
        if speed_mph is None:
            return 'unknown', 20  # Default duration
        
        if speed_mph < 5:
            return 'switching', 60  # 45-75 min avg
        elif speed_mph < 25:
            return 'manifest', 55  # Slow freight
        elif speed_mph < 40:
            return 'intermodal', 18  # Fast freight
        else:
            return 'passenger', 2.5  # Amtrak
    
    def predict_duration(self, train_type, distance_to_yard_km):
        """
        🆕 GEMINI: Duration prediction with yard proximity
        """
        base_durations = {
            'passenger': 2.5,
            'intermodal': 18,
            'manifest': 55,
            'switching': 60,
            'unknown': 20,
        }
        
        duration = base_durations.get(train_type, 20)
        
        # 🆕 GEMINI: Yard overflow adjustment
        if train_type == 'manifest' and distance_to_yard_km < 2.0:
            # 75-85% chance of "doubling over"
            duration = 60  # Increase to max
        
        return duration
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance in kilometers using Haversine formula
        """
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat / 2)**2 + \
            math.cos(lat1_rad) * math.cos(lat2_rad) * \
            math.sin(delta_lon / 2)**2
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def extract_weather_features(self, timestamp):
        """
        Extract weather features for the given timestamp.

        Only fetches live weather if the timestamp is today (no retroactive
        weather data for historical dates).  Returns dict with is_raining,
        precipitation_mm, temperature_c.
        """
        dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
        today = date.today()

        if dt.date() != today:
            # Historical -- no weather data available
            return {'is_raining': 0, 'precipitation_mm': 0.0, 'temperature_c': 10.0}

        return fetch_current_weather()

    def extract_all_features(self, timestamp, crossing_id, crossings_config,
                            train_speed=None, historical_data=None):
        """
        Extract ALL features for a single prediction

        Returns dict with all features combined
        """
        features = {}

        # Temporal features
        features.update(self.extract_temporal_features(timestamp))

        # Weather features
        features.update(self.extract_weather_features(timestamp))
        
        # Location features
        location_features = self.extract_location_features(crossing_id, crossings_config)
        features.update(location_features)
        
        # Speed features
        distance_to_yard = location_features['distance_to_brooklyn_yard_km']
        features.update(self.extract_speed_features(train_speed, distance_to_yard))
        
        # Historical features (if available)
        if historical_data:
            features['blockages_last_hour'] = historical_data.get('blockages_last_hour', 0)
            features['blockages_last_24h'] = historical_data.get('blockages_last_24h', 0)
            features['time_since_last_blockage'] = historical_data.get('time_since_last_blockage', 999)
        else:
            features['blockages_last_hour'] = 0
            features['blockages_last_24h'] = 0
            features['time_since_last_blockage'] = 999
        
        # Temporal peak probabilities (holiday-adjusted)
        # Freight runs heavier on weekends/holidays; passenger is lighter.
        holiday_or_weekend = bool(features['is_holiday'] or features['is_weekend'])

        if features['is_freight_peak']:
            features['freight_peak_probability'] = 0.90 if holiday_or_weekend else 0.85
        else:
            features['freight_peak_probability'] = 0.20 if holiday_or_weekend else 0.15

        if features['is_passenger_peak']:
            features['passenger_peak_probability'] = 0.45 if holiday_or_weekend else 0.70
        else:
            features['passenger_peak_probability'] = 0.20 if holiday_or_weekend else 0.30
        
        return features


# Test the feature extractor
if __name__ == '__main__':
    engine = FeatureEngine()
    
    # Test crossing config
    crossings = {
        'se-11th-ave': {
            'lat': 45.5037,
            'lon': -122.6547,
            'fra_id': '759733U'
        }
    }
    
    # Extract features for a test scenario
    features = engine.extract_all_features(
        timestamp='2026-02-08T09:00:00',
        crossing_id='se-11th-ave',
        crossings_config=crossings,
        train_speed=8.5,  # Slow-moving train
        historical_data={'blockages_last_hour': 2}
    )
    
    print("🧠 Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Test train type inference
    train_type, duration = engine.predict_train_type_from_speed(8.5)
    print(f"\n🚂 Predicted train type: {train_type}")
    print(f"⏱️  Predicted duration: {duration} minutes")
