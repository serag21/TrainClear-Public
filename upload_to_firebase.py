"""
Upload ML predictions to Firebase Firestore

Change Detection: Only writes to Firestore when a prediction has materially
changed (status, duration >2 min, or confidence). Unchanged predictions get
only a timestamp heartbeat to avoid triggering onSnapshot on every client.
"""

import json
import logging
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('TrainClear.upload')


def init_firebase():
    """Initialize Firebase Admin SDK"""

    # Check if running in GitHub Actions
    if 'FIREBASE_CREDENTIALS_FILE' in os.environ:
        # Load from file (GitHub Actions creates this)
        cred_file = os.environ['FIREBASE_CREDENTIALS_FILE']
        cred = credentials.Certificate(cred_file)
    elif os.path.exists('serviceAccountKey.json'):
        # Load from local file (development)
        cred = credentials.Certificate('serviceAccountKey.json')
    else:
        raise FileNotFoundError(
            "Firebase credentials not found. "
            "Download serviceAccountKey.json from Firebase Console."
        )

    # Initialize app if not already initialized
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)

    return firestore.client()


def _load_firestore_state(db):
    """Load current prediction state from Firestore for change comparison."""
    previous = {}
    try:
        docs = db.collection('predictions').stream()
        for doc in docs:
            previous[doc.id] = doc.to_dict()
        logger.debug(f"Loaded previous Firestore state for {len(previous)} crossings")
    except Exception as e:
        logger.warning(f"Could not load previous Firestore state: {e}")
    return previous


def _should_upload(crossing_id, new_pred, prev_pred):
    """
    Return True if prediction differs enough to warrant a full Firestore write.

    Criteria:
    - status changed
    - duration_minutes changed by >2 minutes
    - confidence level changed
    - No previous state exists
    """
    if prev_pred is None:
        return True

    # Status changed
    if new_pred.get('status') != prev_pred.get('status'):
        return True

    # Duration changed by >2 minutes
    new_dur = new_pred.get('duration_minutes')
    prev_dur = prev_pred.get('duration_minutes')
    if new_dur is not None and prev_dur is not None:
        if abs(float(new_dur) - float(prev_dur)) > 2.0:
            return True
    elif (new_dur is None) != (prev_dur is None):
        return True

    # Confidence level changed
    if new_pred.get('confidence') != prev_pred.get('confidence'):
        return True

    return False


def upload_predictions(predictions_file='current_predictions.json'):
    """Upload predictions to Firebase with change detection."""

    print("Uploading predictions to Firebase...")

    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    # Initialize Firebase
    db = init_firebase()

    # Load current Firestore state for comparison
    previous_state = _load_firestore_state(db)

    uploaded = 0
    skipped = 0
    for crossing_id, prediction in predictions.items():
        try:
            doc_ref = db.collection('predictions').document(crossing_id)
            prev = previous_state.get(crossing_id)

            if not _should_upload(crossing_id, prediction, prev):
                # No material change -- only refresh the heartbeat timestamp
                doc_ref.update({
                    'updated_at': firestore.SERVER_TIMESTAMP,
                    'timestamp': prediction.get('timestamp', datetime.now().isoformat()),
                })
                skipped += 1
                logger.info(f"   -- Skipped (no material change): {prediction.get('crossing_name', crossing_id)}")
                continue

            # Material change detected -- full write
            prediction['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(prediction, merge=True)

            print(f"   Uploaded: {prediction.get('crossing_name', crossing_id)}")
            uploaded += 1

        except Exception as e:
            print(f"   Error uploading {crossing_id}: {e}")

    print(f"\nUploaded {uploaded}/{len(predictions)} predictions "
          f"({skipped} skipped, unchanged)")

    return uploaded


if __name__ == '__main__':
    try:
        upload_predictions()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
