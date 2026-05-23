import os
import firebase_admin
from firebase_admin import credentials, firestore, storage


def initialize_firebase():
    """
    Initialize Firebase Admin SDK once and return Firestore + Storage clients.
    """
    if not firebase_admin._apps:
        key_path = os.getenv("FIREBASE_KEY_PATH", "serviceAccountKey.json")

        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Firebase key file not found: {key_path}\n"
                f"Set FIREBASE_KEY_PATH or place serviceAccountKey.json in the project folder."
            )

        cred = credentials.Certificate(key_path)

        firebase_admin.initialize_app(cred, {
            "storageBucket": "agrosentinel-storage-2026"
        })

    db = firestore.client()
    bucket = storage.bucket()
    return db, bucket