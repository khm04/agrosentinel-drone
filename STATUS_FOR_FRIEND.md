# AgroDrone AI / AgroSentinel — Project Status for Review

Hey! Here's a clean breakdown of where the project stands, what works, what's mocked, and what's left to build/fix. Skim the headings — anything marked ❌ or 🐛 needs attention.

---

## 0. The 30-second summary

We have **two pieces** that talk to each other through Firebase:

1. **Python uploader** (`firebase gradproj/`) — runs on a laptop / on the drone companion computer. Takes an image + bbox + GPS, saves a crop, uploads it to Firebase Storage, and writes a document to the Firestore `events` collection.
2. **Flutter mobile app** (`flutter grad/`) — clean-architecture Flutter app (BLoC/Cubit + GoRouter + GetIt). Login with Firebase Auth, then 6 bottom-nav tabs: **Home / Map / Alerts / Events / AI Hub / Settings**.

The **only real end-to-end flow** today is:
> Python script writes to Firestore → app's **Detections** tab shows it live with the image.

Everything else in the app is **mocked** (drone telemetry, map markers, AI diagnosis, push alerts).

---

## 1. Firebase backend (`firebase gradproj/`)

| File | Status | Notes |
|---|---|---|
| `firebase_config.py` | ✅ works | Initializes Admin SDK, points at storage bucket `agrosentinel-storage-2026` |
| `detect_and_upload.py` | ✅ works | `save_crop` / `save_full_image` / `upload_image` / `log_event` / `handle_disease_classification` |
| `test_firebase.py` | ✅ smoke test | Writes a `test_events` doc + uploads `sample.txt` |
| `test_detection.py` | ✅ smoke test | Calls `handle_disease_classification` on `test.jpg` with hardcoded bbox + disease name |
| `testquiz.py` | 🗑️ junk file | `print("hi"*400)` — delete this |
| `serviceAccountKey.json` | ⚠️ SECRET | Make sure this is in `.gitignore` and **never** pushed to GitHub |
| `__pycache__/` | ⚠️ | Gitignore this folder |

### What's missing on the Python side
- ❌ **No actual ML model.** Today only the test script calls `handle_disease_classification` with a hardcoded `("Tomato Early Blight", 0.92, bbox=(20,20,200,200))`. We need to plug in a real YOLO / TFLite / classifier that takes a frame and produces (label, confidence, bbox).
- ❌ **No fire detection function.** Only `handle_disease_classification` exists. We need `handle_fire_detection` with the same shape so the app's red-vs-orange UI works.
- ❌ **No GPS source.** `gps_lat` / `gps_lng` are passed in by hand. Needs to come from the drone telemetry / a MAVLink stream / a phone GPS.
- ❌ **Not running as a service / loop.** Today it's `python test_detection.py` once. Needs to either watch a folder, listen on a socket, or run inside the drone's onboard compute.

### Bugs in the Python
- 🐛 `upload_image()` line 85 — returns `storage_path, storage_path` so the `image_url` field in Firestore is **not actually a URL**, it's just the storage path. The Flutter app handles this OK (it calls `getDownloadURL()` itself), but the field name is misleading. Either rename the Firestore field to `image_path`, or call `blob.generate_signed_url(...)` and return a real URL.

---

## 2. Firestore + Storage rules

```
firestore.rules  →  /events/{doc}    : auth users can read, nobody writes from client (only Admin SDK)
storage.rules    →  /detections/**  : auth users can read, nobody writes from client
```

- ✅ Sensible default — locks the public out, only the Python uploader (Admin SDK) can write.
- ❌ **No rules for any other collection** — when we add user profiles, farms, scan history, etc., remember to write rules per-user (`request.auth.uid == userId`).
- ⚠️ `firebase.json` declares both a `default` storage target and an `agrosentinel` target pointing at the same `storage.rules` — fine but redundant.

---

## 3. Flutter app (`flutter grad/lib/`)

### Architecture & infra — ✅ solid
- Clean Architecture: data / domain / presentation
- State: `flutter_bloc` (BLoC for auth, Cubit for everything else)
- Routing: `go_router` with auth-redirect
- DI: `get_it` ([core/di/injection.dart](flutter%20grad/lib/core/di/injection.dart))
- Theming: 5 named palettes, animated picker, M3 ([core/theme/app_theme.dart](flutter%20grad/lib/core/theme/app_theme.dart))
- i18n: hand-rolled English + Arabic with full RTL ([core/l10n/app_l10n.dart](flutter%20grad/lib/core/l10n/app_l10n.dart))

### Feature-by-feature status

#### 🔐 Auth — ✅ real, mostly done
- Real Firebase Auth: login, signup, logout, password-reset email, auto-login on app start.
- Nice login/signup UI with animations, friendly Firebase error messages, `Forgot password?` dialog.
- ❌ **No profile page** — user can't edit name, farm name, or avatar.
- ❌ `farmName`, `totalScans`, `alertsToday` are hardcoded `0` / `''` in [auth_remote_datasource.dart:67](flutter%20grad/lib/features/auth/data/datasources/auth_remote_datasource.dart#L67) — the Home screen displays them but they're meaningless. We need a `users/{uid}` Firestore document.
- ⚠️ `local_auth` (biometric) and `flutter_secure_storage` are in `pubspec.yaml` but **not used anywhere**. Either wire up Face/Touch ID login or remove the deps.

#### 🏠 Home — ✅ pretty UI, ❌ all mock data
- Lovely hero banner, status card (battery / signal / altitude / speed), quick-action grid, shimmer loaders.
- ❌ `HomeRemoteDataSourceImpl.getDroneStatus()` → returns `DroneStatusModel.mock` (78% battery, 92% signal…) after an 800ms fake delay. **No real telemetry connection.**

#### 🗺️ Map — ⚠️ placeholder
- It's a **stylized fake map**: `CustomPainter` draws a grid + path + circles. No Google Maps, no OSM tiles. ([map_page.dart](flutter%20grad/lib/features/map/presentation/pages/map_page.dart))
- Mock markers + mock flight path.
- ❌ Need to replace with `flutter_map` (free, no API key) or `google_maps_flutter` (needs key + meta-data tag in AndroidManifest).
- ❌ Drone position doesn't move. No real GPS stream. No geofence / NDVI / area drawing.

#### 🔔 Alerts (Notifications) — ⚠️ all mock
- Pretty UI: filter chips (All / Fire / Disease / Warning / Info), shimmer, mark-all-read, tap → `/map`.
- ❌ Mock service that emits one fake notification every **30 seconds** on a static timer ([notification_datasource.dart:22](flutter%20grad/lib/features/notifications/data/datasources/notification_datasource.dart#L22)).
- ❌ **No FCM** (Firebase Cloud Messaging). `firebase_messaging` package is **not in pubspec**. We need to:
  - Add the package + Android service registration.
  - Write a Cloud Function that triggers on new `events` doc and sends FCM.
  - Save the FCM token under `users/{uid}/fcmToken`.
- ❌ Tapping a notification doesn't open a detail page — only takes you to the (mock) map.

#### 📡 Events / Detections — ✅ THE REAL ONE
- This is the **only real Firebase-backed screen**. ([detections_page.dart](flutter%20grad/lib/features/detections/presentation/pages/detections_page.dart))
- Subscribes to Firestore `events` collection (live `snapshots()`), pulls each image via `FirebaseStorage.getDownloadURL()`.
- 🐛 The orange "alertDisease" colored pill is shown for **every** anomaly type. Fire events look identical to disease events. Need a `switch (event.anomalyType)` for color + label.
- 🐛 Storage bucket name is **hardcoded** as `gs://agrosentinel-storage-2026` ([detections_page.dart:68](flutter%20grad/lib/features/detections/presentation/pages/detections_page.dart#L68)). It matches `firebase_options.dart` today, but if either changes the app silently breaks. Pull from `Firebase.app().options.storageBucket` instead.
- ⚠️ List is read-only. No tap-to-detail. No "view on map" link. No way to mark an event as "resolved" — even though Python writes `status: "new"` so the workflow is anticipated.

#### 🤖 AI Hub (Diagnostic) — ⚠️ all mock
- Image picker (camera + gallery) works. UI for analyzing + result card is good.
- ❌ `DiagnosticRemoteDataSourceImpl.analyzeImage()` waits 3 seconds then returns one of two **hardcoded** responses based on whether the file size in KB is even or odd ([diagnostic_remote_datasource.dart:17](flutter%20grad/lib/features/ai_diagnostic/data/datasources/diagnostic_remote_datasource.dart#L17)). 😅
- ❌ No on-device TFLite model, no REST call, no scan history persistence.

#### ⚙️ Settings — ⚠️ partly cosmetic
- 5 themes, EN/AR toggle, push notifications switch, logout, version, architecture info. UI is polished.
- 🐛 The app is hardcoded to `themeMode: ThemeMode.system` ([app.dart:31](flutter%20grad/lib/app.dart#L31)) and sets `theme` and `darkTheme` to the **same** value. So:
  - Toggling dark / light at the OS level does nothing visually.
  - The `themeMode` we save in `SharedPreferences` is loaded but never applied.
  - Decide: either commit to "the palette IS the theme" and delete the unused `themeMode` plumbing, or wire it up properly.
- 🐛 The "Push Notifications" switch saves a `bool` to SharedPreferences that **nothing reads** — purely decorative right now.

---

## 4. Wiring & config issues to fix

- `firebase_options.dart`:
  - ✅ Android keys filled in
  - ❌ **Web** has all `REPLACE_ME` — building for web crashes at startup
  - ❌ **iOS / macOS** has all `REPLACE_ME` — same
  - 🔧 Run `flutterfire configure` and pick the platforms you actually ship.
- ❌ **iOS isn't configured at all** — no `GoogleService-Info.plist`, no `Info.plist` permissions for camera / photo library. Right now this is Android-only.
- ⚠️ AndroidManifest has camera + storage perms but **no FCM service block** — fine until we add `firebase_messaging`, then we need to wire it.
- 🗑️ `flutter_log.txt` (~50KB UTF-16 log) shouldn't be checked in — gitignore it.

---

## 5. Security checklist

- [ ] `serviceAccountKey.json` in `.gitignore` (and rotated if it was ever pushed)
- [ ] `__pycache__/` in `.gitignore`
- [ ] `flutter_log.txt` in `.gitignore`
- [ ] Firestore rules tightened to per-user once we add user docs
- [ ] Storage rules: write should ONLY come from Admin SDK (currently `if false` — good)

---

## 6. What I'd ask the team to do next, in order

### Phase A — make it really work end-to-end (1–2 weeks)
1. Plug a real disease classifier into `detect_and_upload.py` (TFLite or PyTorch with a saved model). Even a 3-class demo model is fine for the defense.
2. Add `handle_fire_detection(...)` in the Python (mirror the disease one, set `anomaly_type="fire"`).
3. Fix the Detections card so fire events render red and disease events render orange.
4. Wire `image_url` in Firestore to be a **real URL** (signed URL or use the path consistently — pick one).
5. Replace the placeholder map with `flutter_map` + OpenStreetMap tiles. Plot the events from Firestore.

### Phase B — push alerts (1 week)
6. Add `firebase_messaging` to the app, save the FCM token to `users/{uid}/fcmToken` in Firestore on login.
7. Write a Cloud Function `onCreate /events/{id}` that fans out an FCM push to all users.
8. Make notifications tap-through to the Event detail (which we'd add to the Detections feature).

### Phase C — nice-to-haves for a polished defense
9. Profile page: editable name + farm name, avatar upload to `users/{uid}/avatar.jpg`.
10. Wire up the dark/light `ThemeMode` toggle, OR delete the unused code.
11. Make the "Push notifications" switch actually subscribe / unsubscribe from an FCM topic.
12. Add a tappable detail page on the Detections list — show the bbox crop full-screen, GPS link to map, "mark resolved" button.
13. Replace the mock 30-second notification stream with a Firestore listener on `events` so the bell-badge actually counts new detections.
14. Either implement biometric login (`local_auth`) or remove the dependency.
15. Write at least 2–3 widget tests so `flutter test` doesn't run empty.

### Phase D — only if there's time
16. AI Hub: replace the size-parity mock with either a TFLite model on-device OR a FastAPI endpoint deployed somewhere cheap.
17. Diagnosis history (SQLite via `drift` or a Firestore subcollection).
18. Charts (`fl_chart`) for trends.
19. PDF export (`pdf` + `printing`).
20. iOS configuration (`GoogleService-Info.plist`, Info.plist perms, `flutterfire configure`).

---

## 7. TL;DR for your friend

- The app **looks finished** but ~80% of it is mock data behind real-looking UI.
- The **one truly working pipeline** is: Python writes to Firestore → app shows it in the Events tab.
- **Top 3 to fix before defense:**
  1. Real classifier driving `detect_and_upload.py` (so it's not running once on `test.jpg`).
  2. Real map (just `flutter_map`, no API key needed) showing the events.
  3. FCM push so a fire detection actually buzzes the phone.
- **Top 3 quick fixes (each <1 hour):**
  1. Make Detections card color depend on `anomalyType` instead of always orange.
  2. Stop hardcoding the storage bucket in the Detections page.
  3. Either wire up `themeMode` in `app.dart` or delete the unused settings code.

Reply with thoughts / what you want to grab first 🙏
