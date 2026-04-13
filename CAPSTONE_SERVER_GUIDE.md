# Running the Seal IP Frontend on the GDMS Capstone Web Server

> **Quick version:** Kill anything on port 8080, build with `--features seal-ip`, run from the `WS/` directory, open `http://localhost:8080/seal`.

---

## Overview

The Seal Photo Processing system can run on **two** web servers:

| Server | Location | Purpose |
|--------|----------|---------|
| **Backup server** (`web_server`) | `AI-Assisted_Seal_…/web_server/` | Lightweight standalone proof-of-concept. No DB, no TLS, no auth. |
| **Capstone server** (`WS`) | `General-Dynamics-Packet-Sniffer-and-Web-Server/WS/` | Full GDMS production server with MySQL, TLS/HTTPS, AES-256 file encryption, user auth, Yew WASM frontend, and the Seal IP integration as a feature flag. |

For the defense presentation, you want the **Capstone server** — it shows the Seal IP system integrated into the real Capstone project architecture.

---

## Prerequisites

1. **Both repos side-by-side** (already the case):
   ```
   C:\Users\user\RustroverProjects\
   ├── AI-Assisted_Seal_Photograph_Image_Processing_and_Correction_System_in_Rust\
   └── General-Dynamics-Packet-Sniffer-and-Web-Server\
   ```
   The Capstone `WS/Cargo.toml` references IP_functions via a relative path:
   ```toml
   IP_functions = { path = "../../AI-Assisted_Seal_.../IP_functions", optional = true }
   ```

2. **MySQL** — The Capstone server uses MySQL for user/file management. The DB pool is lazy-initialized, so the server will start without it, but user/file endpoints will fail. For a **Seal-IP-only demo** this is fine (the `/seal` and `/api/seal/*` endpoints don't touch the database at all). If you want the full dashboard too, start MySQL via Docker Compose (see Step 2a below).

3. **Rust toolchain** — You already have this.

---

## Step-by-Step: Seal IP Demo on the Capstone Server

### Step 1 — Kill Any Existing Server on Port 8080

```powershell
# Check what's on port 8080
netstat -ano | Select-String ":8080"

# Kill it (replace PID with actual process ID from above)
Stop-Process -Id <PID> -Force
```

### Step 2 — Build the Capstone Server with Seal IP Enabled

```powershell
cd "C:\Users\user\RustroverProjects\General-Dynamics-Packet-Sniffer-and-Web-Server"

# Build with the seal-ip feature flag (pulls in IP_functions, image, ndarray, base64)
cargo build --release -p WS --features seal-ip
```

> **Note:** The `seal-ip` feature is defined in `WS/Cargo.toml` and is **off by default** — this is intentional so the Docker build (which doesn't have the thesis repo) still works. You **must** pass `--features seal-ip` or the `/seal` and `/api/seal/*` routes won't be registered.

Build time is ~2-3 minutes for a fresh release build.

### Step 2a — (Optional) Start MySQL for Full Dashboard

Only needed if you want the Yew dashboard (user accounts, file upload/download with AES encryption) in addition to Seal IP. **Skip this if you only need the Seal photo processing UI.**

```powershell
cd "C:\Users\user\RustroverProjects\General-Dynamics-Packet-Sniffer-and-Web-Server\WS"
docker compose up -d db
```

Then set the `DATABASE_URL` environment variable:
```powershell
$env:DATABASE_URL = "mysql://root:changeme123@localhost:3307/webserver"
```

### Step 3 — Start the Server

```powershell
cd "C:\Users\user\RustroverProjects\General-Dynamics-Packet-Sniffer-and-Web-Server\WS"

# Run directly (debug build, slower processing but instant start):
cargo run -p WS --features seal-ip

# Or run the release binary (faster image processing):
..\target\release\WS.exe
```

You should see:
```
[WS] TLS disabled (set WS_TLS_ENABLED=1 to enable HTTPS)
[WS] Starting server on 0.0.0.0:8080 with 16 threads, queue size 256 (TLS: OFF)
Listening on 0.0.0.0:8080
```

> **Important:** The server must be started from the `WS/` directory so it can find the `static/` folder for the Yew frontend assets.

### Step 4 — Open the Frontend

- **Seal Photo Processing UI:** [http://localhost:8080/seal](http://localhost:8080/seal)
- **Yew Dashboard (full Capstone frontend):** [http://localhost:8080/](http://localhost:8080/)
- **Seal API Health Check:** [http://localhost:8080/api/seal/health](http://localhost:8080/api/seal/health)

### Step 5 — Use It / Take Screenshots

1. Open `http://localhost:8080/seal` in your browser
2. Drag & drop a seal image (e.g., `bansui.jpg` from the thesis repo root)
3. Click **🦭 Full Pipeline** (or individual buttons: DCP Dehaze, CLAHE, Gamma)
4. Wait for processing (5-30 seconds depending on image size)
5. Results appear as cards with before/after + download links
6. Screenshot it! (Win+Shift+S on Windows)

---

## API Endpoints (Capstone Server)

All Seal IP endpoints are prefixed with `/api/seal/` (not `/api/` like the backup server):

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/seal` | Interactive upload UI (HTML page) |
| `GET` | `/api/seal/health` | Health check JSON |
| `POST` | `/api/seal/dehaze` | DCP dehazing |
| `POST` | `/api/seal/clahe` | CLAHE contrast enhancement |
| `POST` | `/api/seal/gamma` | Gamma brightness correction |
| `POST` | `/api/seal/process` | Full pipeline (DCP → CLAHE → Gamma) |

All POST endpoints accept: `{ "image": "<base64>", ...optional params }`

---

## Automated Demo Script

There's a ready-made demo script that does everything automatically:

```powershell
cd "C:\Users\user\RustroverProjects\General-Dynamics-Packet-Sniffer-and-Web-Server"
.\demo_seal.ps1
```

This script:
1. Checks if the server is running (starts it if not)
2. Loads `bansui.jpg` and sends it to all 4 endpoints
3. Saves output images to the thesis repo root
4. Opens the results + browser UI automatically

---

## Environment Variables (Reference)

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_ADDR` | `0.0.0.0:8080` | Listen address |
| `WS_THREAD_COUNT` | `4 × CPU cores` (clamped 16-32) | Worker threads |
| `WS_QUEUE_SIZE` | `thread_count × 16` | Job queue capacity |
| `WS_TLS_ENABLED` | `false` | Set `1` for HTTPS |
| `WS_TLS_CERT` | `cert.pem` | TLS certificate path |
| `WS_TLS_KEY` | `key.pem` | TLS private key path |
| `DATABASE_URL` | `mysql://root:my-secret-pw@localhost:3306/filestore` | MySQL connection |

---

## Troubleshooting

- **"Failed to create pool" panic on startup** — MySQL isn't running. The DB pool is lazy so this only happens when a user/file endpoint is hit, NOT at startup. The `/seal` endpoints don't need MySQL.
- **`/seal` returns 404** — You forgot `--features seal-ip` in the build command.
- **Port 8080 already in use** — Kill the other process first (Step 1).
- **Slow image processing** — Use `--release` build. Debug builds are ~10x slower for the image algorithms.
- **Server can't find `static/index.html`** — Make sure you `cd` into the `WS/` directory before running.

