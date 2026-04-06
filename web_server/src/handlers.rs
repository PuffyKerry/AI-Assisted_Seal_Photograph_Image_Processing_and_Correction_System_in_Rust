// =============================================================================
// web_server/src/handlers.rs — API endpoint handlers for Seal IP Processing
//
// Endpoints:
//   GET  /              → HTML upload page with interactive UI
//   GET  /api/health    → JSON health check
//   POST /api/dehaze    → DCP dehazing
//   POST /api/clahe     → CLAHE contrast enhancement
//   POST /api/gamma     → Gamma brightness correction
//   POST /api/process   → Full pipeline: DCP → CLAHE → Gamma
//
// All POST endpoints accept JSON:
//   { "image": "<base64-encoded image>", ...optional params... }
// And return JSON:
//   { "image": "<base64-encoded result>", "width": N, "height": N, ... }
// =============================================================================

use crate::request::Request;
use crate::response::Response;
use crate::convert;

use IP_functions::dehaze::dehaze_with_params;
use IP_functions::enhance::enhance_clahe;
use IP_functions::brightness::brightness_correct;

// ---------------------------------------------------------------------------
// Helper: extract a JSON string field (simple parser, no serde needed for this)
// ---------------------------------------------------------------------------
fn extract_json_string(json: &str, field: &str) -> Option<String> {
    // Look for "field":"..." or "field": "..."
    let pattern = format!(r#""{}""#, field);
    let field_start = json.find(&pattern)?;
    let after_key = &json[field_start + pattern.len()..];
    // Skip optional whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    if after_colon.starts_with('"') {
        // String value — find the closing quote (handle the massive base64 blob)
        let content = &after_colon[1..];
        let end = content.find('"')?;
        Some(content[..end].to_string())
    } else {
        None
    }
}

fn extract_json_f32(json: &str, field: &str) -> Option<f32> {
    let pattern = format!(r#""{}""#, field);
    let field_start = json.find(&pattern)?;
    let after_key = &json[field_start + pattern.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    // Read until comma, brace, or whitespace
    let end = after_colon
        .find(|c: char| c == ',' || c == '}' || c == ' ' || c == '\n' || c == '\r')
        .unwrap_or(after_colon.len());
    after_colon[..end].trim().parse::<f32>().ok()
}

fn extract_json_usize(json: &str, field: &str) -> Option<usize> {
    extract_json_f32(json, field).map(|v| v as usize)
}

/// Extract the base64 image from the request body JSON
fn extract_image_from_request(body: &str) -> Result<ndarray::Array3<f32>, String> {
    let b64 = extract_json_string(body, "image")
        .ok_or_else(|| "Missing \"image\" field in JSON body. Expected: {\"image\": \"<base64>\"}".to_string())?;
    let (_img, array) = convert::decode_base64_image(&b64)?;
    Ok(array)
}

// ===========================================================================
// GET / — Interactive upload page
// ===========================================================================
pub fn home_handler(_request: &Request) -> Response {
    Response::ok()
        .html(UPLOAD_PAGE_HTML.to_string())
        .build()
}

// ===========================================================================
// GET /api/health — Health check
// ===========================================================================
pub fn health_handler(_request: &Request) -> Response {
    Response::ok()
        .json(r#"{"status":"ok","service":"seal-ip-server","version":"0.1.0","endpoints":["/api/dehaze","/api/clahe","/api/gamma","/api/process"]}"#.to_string())
        .build()
}

// ===========================================================================
// POST /api/dehaze — Dark Channel Prior dehazing
// Optional params: patch_size, omega, t0, top_percent, guided_radius, guided_eps
// ===========================================================================
pub fn dehaze_handler(request: &Request) -> Response {
    let body = request.body_as_string();
    let img_array = match extract_image_from_request(&body) {
        Ok(a) => a,
        Err(e) => {
            return Response::bad_request()
                .json(format!(r#"{{"error":"{}"}}"#, e))
                .build();
        }
    };

    let (h, w, _) = img_array.dim();
    println!("[API] /api/dehaze — image {}x{}", w, h);

    // Extract optional custom parameters or use defaults
    let omega = extract_json_f32(&body, "omega").unwrap_or(0.75);
    let t0 = extract_json_f32(&body, "t0").unwrap_or(0.25);
    let patch_size = extract_json_usize(&body, "patch_size").unwrap_or(15);
    let top_percent = extract_json_f32(&body, "top_percent").unwrap_or(0.001);
    let guided_radius = extract_json_usize(&body, "guided_radius").unwrap_or(15);
    let guided_eps = extract_json_f32(&body, "guided_eps").unwrap_or(0.0001);

    let has_custom = extract_json_f32(&body, "omega").is_some();
    if has_custom {
        println!("  Custom params: omega={}, t0={}, patch={}, grad_r={}, grad_e={}",
                 omega, t0, patch_size, guided_radius, guided_eps);
    } else {
        println!("  Using default parameters");
    }
    let result = dehaze_with_params(&img_array, patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    match convert::encode_image_base64_jpeg(&result) {
        Ok(b64) => {
            let json = format!(
                r#"{{"image":"{}","width":{},"height":{},"operation":"dehaze","params":{{"omega":{},"t0":{},"patch_size":{},"top_percent":{},"guided_radius":{},"guided_eps":{}}}}}"#,
                b64, w, h, omega, t0, patch_size, top_percent, guided_radius, guided_eps
            );
            Response::ok().json(json).build()
        }
        Err(e) => Response::internal_server_error()
            .json(format!(r#"{{"error":"Encoding failed: {}"}}"#, e))
            .build(),
    }
}

// ===========================================================================
// POST /api/clahe — CLAHE contrast enhancement
// Optional params: grid_h, grid_w, clip_limit
// ===========================================================================
pub fn clahe_handler(request: &Request) -> Response {
    let body = request.body_as_string();
    let img_array = match extract_image_from_request(&body) {
        Ok(a) => a,
        Err(e) => {
            return Response::bad_request()
                .json(format!(r#"{{"error":"{}"}}"#, e))
                .build();
        }
    };

    let (h, w, _) = img_array.dim();
    println!("[API] /api/clahe — image {}x{}", w, h);

    let grid_h = extract_json_usize(&body, "grid_h").unwrap_or(8);
    let grid_w = extract_json_usize(&body, "grid_w").unwrap_or(8);
    let clip_limit = extract_json_f32(&body, "clip_limit").unwrap_or(2.0);

    if extract_json_f32(&body, "clip_limit").is_some() {
        println!("  Custom params: grid={}x{}, clip_limit={}", grid_h, grid_w, clip_limit);
    } else {
        println!("  Using default parameters (8x8, clip=2.0)");
    }
    let result = enhance_clahe(&img_array, grid_h, grid_w, clip_limit);

    match convert::encode_image_base64_jpeg(&result) {
        Ok(b64) => {
            let json = format!(
                r#"{{"image":"{}","width":{},"height":{},"operation":"clahe","params":{{"grid_h":{},"grid_w":{},"clip_limit":{}}}}}"#,
                b64, w, h, grid_h, grid_w, clip_limit
            );
            Response::ok().json(json).build()
        }
        Err(e) => Response::internal_server_error()
            .json(format!(r#"{{"error":"Encoding failed: {}"}}"#, e))
            .build(),
    }
}

// ===========================================================================
// POST /api/gamma — Gamma brightness correction
// Optional params: gamma (float, <1 brightens, >1 darkens)
// ===========================================================================
pub fn gamma_handler(request: &Request) -> Response {
    let body = request.body_as_string();
    let img_array = match extract_image_from_request(&body) {
        Ok(a) => a,
        Err(e) => {
            return Response::bad_request()
                .json(format!(r#"{{"error":"{}"}}"#, e))
                .build();
        }
    };

    let (h, w, _) = img_array.dim();
    println!("[API] /api/gamma — image {}x{}", w, h);

    let custom_gamma = extract_json_f32(&body, "gamma");
    let gamma_val = custom_gamma.unwrap_or_else(|| {
        IP_functions::gamma::estimate_gamma(&img_array)
    });
    if custom_gamma.is_some() {
        println!("  Custom gamma: {}", gamma_val);
    } else {
        println!("  Using auto-estimated gamma: {:.3}", gamma_val);
    }
    let result = brightness_correct(&img_array, gamma_val);

    match convert::encode_image_base64_jpeg(&result) {
        Ok(b64) => {
            let json = format!(
                r#"{{"image":"{}","width":{},"height":{},"operation":"gamma","params":{{"gamma":{:.4},"auto_estimated":{}}}}}"#,
                b64, w, h, gamma_val, custom_gamma.is_none()
            );
            Response::ok().json(json).build()
        }
        Err(e) => Response::internal_server_error()
            .json(format!(r#"{{"error":"Encoding failed: {}"}}"#, e))
            .build(),
    }
}

// ===========================================================================
// POST /api/process — Full pipeline: DCP → CLAHE → Gamma
// Applies all three operations in sequence with attenuated stacking
// Optional params: same as individual endpoints
// ===========================================================================
pub fn process_handler(request: &Request) -> Response {
    let body = request.body_as_string();
    let img_array = match extract_image_from_request(&body) {
        Ok(a) => a,
        Err(e) => {
            return Response::bad_request()
                .json(format!(r#"{{"error":"{}"}}"#, e))
                .build();
        }
    };

    let (h, w, _) = img_array.dim();
    println!("[API] /api/process — Full pipeline on {}x{}", w, h);

    // === Step 1: DCP Dehazing ===
    let omega = extract_json_f32(&body, "omega").unwrap_or(0.75);
    let t0 = extract_json_f32(&body, "t0").unwrap_or(0.25);
    let patch_size = extract_json_usize(&body, "patch_size").unwrap_or(15);
    let top_percent = extract_json_f32(&body, "top_percent").unwrap_or(0.001);
    let guided_radius = extract_json_usize(&body, "guided_radius").unwrap_or(15);
    let guided_eps = extract_json_f32(&body, "guided_eps").unwrap_or(0.0001);

    println!("  Step 1: DCP dehaze (omega={}, t0={}, patch={})", omega, t0, patch_size);
    let dehazed = dehaze_with_params(&img_array, patch_size, omega, t0, top_percent, guided_radius, guided_eps);

    // === Step 2: CLAHE on dehazed result (attenuated) ===
    let grid_h = extract_json_usize(&body, "grid_h").unwrap_or(8);
    let grid_w = extract_json_usize(&body, "grid_w").unwrap_or(8);
    let clip_limit = extract_json_f32(&body, "clip_limit").unwrap_or(2.0);
    // Attenuate clip_limit when stacking after DCP to avoid overcorrection
    let stacked_clip = 1.0_f32 + (clip_limit - 1.0) * 0.7;
    println!("  Step 2: CLAHE (grid={}x{}, clip {:.2} → {:.2} stacked)", grid_h, grid_w, clip_limit, stacked_clip);
    let enhanced = enhance_clahe(&dehazed, grid_h, grid_w, stacked_clip);

    // === Step 3: Gamma on DCP+CLAHE result (attenuated) ===
    let gamma = extract_json_f32(&body, "gamma");
    let gamma_val = gamma.unwrap_or_else(|| {
        // Auto-estimate from the enhanced result
        IP_functions::gamma::estimate_gamma(&enhanced)
    });
    // Attenuate gamma toward identity when stacking
    let stacked_gamma = 1.0 + (gamma_val - 1.0) * 0.5;
    println!("  Step 3: Gamma ({:.3} → {:.3} stacked)", gamma_val, stacked_gamma);
    let final_result = brightness_correct(&enhanced, stacked_gamma);

    // Build response with the full pipeline result, plus individual stages as base64
    match convert::encode_image_base64_jpeg(&final_result) {
        Ok(b64_final) => {
            // Also encode individual stages for comparison
            let b64_dehazed = convert::encode_image_base64_jpeg(&dehazed).unwrap_or_default();
            let b64_clahe = convert::encode_image_base64_jpeg(&enhance_clahe(&img_array, grid_h, grid_w, clip_limit)).unwrap_or_default();
            let gamma_only_val = IP_functions::gamma::estimate_gamma(&img_array);
            let b64_gamma = convert::encode_image_base64_jpeg(&brightness_correct(&img_array, gamma_only_val)).unwrap_or_default();

            let json = format!(
                r#"{{"image":"{}","dehaze_only":"{}","clahe_only":"{}","gamma_only":"{}","width":{},"height":{},"operation":"full_pipeline","params":{{"omega":{},"t0":{},"patch_size":{},"top_percent":{},"guided_radius":{},"guided_eps":{},"clip_limit":{},"stacked_clip":{:.3},"gamma":{:.4},"stacked_gamma":{:.4},"grid_h":{},"grid_w":{}}}}}"#,
                b64_final, b64_dehazed, b64_clahe, b64_gamma,
                w, h,
                omega, t0, patch_size, top_percent, guided_radius, guided_eps,
                clip_limit, stacked_clip, gamma_val, stacked_gamma, grid_h, grid_w
            );
            Response::ok().json(json).build()
        }
        Err(e) => Response::internal_server_error()
            .json(format!(r#"{{"error":"Encoding failed: {}"}}"#, e))
            .build(),
    }
}

// ===========================================================================
// Upload page HTML — embedded as a const so no external file dependency
// ===========================================================================
const UPLOAD_PAGE_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Seal Photo Processing — AI-Assisted IP System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0c1445 0%, #1a3a5c 50%, #0d4f4f 100%);
            color: #e0e8f0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 {
            text-align: center;
            font-size: 2em;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #4fc3f7, #81d4fa, #4dd0e1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #90a4ae; margin-bottom: 16px; }

        /* ---- Explanation box ---- */
        .explanation {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(79,195,247,0.25);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 22px;
            line-height: 1.55;
            font-size: 0.92em;
            color: #b0c4d8;
        }
        .explanation summary {
            cursor: pointer;
            font-weight: 600;
            color: #81d4fa;
            font-size: 1em;
            margin-bottom: 6px;
        }
        .explanation ul { margin: 8px 0 0 18px; }
        .explanation li { margin-bottom: 4px; }
        .explanation strong { color: #e0ecf4; }

        .upload-area {
            border: 2px dashed #4fc3f7;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            background: rgba(255,255,255,0.04);
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover { background: rgba(79,195,247,0.1); border-color: #81d4fa; }
        .upload-area.dragover { background: rgba(79,195,247,0.15); border-color: #fff; }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            color: #fff;
        }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .btn-dehaze   { background: #1565c0; }
        .btn-dehaze:hover:not(:disabled) { background: #1976d2; }
        .btn-clahe    { background: #00838f; }
        .btn-clahe:hover:not(:disabled) { background: #0097a7; }
        .btn-gamma    { background: #6a1b9a; }
        .btn-gamma:hover:not(:disabled) { background: #7b1fa2; }
        .btn-process  { background: linear-gradient(135deg, #1565c0, #00838f); font-size: 1.1em; }
        .btn-process:hover:not(:disabled) { background: linear-gradient(135deg, #1976d2, #0097a7); }
        .results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }
        .result-card {
            background: rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }
        .result-card h3 { margin-bottom: 8px; font-size: 0.95em; color: #81d4fa; }
        .result-card img {
            max-width: 100%;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .status {
            text-align: center;
            padding: 12px;
            margin: 12px 0;
            border-radius: 8px;
            font-weight: 500;
        }
        .status.loading { background: rgba(255,183,77,0.15); color: #ffb74d; }
        .status.error   { background: rgba(239,83,80,0.15); color: #ef5350; }
        .status.success  { background: rgba(102,187,106,0.15); color: #66bb6a; }
        .preview-container { margin-bottom: 20px; text-align: center; }
        .preview-container img { max-height: 300px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.15); }
        .params-toggle { color: #4fc3f7; cursor: pointer; text-decoration: underline; margin: 10px 0; display: inline-block; }
        .params-panel {
            display: none;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .params-panel.open { display: block; }
        .param-group { display: flex; flex-wrap: wrap; gap: 16px; }
        .param-item { display: flex; flex-direction: column; gap: 4px; font-size: 0.85em; color: #b0bec5; }
        .param-label-row { display: flex; align-items: center; gap: 5px; }
        .param-item input {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #fff;
            padding: 6px 10px;
            border-radius: 4px;
            width: 105px;
        }
        .seal-emoji { font-size: 1.4em; }

        /* ---- Info icon tooltip ---- */
        .info-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px; height: 16px;
            border-radius: 50%;
            background: rgba(79,195,247,0.25);
            color: #4fc3f7;
            font-size: 11px;
            font-weight: 700;
            cursor: help;
            position: relative;
            flex-shrink: 0;
        }
        .info-icon .tip {
            display: none;
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background: #1a2744;
            border: 1px solid #4fc3f7;
            color: #cfd8e8;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 400;
            white-space: normal;
            width: 230px;
            z-index: 100;
            line-height: 1.4;
            box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            text-align: left;
        }
        .info-icon .tip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: #4fc3f7;
        }
        .info-icon:hover .tip { display: block; }

        /* ---- Params used banner ---- */
        .params-used {
            background: rgba(79,195,247,0.08);
            border: 1px solid rgba(79,195,247,0.2);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 16px;
            font-size: 0.85em;
            line-height: 1.6;
            color: #b0c4d8;
        }
        .params-used strong { color: #81d4fa; }
        .params-used .param-val { color: #4fc3f7; font-family: 'Consolas', 'Courier New', monospace; }
        .params-used .default-tag {
            font-size: 0.75em;
            background: rgba(255,183,77,0.18);
            color: #ffb74d;
            padding: 1px 5px;
            border-radius: 3px;
            margin-left: 4px;
        }
        .params-used .custom-tag {
            font-size: 0.75em;
            background: rgba(102,187,106,0.18);
            color: #66bb6a;
            padding: 1px 5px;
            border-radius: 3px;
            margin-left: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><span class="seal-emoji">🦭</span> Seal Photo Processing</h1>
        <p class="subtitle">AI-Assisted Image Processing &amp; Correction System</p>

        <!-- System Explanation -->
        <details class="explanation" open>
            <summary>About This System</summary>
            <p>
                This tool corrects common defects in seal &amp; pinniped photographs — haze, low contrast, and poor lighting —
                using three image processing algorithms that can be run individually or as a combined pipeline.
                A CNN model trained on the <strong>SealID</strong> dataset (GPU-accelerated, wgpu) provides automatic parameter estimation;
                you can also set parameters manually.
            </p>
            <ul>
                <li><strong>DCP Dehaze</strong> — Dark Channel Prior removes atmospheric haze, fog, and mist by estimating a scene transmission map and atmospheric light.</li>
                <li><strong>CLAHE Enhance</strong> — Contrast Limited Adaptive Histogram Equalization boosts local contrast in tile regions without over-amplifying noise.</li>
                <li><strong>Gamma Correct</strong> — Adjusts overall brightness via a power-law curve. Values &lt;1 brighten; values &gt;1 darken. Auto-estimated from mean luminance when not set.</li>
                <li><strong>Full Pipeline</strong> — Runs all three in sequence (DCP &rarr; CLAHE &rarr; Gamma) with attenuated stacking so corrections don't over-compound.</li>
            </ul>
            <p style="margin-top:8px;color:#78909c;font-size:0.9em;">
                Built in <strong>Rust</strong> &middot; Barrett Honors Thesis, ASU &middot; Upload a photo below to get started!
            </p>
        </details>

        <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
            <p style="font-size:1.2em; margin-bottom:8px;">📷 Drop an image here or click to upload</p>
            <p style="color:#78909c; font-size:0.9em;">Supports JPEG, PNG, BMP, TIFF</p>
        </div>

        <div class="preview-container" id="previewContainer" style="display:none">
            <h3 style="margin-bottom:8px; color:#b0bec5;">Original Image</h3>
            <img id="previewImg" alt="Preview">
        </div>

        <span class="params-toggle" onclick="toggleParams()">⚙️ Custom Parameters (optional — leave blank for smart defaults)</span>
        <div class="params-panel" id="paramsPanel">
            <div class="param-group">
                <!-- DCP params -->
                <div class="param-item">
                    <span class="param-label-row">DCP omega
                        <span class="info-icon">?<span class="tip">Haze removal strength (0-1). Higher = more haze removed. 0.75 is a balanced default for seal photos; lower values preserve more of the original atmosphere.</span></span>
                    </span>
                    <input type="number" id="p_omega" step="0.05" placeholder="0.75">
                </div>
                <div class="param-item">
                    <span class="param-label-row">DCP t0
                        <span class="info-icon">?<span class="tip">Minimum transmission floor (0-1). Prevents over-dehazing in very hazy regions. 0.25 keeps dark areas from blowing out. Lower = more aggressive.</span></span>
                    </span>
                    <input type="number" id="p_t0" step="0.05" placeholder="0.25">
                </div>
                <div class="param-item">
                    <span class="param-label-row">Patch size
                        <span class="info-icon">?<span class="tip">Size of the local patch (in pixels) for computing the dark channel. Larger patches handle thicker haze but may blur edges. 15 is standard.</span></span>
                    </span>
                    <input type="number" id="p_patch" step="1" min="3" placeholder="15">
                </div>
                <div class="param-item">
                    <span class="param-label-row">Guide radius
                        <span class="info-icon">?<span class="tip">Radius for the guided filter that refines the transmission map. Larger = smoother transmission. 15 matches patch size by default.</span></span>
                    </span>
                    <input type="number" id="p_gradius" step="1" min="1" placeholder="15">
                </div>
                <div class="param-item">
                    <span class="param-label-row">Guide eps
                        <span class="info-icon">?<span class="tip">Regularization epsilon for the guided filter. Smaller values preserve more edges; larger values smooth more. 0.0001 is a good starting point.</span></span>
                    </span>
                    <input type="number" id="p_geps" step="0.0001" placeholder="0.0001">
                </div>
                <!-- CLAHE params -->
                <div class="param-item">
                    <span class="param-label-row">CLAHE grid H
                        <span class="info-icon">?<span class="tip">Number of horizontal tiles for CLAHE. More tiles = finer local contrast control, but slower. 8 is standard.</span></span>
                    </span>
                    <input type="number" id="p_gridh" step="1" min="1" placeholder="8">
                </div>
                <div class="param-item">
                    <span class="param-label-row">CLAHE grid W
                        <span class="info-icon">?<span class="tip">Number of vertical tiles for CLAHE. More tiles = finer local contrast. 8 is standard; use more for large images.</span></span>
                    </span>
                    <input type="number" id="p_gridw" step="1" min="1" placeholder="8">
                </div>
                <div class="param-item">
                    <span class="param-label-row">CLAHE clip
                        <span class="info-icon">?<span class="tip">Clip limit for histogram equalization. Limits contrast amplification to prevent noise. 2.0 is moderate; higher = more contrast but more noise.</span></span>
                    </span>
                    <input type="number" id="p_clip" step="0.1" min="0.1" placeholder="2.0">
                </div>
                <!-- Gamma -->
                <div class="param-item">
                    <span class="param-label-row">Gamma
                        <span class="info-icon">?<span class="tip">Power-law exponent for brightness. &lt;1 brightens the image, &gt;1 darkens it, 1.0 = no change. Leave blank to auto-estimate from mean luminance.</span></span>
                    </span>
                    <input type="number" id="p_gamma" step="0.05" placeholder="auto">
                </div>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-process" id="btnProcess" disabled onclick="callApi('/api/process')">🦭 Full Pipeline (DCP+CLAHE+Gamma)</button>
            <button class="btn btn-dehaze" id="btnDehaze" disabled onclick="callApi('/api/dehaze')">🌫️ DCP Dehaze</button>
            <button class="btn btn-clahe" id="btnClahe" disabled onclick="callApi('/api/clahe')">📊 CLAHE Enhance</button>
            <button class="btn btn-gamma" id="btnGamma" disabled onclick="callApi('/api/gamma')">☀️ Gamma Correct</button>
        </div>

        <div id="statusArea"></div>
        <div id="paramsUsedArea"></div>
        <div class="results" id="resultsArea"></div>
    </div>

    <script>
        let currentBase64 = null;

        // Drag and drop
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                const dataUrl = e.target.result;
                currentBase64 = dataUrl.split(',')[1];
                document.getElementById('previewContainer').style.display = 'block';
                document.getElementById('previewImg').src = dataUrl;
                document.querySelectorAll('.btn').forEach(b => b.disabled = false);
                document.getElementById('resultsArea').innerHTML = '';
                document.getElementById('paramsUsedArea').innerHTML = '';
                setStatus('Image loaded! Choose a processing operation below.', 'success');
            };
            reader.readAsDataURL(file);
        }

        function toggleParams() {
            document.getElementById('paramsPanel').classList.toggle('open');
        }

        // Which params the user actually typed in
        const paramFields = [
            ['omega', 'p_omega'], ['t0', 'p_t0'], ['patch_size', 'p_patch'],
            ['guided_radius', 'p_gradius'], ['guided_eps', 'p_geps'],
            ['grid_h', 'p_gridh'], ['grid_w', 'p_gridw'],
            ['clip_limit', 'p_clip'], ['gamma', 'p_gamma']
        ];

        function getParams() {
            const p = {};
            for (const [key, id] of paramFields) {
                const val = document.getElementById(id).value;
                if (val !== '') p[key] = parseFloat(val);
            }
            return p;
        }

        function getUserSetKeys() {
            const s = new Set();
            for (const [key, id] of paramFields) {
                if (document.getElementById(id).value !== '') s.add(key);
            }
            return s;
        }

        function setStatus(msg, type) {
            document.getElementById('statusArea').innerHTML =
                '<div class="status ' + type + '">' + msg + '</div>';
        }

        // Human-friendly names & descriptions for the params-used banner
        const paramMeta = {
            omega:          'DCP omega',
            t0:             'DCP t\u2080 (min transmission)',
            patch_size:     'Patch size',
            top_percent:    'Top % for atmospheric light',
            guided_radius:  'Guided filter radius',
            guided_eps:     'Guided filter \u03b5',
            grid_h:         'CLAHE grid rows',
            grid_w:         'CLAHE grid cols',
            clip_limit:     'CLAHE clip limit',
            stacked_clip:   'CLAHE clip (stacked)',
            gamma:          'Gamma',
            stacked_gamma:  'Gamma (stacked)',
            auto_estimated: 'Auto-estimated'
        };

        function showParamsUsed(params, userKeys) {
            if (!params || Object.keys(params).length === 0) {
                document.getElementById('paramsUsedArea').innerHTML = '';
                return;
            }
            let html = '<div class="params-used"><strong>Parameters Used:</strong><br>';
            for (const [k, v] of Object.entries(params)) {
                if (k === 'auto_estimated') continue;
                const name = paramMeta[k] || k;
                const isDefault = !userKeys.has(k);
                const tag = isDefault
                    ? '<span class="default-tag">default</span>'
                    : '<span class="custom-tag">custom</span>';
                const val = (typeof v === 'number') ? (Number.isInteger(v) ? v : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, '')) : v;
                html += name + ': <span class="param-val">' + val + '</span>' + tag + '&ensp; ';
            }
            html += '</div>';
            document.getElementById('paramsUsedArea').innerHTML = html;
        }

        async function callApi(endpoint) {
            if (!currentBase64) return;
            const userKeys = getUserSetKeys();
            document.querySelectorAll('.btn').forEach(b => b.disabled = true);
            setStatus('Processing with ' + endpoint + '... this may take a moment for large images.', 'loading');
            document.getElementById('paramsUsedArea').innerHTML = '';

            const payload = Object.assign({ image: currentBase64 }, getParams());

            try {
                const t0 = performance.now();
                const resp = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

                if (!resp.ok) {
                    const err = await resp.text();
                    throw new Error(err);
                }
                const data = await resp.json();
                setStatus('Done in ' + elapsed + 's!', 'success');
                showParamsUsed(data.params || {}, userKeys);
                showResults(data, endpoint);
            } catch (e) {
                setStatus('Error: ' + e.message, 'error');
            } finally {
                document.querySelectorAll('.btn').forEach(b => b.disabled = false);
            }
        }

        function showResults(data, endpoint) {
            const area = document.getElementById('resultsArea');
            area.innerHTML = '';

            function addCard(title, b64) {
                if (!b64) return;
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = '<h3>' + title + '</h3>'
                    + '<img src="data:image/jpeg;base64,' + b64 + '" alt="' + title + '">'
                    + '<br><a href="data:image/jpeg;base64,' + b64 + '" download="' + title.replace(/ /g,'_') + '.jpg" style="color:#4fc3f7;font-size:0.85em;">Download</a>';
                area.appendChild(card);
            }

            if (endpoint === '/api/process') {
                addCard('Full Pipeline (DCP+CLAHE+Gamma)', data.image);
                addCard('DCP Dehaze Only', data.dehaze_only);
                addCard('CLAHE Only', data.clahe_only);
                addCard('Gamma Only', data.gamma_only);
            } else {
                const opNames = {
                    '/api/dehaze': 'DCP Dehazed',
                    '/api/clahe': 'CLAHE Enhanced',
                    '/api/gamma': 'Gamma Corrected'
                };
                addCard(opNames[endpoint] || 'Result', data.image);
            }
        }
    </script>
</body>
</html>"#;

