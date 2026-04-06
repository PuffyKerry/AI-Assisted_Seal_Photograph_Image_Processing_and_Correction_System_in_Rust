# ============================================================
# Seal IP Web Server — Full API Test Suite
# Tests all endpoints: health, dehaze, clahe, gamma, process
# ============================================================
$ErrorActionPreference = "Continue"
$base = "http://localhost:8080"
$imgDir = "C:\Users\user\RustroverProjects\AI-Assisted_Seal_Photograph_Image_Processing_and_Correction_System_in_Rust"
$passed = 0
$failed = 0
$total = 0

function Test-Endpoint {
    param([string]$Name, [scriptblock]$Block)
    $script:total++
    Write-Host "`n--- TEST $script:total : $Name ---" -ForegroundColor Cyan
    try {
        $result = & $Block
        if ($result) {
            $script:passed++
            Write-Host "  PASS" -ForegroundColor Green
        } else {
            $script:failed++
            Write-Host "  FAIL (returned falsy)" -ForegroundColor Red
        }
    } catch {
        $script:failed++
        Write-Host "  FAIL: $_" -ForegroundColor Red
    }
}

Write-Host "============================================" -ForegroundColor Yellow
Write-Host "  Seal IP Web Server - Full Test Suite" -ForegroundColor Yellow
Write-Host "============================================" -ForegroundColor Yellow

# ---- TEST 1: Health check ----
Test-Endpoint "GET /api/health" {
    $r = Invoke-RestMethod -Uri "$base/api/health" -Method GET -TimeoutSec 10
    Write-Host "  Status: $($r.status), Service: $($r.service)"
    Write-Host "  Endpoints: $($r.endpoints -join ', ')"
    return $r.status -eq "ok"
}

# ---- TEST 2: Home page HTML ----
Test-Endpoint "GET / (upload UI page)" {
    $r = Invoke-WebRequest -Uri "$base/" -Method GET -TimeoutSec 10 -UseBasicParsing
    $hasTitle = $r.Content -match "Seal Photo Processing"
    $hasUpload = $r.Content -match "uploadArea"
    Write-Host "  HTML length: $($r.Content.Length) chars"
    Write-Host "  Has title: $hasTitle, Has upload UI: $hasUpload"
    return ($r.StatusCode -eq 200) -and $hasTitle -and $hasUpload
}

# ---- Load test image ----
$testImg = Join-Path $imgDir "bansui.jpg"
if (-not (Test-Path $testImg)) {
    Write-Host "`nWARNING: bansui.jpg not found, trying fog image..." -ForegroundColor Yellow
    $testImg = Join-Path $imgDir "fog-137794231410y.jpg"
}
if (-not (Test-Path $testImg)) {
    Write-Host "ERROR: No test images found!" -ForegroundColor Red
    exit 1
}
Write-Host "`nUsing test image: $testImg" -ForegroundColor Gray
$imgBytes = [System.IO.File]::ReadAllBytes($testImg)
$b64 = [Convert]::ToBase64String($imgBytes)
Write-Host "  Image size: $([math]::Round($imgBytes.Length / 1024)) KB, base64 length: $($b64.Length)" -ForegroundColor Gray

# ---- TEST 3: CLAHE (fastest, test this first) ----
Test-Endpoint "POST /api/clahe (default params)" {
    $json = '{"image":"' + $b64 + '"}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/clahe" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 120
    $sw.Stop()
    Write-Host "  Operation: $($r.operation), Size: $($r.width)x$($r.height)"
    Write-Host "  Result base64: $($r.image.Length) chars"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    # Save result to verify visually
    $outBytes = [Convert]::FromBase64String($r.image)
    [System.IO.File]::WriteAllBytes((Join-Path $imgDir "test_output_clahe.jpg"), $outBytes)
    Write-Host "  Saved to: test_output_clahe.jpg"
    return ($r.operation -eq "clahe") -and ($r.image.Length -gt 100)
}

# ---- TEST 4: CLAHE with custom params ----
Test-Endpoint "POST /api/clahe (custom: grid 4x4, clip 3.0)" {
    $json = '{"image":"' + $b64 + '","grid_h":4,"grid_w":4,"clip_limit":3.0}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/clahe" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 120
    $sw.Stop()
    Write-Host "  Operation: $($r.operation), Size: $($r.width)x$($r.height)"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    return ($r.operation -eq "clahe") -and ($r.image.Length -gt 100)
}

# ---- TEST 5: Gamma (should be fast) ----
Test-Endpoint "POST /api/gamma (auto-estimated)" {
    $json = '{"image":"' + $b64 + '"}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/gamma" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 120
    $sw.Stop()
    Write-Host "  Operation: $($r.operation), Size: $($r.width)x$($r.height)"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    $outBytes = [Convert]::FromBase64String($r.image)
    [System.IO.File]::WriteAllBytes((Join-Path $imgDir "test_output_gamma.jpg"), $outBytes)
    Write-Host "  Saved to: test_output_gamma.jpg"
    return ($r.operation -eq "gamma") -and ($r.image.Length -gt 100)
}

# ---- TEST 6: Gamma with custom value ----
Test-Endpoint "POST /api/gamma (custom: 0.6 brighten)" {
    $json = '{"image":"' + $b64 + '","gamma":0.6}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/gamma" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 120
    $sw.Stop()
    Write-Host "  Operation: $($r.operation)"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    return ($r.operation -eq "gamma") -and ($r.image.Length -gt 100)
}

# ---- TEST 7: DCP Dehaze (slowest single op) ----
Test-Endpoint "POST /api/dehaze (default params)" {
    $json = '{"image":"' + $b64 + '"}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/dehaze" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 300
    $sw.Stop()
    Write-Host "  Operation: $($r.operation), Size: $($r.width)x$($r.height)"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    $outBytes = [Convert]::FromBase64String($r.image)
    [System.IO.File]::WriteAllBytes((Join-Path $imgDir "test_output_dehaze.jpg"), $outBytes)
    Write-Host "  Saved to: test_output_dehaze.jpg"
    return ($r.operation -eq "dehaze") -and ($r.image.Length -gt 100)
}

# ---- TEST 8: DCP Dehaze with custom params ----
Test-Endpoint "POST /api/dehaze (custom: omega=0.75, t0=0.25)" {
    $json = '{"image":"' + $b64 + '","omega":0.75,"t0":0.25,"patch_size":15}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/dehaze" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 300
    $sw.Stop()
    Write-Host "  Operation: $($r.operation)"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    return ($r.operation -eq "dehaze") -and ($r.image.Length -gt 100)
}

# ---- TEST 9: Full Pipeline ----
Test-Endpoint "POST /api/process (full pipeline: DCP+CLAHE+Gamma)" {
    $json = '{"image":"' + $b64 + '"}'
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "$base/api/process" -Method POST -Body $json -ContentType "application/json" -TimeoutSec 600
    $sw.Stop()
    Write-Host "  Operation: $($r.operation)"
    Write-Host "  Size: $($r.width)x$($r.height)"
    Write-Host "  Has dehaze_only: $($r.dehaze_only.Length -gt 0)"
    Write-Host "  Has clahe_only: $($r.clahe_only.Length -gt 0)"
    Write-Host "  Has gamma_only: $($r.gamma_only.Length -gt 0)"
    Write-Host "  Time: $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
    $outBytes = [Convert]::FromBase64String($r.image)
    [System.IO.File]::WriteAllBytes((Join-Path $imgDir "test_output_full_pipeline.jpg"), $outBytes)
    Write-Host "  Saved to: test_output_full_pipeline.jpg"
    return ($r.operation -eq "full_pipeline") -and ($r.image.Length -gt 100)
}

# ---- TEST 10: 404 handling ----
Test-Endpoint "GET /api/nonexistent (should 404)" {
    try {
        $r = Invoke-WebRequest -Uri "$base/api/nonexistent" -Method GET -TimeoutSec 5 -UseBasicParsing
        Write-Host "  Got status $($r.StatusCode) (expected 404)"
        return $false
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Write-Host "  Got status: $status"
        return $status -eq 404
    }
}

# ---- TEST 11: Bad request handling ----
Test-Endpoint "POST /api/clahe with bad JSON (should error gracefully)" {
    try {
        $r = Invoke-WebRequest -Uri "$base/api/clahe" -Method POST -Body '{"no_image":"oops"}' -ContentType "application/json" -TimeoutSec 10 -UseBasicParsing
        Write-Host "  Got status $($r.StatusCode)"
        return $false  # Should have been an error
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        Write-Host "  Got status: $status (expected 400)"
        return $status -eq 400
    }
}

# ---- SUMMARY ----
Write-Host "`n============================================" -ForegroundColor Yellow
Write-Host "  TEST RESULTS: $passed/$total passed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Red" })
if ($failed -eq 0) {
    Write-Host "  ALL TESTS PASSED!" -ForegroundColor Green
} else {
    Write-Host "  $failed test(s) FAILED" -ForegroundColor Red
}
Write-Host "============================================" -ForegroundColor Yellow
Write-Host "`nOutput images saved to project root:" -ForegroundColor Gray
Write-Host "  test_output_clahe.jpg" -ForegroundColor Gray
Write-Host "  test_output_gamma.jpg" -ForegroundColor Gray
Write-Host "  test_output_dehaze.jpg" -ForegroundColor Gray
Write-Host "  test_output_full_pipeline.jpg" -ForegroundColor Gray
Write-Host "Open them to visually verify the processing looks correct!" -ForegroundColor Gray
Write-Host "`nFor the live demo, open: http://localhost:8080" -ForegroundColor Cyan




