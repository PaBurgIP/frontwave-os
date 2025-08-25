# app/main.py
# -*- coding: utf-8 -*-
import os, uuid, shutil, io, zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
import geopandas as gpd

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.frontwave_os import run_frontwave

BASE = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE / "static"
OUT_BASE = STATIC_DIR / "results"
OUT_BASE.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="FrontWave API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html", status_code=307)

def _to_geojson(gpkg_path: str, layer: str, out_geojson: Path) -> Optional[str]:
    try:
        if not gpkg_path or not os.path.exists(gpkg_path):
            return None
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if len(gdf) == 0:
            return None
        gdf = gdf.to_crs(4326)
        gdf.to_file(out_geojson, driver="GeoJSON")
        return str(out_geojson)
    except Exception:
        return None

def _raster_minmax(path: str) -> Optional[Tuple[float, float]]:
    try:
        with rasterio.open(path) as src:
            arr = src.read(1, masked=True)
            if arr.mask.all():
                return None
            vmin = float(np.nanmin(arr.filled(np.nan)))
            vmax = float(np.nanmax(arr.filled(np.nan)))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                return None
            return vmin, vmax
    except Exception:
        return None

@app.post("/run")
async def run(
    csv: UploadFile = File(...),
    grid_cell_m: float = Form(12000.0),
    krige_cell_m: float = Form(1200.0),
    contour_interval: float = Form(30.0),
    dayfirst: bool = Form(True),
):
    job = str(uuid.uuid4())[:8]
    job_dir = OUT_BASE / job
    job_dir.mkdir(parents=True, exist_ok=True)

    csv_path = job_dir / csv.filename
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(csv.file, f)

    res = run_frontwave(
        str(csv_path), str(job_dir),
        grid_cell_m=grid_cell_m,
        krige_cell_m=krige_cell_m,
        contour_interval=contour_interval,
        dayfirst=dayfirst,
        sep=';'
    )

    contours_geojson = _to_geojson(res.get("contours",""), "contours", job_dir / "contours.geojson")
    ellipse_geojson  = _to_geojson(res.get("ellipse",""), "ellipse", job_dir / "ellipse.geojson")
    selpts_geojson   = _to_geojson(res.get("selected_points",""), "selected_pts", job_dir / "selected_points.geojson")
    grid_geojson     = _to_geojson(res.get("grid",""), "grid", job_dir / "grid.geojson")

    krig_stats = _raster_minmax(res.get("kriging","")) or (None, None)
    slope_stats = _raster_minmax(res.get("slope","")) or (None, None)
    vel_stats = _raster_minmax(res.get("velocity","")) or (None, None)

    base = f"/static/results/{job}/"
    urls = {
        "kriging":  base + Path(res["kriging"]).name  if res.get("kriging")  else None,
        "slope":    base + Path(res["slope"]).name    if res.get("slope")    else None,
        "velocity": base + Path(res["velocity"]).name if res.get("velocity") else None,
        "contours": base + "contours.geojson"         if contours_geojson     else None,
        "ellipse":  base + "ellipse.geojson"          if ellipse_geojson      else None,
        "selected_points": base + "selected_points.geojson" if selpts_geojson else None,
        "grid":     base + "grid.geojson"             if grid_geojson         else None,
    }
    stats = {
        "kriging":  {"min": krig_stats[0], "max": krig_stats[1]},
        "slope":    {"min": slope_stats[0], "max": slope_stats[1]},
        "velocity": {"min": vel_stats[0], "max": vel_stats[1]},
    }
    return JSONResponse({"job": job, "urls": urls, "stats": stats})

@app.get("/download/{job}")
def download_job(job: str):
    job_dir = OUT_BASE / job
    if not job_dir.exists():
        return JSONResponse({"error": "job no encontrado"}, status_code=404)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in job_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(job_dir))
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="frontwave_{job}.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
