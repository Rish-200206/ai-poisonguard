"""
AI PoisonGuard — FastAPI Backend

Multi-layer adversarial training data poisoning detection API
for Indian fintech and government ML systems.

Endpoints:
  POST /api/analyze       — Full multi-layer analysis (CSV + optional model)
  GET  /api/profiles      — List available domain risk profiles
  GET  /api/health        — Health check
"""

import io
import logging
import pickle
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from detection.statistical import StatisticalDetector
from detection.spectral import SpectralDetector
from detection.clustering import ClusteringDetector
from detection.art_integration import ARTDetector, _check_art
from detection.influence import InfluenceFunctionDetector
from detection.backdoor import BackdoorTriggerDetector
from detection.clustering import _check_umap, _check_hdbscan
from domain_profiles import get_profile, list_profiles, DOMAIN_PROFILES
from models import AnalysisResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("poisonguard")

# ─── App ────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI PoisonGuard API",
    description="Adversarial Training Data Poisoning Detector for Indian Fintech & Government ML Systems",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ─────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        art_available=_check_art(),
        umap_available=_check_umap(),
        hdbscan_available=_check_hdbscan(),
    )


# ─── Profiles ───────────────────────────────────────────────────────
@app.get("/api/profiles")
async def get_profiles():
    return {"profiles": list_profiles()}


# ─── Model Loader Helpers ──────────────────────────────────────────
def _load_pkl_model(model_bytes: bytes) -> tuple:
    """Load a scikit-learn .pkl model."""
    model = pickle.loads(model_bytes)
    return model, {
        "type": type(model).__name__,
        "module": type(model).__module__,
    }


def _load_h5_model(model_bytes: bytes) -> tuple:
    """Load a Keras .h5 model."""
    import tempfile, os
    try:
        from tensorflow import keras
    except ImportError:
        try:
            import keras
        except ImportError:
            raise ImportError("keras/tensorflow not installed. Cannot load .h5 models.")
    # Write to temp file since keras loads from path
    tmp = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    try:
        tmp.write(model_bytes)
        tmp.close()
        model = keras.models.load_model(tmp.name)
        info = {
            "type": "KerasModel",
            "module": "keras",
            "layers": len(model.layers) if hasattr(model, 'layers') else 0,
        }
        return model, info
    finally:
        os.unlink(tmp.name)


def _load_onnx_model(model_bytes: bytes) -> tuple:
    """Load an ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Cannot load .onnx models.")
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    try:
        tmp.write(model_bytes)
        tmp.close()
        session = ort.InferenceSession(tmp.name)
        info = {
            "type": "ONNXModel",
            "module": "onnxruntime",
            "inputs": [inp.name for inp in session.get_inputs()],
            "outputs": [out.name for out in session.get_outputs()],
        }
        return session, info
    finally:
        os.unlink(tmp.name)


# ─── Main Analysis ──────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze_dataset(
    dataset: UploadFile = File(...),
    model_file: UploadFile | None = File(default=None),
    domain: str = Form(default="general"),
    label_column: str = Form(default=""),
):
    """
    Run multi-layer poisoning detection on an uploaded dataset.

    - **dataset**: CSV file with features (and optional label column).
    - **model_file**: Optional .pkl/.h5/.onnx model for deeper analysis.
    - **domain**: Domain profile ID (upi_fraud, credit_scoring, kyc_govt_welfare, general).
    - **label_column**: Name of the label column in the CSV.
    """
    t_start = time.time()

    # ── Validate CSV ──
    if not dataset.filename.endswith(".csv"):
        raise HTTPException(400, "Dataset must be a .csv file.")

    try:
        contents = await dataset.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(400, "Dataset is empty.")

    # ── Extract features and labels ──
    y = None
    label_col = label_column.strip()

    # Auto-detect label column if not specified
    if not label_col:
        for candidate in ["target", "label", "class", "y", "is_fraud", "fraud", "default"]:
            if candidate in df.columns:
                label_col = candidate
                break

    if label_col and label_col in df.columns:
        y = df[label_col].values
        df_features = df.drop(columns=[label_col])
    else:
        df_features = df.copy()
        label_col = ""

    # Keep only numeric features
    df_numeric = df_features.select_dtypes(include=[np.number])
    if df_numeric.empty:
        raise HTTPException(400, "No numeric feature columns found in dataset.")

    df_numeric = df_numeric.fillna(df_numeric.mean())
    X = df_numeric.to_numpy()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape
    feature_names = df_numeric.columns.tolist()

    # ── Load domain profile ──
    profile = get_profile(domain)
    logger.info(f"Using domain profile: {profile['name']} ({domain})")

    # ── Load model if provided ──
    model = None
    model_info = None
    if model_file and model_file.filename:
        fname = model_file.filename.lower()
        SUPPORTED_FORMATS = (".pkl", ".h5", ".onnx")
        if not any(fname.endswith(ext) for ext in SUPPORTED_FORMATS):
            raise HTTPException(400, f"Model file must be one of: {', '.join(SUPPORTED_FORMATS)}")
        try:
            model_bytes = await model_file.read()
            if fname.endswith(".pkl"):
                model, info = _load_pkl_model(model_bytes)
            elif fname.endswith(".h5"):
                model, info = _load_h5_model(model_bytes)
            elif fname.endswith(".onnx"):
                model, info = _load_onnx_model(model_bytes)
            model_info = {"filename": model_file.filename, **info}
            logger.info(f"Loaded model: {model_info}")
        except ImportError as e:
            logger.warning(f"Model loader dependency missing: {e}")
            model_info = {"filename": model_file.filename, "error": str(e)}
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            model_info = {"filename": model_file.filename, "error": str(e)}

    # ── Make y integer-encoded if possible ──
    if y is not None:
        try:
            y_unique = np.unique(y)
            if len(y_unique) <= 50:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    # RUN DETECTION LAYERS
    # ═══════════════════════════════════════════════════════════════

    # Layer 1: Statistical Analysis (Z-score + IQR)
    logger.info("Running Layer 1: Statistical Analysis...")
    stat_detector = StatisticalDetector(
        z_threshold=profile["z_threshold"],
        iqr_multiplier=profile["iqr_multiplier"],
        min_anomaly_features=profile["min_anomaly_features"],
    )
    stat_result = stat_detector.detect(X, y)

    # Layer 2: Spectral Signature (SVD)
    logger.info("Running Layer 2: Spectral Signature Analysis...")
    spec_detector = SpectralDetector(
        top_k_singular=profile["spectral_top_k"],
        percentile_threshold=profile["spectral_percentile"],
        mad_multiplier=profile.get("mad_multiplier", 3.5),
        min_score_floor=profile.get("min_score_floor", 0.3),
    )
    spec_result = spec_detector.detect(X, y)

    # Layer 3: Activation Clustering (UMAP + KMeans)
    logger.info("Running Layer 3: Activation Clustering...")
    clust_detector = ClusteringDetector(
        n_clusters=profile["n_clusters"],
        purity_threshold=profile["purity_threshold"],
        umap_n_neighbors=profile["umap_n_neighbors"],
        umap_min_dist=profile["umap_min_dist"],
        auto_k=True,
        max_cluster_frac=profile.get("max_cluster_frac", 0.25),
        min_impurity_margin=profile.get("min_impurity_margin", 0.10),
    )
    clust_result = clust_detector.detect(X, y)

    # Layer 4: IBM ART
    logger.info("Running Layer 4: IBM ART Validation...")
    art_detector = ARTDetector()
    art_result = art_detector.detect(X, y, model)

    # Layer 5: Influence Function Analysis
    logger.info("Running Layer 5: Influence Function Analysis...")
    influence_detector = InfluenceFunctionDetector(
        mad_multiplier=profile.get("mad_multiplier", 3.5),
        min_score_floor=profile.get("min_score_floor", 0.3),
    )
    influence_result = influence_detector.detect(X, y, model)

    # Layer 6: Backdoor Trigger Scanning
    logger.info("Running Layer 6: Backdoor Trigger Scanning...")
    backdoor_detector = BackdoorTriggerDetector(
        mad_multiplier=profile.get("mad_multiplier", 3.5),
        min_score_floor=profile.get("min_score_floor", 0.3),
    )
    backdoor_result = backdoor_detector.detect(X, y, model)

    # ═══════════════════════════════════════════════════════════════
    # ENSEMBLE VOTING AGGREGATION
    # ═══════════════════════════════════════════════════════════════
    #
    # Instead of simple OR logic (any single flag → poisoned), we use
    # an ensemble voting system to reduce false positives:
    #   - A sample must be flagged by ≥2 layers to be "Suspected Poison"
    #   - Each sample gets a weighted risk score based on layer confidence
    #   - Layer 2 (Spectral) and Layer 3 (Clustering) get higher weights
    #     because they are more structurally robust detectors
    #   - Layer 1 (Statistical Z-score) gets a lower confidence weight
    #     because it is inherently noisier
    # ═══════════════════════════════════════════════════════════════

    ENSEMBLE_VOTE_THRESHOLD = 2  # Minimum layers that must flag a sample

    # Layer confidence weights for ensemble voting
    # Higher weight = more trustworthy signal
    LAYER_CONFIDENCE_WEIGHTS = {
        "statistical": 0.10,   # Noisiest — Z-score produces many false positives
        "spectral":    0.25,   # SVD-based, structurally robust
        "clustering":  0.25,   # UMAP+KMeans, structurally robust
        "art":         0.15,   # IBM ART standardised detection
        "influence":   0.15,   # Influence function analysis
        "backdoor":    0.10,   # Backdoor trigger scanning
    }

    weights = profile["layer_weights"]

    # All layer results keyed by name
    layer_results = {
        "statistical": stat_result,
        "spectral":    spec_result,
        "clustering":  clust_result,
        "art":         art_result,
        "influence":   influence_result,
        "backdoor":    backdoor_result,
    }
    all_results = list(layer_results.values())

    LAYER_DISPLAY_NAMES = {
        "statistical": "Statistical Analysis",
        "spectral":    "Spectral Signatures",
        "clustering":  "Activation Clustering",
        "art":         "IBM ART",
        "influence":   "Influence Functions",
        "backdoor":    "Backdoor Trigger Scan",
    }

    # ── Per-sample composite score (weighted sum across all layers) ──
    composite_scores = np.zeros(n_samples)
    for layer_key, result in layer_results.items():
        w = weights.get(layer_key, 0.10)
        scores = np.array(result["scores"])
        composite_scores += w * scores

    # Normalise to 0-1
    if composite_scores.max() > 0:
        composite_scores = composite_scores / composite_scores.max()

    # ── Per-sample vote counting ──
    # For each sample, count how many layers flagged it and compute
    # a weighted ensemble score using confidence weights.
    sample_vote_count = np.zeros(n_samples, dtype=int)
    sample_weighted_score = np.zeros(n_samples)
    sample_flagging_layers = {i: [] for i in range(n_samples)}

    for layer_key, result in layer_results.items():
        confidence_w = LAYER_CONFIDENCE_WEIGHTS[layer_key]
        for idx in result["flagged_indices"]:
            if idx < n_samples:
                sample_vote_count[idx] += 1
                sample_weighted_score[idx] += confidence_w
                sample_flagging_layers[idx].append(layer_key)

    # ── Apply ensemble threshold ──
    # Only samples with votes >= threshold are officially "Suspected Poison"
    ensemble_flagged = set(
        int(i) for i in range(n_samples)
        if sample_vote_count[i] >= ENSEMBLE_VOTE_THRESHOLD
    )

    # Samples with exactly 1 flag are kept as "warnings" (shown but not
    # counted as poisoned) to preserve transparency.
    single_flag_warnings = set(
        int(i) for i in range(n_samples)
        if sample_vote_count[i] == 1
    )

    n_flagged = len(ensemble_flagged)
    n_warnings = len(single_flag_warnings)
    flagged_ratio = n_flagged / n_samples

    def _compute_sample_risk(
        vote_count: int, weighted_score: float
    ) -> tuple:
        """
        Compute per-sample risk category and percentage score
        based on ensemble votes.

        Returns (risk_pct, risk_category)
        """
        # Normalise weighted_score to 0-100 range
        # Max possible weighted_score is sum of all confidence weights = 1.0
        raw_pct = min(100.0, weighted_score * 100)

        if vote_count == 0:
            return (0.0, "Clean")
        elif vote_count == 1:
            return (max(raw_pct, 20.0), "Warning")
        elif vote_count == 2:
            return (max(raw_pct, 40.0), "Compromised")
        elif vote_count == 3:
            return (max(raw_pct, 60.0), "High Risk")
        else:  # 4+ layers
            return (max(raw_pct, 80.0), "Critical")

    # ── Overall risk score (0-100) ──
    # Weighted combination of:
    #   - Ensemble-flagged contamination rate (how many samples pass threshold)
    #   - Mean composite score of flagged samples
    #   - Number of active detection layers that found something
    if n_flagged > 0:
        mean_flagged_composite = np.mean(
            [composite_scores[i] for i in ensemble_flagged]
        )
    else:
        mean_flagged_composite = 0.0

    risk_score = round(
        min(100,
            flagged_ratio * 100 * 3          # Contamination contribution
            + mean_flagged_composite * 50    # Severity contribution
            + composite_scores.mean() * 20   # Global noise contribution
        ),
        1,
    )

    # Risk level
    if risk_score < 15:
        risk_level = "LOW"
        status = "clean"
    elif risk_score < 40:
        risk_level = "MEDIUM"
        status = "suspicious"
    elif risk_score < 70:
        risk_level = "HIGH"
        status = "compromised"
    else:
        risk_level = "CRITICAL"
        status = "compromised"

    # ── Scatter data (from UMAP/clustering coords) ──
    umap_coords = clust_result.get("umap_coords", [])
    cluster_labels = clust_result.get("cluster_labels", [])

    scatter_data = []
    for i in range(n_samples):
        coord = umap_coords[i] if i < len(umap_coords) else [0, 0]
        vote_c = int(sample_vote_count[i])
        risk_pct, risk_cat = _compute_sample_risk(
            vote_c, float(sample_weighted_score[i])
        )
        scatter_data.append({
            "index": i,
            "x": float(coord[0]),
            "y": float(coord[1]),
            "cluster": int(cluster_labels[i]) if i < len(cluster_labels) else 0,
            "is_poisoned": i in ensemble_flagged,   # Ensemble threshold applied
            "is_warning": i in single_flag_warnings,
            "score": float(composite_scores[i]),
            "vote_count": vote_c,
            "ensemble_risk_score": round(risk_pct, 1),
            "risk_category": risk_cat,
            "label": int(y[i]) if y is not None and i < len(y) else None,
        })

    # ── Flagged samples table with RISK REASONS ──
    # Include both ensemble-flagged AND single-flag-warnings for transparency
    all_reportable = ensemble_flagged | single_flag_warnings

    flagged_samples = []
    for idx in sorted(all_reportable):
        layers_flagging = sample_flagging_layers.get(idx, [])
        risk_reasons = []

        if "statistical" in layers_flagging:
            z_max = stat_result.get("z_scores_max", [])
            if idx < len(z_max):
                z_val = z_max[idx]
                risk_reasons.append(
                    f"Z-score outlier (max z={z_val:.2f}, threshold={profile['z_threshold']})"
                )

        if "spectral" in layers_flagging:
            raw_scores = spec_result.get("raw_scores", [])
            if idx < len(raw_scores):
                risk_reasons.append(
                    f"Spectral signature anomaly (score={raw_scores[idx]:.3f})"
                )

        if "clustering" in layers_flagging:
            cl = cluster_labels[idx] if idx < len(cluster_labels) else -1
            cluster_analysis = clust_result.get("cluster_analysis", [])
            for ca in cluster_analysis:
                if ca.get("cluster_id") == cl and ca.get("is_suspicious"):
                    risk_reasons.append(
                        f"Low-purity cluster #{cl} (purity={ca['purity']:.1%})"
                    )
                    break

        if "art" in layers_flagging:
            risk_reasons.append("Flagged by IBM ART Activation Defence")

        if "influence" in layers_flagging:
            inf_scores = influence_result.get("scores", [])
            if idx < len(inf_scores):
                risk_reasons.append(
                    f"High influence score ({inf_scores[idx]:.3f}) — disproportionate model impact"
                )

        if "backdoor" in layers_flagging:
            bd_scores = backdoor_result.get("scores", [])
            if idx < len(bd_scores):
                risk_reasons.append(
                    f"Backdoor trigger pattern detected (score={bd_scores[idx]:.3f})"
                )

        # Build risk reason with clear layer attribution
        n_votes = len(layers_flagging)
        layer_names = [LAYER_DISPLAY_NAMES.get(l, l) for l in layers_flagging]
        flagged_by_str = f"Flagged by: {', '.join(layer_names)}"

        if risk_reasons:
            risk_reason = f"{flagged_by_str}. Details: {'; '.join(risk_reasons)}"
        else:
            risk_reason = flagged_by_str

        # Per-sample ensemble risk
        risk_pct, risk_cat = _compute_sample_risk(
            n_votes, float(sample_weighted_score[idx])
        )

        sample_features = {}
        for j, fname in enumerate(feature_names):
            if j < X.shape[1]:
                sample_features[fname] = float(X[idx, j])

        flagged_samples.append({
            "index": int(idx),
            "score": float(composite_scores[idx]),
            "layers": layers_flagging,
            "n_layers": n_votes,
            "features": sample_features,
            "label": int(y[idx]) if y is not None and idx < len(y) else None,
            "risk_reason": risk_reason,
            "ensemble_risk_score": round(risk_pct, 1),
            "risk_category": risk_cat,
            "is_confirmed_poison": idx in ensemble_flagged,
        })

    # Sort by ensemble risk score (then composite score) descending
    flagged_samples.sort(
        key=lambda x: (x["ensemble_risk_score"], x["score"]),
        reverse=True,
    )

    # ── Influence scores (top 20) ──
    top_indices = np.argsort(composite_scores)[::-1][:20]
    influence_scores = [
        {
            "index": int(idx),
            "score": float(composite_scores[idx]),
            "is_flagged": idx in ensemble_flagged,
            "label": int(y[idx]) if y is not None and idx < len(y) else None,
        }
        for idx in top_indices
    ]

    # ── Label heatmap data ──
    label_heatmap_data = None
    if y is not None and len(feature_names) > 0:
        label_heatmap_data = _build_label_heatmap(X, y, feature_names)

    # ── Summary ──
    elapsed = round(time.time() - t_start, 2)
    n_active_layers = sum(
        1 for r in all_results if r["n_flagged"] > 0
    )
    summary = (
        f"Analysed {n_samples} samples with {n_features} features "
        f"using {profile['name']} profile. "
        f"Ensemble voting (≥{ENSEMBLE_VOTE_THRESHOLD} layers required): "
        f"{n_flagged} confirmed suspicious samples ({flagged_ratio:.1%}), "
        f"{n_warnings} single-layer warnings. "
        f"{n_active_layers} of 6 detection layers active. "
        f"Risk score: {risk_score}/100 ({risk_level}). "
        f"Analysis completed in {elapsed}s."
    )

    logger.info(summary)

    return AnalysisResponse(
        status=status,
        risk_score=risk_score,
        risk_level=risk_level,
        domain_profile=domain,
        n_samples=n_samples,
        n_flagged=n_flagged,
        n_warnings=n_warnings,
        flagged_ratio=round(flagged_ratio, 4),
        summary=summary,
        ensemble_threshold=ENSEMBLE_VOTE_THRESHOLD,
        statistical=stat_result,
        spectral=spec_result,
        clustering=clust_result,
        art=art_result,
        influence=influence_result,
        backdoor=backdoor_result,
        scatter_data=scatter_data,
        flagged_samples=flagged_samples,
        influence_scores=influence_scores,
        label_heatmap_data=label_heatmap_data,
        profile_info=profile,
        model_info=model_info,
        analysis_time=elapsed,
    )


def _build_label_heatmap(
    X: np.ndarray, y: np.ndarray, feature_names: list, n_bins: int = 8
) -> dict:
    """Build label heatmap data: for each feature, bin values and count labels per bin."""
    heatmap = {}
    unique_labels = sorted(np.unique(y).tolist())

    for j, fname in enumerate(feature_names[:10]):  # Limit to 10 features
        col = X[:, j]
        try:
            bin_edges = np.linspace(col.min(), col.max(), n_bins + 1)
            bin_indices = np.digitize(col, bin_edges[1:-1])
        except Exception:
            continue

        bins_data = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                label_counts = {str(l): 0 for l in unique_labels}
            else:
                labels_in_bin = y[mask]
                unique_b, counts_b = np.unique(labels_in_bin, return_counts=True)
                label_counts = {str(l): 0 for l in unique_labels}
                for ul, uc in zip(unique_b, counts_b):
                    label_counts[str(ul)] = int(uc)

            total = int(mask.sum())
            bins_data.append({
                "bin": b,
                "range": f"{bin_edges[b]:.2f}-{bin_edges[b+1]:.2f}",
                "total": total,
                "labels": label_counts,
            })

        heatmap[fname] = bins_data

    return {
        "features": list(heatmap.keys()),
        "labels": [str(l) for l in unique_labels],
        "data": heatmap,
    }


# ─── Run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
