"""
India Domain Risk Profiles — UPI Fraud Detection & Credit Scoring

Provides domain-specific detection thresholds and configurations
optimised for Indian fintech and government ML systems.

Each profile tunes the detection engine parameters based on
the expected characteristics of poisoning attacks in that domain.
"""


DOMAIN_PROFILES = {
    "upi_fraud": {
        "id": "upi_fraud",
        "name": "UPI Fraud Detection",
        "description": (
            "Optimised for detecting poisoning in UPI transaction fraud models. "
            "Indian UPI systems process 10B+ monthly transactions. Poisoning attacks "
            "may target high-value merchant clusters or specific time-window patterns "
            "to evade fraud detection. Uses aggressive thresholds due to high-stakes "
            "financial impact."
        ),
        "icon": "₹",
        "regulatory_context": (
            "RBI Digital Payment Security Controls (2021), "
            "NPCI UPI Fraud Monitoring Guidelines, "
            "CERT-In Cyber Security Framework"
        ),
        "risk_category": "HIGH",
        # Layer 1: Statistical Analysis thresholds
        "z_threshold": 2.5,          # Lower = more aggressive (UPI fraud is high-stakes)
        "iqr_multiplier": 1.3,       # Tighter fences
        "min_anomaly_features": 1,
        # Layer 2: Spectral Analysis
        "spectral_top_k": 5,
        "spectral_percentile": 93.0,  # More sensitive
        # Layer 3: Clustering
        "n_clusters": 6,
        "purity_threshold": 0.80,     # Lower purity tolerance
        "umap_n_neighbors": 20,
        "umap_min_dist": 0.05,
        # Risk score weights
        "layer_weights": {
            "statistical": 0.20,
            "spectral": 0.25,
            "clustering": 0.20,
            "art": 0.10,
            "influence": 0.15,
            "backdoor": 0.10,
        },
        # Expected dataset characteristics
        "expected_features": [
            "transaction_amount", "merchant_category", "time_delta",
            "device_type", "location_cluster", "tx_frequency",
            "avg_amount_7d", "is_new_merchant", "hour_of_day",
        ],
        "contamination_baseline": 0.03,
    },
    "credit_scoring": {
        "id": "credit_scoring",
        "name": "Credit Scoring",
        "description": (
            "Optimised for detecting poisoning in credit scoring / creditworthiness "
            "models used by Indian NBFCs and banks. Poisoning here may target "
            "edge-case applicants (near approval thresholds) or introduce systematic "
            "bias against protected groups. Uses balanced thresholds with emphasis "
            "on label integrity."
        ),
        "icon": "📊",
        "regulatory_context": (
            "RBI Fair Practices Code for NBFCs, "
            "RBI ML Model Governance Guidelines (2023), "
            "DPDP Act 2023 — Fair Processing Obligations"
        ),
        "risk_category": "MEDIUM",
        # Layer 1: Statistical Analysis thresholds
        "z_threshold": 3.0,           # Standard thresholds
        "iqr_multiplier": 1.5,
        "min_anomaly_features": 2,    # Require multiple anomalous features
        # Layer 2: Spectral Analysis
        "spectral_top_k": 3,
        "spectral_percentile": 95.0,
        # Layer 3: Clustering
        "n_clusters": 4,
        "purity_threshold": 0.85,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        # Risk score weights
        "layer_weights": {
            "statistical": 0.25,
            "spectral": 0.20,
            "clustering": 0.20,
            "art": 0.10,
            "influence": 0.15,
            "backdoor": 0.10,
        },
        # Expected dataset characteristics
        "expected_features": [
            "annual_income", "debt_to_income", "credit_history_months",
            "num_open_accounts", "num_defaults", "loan_amount",
            "employment_years", "home_ownership", "credit_score_band",
        ],
        "contamination_baseline": 0.05,
    },
    "kyc_govt_welfare": {
        "id": "kyc_govt_welfare",
        "name": "KYC / Government Welfare",
        "description": (
            "Optimised for detecting poisoning in Aadhaar KYC verification and "
            "government welfare scheme (PMJAY, PM-KISAN, MGNREGA) fraud detection models. "
            "These systems process billions of identity verifications annually. Poisoning may "
            "target demographic subgroups to create systematic exclusion or inclusion errors. "
            "Uses conservative thresholds to avoid false positives on legitimate demographic "
            "variance while remaining sensitive to coordinated label-flipping attacks."
        ),
        "icon": "🏛️",
        "regulatory_context": (
            "UIDAI Aadhaar Act 2016, "
            "MeitY Digital India Guidelines, "
            "DPDP Act 2023 — Sensitive Personal Data provisions, "
            "CERT-In Cyber Security Framework"
        ),
        "risk_category": "HIGH",
        # Layer 1: Statistical Analysis thresholds
        "z_threshold": 2.8,
        "iqr_multiplier": 1.4,
        "min_anomaly_features": 2,
        # Layer 2: Spectral Analysis
        "spectral_top_k": 4,
        "spectral_percentile": 94.0,
        # Layer 3: Clustering
        "n_clusters": 5,
        "purity_threshold": 0.82,
        "umap_n_neighbors": 18,
        "umap_min_dist": 0.08,
        # Risk score weights
        "layer_weights": {
            "statistical": 0.20,
            "spectral": 0.25,
            "clustering": 0.25,
            "art": 0.10,
            "influence": 0.10,
            "backdoor": 0.10,
        },
        # Expected dataset characteristics
        "expected_features": [
            "age", "gender_code", "state_code", "district_code",
            "scheme_id", "verification_score", "num_dependents",
            "income_bracket", "biometric_match_score",
        ],
        "contamination_baseline": 0.04,
    },
    "general": {
        "id": "general",
        "name": "General Purpose",
        "description": (
            "Default detection profile for generic ML datasets. "
            "Uses balanced thresholds suitable for exploratory analysis."
        ),
        "icon": "🔍",
        "regulatory_context": "CERT-In Guidelines for Responsible AI",
        "risk_category": "MEDIUM",
        "z_threshold": 3.0,
        "iqr_multiplier": 1.5,
        "min_anomaly_features": 1,
        "spectral_top_k": 3,
        "spectral_percentile": 95.0,
        "n_clusters": 5,
        "purity_threshold": 0.85,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "layer_weights": {
            "statistical": 0.20,
            "spectral": 0.20,
            "clustering": 0.20,
            "art": 0.10,
            "influence": 0.15,
            "backdoor": 0.15,
        },
        "expected_features": [],
        "contamination_baseline": 0.05,
    },
}


def get_profile(profile_id: str) -> dict:
    """Get a domain profile by ID. Falls back to 'general' if not found."""
    return DOMAIN_PROFILES.get(profile_id, DOMAIN_PROFILES["general"])


def list_profiles() -> list:
    """Return list of available profile summaries."""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "description": p["description"],
            "icon": p["icon"],
            "risk_category": p["risk_category"],
            "regulatory_context": p["regulatory_context"],
        }
        for p in DOMAIN_PROFILES.values()
    ]
