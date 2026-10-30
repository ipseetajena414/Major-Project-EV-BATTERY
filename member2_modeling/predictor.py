"""
Member 2 → Member 4 Inference API
Autonomous Multi-Agent EV Battery Life & Range Predictor

Usage:
    from member2_modeling.predictor import Member2Predictor
    
    predictor = Member2Predictor("path/to/member2_modeling/")
    result = predictor.predict_future_soh(features, chemistry="NMC", horizon_cycles=1)
    result = predictor.estimate_rul(features, chemistry="NMC")
    result = predictor.predict_capacity(features, chemistry="NMC", horizon_cycles=1)
"""

import os
import json
import warnings
import numpy as np

try:
    import joblib
except ImportError:
    raise ImportError("joblib is required: pip install joblib")


class PredictionResult:
    """Structured prediction result for Member 4 integration."""
    
    def __init__(self, status, prediction=None, unit=None, horizon_cycles=None,
                 chemistry=None, model_name=None, schema_version=None,
                 model_status=None, caveat=None, error=None):
        self.status = status           # "ok", "unsupported", "error"
        self.prediction = prediction   # float or None
        self.unit = unit               # e.g. "%", "Ah", "cycles"
        self.horizon_cycles = horizon_cycles
        self.chemistry = chemistry
        self.model_name = model_name
        self.schema_version = schema_version
        self.model_status = model_status  # "experimental_research_stage" / "unvalidated_pre_eol"
        self.caveat = caveat           # warning string or None
        self.error = error             # error string or None
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def __repr__(self):
        return f"PredictionResult({json.dumps(self.to_dict(), indent=2)})"


class Member2Predictor:
    """
    Inference wrapper for Member 2 trained models.

    Validates feature names, order, types, supported chemistry,
    and returns structured results with metadata.

    All served models are research-stage. The SOH models underperform a
    persistence baseline, and the RUL model is unvalidated for pre-EOL use;
    see model_manifest.json["model_readiness"].
    """
    
    RATED_CAPACITY = {"NMC": 2.0, "LFP": 1.25}
    
    def __init__(self, model_dir):
        """
        Load all model artifacts from model_dir.
        
        Args:
            model_dir: Path to the member2_modeling directory containing
                       model files, scalers, and schema JSON.
        """
        self.model_dir = model_dir
        
        # Load schema
        schema_path = os.path.join(model_dir, "feature_schema.json")
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"feature_schema.json not found in {model_dir}")
        with open(schema_path, "r") as f:
            self.schema = json.load(f)
        
        self.schema_version = self.schema.get("schema_version", "unknown")
        
        # Load manifest
        manifest_path = os.path.join(model_dir, "model_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {}
        
        # Lazy-load models
        self._models = {}
        self._scalers = {}
    
    def _load_model(self, name):
        """Lazy-load a model by name."""
        if name in self._models:
            return self._models[name]
        
        model_info = self.schema.get("models", {}).get(name, {})
        model_file = model_info.get("model_file")
        if not model_file:
            return None
        
        path = os.path.join(self.model_dir, model_file)
        if not os.path.exists(path):
            return None
        
        if path.endswith(".joblib"):
            self._models[name] = joblib.load(path)
        elif path.endswith(".pt"):
            import torch
            # Need to reconstruct the BiLSTM architecture
            # This is handled separately
            return None
        
        return self._models[name]
    
    def _load_scaler(self, name):
        """Lazy-load a scaler by name."""
        if name in self._scalers:
            return self._scalers[name]
        
        path = os.path.join(self.model_dir, f"scaler_{name}.joblib")
        if os.path.exists(path):
            self._scalers[name] = joblib.load(path)
            return self._scalers[name]
        return None
    
    def _validate_features(self, features, expected_names):
        """
        Validate input features against expected schema.
        
        Args:
            features: dict of {feature_name: value}
            expected_names: list of expected feature names in order
        
        Returns:
            (is_valid, error_message, feature_array)
        """
        if not isinstance(features, dict):
            return False, "features must be a dict of {name: value}", None
        
        # Check for missing features
        missing = [f for f in expected_names if f not in features]
        if missing:
            return False, f"Missing features: {missing}", None
        
        # Check for unexpected features (warning only)
        extra = [f for f in features if f not in expected_names]
        if extra:
            warnings.warn(f"Ignoring unexpected features: {extra}")
        
        # Build ordered array
        try:
            values = np.array([[float(features[f]) for f in expected_names]])
        except (ValueError, TypeError) as e:
            return False, f"Feature type error: {e}", None
        
        # Check for NaN
        nan_features = [expected_names[i] for i in range(len(expected_names))
                        if np.isnan(values[0, i])]
        if nan_features:
            return False, f"NaN values in features: {nan_features}", None
        
        return True, None, values
    
    def _get_model_key(self, chemistry, task):
        """Map chemistry + task to model key."""
        chem_lower = chemistry.lower()
        if chem_lower == "nmc":
            if task == "soh":
                return "nmc_lgbm"
            elif task == "rul":
                return "nmc_rul_lgbm"
        elif chem_lower == "lfp":
            if task == "soh":
                return "lfp_lgbm"
        return None
    
    def predict_future_soh(self, features, chemistry, horizon_cycles=1):
        """
        Predict SOH of the next observed cycle record given the current
        cycle record's features.

        horizon_cycles is positional (1 = next observed record). It does NOT
        guarantee the next physical cycle: for LFP B5 the observed transitions
        100->151 and 200->401 are many physical cycles apart.

        Args:
            features: dict of feature_name → float value
            chemistry: "NMC" or "LFP"
            horizon_cycles: prediction horizon in observed records (only 1 supported)
        
        Returns:
            PredictionResult
        """
        # Validate chemistry
        chem_upper = chemistry.upper()
        if chem_upper not in ["NMC", "LFP"]:
            return PredictionResult(
                status="unsupported",
                chemistry=chemistry,
                error=f"Chemistry '{chemistry}' not supported. Supported: NMC, LFP.",
                schema_version=self.schema_version,
            )
        
        # Validate horizon
        if horizon_cycles != 1:
            return PredictionResult(
                status="unsupported",
                chemistry=chem_upper,
                horizon_cycles=horizon_cycles,
                error="Only horizon_cycles=1 is currently supported.",
                schema_version=self.schema_version,
            )
        
        # Get model
        model_key = self._get_model_key(chem_upper, "soh")
        model = self._load_model(model_key)
        if model is None:
            return PredictionResult(
                status="error",
                chemistry=chem_upper,
                error=f"Model '{model_key}' not found or failed to load.",
                schema_version=self.schema_version,
            )
        
        # Get expected features
        model_info = self.schema["models"][model_key]
        expected_features = model_info["feature_names"]
        
        # Validate
        is_valid, err_msg, X = self._validate_features(features, expected_features)
        if not is_valid:
            return PredictionResult(
                status="error",
                chemistry=chem_upper,
                error=err_msg,
                schema_version=self.schema_version,
            )
        
        # Predict
        pred = float(model.predict(X)[0])
        
        if chem_upper == "NMC":
            caveat = ("RESEARCH-STAGE MODEL: on the chronological holdout (n=95), "
                      "the persistence baseline MAE was 0.504% versus 7.968% for "
                      "this model. Do not treat this output as production-ready.")
        else:
            caveat = ("LFP B5 labels are CONDITIONAL (documented charge-balance "
                      "discrepancy; cycle_id=40 has a null SOH label). Horizon is the "
                      "next observed record, not necessarily the next physical cycle. "
                      "On the chronological holdout (n=51, adjacent observed pairs "
                      "only), persistence MAE was 0.131% versus 0.926% for this model.")
        
        return PredictionResult(
            status="ok",
            prediction=round(pred, 4),
            unit="%",
            horizon_cycles=horizon_cycles,
            chemistry=chem_upper,
            model_name=model_key,
            schema_version=self.schema_version,
            model_status="experimental_research_stage",
            caveat=caveat,
        )
    
    def predict_capacity(self, features, chemistry, horizon_cycles=1):
        """
        Predict discharge capacity (Ah) for the next observed cycle record.
        Derived algebraically from the SOH prediction:
        capacity = soh_pct / 100 * rated_capacity (NMC 2.0 Ah, LFP 1.25 Ah).
        This is not an independently validated capacity measurement.
        
        Args:
            features: dict of feature_name → float value
            chemistry: "NMC" or "LFP"
            horizon_cycles: prediction horizon in observed records
        
        Returns:
            PredictionResult
        """
        soh_result = self.predict_future_soh(features, chemistry, horizon_cycles)
        
        if soh_result.status != "ok":
            return soh_result  # Pass through error
        
        chem_upper = chemistry.upper()
        rated = self.RATED_CAPACITY.get(chem_upper)
        if rated is None:
            return PredictionResult(
                status="unsupported",
                chemistry=chem_upper,
                error=f"No rated capacity defined for {chem_upper}",
                schema_version=self.schema_version,
            )
        
        cap_ah = round(soh_result.prediction / 100.0 * rated, 4)
        
        return PredictionResult(
            status="ok",
            prediction=cap_ah,
            unit="Ah",
            horizon_cycles=horizon_cycles,
            chemistry=chem_upper,
            model_name=soh_result.model_name,
            schema_version=self.schema_version,
            model_status=soh_result.model_status,
            caveat=(soh_result.caveat + " Capacity is derived as SOH/100*C_rated "
                    "and is not an independently validated measurement."),
        )
    
    def estimate_rul(self, features, chemistry):
        """
        Estimate Remaining Useful Life (cycles to 80% EOL).
        Only supported for NMC chemistry.

        UNVALIDATED: all 97 chronological test examples were post-EOL
        (ground truth RUL = 0), so this output has no established pre-EOL
        forecasting capability.
        
        Args:
            features: dict of feature_name → float value
            chemistry: "NMC" (only supported)
        
        Returns:
            PredictionResult
        """
        chem_upper = chemistry.upper()
        if chem_upper != "NMC":
            return PredictionResult(
                status="unsupported",
                chemistry=chem_upper,
                error=("RUL estimation only supported for NMC. "
                       "LFP B5 never crosses 80% EOL in observed data. "
                       "Na-ion has insufficient degradation."),
                schema_version=self.schema_version,
            )
        
        model = self._load_model("nmc_rul_lgbm")
        if model is None:
            return PredictionResult(
                status="error",
                chemistry=chem_upper,
                error="RUL model not found or failed to load.",
                schema_version=self.schema_version,
            )
        
        model_info = self.schema["models"]["nmc_rul_lgbm"]
        expected_features = model_info["feature_names"]
        
        is_valid, err_msg, X = self._validate_features(features, expected_features)
        if not is_valid:
            return PredictionResult(
                status="error",
                chemistry=chem_upper,
                error=err_msg,
                schema_version=self.schema_version,
            )
        
        pred = float(model.predict(X)[0])
        pred = max(0.0, round(pred, 1))  # RUL cannot be negative
        
        return PredictionResult(
            status="ok",
            prediction=pred,
            unit="cycles",
            chemistry=chem_upper,
            model_name="nmc_rul_lgbm",
            schema_version=self.schema_version,
            model_status="unvalidated_pre_eol",
            caveat="UNVALIDATED FOR PRE-EOL FORECASTING: all 97 chronological "
                   "test examples are post-80% EOL (ground truth RUL=0). Model is "
                   "NOT verified for active pre-EOL degradation forecasting.",
        )
    
    def get_supported_chemistries(self):
        """Return supported chemistry info."""
        return self.manifest.get("supported_chemistries", {})
    
    def get_feature_schema(self, chemistry):
        """Return feature names and definitions for a chemistry."""
        chem_upper = chemistry.upper()
        model_key = self._get_model_key(chem_upper, "soh")
        if model_key and model_key in self.schema.get("models", {}):
            info = self.schema["models"][model_key]
            return {
                "feature_names": info["feature_names"],
                "feature_count": info["feature_count"],
                "definitions": {
                    f: self.schema["feature_definitions"].get(f, {})
                    for f in info["feature_names"]
                },
            }
        return None
    
    def get_blocked_features(self):
        """Return Member 4 candidate features that cannot be supported."""
        return self.schema.get("blocked_features", {})


# ─── Smoke Test ──────────────────────────────────────────────────────────────

def smoke_test(model_dir=None):
    """
    Smoke test: reload models, run inference, verify outputs.
    """
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 50)
    print("SMOKE TEST: Member 2 Predictor")
    print("=" * 50)
    
    pred = Member2Predictor(model_dir)
    print(f"  Schema version: {pred.schema_version}")
    print(f"  Supported chemistries: {list(pred.get_supported_chemistries().keys())}")
    print(f"  Blocked features: {list(pred.get_blocked_features().keys())}")
    
    # Load sample I/O from manifest
    sample = pred.manifest.get("sample_io", {})
    if sample:
        print(f"\n  Sample input chemistry: {sample['chemistry']}")
        features = sample["input_features"]
        
        # Test predict_future_soh
        result = pred.predict_future_soh(features, sample["chemistry"], horizon_cycles=1)
        print(f"  predict_future_soh: {result}")
        assert result.status == "ok", f"Expected ok, got {result.status}: {result.error}"
        assert result.unit == "%"
        assert isinstance(result.prediction, float)
        print(f"    [OK] SOH prediction: {result.prediction}%")
        
        # Test predict_capacity
        result_cap = pred.predict_capacity(features, sample["chemistry"], horizon_cycles=1)
        print(f"  predict_capacity: {result_cap}")
        assert result_cap.status == "ok"
        assert result_cap.unit == "Ah"
        print(f"    [OK] Capacity prediction: {result_cap.prediction} Ah")
        
        # Test estimate_rul (NMC only)
        result_rul = pred.estimate_rul(features, "NMC")
        print(f"  estimate_rul: {result_rul}")
        if result_rul.status == "ok":
            assert result_rul.unit == "cycles"
            print(f"    [OK] RUL prediction: {result_rul.prediction} cycles")
        else:
            print(f"    [WARN] RUL: {result_rul.error}")
        
        # Test unsupported chemistry
        result_na = pred.predict_future_soh(features, "Na-ion", horizon_cycles=1)
        assert result_na.status == "unsupported"
        print(f"    [OK] Na-ion correctly returns unsupported")
        
        # Test unsupported horizon
        result_h5 = pred.predict_future_soh(features, "NMC", horizon_cycles=5)
        assert result_h5.status == "unsupported"
        print(f"    [OK] horizon=5 correctly returns unsupported")
        
        # Test missing features
        bad_features = {k: v for k, v in list(features.items())[:3]}
        result_bad = pred.predict_future_soh(bad_features, "NMC")
        assert result_bad.status == "error"
        assert "Missing features" in result_bad.error
        print(f"    [OK] Missing features correctly returns error")
        
        print(f"\n  ALL SMOKE TESTS PASSED [OK]")
    else:
        print("  [WARN] No sample I/O in manifest — cannot run smoke test")
    
    return True


if __name__ == "__main__":
    smoke_test()
