"""
ROX Proven Edge Engine v3.0 - ML Models
======================================
Machine learning models for trade prediction.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import pickle
import json
import os


class ModelType(Enum):
    """ML model types"""
    XGBOOST = "XGBOOST"
    RANDOM_FOREST = "RANDOM_FOREST"
    LSTM = "LSTM"
    TRANSFORMER = "TRANSFORMER"
    ENSEMBLE = "ENSEMBLE"


@dataclass
class PredictionResult:
    """ML prediction result"""
    model_name: str
    prediction: str  # LONG, SHORT, NEUTRAL
    probability: float
    confidence: float
    features_used: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    raw_scores: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "features_used": self.features_used,
            "timestamp": self.timestamp.isoformat()
        }


class MLModel(ABC):
    """Abstract base class for ML models"""
    
    def __init__(self, name: str, model_type: ModelType):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names: List[str] = []
        self.logger = logging.getLogger(f"MLModel.{name}")
    
    @abstractmethod
    def train(self, X: Any, y: Any) -> bool:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, features: Dict) -> PredictionResult:
        """Make prediction"""
        pass
    
    @abstractmethod
    def predict_proba(self, features: Dict) -> Dict[str, float]:
        """Get prediction probabilities"""
        pass
    
    def save(self, path: str) -> bool:
        """Save model to file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_names': self.feature_names,
                    'is_trained': self.is_trained
                }, f)
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, path: str) -> bool:
        """Load model from file"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data['feature_names']
                self.is_trained = data['is_trained']
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False


class XGBoostModel(MLModel):
    """XGBoost classifier for trade direction prediction"""
    
    def __init__(self, name: str = "xgboost_direction"):
        super().__init__(name, ModelType.XGBOOST)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model"""
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                num_class=3,
                random_state=42
            )
        except ImportError:
            self.logger.warning("XGBoost not installed, using placeholder")
            self.model = None
    
    def train(self, X: Any, y: Any) -> bool:
        """Train XGBoost model"""
        if self.model is None:
            return False
        
        try:
            self.model.fit(X, y)
            self.is_trained = True
            return True
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return False
    
    def predict(self, features: Dict) -> PredictionResult:
        """Make prediction"""
        if not self.is_trained or self.model is None:
            return PredictionResult(
                model_name=self.name,
                prediction="NEUTRAL",
                probability=0.33,
                confidence=0.0,
                features_used=list(features.keys())
            )
        
        try:
            import numpy as np
            
            # Prepare feature vector
            X = np.array([[features.get(f, 0) for f in self.feature_names]])
            
            # Get prediction
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            
            label_map = {0: "SHORT", 1: "NEUTRAL", 2: "LONG"}
            prediction = label_map.get(pred, "NEUTRAL")
            probability = max(proba)
            confidence = probability - 0.33  # Above random
            
            return PredictionResult(
                model_name=self.name,
                prediction=prediction,
                probability=probability,
                confidence=confidence,
                features_used=self.feature_names,
                raw_scores={label_map[i]: float(p) for i, p in enumerate(proba)}
            )
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return PredictionResult(
                model_name=self.name,
                prediction="NEUTRAL",
                probability=0.33,
                confidence=0.0,
                features_used=list(features.keys())
            )
    
    def predict_proba(self, features: Dict) -> Dict[str, float]:
        """Get probabilities"""
        result = self.predict(features)
        return result.raw_scores if result.raw_scores else {
            "LONG": 0.33, "SHORT": 0.33, "NEUTRAL": 0.33
        }


class LSTMModel(MLModel):
    """LSTM model for sequence prediction"""
    
    def __init__(self, name: str = "lstm_price", sequence_length: int = 60):
        super().__init__(name, ModelType.LSTM)
        self.sequence_length = sequence_length
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize LSTM model"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            self.model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(3, activation='softmax')  # 3 classes: LONG, SHORT, NEUTRAL
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        except ImportError:
            self.logger.warning("TensorFlow not installed, using placeholder")
            self.model = None
    
    def train(self, X: Any, y: Any) -> bool:
        """Train LSTM model"""
        if self.model is None:
            return False
        
        try:
            self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            self.is_trained = True
            return True
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return False
    
    def predict(self, features: Dict) -> PredictionResult:
        """Make prediction"""
        if not self.is_trained or self.model is None:
            return PredictionResult(
                model_name=self.name,
                prediction="NEUTRAL",
                probability=0.33,
                confidence=0.0,
                features_used=list(features.keys())
            )
        
        # Placeholder for actual prediction
        return PredictionResult(
            model_name=self.name,
            prediction="NEUTRAL",
            probability=0.33,
            confidence=0.0,
            features_used=list(features.keys())
        )
    
    def predict_proba(self, features: Dict) -> Dict[str, float]:
        """Get probabilities"""
        return {"LONG": 0.33, "SHORT": 0.33, "NEUTRAL": 0.33}


class EnsembleModel(MLModel):
    """Ensemble model combining multiple models"""
    
    def __init__(self, name: str = "ensemble"):
        super().__init__(name, ModelType.ENSEMBLE)
        self.models: List[MLModel] = []
        self.weights: List[float] = []
    
    def add_model(self, model: MLModel, weight: float = 1.0):
        """Add model to ensemble"""
        self.models.append(model)
        self.weights.append(weight)
    
    def train(self, X: Any, y: Any) -> bool:
        """Train all models in ensemble"""
        success = True
        for model in self.models:
            if not model.train(X, y):
                success = False
        self.is_trained = any(m.is_trained for m in self.models)
        return success
    
    def predict(self, features: Dict) -> PredictionResult:
        """Combine predictions from all models"""
        if not self.models:
            return PredictionResult(
                model_name=self.name,
                prediction="NEUTRAL",
                probability=0.33,
                confidence=0.0,
                features_used=list(features.keys())
            )
        
        # Collect predictions
        predictions = []
        for model in self.models:
            pred = model.predict(features)
            predictions.append(pred)
        
        # Weighted voting
        weighted_scores = {"LONG": 0.0, "SHORT": 0.0, "NEUTRAL": 0.0}
        
        for pred, weight in zip(predictions, self.weights):
            weighted_scores[pred.prediction] += pred.probability * weight
        
        # Normalize
        total_weight = sum(self.weights)
        if total_weight > 0:
            for key in weighted_scores:
                weighted_scores[key] /= total_weight
        
        # Get best prediction
        best_pred = max(weighted_scores, key=weighted_scores.get)
        best_prob = weighted_scores[best_pred]
        
        return PredictionResult(
            model_name=self.name,
            prediction=best_pred,
            probability=best_prob,
            confidence=best_prob - 0.33,
            features_used=list(features.keys()),
            raw_scores=weighted_scores
        )
    
    def predict_proba(self, features: Dict) -> Dict[str, float]:
        """Get combined probabilities"""
        result = self.predict(features)
        return result.raw_scores


class ModelManager:
    """
    Manager for all ML models.
    
    Features:
    - Model registration and versioning
    - Model serving and inference
    - Model monitoring
    - A/B testing support
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("ModelManager")
        
        # Model storage
        self.models: Dict[str, MLModel] = {}
        self.model_versions: Dict[str, List[str]] = {}
        
        # Model paths
        self.model_dir = config.get("model_dir", "models")
        
        # Default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default model set"""
        # Direction prediction model
        self.models["xgboost_direction"] = XGBoostModel()
        
        # Price prediction model
        self.models["lstm_price"] = LSTMModel()
        
        # Ensemble
        ensemble = EnsembleModel()
        ensemble.add_model(self.models["xgboost_direction"], weight=0.6)
        ensemble.add_model(self.models["lstm_price"], weight=0.4)
        self.models["ensemble"] = ensemble
    
    def register_model(self, name: str, model: MLModel):
        """Register a new model"""
        self.models[name] = model
        self.logger.info(f"Registered model: {name}")
    
    def get_model(self, name: str) -> Optional[MLModel]:
        """Get model by name"""
        return self.models.get(name)
    
    def predict(self, model_name: str, features: Dict) -> PredictionResult:
        """Make prediction using specified model"""
        model = self.get_model(model_name)
        if model is None:
            self.logger.warning(f"Model not found: {model_name}")
            return PredictionResult(
                model_name=model_name,
                prediction="NEUTRAL",
                probability=0.33,
                confidence=0.0,
                features_used=list(features.keys())
            )
        
        return model.predict(features)
    
    def predict_all(self, features: Dict) -> Dict[str, PredictionResult]:
        """Get predictions from all models"""
        results = {}
        for name, model in self.models.items():
            results[name] = model.predict(features)
        return results
    
    def get_consensus_prediction(self, features: Dict) -> PredictionResult:
        """Get consensus prediction from ensemble"""
        if "ensemble" in self.models:
            return self.models["ensemble"].predict(features)
        
        # Fallback: simple voting
        predictions = self.predict_all(features)
        
        scores = {"LONG": 0.0, "SHORT": 0.0, "NEUTRAL": 0.0}
        for pred in predictions.values():
            scores[pred.prediction] += pred.probability
        
        for key in scores:
            scores[key] /= len(predictions)
        
        best = max(scores, key=scores.get)
        
        return PredictionResult(
            model_name="consensus",
            prediction=best,
            probability=scores[best],
            confidence=scores[best] - 0.33,
            features_used=list(features.keys()),
            raw_scores=scores
        )
    
    def save_all_models(self, directory: str = None):
        """Save all models to directory"""
        directory = directory or self.model_dir
        
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}.pkl")
            model.save(path)
    
    def load_all_models(self, directory: str = None):
        """Load all models from directory"""
        directory = directory or self.model_dir
        
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}.pkl")
            if os.path.exists(path):
                model.load(path)
    
    def get_model_status(self) -> Dict:
        """Get status of all models"""
        return {
            name: {
                "type": model.model_type.value,
                "is_trained": model.is_trained,
                "feature_count": len(model.feature_names)
            }
            for name, model in self.models.items()
        }
