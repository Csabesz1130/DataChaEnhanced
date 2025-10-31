"""
Model Manager for Excel AI

This module manages model versioning, deployment, and lifecycle management
for the AI Excel learning system.
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import shutil
import hashlib
import pickle
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Represents a model version"""
    version_id: str
    model_name: str
    model_type: str
    created_at: str
    performance_metrics: Dict[str, Any]
    training_data_hash: str
    model_path: str
    dependencies: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class ModelDeployment:
    """Represents a model deployment"""
    deployment_id: str
    model_version_id: str
    environment: str
    deployed_at: str
    status: str  # 'active', 'inactive', 'failed'
    performance_monitoring: Dict[str, Any]
    rollback_version: Optional[str] = None

class ModelManager:
    """
    Manages model versioning, deployment, and lifecycle
    """
    
    def __init__(self, models_dir: str = "models", 
                 versions_dir: str = "model_versions",
                 deployments_dir: str = "deployments"):
        self.models_dir = Path(models_dir)
        self.versions_dir = Path(versions_dir)
        self.deployments_dir = Path(deployments_dir)
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)
        self.deployments_dir.mkdir(exist_ok=True)
        
        # Load existing data
        self.versions = self._load_versions()
        self.deployments = self._load_deployments()
        
    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load existing model versions"""
        versions = {}
        versions_file = self.versions_dir / "versions.json"
        
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                data = json.load(f)
                for version_id, version_data in data.items():
                    versions[version_id] = ModelVersion(**version_data)
        
        return versions
    
    def _load_deployments(self) -> Dict[str, ModelDeployment]:
        """Load existing deployments"""
        deployments = {}
        deployments_file = self.deployments_dir / "deployments.json"
        
        if deployments_file.exists():
            with open(deployments_file, 'r') as f:
                data = json.load(f)
                for deployment_id, deployment_data in data.items():
                    deployments[deployment_id] = ModelDeployment(**deployment_data)
        
        return deployments
    
    def _save_versions(self):
        """Save versions to file"""
        versions_file = self.versions_dir / "versions.json"
        data = {version_id: asdict(version) for version_id, version in self.versions.items()}
        
        with open(versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_deployments(self):
        """Save deployments to file"""
        deployments_file = self.deployments_dir / "deployments.json"
        data = {deployment_id: asdict(deployment) for deployment_id, deployment in self.deployments.items()}
        
        with open(deployments_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_model_version(self, model_name: str, model_type: str,
                           model_path: str, training_data: pd.DataFrame,
                           performance_metrics: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> str:
        """
        Create a new model version
        
        Args:
            model_name: Name of the model
            model_type: Type of model
            model_path: Path to the model file
            training_data: Training data used
            performance_metrics: Performance metrics
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.now().isoformat()
        version_id = f"{model_name}_v{len(self._get_model_versions(model_name)) + 1}_{int(datetime.now().timestamp())}"
        
        # Calculate training data hash
        training_data_hash = self._calculate_data_hash(training_data)
        
        # Copy model to versions directory
        version_model_path = self.versions_dir / f"{version_id}.pkl"
        shutil.copy2(model_path, version_model_path)
        
        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_type=model_type,
            created_at=timestamp,
            performance_metrics=performance_metrics,
            training_data_hash=training_data_hash,
            model_path=str(version_model_path),
            dependencies=self._get_dependencies(),
            metadata=metadata or {}
        )
        
        # Save version
        self.versions[version_id] = version
        self._save_versions()
        
        logger.info(f"Created model version: {version_id}")
        return version_id
    
    def _get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        return [version for version in self.versions.values() if version.model_name == model_name]
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data"""
        # Convert to string and hash
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_dependencies(self) -> Dict[str, str]:
        """Get current dependencies"""
        return {
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit-learn': '1.4.0',  # This should be dynamically detected
            'tensorflow': '2.15.0'    # This should be dynamically detected
        }
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self._get_model_versions(model_name)
        if not versions:
            return None
        
        # Sort by creation time and return latest
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[0]
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version"""
        return self.versions.get(version_id)
    
    def list_model_versions(self, model_name: str = None) -> List[ModelVersion]:
        """List model versions"""
        if model_name:
            return self._get_model_versions(model_name)
        else:
            return list(self.versions.values())
    
    def deploy_model(self, version_id: str, environment: str = "production") -> str:
        """
        Deploy a model version
        
        Args:
            version_id: Version ID to deploy
            environment: Deployment environment
            
        Returns:
            Deployment ID
        """
        # Check if version exists
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        # Generate deployment ID
        deployment_id = f"deployment_{version_id}_{environment}_{int(datetime.now().timestamp())}"
        
        # Create deployment record
        deployment = ModelDeployment(
            deployment_id=deployment_id,
            model_version_id=version_id,
            environment=environment,
            deployed_at=datetime.now().isoformat(),
            status='active',
            performance_monitoring={},
            rollback_version=None
        )
        
        # Deactivate previous deployments in the same environment
        self._deactivate_environment_deployments(environment)
        
        # Save deployment
        self.deployments[deployment_id] = deployment
        self._save_deployments()
        
        logger.info(f"Deployed model version {version_id} to {environment}")
        return deployment_id
    
    def _deactivate_environment_deployments(self, environment: str):
        """Deactivate all deployments in an environment"""
        for deployment in self.deployments.values():
            if deployment.environment == environment and deployment.status == 'active':
                deployment.status = 'inactive'
    
    def rollback_deployment(self, deployment_id: str, rollback_version_id: str):
        """
        Rollback a deployment to a previous version
        
        Args:
            deployment_id: Deployment to rollback
            rollback_version_id: Version to rollback to
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        # Check if rollback version exists
        rollback_version = self.get_version(rollback_version_id)
        if not rollback_version:
            raise ValueError(f"Rollback version {rollback_version_id} not found")
        
        # Update deployment
        deployment.rollback_version = rollback_version_id
        deployment.status = 'active'
        deployment.deployed_at = datetime.now().isoformat()
        
        # Deactivate other deployments in the same environment
        self._deactivate_environment_deployments(deployment.environment)
        
        self._save_deployments()
        
        logger.info(f"Rolled back deployment {deployment_id} to version {rollback_version_id}")
    
    def get_active_deployment(self, environment: str = "production") -> Optional[ModelDeployment]:
        """Get the active deployment for an environment"""
        for deployment in self.deployments.values():
            if deployment.environment == environment and deployment.status == 'active':
                return deployment
        return None
    
    def list_deployments(self, environment: str = None) -> List[ModelDeployment]:
        """List deployments"""
        if environment:
            return [d for d in self.deployments.values() if d.environment == environment]
        else:
            return list(self.deployments.values())
    
    def update_performance_monitoring(self, deployment_id: str, 
                                    metrics: Dict[str, Any]):
        """Update performance monitoring for a deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment.performance_monitoring.update(metrics)
        self._save_deployments()
    
    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict[str, Any]:
        """
        Compare two model versions
        
        Args:
            version_id_1: First version ID
            version_id_2: Second version ID
            
        Returns:
            Comparison results
        """
        version_1 = self.get_version(version_id_1)
        version_2 = self.get_version(version_id_2)
        
        if not version_1 or not version_2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'version_1': {
                'version_id': version_1.version_id,
                'created_at': version_1.created_at,
                'performance_metrics': version_1.performance_metrics
            },
            'version_2': {
                'version_id': version_2.version_id,
                'created_at': version_2.created_at,
                'performance_metrics': version_2.performance_metrics
            },
            'differences': {}
        }
        
        # Compare performance metrics
        metrics_1 = version_1.performance_metrics
        metrics_2 = version_2.performance_metrics
        
        for metric in set(metrics_1.keys()) | set(metrics_2.keys()):
            val_1 = metrics_1.get(metric)
            val_2 = metrics_2.get(metric)
            
            if val_1 != val_2:
                comparison['differences'][metric] = {
                    'version_1': val_1,
                    'version_2': val_2,
                    'improvement': self._calculate_improvement(val_1, val_2, metric)
                }
        
        return comparison
    
    def _calculate_improvement(self, val_1: Any, val_2: Any, metric: str) -> str:
        """Calculate improvement between two values"""
        if val_1 is None or val_2 is None:
            return "unknown"
        
        try:
            if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                if 'accuracy' in metric.lower() or 'score' in metric.lower():
                    # Higher is better
                    if val_2 > val_1:
                        return f"improved by {((val_2 - val_1) / val_1 * 100):.2f}%"
                    elif val_2 < val_1:
                        return f"degraded by {((val_1 - val_2) / val_1 * 100):.2f}%"
                    else:
                        return "no change"
                elif 'error' in metric.lower() or 'loss' in metric.lower():
                    # Lower is better
                    if val_2 < val_1:
                        return f"improved by {((val_1 - val_2) / val_1 * 100):.2f}%"
                    elif val_2 > val_1:
                        return f"degraded by {((val_2 - val_1) / val_1 * 100):.2f}%"
                    else:
                        return "no change"
                else:
                    return "unknown"
            else:
                return "different values"
        except:
            return "unknown"
    
    def delete_version(self, version_id: str):
        """Delete a model version"""
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        # Check if version is deployed
        for deployment in self.deployments.values():
            if deployment.model_version_id == version_id and deployment.status == 'active':
                raise ValueError(f"Cannot delete version {version_id} - it is currently deployed")
        
        # Remove model file
        model_path = Path(version.model_path)
        if model_path.exists():
            model_path.unlink()
        
        # Remove from versions
        del self.versions[version_id]
        self._save_versions()
        
        logger.info(f"Deleted model version: {version_id}")
    
    def export_model(self, version_id: str, export_path: str):
        """Export a model version"""
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Model version {version_id} not found")
        
        # Create export package
        export_data = {
            'version': asdict(version),
            'exported_at': datetime.now().isoformat()
        }
        
        # Copy model file
        model_path = Path(version.model_path)
        export_model_path = Path(export_path) / f"{version_id}.pkl"
        shutil.copy2(model_path, export_model_path)
        
        # Save metadata
        metadata_path = Path(export_path) / f"{version_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported model version {version_id} to {export_path}")
    
    def import_model(self, import_path: str) -> str:
        """Import a model version"""
        # Find metadata file
        metadata_files = list(Path(import_path).glob("*_metadata.json"))
        if not metadata_files:
            raise ValueError("No metadata file found in import path")
        
        metadata_file = metadata_files[0]
        with open(metadata_file, 'r') as f:
            import_data = json.load(f)
        
        version_data = import_data['version']
        version_id = version_data['version_id']
        
        # Check if version already exists
        if version_id in self.versions:
            raise ValueError(f"Model version {version_id} already exists")
        
        # Copy model file
        model_filename = f"{version_id}.pkl"
        source_model_path = Path(import_path) / model_filename
        if not source_model_path.exists():
            raise ValueError(f"Model file {model_filename} not found")
        
        target_model_path = self.versions_dir / model_filename
        shutil.copy2(source_model_path, target_model_path)
        
        # Update model path
        version_data['model_path'] = str(target_model_path)
        
        # Create version
        version = ModelVersion(**version_data)
        self.versions[version_id] = version
        self._save_versions()
        
        logger.info(f"Imported model version: {version_id}")
        return version_id
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about models and versions"""
        total_versions = len(self.versions)
        total_deployments = len(self.deployments)
        active_deployments = len([d for d in self.deployments.values() if d.status == 'active'])
        
        # Count by model type
        model_types = {}
        for version in self.versions.values():
            model_types[version.model_type] = model_types.get(version.model_type, 0) + 1
        
        # Count by environment
        environments = {}
        for deployment in self.deployments.values():
            environments[deployment.environment] = environments.get(deployment.environment, 0) + 1
        
        return {
            'total_versions': total_versions,
            'total_deployments': total_deployments,
            'active_deployments': active_deployments,
            'model_types': model_types,
            'environments': environments,
            'average_versions_per_model': total_versions / len(set(v.model_name for v in self.versions.values())) if self.versions else 0
        }
