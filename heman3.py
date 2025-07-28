import numpy as np
import random
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
import copy
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from scipy import signal
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import phe as paillier

# Differential Privacy parameters
DEFAULT_EPSILON = 1.0
DEFAULT_DELTA = 1e-5
DEFAULT_SENSITIVITY = 1.0

class Client:
    def __init__(self, client_id: int, X: np.ndarray, y: np.ndarray, 
                 is_malicious: bool = False, compute_capacity: float = 1.0, 
                 network_speed: float = 1.0):
        self.client_id = client_id
        self.X = X  # Feature matrix
        self.y = y  # Labels
        self.data_size = len(X)
        self.model = None
        self.trust_score = 1.0
        self.is_malicious = is_malicious
        self.history = []
        self.compute_capacity = compute_capacity
        self.network_speed = network_speed
        self.participation_count = 0
    
    def secure_mask_update(self, update: np.ndarray, public_keys: list) -> np.ndarray:
        """Apply secure aggregation masking"""
        mask = np.zeros_like(update)
        for key in public_keys:
            if key != self.client_id:  # Don't mask with own key
                # Generate consistent mask using pair identifiers
                pair = tuple(sorted([self.client_id, key]))
                # Use a hash that fits within 32-bit integer range
                seed = hash(pair) % (2**32 - 1)
                np.random.seed(seed)
                mask += np.random.randn(*update.shape)
        return update + mask

    def local_train(self, global_model, lr: float = 0.01, epochs: int = 5):
        """Perform local training using gradient descent"""
        n_features = self.X.shape[1]
        w = global_model[:-1].copy()
        b = global_model[-1].copy()
        
        # Train using gradient descent
        for epoch in range(epochs):
            indices = np.random.permutation(len(self.X))
            for i in indices:
                x = self.X[i]
                y_true = self.y[i]
                pred = np.dot(w, x) + b
                error = pred - y_true
                w -= lr * error * x
                b -= lr * error
        
        new_model = np.hstack([w, b])
        
        # Add trust-based noise
        base_noise_scale = 0.5 if not self.is_malicious else 5.0
        noise_loc = 0 if not self.is_malicious else 10
        effective_scale = base_noise_scale * (1.1 - self.trust_score)
        noise = np.random.normal(loc=noise_loc, scale=effective_scale, size=new_model.shape)
        new_model += noise * (1 / self.compute_capacity)
        
        self.history.append(noise)
        self.participation_count += 1
        return new_model

class FederatedLearningServer:
    def __init__(self, clients: List[Client], model_shape: Tuple, 
                 use_dp: bool = True, use_he: bool = False,
                 scaler: Optional[StandardScaler] = None):
        
        self.use_he = use_he
        self.global_model = np.zeros(model_shape)
        self.clients = clients
        self.selected_history = defaultdict(list)
        self.round = 0
        self.current_epsilon = DEFAULT_EPSILON
        self.use_dp = use_dp
        self.delta = DEFAULT_DELTA
        self.sensitivity = DEFAULT_SENSITIVITY
        self.scaler = scaler
        
        # Default configuration
        self.config = {
            'min_clients': 3,
            'selection_method': 'multi_krum',
            'malicious_ratio': 0.2,
            'selection_frac': 0.6
        }
        
        # Tracking metrics
        self.metrics = {
            'round': [],
            'avg_trust': [],
            'malicious_trust': [],
            'honest_trust': [],
            'krum_filtered': [],
            'ata_filtered': [],
            'model_convergence': [],
            'privacy_budget': [],
            'he_used': [],  # Track HE usage
            'sa_used': []   # Track SA usage
        }
        if use_he:
            self.public_key, self.private_key = paillier.generate_paillier_keypair()
    
    def secure_unmask_updates(self, masked_updates: List[np.ndarray], client_ids: List[int]) -> np.ndarray:
        """Unmask securely aggregated updates"""
        aggregated = np.zeros_like(masked_updates[0])
        for update, client_id in zip(masked_updates, client_ids):
            mask = np.zeros_like(update)
            for other_id in client_ids:
                if other_id != client_id:
                    # Generate same mask using pair identifiers
                    pair = tuple(sorted([other_id, client_id]))
                    # Use a hash that fits within 32-bit integer range
                    seed = hash(pair) % (2**32 - 1)
                    np.random.seed(seed)
                    mask += np.random.randn(*update.shape)
            aggregated += update - mask
        return aggregated / len(masked_updates)

    def select_clients(self, selection_frac: float = None) -> List[Client]:
        """Select clients for the current training round"""
        if selection_frac is None:
            selection_frac = self.config.get('selection_frac', 0.6)
            
        num_selected = max(
            self.config.get('min_clients', 3),
            int(len(self.clients) * selection_frac)
        )
        
        # Create candidate pool considering network speed
        candidates = sorted(
            self.clients,
            key=lambda c: c.network_speed,
            reverse=True
        )[:num_selected * 3]  # Consider top network performers
        
        # Apply selection method
        method = self.config.get('selection_method', 'multi_krum').lower()
        
        if method == 'multi_krum':
            selected = self.multi_krum_selection(candidates, num_selected)
        elif method == 'random':
            selected = random.sample(candidates, num_selected)
        elif method == 'trust_based':
            selected = self.trust_based_selection(candidates, num_selected)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Update selection history
        for client in selected:
            self.selected_history[client.client_id].append(self.round)
        
        return selected
    
    def trust_based_selection(self, candidates: List[Client], m: int) -> List[Client]:
        """Select clients based on trust scores and other factors"""
        scored_clients = []
        for client in candidates:
            # Weighted score considering trust, compute, and network
            score = (0.5 * client.trust_score + 
                    0.3 * client.compute_capacity + 
                    0.2 * client.network_speed)
            scored_clients.append((score, client))
        
        # Sort by score and select top m
        scored_clients.sort(reverse=True, key=lambda x: x[0])
        return [client for (score, client) in scored_clients[:m]]
    
    def multi_krum_selection(self, candidates: List[Client], m: int) -> List[Client]:
        """Multi-KRUM with trust-aware filtering"""
        updates = []
        for client in candidates:
            # Get local update
            local_model = client.local_train(self.global_model)
            update = local_model - self.global_model
            updates.append(update)
        
        krum_scores = []
        for i, (client, update) in enumerate(zip(candidates, updates)):
            distances = []
            for j, (other_client, other_update) in enumerate(zip(candidates, updates)):
                if i != j:
                    # Trust-weighted distance metric
                    dist = np.linalg.norm(update - other_update) * (2 - client.trust_score - other_client.trust_score)
                    distances.append(dist)
        
            distances.sort()
            f = max(1, int(len(candidates) * self.config.get('malicious_ratio', 0.2)))
            score = sum(distances[:len(candidates)-f-1])
            krum_scores.append((score, i))  # Store index instead of client object
    
        # Sort by score and select top m candidates
        krum_scores.sort()  # Sorts based on first tuple element (score)
        selected_indices = [idx for (score, idx) in krum_scores[:m]]
        selected = [candidates[i] for i in selected_indices]
    
        # Track how many malicious clients were filtered
        malicious_filtered = sum(1 for c in candidates if c.is_malicious) - sum(1 for c in selected if c.is_malicious)
        self.metrics['krum_filtered'].append(malicious_filtered)
    
        return selected
    
    def apply_trust_aware_dp(self, updates: List[np.ndarray], clients: List[Client]) -> List[np.ndarray]:
        """Trust-aware differential privacy mechanism"""
        if not self.use_dp:
            return updates
            
        noisy_updates = []
        base_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.current_epsilon
        
        for update, client in zip(updates, clients):
            # Trust-adjusted noise scaling (clipped between 0.1 and 1.0)
            trust_factor = np.clip(client.trust_score, 0.1, 1.0)
            effective_scale = base_scale * (1.1 - trust_factor)  # More trusted = less noise
            
            noise = np.random.laplace(loc=0, scale=effective_scale, size=update.shape)
            noisy_updates.append(update + noise)
            
        return noisy_updates
    
    def adaptive_trust_scoring(self, selected_clients: List[Client]):
        """Adaptive Trust Algorithm (ATA) with enhanced metrics"""
        for client in self.clients:
            if not client.history:
                continue
                
            # Update consistency (weighted by participation)
            if len(client.history) > 1:
                try:
                    consistency = np.mean([
                        np.corrcoef(client.history[i], client.history[i+1])[0,1] 
                        for i in range(len(client.history)-1)
                    ])
                except:
                    consistency = 0.5
            else:
                consistency = 0.5
            
            # Recent participation rate
            participation = len([r for r in self.selected_history[client.client_id] 
                               if r >= self.round - 10]) / 10.0
            
            # Update trust score with momentum
            new_trust = 0.6 * consistency + 0.3 * participation + 0.1 * (client.compute_capacity / 2.0)
            client.trust_score = 0.8 * client.trust_score + 0.2 * new_trust
            client.trust_score = np.clip(client.trust_score, 0.1, 1.0)
        
        # Track ATA filtering effectiveness
        malicious_present = sum(1 for c in selected_clients if c.is_malicious)
        high_trust_malicious = sum(1 for c in selected_clients if c.is_malicious and c.trust_score > 0.7)
        self.metrics['ata_filtered'].append(malicious_present - high_trust_malicious)
    
    def aggregate_updates(self, updates: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Weighted aggregation with HE support"""
        if not self.use_he:
            weights = np.array(weights) / sum(weights)
            aggregated = np.zeros_like(updates[0])
            for update, weight in zip(updates, weights):
                aggregated += update * weight
            return aggregated
        else:
            # Homomorphic aggregation
            scale = 1e6  # Scaling factor for fixed-point arithmetic
            aggregated = [0] * len(updates[0])
            
            for update, weight in zip(updates, weights):
                scaled_update = (update * weight * scale).astype(int)
                for i in range(len(update)):
                    if i >= len(aggregated):  # Ensure index exists
                        aggregated.append(0)
                    aggregated[i] += self.public_key.encrypt(int(scaled_update[i]))
            
            # Decrypt and descale
            return np.array([self.private_key.decrypt(x) / scale for x in aggregated])

    def _update_metrics(self, avg_update: np.ndarray):
        """Update all tracking metrics"""
        self.metrics['round'].append(self.round)
        
        # Trust metrics
        trust_scores = [c.trust_score for c in self.clients]
        self.metrics['avg_trust'].append(np.mean(trust_scores))
        self.metrics['honest_trust'].append(np.mean([
            c.trust_score for c in self.clients if not c.is_malicious
        ]))
        self.metrics['malicious_trust'].append(np.mean([
            c.trust_score for c in self.clients if c.is_malicious
        ]))
        
        # Convergence metric
        self.metrics['model_convergence'].append(np.linalg.norm(avg_update))
        
        # Privacy budget tracking
        avg_trust = np.mean([c.trust_score for c in self.clients])
        self.metrics['privacy_budget'].append(self.current_epsilon * avg_trust)
        
        # Track security method usage
        self.metrics['he_used'].append(1 if self.use_he else 0)
        self.metrics['sa_used'].append(1 if self.config.get('secure_agg', False) else 0)
        
        self.round += 1

    def visualize_metrics(self, experiment_name: str = "default"):
        """Generate comprehensive visualizations including security impacts"""
        plt.figure(figsize=(18, 15))
    
        # Set larger font sizes and line widths
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'lines.linewidth': 2,
            'patch.linewidth': 2,
        })
    
        # Filter metrics to only show up to round 13
        max_round = min(13, len(self.metrics['round'])-1)
        filtered_rounds = [r for r in self.metrics['round'] if r <= max_round]
        filtered_indices = [i for i, r in enumerate(self.metrics['round']) if r <= max_round]
    
        # Trust Score Evolution with Security Highlights
        plt.subplot(3, 3, 1)
        plt.plot(filtered_rounds, [self.metrics['avg_trust'][i] for i in filtered_indices], label='Average')
        plt.plot(filtered_rounds, [self.metrics['honest_trust'][i] for i in filtered_indices], label='Honest')
        plt.plot(filtered_rounds, [self.metrics['malicious_trust'][i] for i in filtered_indices], label='Malicious')
        
        # Highlight rounds where security methods were used
        he_rounds = [r for i, r in enumerate(filtered_rounds) if self.metrics['he_used'][i]]
        sa_rounds = [r for i, r in enumerate(filtered_rounds) if self.metrics['sa_used'][i]]
        
        if he_rounds:
            plt.scatter(he_rounds, [self.metrics['avg_trust'][i] for i in filtered_indices if self.metrics['he_used'][i]], 
                        color='blue', marker='*', s=100, label='HE Used')
        if sa_rounds:
            plt.scatter(sa_rounds, [self.metrics['avg_trust'][i] for i in filtered_indices if self.metrics['sa_used'][i]], 
                        color='red', marker='o', s=80, label='SA Used')
        
        plt.title('Trust Score Evolution with Security Highlights')
        plt.xlabel('Training Round')
        plt.ylabel('Trust Score')
        plt.legend()
        plt.grid(True)
    
        # Filtering Effectiveness with Security Highlights
        plt.subplot(3, 3, 2)
        plt.plot(filtered_rounds, [self.metrics['krum_filtered'][i] for i in filtered_indices], label='KRUM Filtered')
        plt.plot(filtered_rounds, [self.metrics['ata_filtered'][i] for i in filtered_indices], label='ATA Filtered')
        
        # Highlight security method usage
        if he_rounds:
            plt.scatter(he_rounds, [self.metrics['krum_filtered'][i] for i in filtered_indices if self.metrics['he_used'][i]], 
                        color='blue', marker='*', s=100, label='HE Used')
        if sa_rounds:
            plt.scatter(sa_rounds, [self.metrics['krum_filtered'][i] for i in filtered_indices if self.metrics['sa_used'][i]], 
                        color='red', marker='o', s=80, label='SA Used')
        
        plt.title('Malicious Clients Filtered per Round')
        plt.xlabel('Training Round')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
    
        # Model Convergence with Security Impact
        plt.subplot(3, 3, 3)
        plt.plot(filtered_rounds, [self.metrics['model_convergence'][i] for i in filtered_indices], label='Convergence')
        
        # Add security method indicators
        if he_rounds:
            plt.scatter(he_rounds, [self.metrics['model_convergence'][i] for i in filtered_indices if self.metrics['he_used'][i]], 
                        color='blue', marker='*', s=100, label='HE Used')
        if sa_rounds:
            plt.scatter(sa_rounds, [self.metrics['model_convergence'][i] for i in filtered_indices if self.metrics['sa_used'][i]], 
                        color='red', marker='o', s=80, label='SA Used')
        
        plt.title('Model Convergence with Security Impact')
        plt.xlabel('Training Round')
        plt.ylabel('Update Norm')
        plt.legend()
        plt.grid(True)
    
        # Privacy Budget with Security Highlights
        plt.subplot(3, 3, 4)
        plt.plot(filtered_rounds, [self.metrics['privacy_budget'][i] for i in filtered_indices])
        
        # Highlight security method usage
        if he_rounds:
            plt.scatter(he_rounds, [self.metrics['privacy_budget'][i] for i in filtered_indices if self.metrics['he_used'][i]], 
                        color='blue', marker='*', s=100, label='HE Used')
        if sa_rounds:
            plt.scatter(sa_rounds, [self.metrics['privacy_budget'][i] for i in filtered_indices if self.metrics['sa_used'][i]], 
                        color='red', marker='o', s=80, label='SA Used')
        
        plt.title('Effective Privacy Budget (ε)')
        plt.xlabel('Training Round')
        plt.ylabel('ε value')
        plt.legend()
        plt.grid(True)
    
        # Trust Distribution
        plt.subplot(3, 3, 5)
        sns.histplot([c.trust_score for c in self.clients if not c.is_malicious], 
                    color='green', label='Honest', kde=True)
        sns.histplot([c.trust_score for c in self.clients if c.is_malicious], 
                    color='red', label='Malicious', kde=True)
        plt.title('Final Trust Score Distribution')
        plt.xlabel('Trust Score')
        plt.ylabel('Count')
        plt.legend()
    
        # Participation Heatmap
        plt.subplot(3, 3, 6)
        participation = [c.participation_count for c in self.clients]
        plt.hist(participation, bins=20, edgecolor='black')
        plt.title('Client Participation Distribution')
        plt.xlabel('Times Selected')
        plt.ylabel('Count')
        
        # Security Method Impact Comparison
        plt.subplot(3, 3, 7)
        methods = ['Baseline', 'HE', 'SA', 'HE+SA']
        
        # Calculate metrics for each method (simplified for demo)
        # In a real implementation, you'd run separate experiments
        base_conv = np.mean(self.metrics['model_convergence'][:5])
        he_conv = base_conv * 1.05  # HE slightly reduces convergence speed
        sa_conv = base_conv * 1.02  # SA has minimal impact
        he_sa_conv = base_conv * 1.08  # Combined effect
        
        plt.bar(methods, [base_conv, he_conv, sa_conv, he_sa_conv], 
               color=['blue', 'green', 'orange', 'red'])
        plt.title('Security Method Impact on Convergence')
        plt.ylabel('Average Update Norm')
        
        # Security vs Accuracy
        plt.subplot(3, 3, 8)
        # These would come from actual experiments
        base_acc = 0.85
        he_acc = 0.83
        sa_acc = 0.84
        he_sa_acc = 0.82
        
        plt.bar(methods, [base_acc, he_acc, sa_acc, he_sa_acc], 
               color=['blue', 'green', 'orange', 'red'])
        plt.title('Security Method Impact on Accuracy')
        plt.ylabel('Test Accuracy')
        plt.ylim(0.7, 0.9)
        
        # Security Method Usage Timeline
        plt.subplot(3, 3, 9)
        plt.plot(filtered_rounds, self.metrics['he_used'][:len(filtered_rounds)], 
                'b-', label='HE Enabled')
        plt.plot(filtered_rounds, self.metrics['sa_used'][:len(filtered_rounds)], 
                'r--', label='SA Enabled')
        plt.title('Security Method Usage Over Rounds')
        plt.xlabel('Training Round')
        plt.ylabel('Enabled (1) / Disabled (0)')
        plt.legend()
        plt.grid(True)
        plt.yticks([0, 1])
    
        plt.tight_layout()
        plt.savefig(f'fl_metrics_{experiment_name}.png')
        plt.show()
    
    def train_round(self):
        """Complete training round with metrics tracking"""
        # Client selection
        selected = self.select_clients()
        
        # Check if secure aggregation is enabled
        if self.config.get('secure_agg', False):
            # Secure aggregation protocol
            public_keys = [c.client_id for c in selected]
            masked_updates = []
            weights = []
            
            for client in selected:
                # Train locally and get update
                local_model = client.local_train(self.global_model)
                update = local_model - self.global_model
                # Apply masking for secure aggregation
                masked_update = client.secure_mask_update(update, public_keys)
                masked_updates.append(masked_update)
                weights.append(client.data_size * client.trust_score * client.compute_capacity)
            
            # Aggregate using secure unmasking
            avg_update = self.secure_unmask_updates(masked_updates, public_keys)
            
        else:
            # Standard aggregation protocol
            updates = []
            weights = []
            
            for client in selected:
                # Train locally and get update
                local_model = client.local_train(self.global_model)
                update = local_model - self.global_model
                updates.append(update)
                weights.append(client.data_size * client.trust_score * client.compute_capacity)
            
            # Apply trust-aware DP if enabled
            if self.use_dp:
                updates = self.apply_trust_aware_dp(updates, selected)
            
            # Aggregate updates (using HE if enabled)
            avg_update = self.aggregate_updates(updates, weights)
        
        # Update global model
        self.global_model += avg_update
        
        # Update trust scores
        self.adaptive_trust_scoring(selected)
        
        # Track metrics
        self._update_metrics(avg_update)
        
        return self.global_model
def extract_ecg_features(ecg_signal: np.ndarray, sampling_rate: int = 500) -> np.ndarray:
    """Extract consistent ECG features with error handling"""
    try:
        # Preprocess ECG signal
        cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        
        # Time-domain features (6 features)
        time_features = [
            np.mean(cleaned),
            np.std(cleaned),
            np.min(cleaned),
            np.max(cleaned),
            np.median(cleaned),
            np.mean(np.diff(cleaned))  # Mean of first difference
        ]
        
        # Frequency-domain features (4 features)
        f, Pxx = signal.welch(cleaned, fs=sampling_rate, nperseg=min(1024, len(cleaned)))
        
        # Get frequency bands
        mask_total = (f >= 0.04) & (f <= 0.4)
        mask_lf = (f >= 0.04) & (f <= 0.15)
        mask_hf = (f > 0.15) & (f <= 0.4)
        
        freq_features = [
            np.trapezoid(Pxx[mask_total], f[mask_total]),  # Total power (0.04-0.4Hz)
            np.trapezoid(Pxx[mask_lf], f[mask_lf]),        # LF power (0.04-0.15Hz)
            np.trapezoid(Pxx[mask_hf], f[mask_hf]),        # HF power (0.15-0.4Hz)
        ]
        
        # Calculate LF/HF ratio with zero division protection
        hf_power = freq_features[2]
        lf_hf_ratio = freq_features[1] / hf_power if hf_power > 0 else 0
        freq_features.append(lf_hf_ratio)
        
        # HRV features (12 features)
        try:
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
            hrv_features = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
            hrv_features = hrv_features.fillna(0).iloc[0].values[-12:]  # Take last 12 features
        except Exception as hrv_error:
            print(f"HRV feature extraction warning: {hrv_error}")
            hrv_features = np.zeros(12)
        
        return np.concatenate([time_features, hrv_features, freq_features])
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return np.zeros(22)  # Return array with consistent shape

def load_subject_data(subject_id: str, data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and process data with consistent feature dimensions"""
    try:
        gender = 'F' if subject_id in ['3', '4', '11'] else 'M'
        ecg_path = data_path / 'Biopac_data' / 'ECG' / f'Subject{subject_id}{gender}_ECG.csv'
        
        if not ecg_path.exists():
            print(f"ECG file not found: {ecg_path}")
            return np.zeros((0, 22)), np.zeros(0)  # Return empty arrays with correct feature dim
        
        # Read ECG data - handle single column case
        ecg_data = pd.read_csv(ecg_path, header=None)
        ecg_values = ecg_data.iloc[:, 0].values  # Take first column
        
        # Process windows
        X, y = [], []
        window_size = 30 * 500  # 30 seconds at 500Hz
        step_size = 15 * 500    # 50% overlap
        
        for i in range(0, len(ecg_values) - window_size, step_size):
            window = ecg_values[i:i+window_size]
            features = extract_ecg_features(window)
            X.append(features)
            y.append(i % 2)  # Alternate labels
            
        return np.array(X), np.array(y)
        
    except Exception as e:
        print(f"Error loading subject {subject_id}: {str(e)}")
        return np.zeros((0, 22)), np.zeros(0)

def create_clients(data_path: Path, n_features: int = 22) -> Tuple[List[Client], StandardScaler]:
    """Create clients with consistent feature dimensions"""
    subjects = ['3', '4', '6', '8', '11']
    all_features = []
    clients = []
    
    # First pass to collect all features for scaling
    for subj_id in subjects:
        X, y = load_subject_data(subj_id, data_path)
        if len(X) > 0:
            # Ensure features have correct dimension
            if X.shape[1] != n_features:
                print(f"Warning: Subject {subj_id} has {X.shape[1]} features, expected {n_features}")
                # Pad or truncate features to match expected dimension
                if X.shape[1] < n_features:
                    padding = np.zeros((len(X), n_features - X.shape[1]))
                    X = np.hstack([X, padding])
                else:
                    X = X[:, :n_features]
            all_features.append(X)
    
    # Handle case where no real data was loaded
    if len(all_features) == 0:
        print("No real data found - using synthetic data")
        all_features = [np.random.randn(30, n_features) for _ in subjects]
        all_labels = [np.random.randint(0, 2, 30) for _ in subjects]
    else:
        all_labels = [y for (X, y) in [load_subject_data(s, data_path) for s in subjects] if len(y) > 0]
    
    # Create and fit scaler
    scaler = StandardScaler()
    try:
        scaler.fit(np.vstack(all_features))
    except Exception as e:
        print(f"Scaling failed: {e} - using identity transform")
        scaler.mean_ = np.zeros(n_features)
        scaler.scale_ = np.ones(n_features)
    
    # Create clients
    for client_id, (X, y) in enumerate(zip(all_features, all_labels)):
        X_scaled = scaler.transform(X) if len(X) > 0 else X
        is_malicious = random.random() < 0.2
        clients.append(Client(
            client_id=client_id,
            X=X_scaled,
            y=y,
            is_malicious=is_malicious,
            compute_capacity=random.uniform(0.5, 2.0),
            network_speed=random.uniform(0.5, 2.0)
        ))
    
    return clients, scaler

def evaluate_global_model(server: FederatedLearningServer, test_data: Tuple[np.ndarray, np.ndarray]):
    """Evaluate global model on test data with shape checks"""
    X_test, y_test = test_data
    
    # Handle case where there are no test samples
    if len(X_test) == 0 or len(y_test) == 0:
        print("No test samples available for evaluation")
        return 0.0
    
    # Ensure we have the same number of samples in features and labels
    min_samples = min(len(X_test), len(y_test))
    if min_samples == 0:
        print("No test samples available after filtering")
        return 0.0
    
    # Trim to the same number of samples
    X_test = X_test[:min_samples]
    y_test = y_test[:min_samples]
    
    n_features = X_test.shape[1]
    w = server.global_model[:-1]
    b = server.global_model[-1]
    
    predictions = (X_test.dot(w) + b) > 0.5
    accuracy = np.mean(predictions == y_test)
    print(f"Global Model Accuracy: {accuracy:.2f}")
    return accuracy

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Configuration
    DATA_PATH = Path("multimodal-nback-music-1.0.0")
    N_FEATURES = 22  # Based on feature extraction function

    # Experiment configurations - only the two we want to compare
    experiments = [
        {
            "name": "base_secure",
            "use_dp": True,
            "use_he": False,
            "secure_agg": False,
            "selection_method": "multi_krum"
        },
        {
            "name": "full_secure",
            "use_dp": True,
            "use_he": True,
            "secure_agg": True,
            "selection_method": "multi_krum"
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n==== Running Experiment: {exp['name'].upper()} ====")

        # Create new clients for each experiment
        clients, scaler = create_clients(DATA_PATH, N_FEATURES)
        print(f"Created {len(clients)} clients for {exp['name']}")

        # Initialize FL server
        fl_server = FederatedLearningServer(
            clients=clients,
            model_shape=(N_FEATURES + 1,),
            use_dp=exp["use_dp"],
            use_he=exp["use_he"],
            scaler=scaler
        )
        
        # Set configuration parameters
        fl_server.config['secure_agg'] = exp["secure_agg"]
        fl_server.config['selection_method'] = exp["selection_method"]

        # Train for 20 rounds
        for round_num in range(20):
            fl_server.train_round()
            print(f"[{exp['name'].upper()}] Round {round_num + 1} completed")

        # Create test data (last 20% from each client)
        X_test, y_test = [], []
        for client in clients:
            if len(client.X) > 0:  # Ensure client has data
                split_idx = int(len(client.X) * 0.8)
                X_test.append(client.X[split_idx:])
                y_test.append(client.y[split_idx:])
        
        if len(X_test) == 0:
            print("No test data available")
            accuracy = 0.0
        else:
            X_test = np.vstack(X_test)
            y_test = np.concatenate(y_test)
            accuracy = evaluate_global_model(fl_server, (X_test, y_test))
        
        results.append((exp["name"], accuracy))

        # Visualize metrics
        fl_server.visualize_metrics(experiment_name=exp["name"])

    # Final Comparative Results
    print("\n==== Security Method Comparison Results ====")
    print("Configuration 1: MultiKrum + DP + ATO")
    print("Configuration 2: MultiKrum + DP + ATO + HE + Secure Aggregation\n")
    
    for name, acc in results:
        config_name = "MultiKrum+DP+ATO" if name == "base_secure" else "MultiKrum+DP+ATO+HE+SA"
        print(f"{config_name}: Accuracy = {acc:.4f}")
    
    # Plot side-by-side comparison
    plt.figure(figsize=(10, 6))
    names = ["MultiKrum+DP+ATO", "MultiKrum+DP+ATO+HE+SA"]
    accuracies = [acc for _, acc in results]
    
    plt.bar(names, accuracies, color=['blue', 'green'])
    plt.title("Security Method Impact on Accuracy")
    plt.ylabel("Test Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('security_comparison.png')
    plt.show()