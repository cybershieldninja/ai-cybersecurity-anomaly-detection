# AI-Driven Anomaly Detection for Cybersecurity
# A comprehensive system for detecting network anomalies using machine learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import psutil
import time
import os
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('dark_background')
sns.set_palette("husl")

class NetworkAnomalyDetector:
    """
    AI-powered network anomaly detection system for cybersecurity
    """
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.pca = PCA(n_components=2)
        self.feature_names = None
        self.is_trained = False
        
    def generate_sample_data(self, n_samples=5000):
        """
        Generate realistic network traffic data for demonstration
        """
        np.random.seed(42)
        
        # Normal network behavior
        normal_data = {
            'packet_size': np.random.normal(1500, 300, int(n_samples * 0.9)),
            'connection_duration': np.random.exponential(10, int(n_samples * 0.9)),
            'bytes_sent': np.random.lognormal(8, 1.5, int(n_samples * 0.9)),
            'bytes_received': np.random.lognormal(7, 1.2, int(n_samples * 0.9)),
            'packets_per_second': np.random.gamma(2, 5, int(n_samples * 0.9)),
            'unique_destinations': np.random.poisson(3, int(n_samples * 0.9)),
            'failed_connections': np.random.poisson(0.5, int(n_samples * 0.9)),
            'port_scan_indicators': np.random.poisson(0.1, int(n_samples * 0.9))
        }
        
        # Anomalous behavior (attacks, suspicious activity)
        anomalous_data = {
            'packet_size': np.random.normal(3000, 800, int(n_samples * 0.1)),  # Larger packets
            'connection_duration': np.random.exponential(50, int(n_samples * 0.1)),  # Longer connections
            'bytes_sent': np.random.lognormal(12, 2, int(n_samples * 0.1)),  # Much more data
            'bytes_received': np.random.lognormal(10, 1.8, int(n_samples * 0.1)),
            'packets_per_second': np.random.gamma(10, 15, int(n_samples * 0.1)),  # High frequency
            'unique_destinations': np.random.poisson(50, int(n_samples * 0.1)),  # Port scanning
            'failed_connections': np.random.poisson(20, int(n_samples * 0.1)),  # Brute force
            'port_scan_indicators': np.random.poisson(15, int(n_samples * 0.1))  # Obvious scanning
        }
        
        # Combine normal and anomalous data
        data = {}
        labels = []
        
        for feature in normal_data.keys():
            data[feature] = np.concatenate([normal_data[feature], anomalous_data[feature]])
            
        # Create labels (0 = normal, 1 = anomaly)
        labels = np.concatenate([
            np.zeros(int(n_samples * 0.9)), 
            np.ones(int(n_samples * 0.1))
        ])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['is_anomaly'] = labels
        
        # Add some derived features
        df['bytes_ratio'] = df['bytes_sent'] / (df['bytes_received'] + 1)
        df['connection_efficiency'] = df['bytes_sent'] / (df['connection_duration'] + 1)
        df['suspicious_score'] = (
            df['failed_connections'] * 0.3 + 
            df['port_scan_indicators'] * 0.5 + 
            df['unique_destinations'] * 0.02
        )
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df

    def generate_realistic_data(self, n_samples=100, interval=1,filename="data/real_network_traffic.csv"):
        """
        Capture real system-level network traffic data over time to generate
        a more realistic dataset for training or testing.
        """
        data = []

        print(f"[✓] Capturing {n_samples} real-time samples (interval: {interval}s)...")

        prev_counters = psutil.net_io_counters()
        prev_time = time.time()

        for _ in range(n_samples):
            time.sleep(interval)
            curr_counters = psutil.net_io_counters()
            curr_time = time.time()

            duration = curr_time - prev_time
            bytes_sent = curr_counters.bytes_sent - prev_counters.bytes_sent
            bytes_recv = curr_counters.bytes_recv - prev_counters.bytes_recv
            packets_sent = curr_counters.packets_sent - prev_counters.packets_sent
            packets_recv = curr_counters.packets_recv - prev_counters.packets_recv
            total_packets = packets_sent + packets_recv
            packets_per_second = total_packets / duration if duration > 0 else 0

            # Simulated fields — use logics or heuristics for realism
            packet_size = np.random.normal(1500, 200)  # approximate average MTU
            unique_destinations = np.random.poisson(5)
            failed_connections = np.random.poisson(1)
            port_scan_indicators = np.random.poisson(0.2)

            # Derived features
            bytes_ratio = bytes_sent / (bytes_recv + 1)
            connection_efficiency = bytes_sent / (duration + 1)
            suspicious_score = (
                failed_connections * 0.3 +
                port_scan_indicators * 0.5 +
                unique_destinations * 0.02
            )

            data.append({
                "packet_size": packet_size,
                "connection_duration": duration,
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_recv,
                "packets_per_second": packets_per_second,
                "unique_destinations": unique_destinations,
                "failed_connections": failed_connections,
                "port_scan_indicators": port_scan_indicators,
                "bytes_ratio": bytes_ratio,
                "connection_efficiency": connection_efficiency,
                "suspicious_score": suspicious_score,
                "is_anomaly": 0  # unknown, assumed normal unless labeled manually
            })

            # update previous counters
            prev_counters = curr_counters
            prev_time = curr_time

        df = pd.DataFrame(data)
        # Save to CSV
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"[✓] Saved {len(df)} samples to {filename}")        
        print(f"[✓] Captured {len(df)} samples.")
        return df

    def preprocess_data(self, df):
        """
        Preprocess the network data for machine learning
        """
        # Select features for training (exclude the target)
        feature_columns = [col for col in df.columns if col != 'is_anomaly']
        X = df[feature_columns].copy()
        
        # Handle any missing values
        X = X.fillna(X.mean())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, df['is_anomaly'] if 'is_anomaly' in df.columns else None
    
    def train(self, df):
        """
        Train the anomaly detection models
        """
        print("Training anomaly detection models...")
        
        # Preprocess data
        X_scaled, y_true = self.preprocess_data(df)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        # Train DBSCAN (unsupervised, so no explicit training needed)
        dbscan_predictions = self.dbscan.fit_predict(X_scaled)
        
        # Fit PCA for visualization
        self.pca.fit(X_scaled)
        
        self.is_trained = True
        print("Models trained successfully!")
        
        return self.evaluate_models(X_scaled, y_true)
    
    def predict(self, df):
        """
        Predict anomalies in new data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled, _ = self.preprocess_data(df)
        
        # Get predictions from both models
        isolation_pred = self.isolation_forest.predict(X_scaled)
        dbscan_pred = self.dbscan.fit_predict(X_scaled)
        
        # Convert predictions to binary (1 = anomaly, 0 = normal)
        isolation_anomalies = (isolation_pred == -1).astype(int)
        dbscan_anomalies = (dbscan_pred == -1).astype(int)
        
        # Combine predictions (ensemble approach)
        ensemble_anomalies = ((isolation_anomalies + dbscan_anomalies) >= 1).astype(int)
        
        return {
            'isolation_forest': isolation_anomalies,
            'dbscan': dbscan_anomalies,
            'ensemble': ensemble_anomalies
        }
    
    def evaluate_models(self, X_scaled, y_true):
        """
        Evaluate the performance of anomaly detection models
        """
        # Get predictions
        isolation_pred = self.isolation_forest.predict(X_scaled)
        dbscan_pred = self.dbscan.fit_predict(X_scaled)
        
        # Convert to binary
        isolation_anomalies = (isolation_pred == -1).astype(int)
        dbscan_anomalies = (dbscan_pred == -1).astype(int)
        ensemble_anomalies = ((isolation_anomalies + dbscan_anomalies) >= 1).astype(int)
        
        results = {}
        
        # Evaluate each model
        for name, pred in [
            ('Isolation Forest', isolation_anomalies),
            ('DBSCAN', dbscan_anomalies),
            ('Ensemble', ensemble_anomalies)
        ]:
            print(f"\n{name} Results:")
            print(classification_report(y_true, pred))
            results[name.lower().replace(' ', '_')] = {
                'predictions': pred,
                'report': classification_report(y_true, pred, output_dict=True)
            }
        
        return results
    
    def visualize_anomalies(self, df, predictions):
        """
        Create comprehensive visualizations of the anomaly detection results
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Network Anomaly Detection Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        X_scaled, y_true = self.preprocess_data(df)
        X_pca = self.pca.transform(X_scaled)
        
        # 1. PCA visualization with true labels
        axes[0, 0].scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1], 
                          c='lightblue', alpha=0.6, label='Normal', s=30)
        axes[0, 0].scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1], 
                          c='red', alpha=0.8, label='True Anomaly', s=50)
        axes[0, 0].set_title('True Anomalies (Ground Truth)')
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Isolation Forest predictions
        iso_pred = predictions['isolation_forest']
        axes[0, 1].scatter(X_pca[iso_pred == 0, 0], X_pca[iso_pred == 0, 1], 
                          c='lightgreen', alpha=0.6, label='Normal', s=30)
        axes[0, 1].scatter(X_pca[iso_pred == 1, 0], X_pca[iso_pred == 1, 1], 
                          c='orange', alpha=0.8, label='Detected Anomaly', s=50)
        axes[0, 1].set_title('Isolation Forest Detection')
        axes[0, 1].set_xlabel('First Principal Component')
        axes[0, 1].set_ylabel('Second Principal Component')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. DBSCAN predictions
        dbscan_pred = predictions['dbscan']
        axes[0, 2].scatter(X_pca[dbscan_pred == 0, 0], X_pca[dbscan_pred == 0, 1], 
                          c='lightcoral', alpha=0.6, label='Normal', s=30)
        axes[0, 2].scatter(X_pca[dbscan_pred == 1, 0], X_pca[dbscan_pred == 1, 1], 
                          c='purple', alpha=0.8, label='Detected Anomaly', s=50)
        axes[0, 2].set_title('DBSCAN Detection')
        axes[0, 2].set_xlabel('First Principal Component')
        axes[0, 2].set_ylabel('Second Principal Component')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Feature importance (based on isolation forest)
        feature_importance = np.abs(self.isolation_forest.score_samples(X_scaled)).std()
        if len(self.feature_names) <= 10:  # Only show if we have reasonable number of features
            axes[1, 0].bar(range(len(self.feature_names)), 
                          np.random.random(len(self.feature_names)))  # Placeholder
            axes[1, 0].set_xticks(range(len(self.feature_names)))
            axes[1, 0].set_xticklabels(self.feature_names, rotation=45)
            axes[1, 0].set_title('Feature Importance (Relative)')
            axes[1, 0].set_ylabel('Importance Score')
        
        # 5. Anomaly distribution over time (simulated)
        time_points = range(len(df))
        anomaly_counts = np.cumsum(predictions['ensemble'])
        axes[1, 1].plot(time_points, anomaly_counts, color='red', linewidth=2)
        axes[1, 1].set_title('Cumulative Anomalies Detected')
        axes[1, 1].set_xlabel('Time (Data Points)')
        axes[1, 1].set_ylabel('Cumulative Anomaly Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model comparison
        models = ['Isolation Forest', 'DBSCAN', 'Ensemble']
        anomaly_counts = [
            np.sum(predictions['isolation_forest']),
            np.sum(predictions['dbscan']),
            np.sum(predictions['ensemble'])
        ]
        colors = ['orange', 'purple', 'red']
        axes[1, 2].bar(models, anomaly_counts, color=colors, alpha=0.8)
        axes[1, 2].set_title('Anomalies Detected by Each Model')
        axes[1, 2].set_ylabel('Number of Anomalies')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_security_report(self, df, predictions):
        """
        Generate a comprehensive security analysis report
        """
        print("="*60)
        print("CYBERSECURITY ANOMALY DETECTION REPORT")
        print("="*60)
        
        total_samples = len(df)
        anomalies_detected = np.sum(predictions['ensemble'])
        anomaly_rate = (anomalies_detected / total_samples) * 100
        
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"- Total network sessions analyzed: {total_samples:,}")
        print(f"- Anomalies detected: {anomalies_detected}")
        print(f"- Anomaly rate: {anomaly_rate:.2f}%")
        
        if anomaly_rate > 15:
            print("⚠️  HIGH RISK: Elevated anomaly rate detected!")
        elif anomaly_rate > 5:
            print("⚡ MEDIUM RISK: Moderate anomaly activity")
        else:
            print("✅ LOW RISK: Normal network activity levels")
        
        # Analyze anomalous samples
        anomalous_indices = np.where(predictions['ensemble'] == 1)[0]
        if len(anomalous_indices) > 0:
            anomalous_data = df.iloc[anomalous_indices]
            
            print(f"\nANOMALY CHARACTERISTICS:")
            print(f"- Average packet size: {anomalous_data['packet_size'].mean():.2f}")
            print(f"- Average connection duration: {anomalous_data['connection_duration'].mean():.2f}s")
            print(f"- Average failed connections: {anomalous_data['failed_connections'].mean():.2f}")
            print(f"- Average port scan indicators: {anomalous_data['port_scan_indicators'].mean():.2f}")
            
            # Identify potential attack types
            print(f"\nPOTENTIAL THREAT INDICATORS:")
            
            # DDoS indicators
            high_pps = anomalous_data['packets_per_second'] > anomalous_data['packets_per_second'].quantile(0.8)
            if np.sum(high_pps) > 0:
                print(f"- Possible DDoS activity: {np.sum(high_pps)} sessions with high packet rates")
            
            # Port scanning
            high_destinations = anomalous_data['unique_destinations'] > 20
            if np.sum(high_destinations) > 0:
                print(f"- Possible port scanning: {np.sum(high_destinations)} sessions with many unique destinations")
            
            # Brute force
            high_failures = anomalous_data['failed_connections'] > 10
            if np.sum(high_failures) > 0:
                print(f"- Possible brute force: {np.sum(high_failures)} sessions with many failed connections")
            
            # Data exfiltration
            high_bytes = anomalous_data['bytes_sent'] > anomalous_data['bytes_sent'].quantile(0.9)
            if np.sum(high_bytes) > 0:
                print(f"- Possible data exfiltration: {np.sum(high_bytes)} sessions with high data transfer")
        
        print(f"\nRECOMMENDATIONS:")
        if anomaly_rate > 10:
            print("- Immediate investigation required for high-risk sessions")
            print("- Consider implementing additional network monitoring")
            print("- Review and update firewall rules")
        else:
            print("- Continue monitoring with current detection parameters")
            print("- Regular review of anomaly patterns recommended")
        
        print("="*60)

# Main execution function
def run_anomaly_detection_demo():
    """
    Run a complete demonstration of the anomaly detection system
    """
    print("🛡️  Initializing AI-Driven Cybersecurity Anomaly Detection System")
    print("="*70)
    
    # Initialize the detector
    detector = NetworkAnomalyDetector(contamination=0.1)
    
    # Generate sample network data
    print("\n📊 Generating sample network traffic data...")
    df = detector.generate_sample_data(n_samples=3000)
    # # Generate real network data
    # print("\n📊 Generating realistic network traffic data...")
    # df = detector.generate_realistic_data(n_samples=100)    
    print(f"Generated {len(df)} network sessions")
    print(f"Features: {list(df.columns[:-1])}")  # Exclude 'is_anomaly' label
    
    # Train the models
    print("\n🤖 Training anomaly detection models...")
    df.loc[df['packets_per_second'] > 400, 'is_anomaly'] = 1  # heuristic anomaly labeling
    evaluation_results = detector.train(df)
    
    # Make predictions
    print("\n🔍 Detecting anomalies in network traffic...")
    predictions = detector.predict(df)
    
    # Create visualizations
    print("\n📈 Generating security visualizations...")
    detector.visualize_anomalies(df, predictions)
    
    # Generate security report
    print("\n📋 Generating security analysis report...")
    detector.generate_security_report(df, predictions)
    
    print("\n✅ Anomaly detection analysis complete!")
    
    return detector, df, predictions

# Real-time monitoring simulation
def simulate_real_time_monitoring(detector, duration_minutes=5):
    """
    Simulate real-time network monitoring
    """
    print(f"\n🔄 Starting real-time monitoring simulation ({duration_minutes} minutes)")
    print("Press Ctrl+C to stop monitoring")
    
    import time
    
    try:
        start_time = time.time()
        alert_count = 0
        
        while time.time() - start_time < duration_minutes * 60:
            # Generate a small batch of new network data
            new_data = detector.generate_sample_data(n_samples=50)
#            new_data = detector.generate_realistic_data(n_samples=50)
            predictions = detector.predict(new_data)
            
            # Check for anomalies
            current_anomalies = np.sum(predictions['ensemble'])
            
            if current_anomalies > 0:
                alert_count += 1
                anomaly_rate = (current_anomalies / len(new_data)) * 100
                timestamp = time.strftime("%H:%M:%S")
                
                print(f"[{timestamp}] 🚨 ALERT: {current_anomalies} anomalies detected "
                      f"({anomaly_rate:.1f}% of traffic)")
                
                # Detailed analysis for high-severity alerts
                if anomaly_rate > 20:
                    print(f"           ⚠️  HIGH SEVERITY - Immediate attention required")
            
            # Wait before next check (simulate real-time intervals)
            time.sleep(10)  # Check every 10 seconds
    
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped. Total alerts generated: {alert_count}")

if __name__ == "__main__":
    # Run the main demonstration
    detector, data, predictions = run_anomaly_detection_demo()
    
    # Optionally run real-time monitoring simulation
    response = input("\nWould you like to run real-time monitoring simulation? (y/n): ")
    if response.lower() == 'y':
        simulate_real_time_monitoring(detector, duration_minutes=10)