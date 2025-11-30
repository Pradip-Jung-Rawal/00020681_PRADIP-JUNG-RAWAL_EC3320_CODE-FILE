import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class UniversityRecommendationSystem:
    def __init__(self):
        self.df = None
        self.y = None
        self.mlb = None
        self.scaler = None
        self.tfidf_programs = None
        self.svd = None
        self.classifier = None
        self.knn_recommender = None
        self.data_loss_percentage = 0.0
        self.location_features = None
        self.X_classification = None
        self.X_recommender = None

    def load_and_inspect_data(self, filepath="collegedata.csv"):
        """Load and inspect the dataset"""
        print("="*60)
        print("LOADING AND INSPECTING DATA")
        print("="*60)
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_file_path = os.path.join(current_dir, filepath)
            
            print(f"Looking for data file at: {data_file_path}")
            
            if not os.path.exists(data_file_path):
                raise FileNotFoundError(f"Data file not found at: {data_file_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            self.df = None
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(data_file_path, encoding=encoding)
                    print(f"✅ Successfully loaded with encoding: {encoding}")
                    break
                except Exception as e:
                    continue
            
            if self.df is None:
                raise Exception("Could not load CSV with any encoding")
            
            print(f"✅ Dataset shape: {self.df.shape}")
            
            total_cells = self.df.shape[0] * self.df.shape[1]
            missing_cells = self.df.isnull().sum().sum()
            self.data_loss_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            
            print(f"Missing Values: {missing_cells}")
            print(f"Data Loss: {self.data_loss_percentage:.2f}%")
            print(f"Columns: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def preprocess_data(self):
        """Preprocess the dataset"""
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        if self.df is None or self.df.empty:
            print("❌ No data available for preprocessing")
            return
        
        # Handle numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        # Handle categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna('Unknown')

        # Create program list and count
        if 'Programs' in self.df.columns:
            self.df['Programs_list'] = self.df['Programs'].str.split(';')
            self.df['Program_Count'] = self.df['Programs_list'].apply(len)
        else:
            self.df['Programs_list'] = [[] for _ in range(len(self.df))]
            self.df['Program_Count'] = 0

        # Create fee categories
        if 'Fee (USD)' in self.df.columns:
            percentiles = np.percentile(self.df['Fee (USD)'], [20, 40, 60, 80])
            def fee_label(fee):
                if fee < percentiles[0]: return 0
                elif fee < percentiles[1]: return 1
                elif fee < percentiles[2]: return 2
                elif fee < percentiles[3]: return 3
                else: return 4
            self.df['Fee_Category'] = self.df['Fee (USD)'].apply(fee_label)
            self.y = self.df['Fee_Category'].values
        else:
            print("⚠️  'Fee (USD)' column not found")
            self.y = np.zeros(len(self.df))
        
        print("✅ Data preprocessing completed successfully")

    def visualize_data(self):
        """Visualize data distributions"""
        print("\n" + "="*60)
        print("DATA VISUALIZATION")
        print("="*60)
        
        if self.df is None or self.df.empty:
            print("❌ No data available for visualization")
            return
        
        try:
            plt.figure(figsize=(15, 5))
            
            if 'Fee (USD)' in self.df.columns:
                plt.subplot(1, 3, 1)
                sns.histplot(self.df['Fee (USD)'], bins=30, kde=True)
                plt.title('Fee Distribution')
            
            if 'Fee_Category' in self.df.columns:
                plt.subplot(1, 3, 2)
                sns.countplot(x='Fee_Category', data=self.df)
                plt.title('Fee Category Counts')
            
            if 'Program_Count' in self.df.columns:
                plt.subplot(1, 3, 3)
                sns.histplot(self.df['Program_Count'], bins=20, kde=True)
                plt.title('Program Count Distribution')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️  Could not generate visualizations: {e}")

    def create_features(self, train_idx=None):
        """Create features for the model"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        if self.df is None or self.df.empty:
            print("❌ No data available for feature engineering")
            return

        try:
            # Location features
            if 'City' in self.df.columns and 'Country' in self.df.columns:
                self.location_features = pd.get_dummies(self.df[['City', 'Country']], drop_first=True)
            else:
                self.location_features = pd.DataFrame(np.zeros((len(self.df), 1)))

            # Scholarship feature
            scholarship_feature = np.zeros((len(self.df), 1))
            if 'Scholarship' in self.df.columns:
                scholarship_feature = (self.df['Scholarship'].str.lower() == 'yes').astype(int).values.reshape(-1, 1)

            # Program features
            self.mlb = MultiLabelBinarizer()
            if 'Programs_list' in self.df.columns:
                program_bin = self.mlb.fit_transform(self.df['Programs_list'])
            else:
                program_bin = np.zeros((len(self.df), 1))

            # TF-IDF features
            self.tfidf_programs = TfidfVectorizer(max_features=200)
            if 'Programs' in self.df.columns:
                programs_text = self.df['Programs'].str.replace(';', ' ')
            else:
                programs_text = pd.Series([''] * len(self.df))

            if train_idx is not None:
                program_tfidf_train = self.tfidf_programs.fit_transform(programs_text.iloc[train_idx])
                program_tfidf_full = self.tfidf_programs.transform(programs_text)
            else:
                program_tfidf_full = self.tfidf_programs.fit_transform(programs_text)

            # SVD reduction
            n_features = program_tfidf_full.shape[1]
            n_components = min(50, n_features - 1)
            if n_components < 2:
                n_components = 2

            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            if train_idx is not None:
                self.svd.fit(program_tfidf_train)
            program_tfidf_reduced = self.svd.transform(program_tfidf_full)

            # Program count feature
            program_count_feature = np.sqrt(self.df['Program_Count'].values).reshape(-1, 1)

            # Classification features
            self.X_classification = np.hstack([
                self.location_features.values,
                scholarship_feature,
                program_count_feature,
                program_bin,
                program_tfidf_reduced
            ])

            # Recommender features
            fee_feature = np.zeros((len(self.df), 1))
            if 'Fee (USD)' in self.df.columns:
                fee_feature = np.log1p(self.df['Fee (USD)'].values).reshape(-1, 1)

            numeric_features = np.hstack([fee_feature, program_count_feature])
            self.scaler = StandardScaler()
            numeric_scaled = self.scaler.fit_transform(numeric_features)

            self.X_recommender = np.hstack([
                numeric_scaled,
                self.location_features.values,
                scholarship_feature,
                program_bin,
                program_tfidf_reduced
            ])
            
            print("✅ Feature engineering completed successfully")
        except Exception as e:
            print(f"❌ Error in feature engineering: {e}")

    def train_and_evaluate(self):
        """Train and evaluate the models"""
        print("\n" + "="*60)
        print("MODEL TRAINING & EVALUATION")
        print("="*60)

        if self.df is None or self.df.empty:
            print("❌ No data available for training")
            return

        try:
            X_train_idx, X_val_idx, y_train, y_val = train_test_split(
                np.arange(len(self.df)), self.y, test_size=0.3, random_state=42, stratify=self.y
            )

            self.create_features(train_idx=X_train_idx)
            X_train = self.X_classification[X_train_idx]
            X_val = self.X_classification[X_val_idx]

            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            )
            self.classifier.fit(X_train, y_train)

            train_pred = self.classifier.predict(X_train)
            val_pred = self.classifier.predict(X_val)

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            corr, _ = pearsonr(y_val, val_pred)

            print("\n📊 TRAINING RESULTS:")
            print(f"Data Loss: {self.data_loss_percentage:.2f}%")
            print(f"Epoch Correlation: {corr:.4f}")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Training Loss (MSE): {train_mse:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            print(f"Validation Loss (MSE): {val_mse:.4f}")
            
            self.knn_recommender = NearestNeighbors(n_neighbors=min(15, len(self.df)), metric='cosine')
            self.knn_recommender.fit(self.X_recommender)
            
            print("✅ Model training completed successfully")
        except Exception as e:
            print(f"❌ Error in training: {e}")

    def recommend(self, preferences, top_n=5):
        """Generate recommendations based on user preferences"""
        if self.df is None or self.df.empty:
            return []
        
        try:
            user_vec = np.zeros(self.X_recommender.shape[1])

            if 'fee_range' in preferences:
                avg_fee = sum(preferences['fee_range']) / 2
                fee_num = np.log1p(avg_fee)
                prog_count = 1
                scaled_fee = self.scaler.transform([[fee_num, np.sqrt(prog_count)]])[0]
                user_vec[0:2] = scaled_fee

            distances, indices = self.knn_recommender.kneighbors([user_vec], min(top_n, len(self.df)))
            
            recommendations = []
            for i, idx in enumerate(indices[0]):
                rec = {
                    'university': self.df.iloc[idx].get('University Name', 'Unknown'),
                    'city': self.df.iloc[idx].get('City', 'Unknown'),
                    'country': self.df.iloc[idx].get('Country', 'Unknown'),
                    'programs': self.df.iloc[idx].get('Programs', 'N/A'),
                    'fee': float(self.df.iloc[idx].get('Fee (USD)', 0)),
                    'score': float(1 - distances[0][i])
                }
                recommendations.append(rec)
            
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def save_models(self):
        """Save trained models"""
        try:
            artifacts = {
                'classifier': self.classifier,
                'knn_recommender': self.knn_recommender,
                'scaler': self.scaler,
                'tfidf': self.tfidf_programs,
                'mlb': self.mlb,
                'svd': self.svd
            }
            for name, obj in artifacts.items():
                if obj is not None:
                    joblib.dump(obj, f'uni_recommender_{name}.pkl')
            print("✅ Models saved successfully")
        except Exception as e:
            print(f"Error saving models: {e}")


if __name__ == "__main__":
    system = UniversityRecommendationSystem()
    if system.load_and_inspect_data():
        system.preprocess_data()
        system.visualize_data()
        system.train_and_evaluate()
        system.save_models()
    else:
        print("❌ Failed to load data.")
