import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from sklearn.metrics import mean_squared_error
import os
from matplotlib import rcParams

class RainfallVisualizer:
    def __init__(self, model_path='models/rainfall_model.pkl'):
        """Initialize with trained model"""
        os.makedirs('assets', exist_ok=True)
        try:
            self.model = joblib.load(model_path)
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            print("Using dummy model for feature importance visualization")
            class DummyModel:
                feature_names_in_ = ['temperature','humidity','pressure','windspeed']
                feature_importances_ = [0.25, 0.35, 0.2, 0.2]
            self.model = DummyModel()
            
        rcParams['font.family'] = 'sans-serif'
        plt.style.use('ggplot')
    
    def load_data(self, data_path='data/predictions.csv'):
        """Load prediction data"""
        try:
            self.df = pd.read_csv(data_path, parse_dates=['date'])
            return self.df
        except FileNotFoundError:
            print(f"Data file not found at {data_path}")
            print("Generating sample data...")
            dates = pd.date_range(start='2023-01-01', periods=10)
            self.df = pd.DataFrame({
                'date': dates,
                'actual': np.random.uniform(0, 50, 10),
                'predicted': np.random.uniform(0, 50, 10),
                'temperature': np.random.uniform(10, 35, 10),
                'humidity': np.random.uniform(40, 100, 10),
                'pressure': np.random.uniform(995, 1025, 10),
                'windspeed': np.random.uniform(0, 25, 10)
            })
            return self.df
    
    def plot_predictions_vs_actual(self):
        """Create prediction comparison plot"""
        plt.figure(figsize=(12, 6))

        # Line plot for actual vs predicted
        plt.plot(self.df['date'], self.df['actual'], 
                label='Actual Rainfall', linewidth=2, color='#1f77b4')
        plt.plot(self.df['date'], self.df['predicted'], 
                label='Predicted', linestyle='--', color='#ff7f0e')

        # Calculate RMSE (compatible with all scikit-learn versions)
        mse = mean_squared_error(self.df['actual'], self.df['predicted'])
        rmse = np.sqrt(mse)
        
        plt.annotate(f'RMSE: {rmse:.2f} mm', 
                    xy=(0.05, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.title('Actual vs Predicted Rainfall', fontsize=16, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Rainfall (mm)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/predictions_plot.png', dpi=120)
        plt.close()

        # Interactive version
        fig = px.line(self.df, x='date', y=['actual', 'predicted'],
                     title=f'Rainfall Predictions (RMSE: {rmse:.2f}mm)')
        fig.write_html('assets/predictions_interactive.html')

    def plot_residuals(self):
        """Create residual analysis plot"""
        plt.figure(figsize=(12, 6))

        # Residual scatter plot
        residuals = self.df['actual'] - self.df['predicted']
        sns.scatterplot(x=self.df['predicted'], y=residuals,
                       alpha=0.6, color='#2ca02c')

        # Reference lines
        plt.axhline(y=0, color='#d62728', linestyle='--')
        plt.axhline(y=residuals.std(), color='#7f7f7f', linestyle=':')
        plt.axhline(y=-residuals.std(), color='#7f7f7f', linestyle=':')

        plt.title('Residual Analysis', fontsize=16, pad=20)
        plt.xlabel('Predicted Rainfall (mm)', fontsize=12)
        plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/residuals_plot.png', dpi=120)
        plt.close()

    def plot_feature_importance(self):
        """Visualize feature importance"""
        try:
            importance = pd.read_csv('data/feature_importances.csv')
        except FileNotFoundError:
            # Fallback for model feature importance
            importance = pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance,
                   palette='viridis_r')

        plt.title('Feature Importance', fontsize=16, pad=20)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('assets/feature_importance.png', dpi=120)
        plt.close()

if __name__ == "__main__":
    visualizer = RainfallVisualizer()
    data = visualizer.load_data()

    print("Generating visualizations...")
    visualizer.plot_predictions_vs_actual()
    visualizer.plot_residuals()
    visualizer.plot_feature_importance()
    print("Visualizations saved to assets/ folder")