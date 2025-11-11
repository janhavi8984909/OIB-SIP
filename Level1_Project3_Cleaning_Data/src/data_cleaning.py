import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class AirbnbDataCleaner:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.cleaning_report = {}
    
    def basic_info(self):
        """Display basic dataset information"""
        print("Dataset Shape:", self.df.shape)
        print("\nColumns:", self.df.columns.tolist())
        print("\nData Types:")
        print(self.df.dtypes)
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        self.cleaning_report['original_shape'] = self.df.shape
        self.cleaning_report['original_columns'] = self.df.columns.tolist()
        
        return self.df.info()
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        missing_before = self.df.isnull().sum().sum()
        
        # Fill missing values
        self.df['reviews_per_month'] = self.df['reviews_per_month'].fillna(0)
        self.df['last_review'] = self.df['last_review'].fillna('No Review')
        
        # For numerical columns with few missing values, fill with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        missing_after = self.df.isnull().sum().sum()
        
        self.cleaning_report['missing_values_before'] = missing_before
        self.cleaning_report['missing_values_after'] = missing_after
        
        print(f"Missing values handled: {missing_before} -> {missing_after}")
    
    def remove_duplicates(self):
        """Remove duplicate entries"""
        duplicates_before = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        duplicates_after = self.df.duplicated().sum()
        
        self.cleaning_report['duplicates_before'] = duplicates_before
        self.cleaning_report['duplicates_after'] = duplicates_after
        
        print(f"Duplicates removed: {duplicates_before} -> {duplicates_after}")
    
    def handle_outliers(self):
        """Handle outliers in numerical columns"""
        numerical_cols = ['price', 'minimum_nights', 'number_of_reviews', 
                         'reviews_per_month', 'calculated_host_listings_count', 
                         'availability_365']
        
        outlier_report = {}
        
        for col in numerical_cols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                # Cap outliers instead of removing them to preserve data
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                
                outliers_after = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                outlier_report[col] = {
                    'before': outliers_before,
                    'after': outliers_after
                }
        
        self.cleaning_report['outliers'] = outlier_report
        print("Outliers handled for numerical columns")
    
    def create_new_features(self):
        """Create new features for analysis"""
        # Price categories
        self.df['price_category'] = pd.cut(self.df['price'], 
                                         bins=[0, 100, 200, 500, float('inf')],
                                         labels=['Budget', 'Moderate', 'Expensive', 'Luxury'])
        
        # Availability status
        self.df['availability_status'] = np.where(self.df['availability_365'] > 180, 
                                                'High', 'Low')
        
        # Host experience (based on number of reviews)
        self.df['host_experience'] = pd.cut(self.df['number_of_reviews'],
                                          bins=[-1, 0, 10, 50, float('inf')],
                                          labels=['New', 'Beginner', 'Experienced', 'Veteran'])
        
        print("New features created: price_category, availability_status, host_experience")
    
    def data_quality_check(self):
        """Perform data quality checks"""
        quality_report = {}
        
        # Check for negative values in numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            negative_count = (self.df[col] < 0).sum()
            quality_report[f'{col}_negative_values'] = negative_count
        
        # Check for unrealistic values
        unrealistic_price = (self.df['price'] == 0).sum()
        unrealistic_min_nights = (self.df['minimum_nights'] > 365).sum()
        
        quality_report['zero_prices'] = unrealistic_price
        quality_report['min_nights_over_year'] = unrealistic_min_nights
        
        self.cleaning_report['quality_check'] = quality_report
        
        print("Data quality check completed")
    
    def clean_data(self):
        """Execute complete cleaning pipeline"""
        print("Starting data cleaning process...")
        print("=" * 50)
        
        self.basic_info()
        print("\n" + "=" * 50)
        
        self.remove_duplicates()
        print("=" * 50)
        
        self.handle_missing_values()
        print("=" * 50)
        
        self.handle_outliers()
        print("=" * 50)
        
        self.create_new_features()
        print("=" * 50)
        
        self.data_quality_check()
        print("=" * 50)
        
        print("Data cleaning completed!")
        print(f"Final dataset shape: {self.df.shape}")
        
        return self.df
    
    def save_cleaned_data(self, output_path):
        """Save cleaned dataset"""
        self.df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    
    def get_cleaning_report(self):
        """Return detailed cleaning report"""
        return self.cleaning_report

# Example usage
if __name__ == "__main__":
    cleaner = AirbnbDataCleaner('data/raw/AB_NYC_2019.csv')
    cleaned_df = cleaner.clean_data()
    cleaner.save_cleaned_data('data/processed/cleaned_airbnb.csv')
    
    report = cleaner.get_cleaning_report()
    print("\nCleaning Report Summary:")
    for key, value in report.items():
        print(f"{key}: {value}")