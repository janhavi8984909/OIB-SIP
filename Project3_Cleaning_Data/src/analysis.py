import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class AirbnbAnalyzer:
    def __init__(self, df):
        self.df = df
        self.analysis_results = {}
    
    def price_analysis(self):
        """Analyze price distribution and patterns"""
        price_stats = self.df['price'].describe()
        
        # Price by neighbourhood group
        price_by_area = self.df.groupby('neighbourhood_group')['price'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(2)
        
        # Price by room type
        price_by_room_type = self.df.groupby('room_type')['price'].agg([
            'mean', 'median', 'std'
        ]).round(2)
        
        self.analysis_results['price_stats'] = price_stats
        self.analysis_results['price_by_area'] = price_by_area
        self.analysis_results['price_by_room_type'] = price_by_room_type
        
        return {
            'overall_stats': price_stats,
            'by_area': price_by_area,
            'by_room_type': price_by_room_type
        }
    
    def availability_analysis(self):
        """Analyze availability patterns"""
        availability_stats = self.df['availability_365'].describe()
        
        # Availability by neighbourhood group
        availability_by_area = self.df.groupby('neighbourhood_group')['availability_365'].mean().round(2)
        
        # Correlation between price and availability
        price_availability_corr = self.df['price'].corr(self.df['availability_365'])
        
        self.analysis_results['availability_stats'] = availability_stats
        self.analysis_results['availability_by_area'] = availability_by_area
        self.analysis_results['price_availability_corr'] = price_availability_corr
        
        return {
            'availability_stats': availability_stats,
            'availability_by_area': availability_by_area,
            'price_availability_correlation': price_availability_corr
        }
    
    def host_analysis(self):
        """Analyze host behavior and patterns"""
        # Host listing counts
        host_listings_stats = self.df['calculated_host_listings_count'].describe()
        
        # Top hosts by number of listings
        top_hosts = self.df.groupby('host_id').agg({
            'id': 'count',
            'price': 'mean',
            'number_of_reviews': 'sum'
        }).nlargest(10, 'id')
        
        # Host distribution
        single_listing_hosts = (self.df['calculated_host_listings_count'] == 1).sum()
        multiple_listing_hosts = (self.df['calculated_host_listings_count'] > 1).sum()
        
        self.analysis_results['host_listings_stats'] = host_listings_stats
        self.analysis_results['top_hosts'] = top_hosts
        self.analysis_results['host_distribution'] = {
            'single_listing_hosts': single_listing_hosts,
            'multiple_listing_hosts': multiple_listing_hosts
        }
        
        return {
            'host_listings_stats': host_listings_stats,
            'top_hosts': top_hosts,
            'host_distribution': {
                'single_listing': single_listing_hosts,
                'multiple_listings': multiple_listing_hosts
            }
        }
    
    def review_analysis(self):
        """Analyze review patterns"""
        review_stats = self.df['number_of_reviews'].describe()
        monthly_review_stats = self.df['reviews_per_month'].describe()
        
        # Reviews by neighbourhood group
        reviews_by_area = self.df.groupby('neighbourhood_group')['number_of_reviews'].mean().round(2)
        
        # Correlation between reviews and price
        review_price_corr = self.df['number_of_reviews'].corr(self.df['price'])
        
        self.analysis_results['review_stats'] = review_stats
        self.analysis_results['monthly_review_stats'] = monthly_review_stats
        self.analysis_results['reviews_by_area'] = reviews_by_area
        self.analysis_results['review_price_corr'] = review_price_corr
        
        return {
            'review_stats': review_stats,
            'monthly_review_stats': monthly_review_stats,
            'reviews_by_area': reviews_by_area,
            'review_price_correlation': review_price_corr
        }
    
    def neighbourhood_analysis(self):
        """Analyze neighbourhood patterns"""
        # Listing distribution by neighbourhood group
        area_distribution = self.df['neighbourhood_group'].value_counts()
        
        # Top neighbourhoods by number of listings
        top_neighbourhoods = self.df['neighbourhood'].value_counts().head(10)
        
        # Price distribution by neighbourhood
        neighbourhood_prices = self.df.groupby('neighbourhood')['price'].agg([
            'mean', 'median', 'count'
        ]).round(2).nlargest(10, 'count')
        
        self.analysis_results['area_distribution'] = area_distribution
        self.analysis_results['top_neighbourhoods'] = top_neighbourhoods
        self.analysis_results['neighbourhood_prices'] = neighbourhood_prices
        
        return {
            'area_distribution': area_distribution,
            'top_neighbourhoods': top_neighbourhoods,
            'neighbourhood_prices': neighbourhood_prices
        }
    
    def comprehensive_analysis(self):
        """Run all analyses"""
        print("Starting comprehensive analysis...")
        print("=" * 50)
        
        results = {}
        
        print("1. Price Analysis...")
        results['price'] = self.price_analysis()
        
        print("2. Availability Analysis...")
        results['availability'] = self.availability_analysis()
        
        print("3. Host Analysis...")
        results['host'] = self.host_analysis()
        
        print("4. Review Analysis...")
        results['review'] = self.review_analysis()
        
        print("5. Neighbourhood Analysis...")
        results['neighbourhood'] = self.neighbourhood_analysis()
        
        print("=" * 50)
        print("Analysis completed!")
        
        return results

# Example usage
if __name__ == "__main__":
    df = pd.read_csv('data/processed/cleaned_airbnb.csv')
    analyzer = AirbnbAnalyzer(df)
    results = analyzer.comprehensive_analysis()