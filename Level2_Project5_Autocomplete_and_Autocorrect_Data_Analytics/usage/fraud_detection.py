from src.fraud_detection.data_loader import load_credit_card_data
from src.fraud_detection.models import FraudDetector

# Load data
data = load_credit_card_data()
detector = FraudDetector()
detector.train(data)
