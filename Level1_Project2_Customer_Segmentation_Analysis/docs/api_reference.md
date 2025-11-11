# API Reference

## Data Preprocessing Module

### `load_data(file_path)`
Load the customer segmentation dataset.

**Parameters:**
- `file_path` (str): Path to the CSV file

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
df = load_data('data/raw/customer_data.csv')
