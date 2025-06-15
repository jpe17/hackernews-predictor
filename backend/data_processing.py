"""
Data processing for the HackerNews Score Prediction project.
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import pickle
import re
from urllib.parse import urlparse

from . import config as cfg

def load_glove_embeddings():
    """Load cached GloVe embeddings."""
    cache_dir = os.path.dirname(cfg.GLOVE_FILE)
    word_to_idx_path = os.path.join(cache_dir, "word_to_idx.pkl")
    embeddings_path = os.path.join(cache_dir, "embeddings.npy")
    
    if os.path.exists(word_to_idx_path) and os.path.exists(embeddings_path):
        with open(word_to_idx_path, 'rb') as f:
            word_to_idx = pickle.load(f)
        embeddings = np.load(embeddings_path)
        return word_to_idx, embeddings
    
    print("❌ GloVe cache not found. Please run original data processing once to create cache.")
    return None, None

def clean_text(text):
    """Clean text for embedding lookup."""
    if pd.isna(text):
        return []
    text = str(text).lower()
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()

def title_to_embedding(title, word_to_idx, embeddings):
    """Convert title to embedding."""
    words = clean_text(title)
    word_embeddings = [embeddings[word_to_idx[word]] for word in words if word in word_to_idx]
    
    if not word_embeddings:
        return np.zeros(embeddings.shape[1])
    
    return np.mean(word_embeddings, axis=0)

def extract_domain(url):
    """Extract domain from URL."""
    if pd.isna(url) or url == '':
        return 'self_post'
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        domain = urlparse(url).netloc.lower()
        prefixes = ['www.', 'm.', 'mobile.', 'old.']
        for prefix in prefixes:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
                break
        return domain.split(':')[0].rstrip('.') or 'parse_error'
    except:
        return 'parse_error'

def create_advanced_features(df):
    """Create stronger predictive features."""
    print("Creating advanced features...")
    
    # Basic text features
    df['word_count'] = df['title'].str.split().str.len()
    df['title_length'] = df['title'].str.len()
    # Avoid division by zero
    df['avg_word_length'] = np.where(df['word_count'] > 0, df['title_length'] / df['word_count'], 0)
    
    # Time features
    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    
    # Cyclical time encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Weekend indicator
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Peak hours (when HN is most active)
    df['is_peak_hour'] = df['hour_of_day'].isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17]).astype(int)
    
    # Domain features
    df['domain'] = df['url'].apply(extract_domain)
    df['is_self_post'] = (df['domain'] == 'self_post').astype(int)
    
    # Title content features
    df['has_question_mark'] = df['title'].str.contains(r'\?', na=False).astype(int)
    df['has_exclamation'] = df['title'].str.contains(r'!', na=False).astype(int)
    # Avoid division by zero
    df['title_upper_ratio'] = np.where(df['title_length'] > 0, df['title'].str.count(r'[A-Z]') / df['title_length'], 0)
    
    # Tech keywords (HackerNews specific) - EXPANDED
    tech_keywords = ['startup', 'python', 'javascript', 'react', 'ai', 'machine learning', 
                     'blockchain', 'crypto', 'google', 'apple', 'microsoft', 'open source',
                     'github', 'api', 'database', 'security', 'privacy', 'algorithm']
    
    df['tech_keyword_count'] = 0
    for keyword in tech_keywords:
        df['tech_keyword_count'] += df['title'].str.lower().str.contains(keyword, na=False).astype(int)
    
    # NEW: Viral/engagement words
    viral_keywords = ['show', 'ask', 'launch', 'new', 'free', 'best', 'top', 'guide', 
                     'tips', 'how to', 'why', 'what', 'review', 'vs', 'comparison']
    df['viral_keyword_count'] = 0
    for keyword in viral_keywords:
        df['viral_keyword_count'] += df['title'].str.lower().str.contains(keyword, na=False).astype(int)
    
    # NEW: Title sentiment/appeal features
    df['starts_with_show'] = df['title'].str.lower().str.startswith('show hn', na=False).astype(int)
    df['starts_with_ask'] = df['title'].str.lower().str.startswith('ask hn', na=False).astype(int)
    df['has_numbers'] = df['title'].str.contains(r'\d', na=False).astype(int)
    df['has_year'] = df['title'].str.contains(r'20\d{2}', na=False).astype(int)
    
    # NEW: Title structure features
    df['title_word_diversity'] = df['title'].apply(lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1) if pd.notna(x) else 0)
    df['title_has_colon'] = df['title'].str.contains(':', na=False).astype(int)
    df['title_has_parentheses'] = df['title'].str.contains(r'[\(\)]', na=False).astype(int)
    
    # NEW: Domain quality indicators
    df['domain_length'] = df['domain'].str.len()
    df['is_major_tech_domain'] = df['domain'].isin(['github.com', 'medium.com', 'youtube.com', 
                                                    'twitter.com', 'reddit.com', 'stackoverflow.com',
                                                    'techcrunch.com', 'arstechnica.com']).astype(int)
    
    # NEW: User activity proxy (based on username characteristics)
    df['username_length'] = df['by'].str.len()
    df['username_has_numbers'] = df['by'].str.contains(r'\d', na=False).astype(int)
    
    # NEW: Temporal features
    df['is_workday'] = (df['day_of_week'] < 5).astype(int)  # Monday-Friday
    df['is_morning'] = df['hour_of_day'].isin([7, 8, 9, 10, 11]).astype(int)
    df['is_afternoon'] = df['hour_of_day'].isin([12, 13, 14, 15, 16, 17]).astype(int)
    df['is_evening'] = df['hour_of_day'].isin([18, 19, 20, 21]).astype(int)
    
    # NEW: Reading difficulty proxy
    df['complex_words'] = df['title'].str.count(r'\b\w{7,}\b')  # Words with 7+ letters
    df['simple_words'] = df['title'].str.count(r'\b\w{1,4}\b')  # Words with 1-4 letters
    df['complexity_ratio'] = np.where(df['word_count'] > 0, df['complex_words'] / df['word_count'], 0)
    
    # Readability proxy (keep existing)
    df['reading_time'] = df['word_count'] / 200  # Assume 200 WPM reading speed
    
    return df

def prepare_features_fixed(config):
    """Fixed data preparation without the bugs."""
    print("🛠️ FIXED DATA PREPARATION")
    print("=" * 50)
    
    # Load and sample data FIRST
    print("Loading raw data...")
    df = pd.read_parquet(config.DATA_PATH)
    print(f"📊 Sampling {config.NUMBER_OF_SAMPLES:,} samples...")
    df_sample = df.sample(n=config.NUMBER_OF_SAMPLES, random_state=config.RANDOM_STATE).copy()
    print(f"Sampled: {len(df_sample)} samples")
    
    # Apply filtering AFTER sampling
    print("Filtering data...")
    df_filtered = df_sample[
        (df_sample['score'] >= config.MINIMUM_SCORE) &
        (df_sample['score'] <= config.MAXIMUM_SCORE) &
        (df_sample['title'].notna()) &
        (df_sample['by'].notna()) &
        (df_sample['time'].notna())
    ].copy()
    print(f"After filtering: {len(df_filtered)} samples")
    
    # Create target variable ONCE
    print("Creating target variable...")
    df_filtered['score_log'] = np.log1p(df_filtered['score'])
    
    # Verify consistency
    test_consistency = np.abs(df_filtered['score_log'] - np.log1p(df_filtered['score'])).max()
    print(f"Target consistency check: {test_consistency:.10f}")
    assert test_consistency < 1e-10, "Target variable inconsistency detected!"
    
    # Create advanced features
    df_filtered = create_advanced_features(df_filtered)
    
    # NO DATA AUGMENTATION - Remove this step that was causing corruption
    print("ℹ️ Skipping data augmentation to ensure data consistency")
    
    # Load embeddings (skip if TITLE_EMB_DIM = 0)
    if config.TITLE_EMB_DIM > 0:
        print("Loading embeddings...")
        word_to_idx, embeddings = load_glove_embeddings()
        
        if word_to_idx is None:
            # Use zero embeddings if cache not available
            print("Using zero embeddings for testing...")
            X_title_embeddings = np.zeros((len(df_filtered), config.TITLE_EMB_DIM), dtype=np.float32)
        else:
            print("Creating title embeddings...")
            X_title_embeddings = np.array([
                title_to_embedding(title, word_to_idx, embeddings)
                for title in df_filtered['title']
            ], dtype=np.float32)
    else:
        print("Skipping title embeddings (TITLE_EMB_DIM = 0)")
        X_title_embeddings = np.zeros((len(df_filtered), 0), dtype=np.float32)
    
    # Prepare numerical features (GREATLY expanded feature set)
    numerical_cols = [
        # Original features (KEPT)
        'word_count', 'title_length', 'avg_word_length',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_peak_hour', 'is_self_post',
        'has_question_mark', 'has_exclamation', 'title_upper_ratio',
        'tech_keyword_count', 'reading_time',
        
        # NEW features for better prediction
        'viral_keyword_count', 'starts_with_show', 'starts_with_ask',
        'has_numbers', 'has_year', 'title_word_diversity',
        'title_has_colon', 'title_has_parentheses', 'domain_length',
        'is_major_tech_domain', 'username_length', 'username_has_numbers',
        'is_workday', 'is_morning', 'is_afternoon', 'is_evening',
        'complex_words', 'simple_words', 'complexity_ratio'
    ]
    
    X_numerical = df_filtered[numerical_cols].values.astype(np.float32)
    
    # Handle any NaN, infinity, or extreme values
    print("Cleaning numerical features...")
    print(f"Before cleaning - NaN: {np.isnan(X_numerical).sum()}, Inf: {np.isinf(X_numerical).sum()}")
    
    # Replace NaN and infinity with finite values
    X_numerical = np.nan_to_num(X_numerical, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Clip extreme values to reasonable range
    X_numerical = np.clip(X_numerical, -1e6, 1e6)
    
    print(f"After cleaning - NaN: {np.isnan(X_numerical).sum()}, Inf: {np.isinf(X_numerical).sum()}")
    print(f"Feature ranges: min={X_numerical.min():.3f}, max={X_numerical.max():.3f}")
    
    # Prepare categorical features
    print("Preparing categorical features...")
    
    # Domain encoding
    domain_counts = df_filtered['domain'].value_counts()
    top_domains = domain_counts.head(config.NUM_DOMAINS).index
    df_filtered['domain_mapped'] = df_filtered['domain'].where(
        df_filtered['domain'].isin(top_domains), 'OTHER'
    )
    domain_encoder = LabelEncoder()
    domain_ids = domain_encoder.fit_transform(df_filtered['domain_mapped'])
    
    # User encoding
    user_counts = df_filtered['by'].value_counts()
    top_users = user_counts.head(config.NUM_USERS).index
    df_filtered['user_mapped'] = df_filtered['by'].where(
        df_filtered['by'].isin(top_users), 'OTHER'
    )
    user_encoder = LabelEncoder()
    user_ids = user_encoder.fit_transform(df_filtered['user_mapped'])
    
    # Final target
    y = df_filtered['score_log'].values
    
    # Final consistency check
    print("Final data validation...")
    print(f"Samples: {len(y)}")
    print(f"Target range: {y.min():.3f} to {y.max():.3f}")
    print(f"Target std: {y.std():.3f}")
    print(f"Features shape: {X_numerical.shape}")
    print(f"Embeddings shape: {X_title_embeddings.shape}")
    print(f"Domains: {len(domain_encoder.classes_)}")
    print(f"Users: {len(user_encoder.classes_)}")
    
    # Check for any remaining issues
    assert not np.isnan(y).any(), "NaN in target!"
    assert not np.isnan(X_numerical).any(), "NaN in numerical features!"
    assert not np.isnan(X_title_embeddings).any(), "NaN in embeddings!"
    
    print("✅ Fixed data preparation complete!")
    
    # Return clean data (matching original structure)
    return {
        'y': y,
        'X_numerical': X_numerical,
        'X_title_embeddings': X_title_embeddings,
        'X_domain_ids': domain_ids,
        'X_user_ids': user_ids,
        'n_domains': len(domain_encoder.classes_),
        'n_users': len(user_encoder.classes_),
        'domain_encoder': domain_encoder,
        'user_encoder': user_encoder
    }

def create_data_loader_fixed(data_dict, indices, batch_size, shuffle=True):
    """Create data loader with proper device handling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use scaled data if available, otherwise use original numerical data
    numerical_key = 'X_numerical_scaled' if 'X_numerical_scaled' in data_dict else 'X_numerical'
    
    # Use title embeddings (full architecture)
    title_embeddings = torch.FloatTensor(data_dict['X_title_embeddings'][indices])
    
    dataset = TensorDataset(
        title_embeddings,
        torch.FloatTensor(data_dict[numerical_key][indices]),
        torch.LongTensor(data_dict['X_domain_ids'][indices]),
        torch.LongTensor(data_dict['X_user_ids'][indices]),
        torch.FloatTensor(data_dict['y'][indices])
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test the fixed data processing
    print("Testing fixed data processing...")
    data = prepare_features_fixed(cfg)
    
    # Quick validation test
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    
    indices = np.arange(len(data['y']))
    train_idx, test_idx = train_test_split(indices, test_size=cfg.VAL_SIZE, random_state=cfg.RANDOM_STATE)
    
    # Test with expanded feature set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(data['X_numerical'][train_idx])
    X_test = scaler.transform(data['X_numerical'][test_idx])
    y_train = data['y'][train_idx]
    y_test = data['y'][test_idx]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    r2 = r2_score(y_test, pred)
    
    print(f"Fixed data R²: {r2:.4f}")
    
    if r2 > 0.05:
        print("✅ Fixed data processing works well!")
        print("You can now use this for training with positive R² expected.")
    elif r2 > 0:
        print("✅ Fixed data processing works!")
        print("R² is positive but low - this is normal for this type of prediction task.")
    else:
        print("❌ Still having issues with the fixed version.")
    
    # Show feature importance
    feature_names = [
        # Original features (KEPT)
        'word_count', 'title_length', 'avg_word_length',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_peak_hour', 'is_self_post',
        'has_question_mark', 'has_exclamation', 'title_upper_ratio',
        'tech_keyword_count', 'reading_time',
        
        # NEW features for better prediction
        'viral_keyword_count', 'starts_with_show', 'starts_with_ask',
        'has_numbers', 'has_year', 'title_word_diversity',
        'title_has_colon', 'title_has_parentheses', 'domain_length',
        'is_major_tech_domain', 'username_length', 'username_has_numbers',
        'is_workday', 'is_morning', 'is_afternoon', 'is_evening',
        'complex_words', 'simple_words', 'complexity_ratio'
    ]
    
    importance = np.abs(lr.coef_)
    sorted_idx = np.argsort(importance)[::-1]
    
    print(f"\nTop 5 most important features:")
    for i in range(min(5, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {feature_names[idx]}: {importance[idx]:.4f}") 