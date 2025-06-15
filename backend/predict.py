"""
Prediction script to load the trained model and artifacts for inference.
"""
import argparse
import glob
import os
import pickle
from datetime import datetime

import numpy as np
import torch

from . import config as cfg
from .data_processing import (clean_text, extract_domain, load_glove_embeddings, title_to_embedding)
from .train import NumericalPlusTitleNN


def find_most_recent_run():
    """Find the most recent training run directory."""
    runs_dir = "artifacts/training_runs"
    if not os.path.exists(runs_dir):
        return None
    
    # Find all run directories matching the pattern
    run_pattern = os.path.join(runs_dir, "run_*")
    run_dirs = glob.glob(run_pattern)
    
    if not run_dirs:
        return None
    
    # Sort by modification time (most recent first)
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    most_recent = run_dirs[0]
    
    return most_recent


class Scorer:
    def __init__(self, run_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine artifacts directory - ALWAYS use timestamped directories
        if run_dir:
            if not os.path.exists(run_dir):
                raise FileNotFoundError(f"Specified run directory does not exist: {run_dir}")
            artifacts_dir = run_dir
            print(f"--- Loading artifacts from specified directory: {run_dir} ---")
        else:
            # ALWAYS try to find the most recent training run
            most_recent_run = find_most_recent_run()
            if most_recent_run:
                artifacts_dir = most_recent_run
                print(f"--- Loading artifacts from most recent run: {most_recent_run} ---")
            else:
                raise FileNotFoundError(
                    "No training runs found! Please run training first or specify --run-dir.\n"
                    f"Expected training runs in: artifacts/training_runs/run_YYYYMMDD_HHMMSS/"
                )
        
        # Load artifacts ONLY from timestamped directory
        self.word_to_idx, self.embeddings = load_glove_embeddings()
        
        # Load scaler from timestamped run directory ONLY
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found in run directory: {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"✅ Loaded scaler from: {scaler_path}")
        
        # Load domain and user statistics (computed during training)
        self.domain_stats = {}
        self.user_stats = {}
        self.global_mean = 3.0  # Reasonable default for log(score)
        
        # Load pre-computed statistics from timestamped directory
        domain_stats_path = os.path.join(artifacts_dir, "domain_stats.pkl")
        user_stats_path = os.path.join(artifacts_dir, "user_stats.pkl")
        global_mean_path = os.path.join(artifacts_dir, "global_mean.pkl")
        
        try:
            with open(domain_stats_path, 'rb') as f:
                self.domain_stats = pickle.load(f)
            with open(user_stats_path, 'rb') as f:
                self.user_stats = pickle.load(f)
            with open(global_mean_path, 'rb') as f:
                self.global_mean = pickle.load(f)
            print("✅ Loaded pre-computed domain/user statistics")
        except FileNotFoundError as e:
            print(f"⚠️ Pre-computed statistics not found: {e}")
            print(f"⚠️ Using defaults - predictions may be less accurate")
        
        # Load model configuration from timestamped directory
        config_path = os.path.join(artifacts_dir, "config.json")
        try:
            with open(config_path, 'r') as f:
                import json
                config_data = json.load(f)
                numerical_dim = config_data.get('numerical_dim', cfg.NUMERICAL_DIM)
                title_dim = config_data.get('title_dim', cfg.TITLE_EMB_DIM)
                hidden_dim = config_data.get('hidden_dim', cfg.HIDDEN_DIM)
                dropout = config_data.get('dropout', cfg.DROPOUT_RATE)
                print(f"✅ Loaded model config: {numerical_dim}D numerical, {title_dim}D title, {hidden_dim}D hidden")
        except FileNotFoundError:
            # Default model dimensions from config
            numerical_dim = cfg.NUMERICAL_DIM
            title_dim = cfg.TITLE_EMB_DIM
            hidden_dim = cfg.HIDDEN_DIM
            dropout = cfg.DROPOUT_RATE
            print("⚠️ Model config not found, using defaults from cfg")
        
        # Load model with architecture
        self.model = NumericalPlusTitleNN(
            numerical_dim=numerical_dim,
            title_dim=title_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(self.device)
        
        # Load model weights ONLY from timestamped directory
        model_path = os.path.join(artifacts_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found in run directory: {model_path}")
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded model from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        self.model.eval()
        print(f"✅ All artifacts loaded successfully from: {artifacts_dir}")

    def create_enhanced_features(self, title: str, url: str, user: str, submission_time: datetime):
        """Create the enhanced 34D numerical features used in training."""
        
        # 1. Basic title features
        title_words = title.strip().split()
        word_count = len(title_words)
        title_length = len(title)
        avg_word_length = title_length / word_count if word_count > 0 else 0
        
        # 2. Time features
        hour_of_day = submission_time.hour
        day_of_week = submission_time.weekday()  # Monday is 0 and Sunday is 6
        month = submission_time.month
        year = submission_time.year
        
        # Cyclical time encoding
        hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Weekend and peak hour indicators
        is_weekend = int(day_of_week >= 5)
        is_peak_hour = int(hour_of_day in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        
        # 3. Domain features
        domain = extract_domain(url)
        is_self_post = int(domain == 'self_post')
        domain_length = len(domain)
        is_major_tech_domain = int(domain in ['github.com', 'medium.com', 'youtube.com', 
                                             'twitter.com', 'reddit.com', 'stackoverflow.com',
                                             'techcrunch.com', 'arstechnica.com'])
        
        # 4. Title content features
        title_lower = title.lower()
        has_question_mark = int('?' in title)
        has_exclamation = int('!' in title)
        title_upper_ratio = sum(1 for c in title if c.isupper()) / title_length if title_length > 0 else 0
        
        # Tech keywords
        tech_keywords = ['startup', 'python', 'javascript', 'react', 'ai', 'machine learning', 
                        'blockchain', 'crypto', 'google', 'apple', 'microsoft', 'open source',
                        'github', 'api', 'database', 'security', 'privacy', 'algorithm']
        tech_keyword_count = sum(1 for keyword in tech_keywords if keyword in title_lower)
        
        # Viral/engagement words
        viral_keywords = ['show', 'ask', 'launch', 'new', 'free', 'best', 'top', 'guide', 
                         'tips', 'how to', 'why', 'what', 'review', 'vs', 'comparison']
        viral_keyword_count = sum(1 for keyword in viral_keywords if keyword in title_lower)
        
        # Title sentiment/appeal features
        starts_with_show = int(title_lower.startswith('show hn'))
        starts_with_ask = int(title_lower.startswith('ask hn'))
        has_numbers = int(any(c.isdigit() for c in title))
        has_year = int(bool(__import__('re').search(r'20\d{2}', title)))
        
        # Title structure features
        title_words_set = set(title_lower.split())
        title_word_diversity = len(title_words_set) / max(len(title_words), 1)
        title_has_colon = int(':' in title)
        title_has_parentheses = int('(' in title or ')' in title)
        
        # User features
        username_length = len(user)
        username_has_numbers = int(any(c.isdigit() for c in user))
        
        # Temporal features
        is_workday = int(day_of_week < 5)  # Monday-Friday
        is_morning = int(hour_of_day in [7, 8, 9, 10, 11])
        is_afternoon = int(hour_of_day in [12, 13, 14, 15, 16, 17])
        is_evening = int(hour_of_day in [18, 19, 20, 21])
        
        # Reading difficulty proxy
        complex_words = len([w for w in title_words if len(w) >= 7])
        simple_words = len([w for w in title_words if len(w) <= 4])
        complexity_ratio = complex_words / word_count if word_count > 0 else 0
        
        # Reading time
        reading_time = word_count / 200  # Assume 200 WPM reading speed
        
        # Combine all 34 original numerical features
        numerical_features = np.array([
            word_count, title_length, avg_word_length,
            hour_sin, hour_cos, day_sin, day_cos,
            is_weekend, is_peak_hour, is_self_post,
            has_question_mark, has_exclamation, title_upper_ratio,
            tech_keyword_count, reading_time,
            viral_keyword_count, starts_with_show, starts_with_ask,
            has_numbers, has_year, title_word_diversity,
            title_has_colon, title_has_parentheses, domain_length,
            is_major_tech_domain, username_length, username_has_numbers,
            is_workday, is_morning, is_afternoon, is_evening,
            complex_words, simple_words, complexity_ratio
        ])
        
        return numerical_features, domain, user

    def predict(self, title: str, url: str, user: str, submission_time: datetime):
        """Predicts the score for a new story using the same architecture as training."""
        
        # 1. Create enhanced numerical features (34D) - same as training
        numerical_features, domain, user_name = self.create_enhanced_features(title, url, user, submission_time)
        
        # 2. Add statistical domain/user features (2D more = 36D total) - same as training
        # Use domain string directly as key (same as training saves it)
        domain_mean = self.domain_stats.get(domain, self.global_mean)
        user_mean = self.user_stats.get(user_name, self.global_mean)
        
        # Enhanced numerical features (36D) - same as training
        numerical_enhanced = np.append(numerical_features, [domain_mean, user_mean])
        
        # 3. Title embeddings (200D) - same as training
        title_emb = title_to_embedding(title, self.word_to_idx, self.embeddings)
        
        # 4. Scale numerical features - same as training
        numerical_scaled = self.scaler.transform(numerical_enhanced.reshape(1, -1))
        
        # 5. Create tensors - same as training
        numerical_tensor = torch.FloatTensor(numerical_scaled).to(self.device)
        title_tensor = torch.FloatTensor(title_emb).unsqueeze(0).to(self.device)
        
        # 6. Predict using architecture - same as training
        with torch.no_grad():
            pred_log = self.model(numerical_tensor, title_tensor)
            pred_orig = np.expm1(pred_log.cpu().item())
        
        return max(1, int(round(pred_orig)))


def main():
    parser = argparse.ArgumentParser(description='Predict HackerNews scores')
    parser.add_argument('--run-dir', type=str, help='Path to training run directory (default: most recent run)')
    parser.add_argument('--title', type=str, required=True, help='Article title')
    parser.add_argument('--url', type=str, default='', help='Article URL (empty for self-posts)')
    parser.add_argument('--user', type=str, required=True, help='Username')
    parser.add_argument('--timestamp', type=int, help='UNIX timestamp (default: current time)')
    
    args = parser.parse_args()
    
    # Use current timestamp if not provided
    if args.timestamp is None:
        args.timestamp = int(datetime.now().timestamp())
    
    # Load scorer
    scorer = Scorer(run_dir=args.run_dir)
    
    # Make prediction
    submission_time = datetime.fromtimestamp(args.timestamp)
    predicted_score = scorer.predict(args.title, args.url, args.user, submission_time)
    
    # Output results
    print(f"\n🎯 PREDICTION RESULTS")
    print(f"=" * 40)
    print(f"Title: '{args.title}'")
    print(f"URL: {args.url if args.url else '(self-post)'}")
    print(f"User: {args.user}")
    print(f"Time: {submission_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Predicted Score: {predicted_score}")
    print(f"=" * 40)
    
    return predicted_score


if __name__ == '__main__':
    if len(__import__('sys').argv) == 1:
        # No arguments provided, run example
        print("\n--- Example Prediction (using most recent training run) ---")
        scorer = Scorer()  # Will automatically find most recent run
        test_title = "Sami nami na eh eh"
        test_url = "https://wakawaka.com"
        test_user = "eh eh"
        test_timestamp = int(datetime.now().timestamp())
        
        submission_time = datetime.fromtimestamp(test_timestamp)
        predicted_score = scorer.predict(test_title, test_url, test_user, submission_time)
        print(f"Title: '{test_title}'")
        print(f"URL: {test_url}")
        print(f"User: {test_user}")
        print(f"Predicted Score: {predicted_score}")
    else:
        # Arguments provided, run main function
        main()


    # python predict.py --run-dir artifacts/training_runs/run_20250613_103621 --title "Show HN: My awesome project" --user "myusername" --url "https://github.com/user/project"