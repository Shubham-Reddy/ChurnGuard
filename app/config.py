import os
from dotenv import load_dotenv
load_dotenv('secrets.env')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq API Key
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Paths
DATA_PATH = "../data/telco_churn.csv"
MODEL_PATH = "../models/"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# App settings
APP_TITLE = "ChurnGuard"
APP_SUBTITLE = "Customer Retention Intelligence Platform"



