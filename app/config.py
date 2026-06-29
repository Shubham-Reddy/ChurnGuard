import os
from dotenv import load_dotenv
load_dotenv('secrets.env')
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Groq API Key
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models") + "/"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# App settings
APP_TITLE = "ChurnGuard"
APP_SUBTITLE = "Customer Retention Intelligence Platform"



