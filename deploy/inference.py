from typing import Any

import joblib
import numpy as np
import pandas as pd

from deploy.config import MODEL_DIR
from deploy.preprocessing import get_feature

class InferencePipeline:
    
