# Machine Learning Group Assignment: App Reviews Analysis & Classification

## Dashboard API & Client (FastAPI + React)

The **client** (`client/`) is a React dashboard that displays analysis charts and KPIs. The **API** (`api/`) is a FastAPI backend that runs the analysis pipeline and serves dashboard data as JSON.

**Run the backend** (from project root; ensure `output/` has converted CSVs):

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Run the frontend** (with dev proxy to the API):

```bash
cd client && npm install && npm run dev
```

Open http://localhost:5173. The dashboard fetches data from `GET /api/dashboard/analytics`. For production, set `VITE_API_URL` to your API base URL.

---

## Documentation (AI conversion pipeline)

For the **AI review conversion** pipeline (convert CSVs to a target domain and generate reports), see:

| Doc | Purpose |
|-----|--------|
| **[commands.md](commands.md)** | Quick commands and step-by-step workflow |
| **[developer.md](developer.md)** | Setup, project structure, options |
| **[PROCESS.md](PROCESS.md)** | Full process, data flow, coding details, troubleshooting |

---

## Assignment Overview

**Type**: Group Assignment (2 students per group)  
**Duration**: **1 Week (7 Days)**  
**Difficulty**: Intermediate  
**Tools Required**: Python, Cursor AI, Pandas, Scikit-learn, Streamlit

## Learning Objectives

By completing this assignment, you will:

1. Generate synthetic datasets using AI tools (Cursor AI)
2. Perform exploratory data analysis (EDA) on real app review data
3. Engineer features from text data (NLP/text processing)
4. Build classification models for business use cases
5. Create interactive dashboards for insights
6. Work with real-world messy data (missing values, class imbalance)

## Datasets

### Part 1: Netflix PlayStore Reviews (Provided)

**Dataset Link**: https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated

**Description**: Real Netflix app reviews from Google Play Store (updated daily)

**Dataset Statistics**:

- **Total Reviews**: 145,791 reviews
- **Time Period**: 2020-2026 (continuously updated)
- **Rating Distribution**:
  - ⭐ 1 star: 57,113 reviews (39%)
  - ⭐⭐ 2 stars: 12,771 reviews (9%)
  - ⭐⭐⭐ 3 stars: 13,974 reviews (10%)
  - ⭐⭐⭐⭐ 4 stars: 16,030 reviews (11%)
  - ⭐⭐⭐⭐⭐ 5 stars: 45,903 reviews (31%)

**Exact Column Structure**:

```
reviewId               - Unique UUID for each review (no missing values)
userName               - Reviewer's name (2 missing values)
content                - Review text (6 missing values)
score                  - Rating 1-5 stars (no missing values)
thumbsUpCount          - Number of helpful votes (no missing values)
reviewCreatedVersion   - App version when reviewed (25,681 missing values)
at                     - Timestamp of review (no missing values)
appVersion             - Current app version (25,681 missing values)
```

**Example Row**:

```
reviewId: 9594fc40-1280-49cc-b020-816ab174770c
userName: Arif Rahman Hakim
content: "One of the best streaming apps out there! High-quality originals..."
score: 5
thumbsUpCount: 0
reviewCreatedVersion: 9.49.1 build 6 63792
at: 2026-01-27 10:27:46
appVersion: 9.49.1 build 6 63792
```

### Part 2: Banking App Reviews (You Generate)

**Task**: Create a similar dataset for a fictional banking app using **Cursor AI**

**Banking App Profile**:

- **Name**: SecureBank Mobile
- **Category**: Finance & Banking
- **Features**: Account management, transfers, bill pay, mobile deposits, budgeting
- **Size Requirement**: ~50,000 reviews (minimum 10,000 if time-constrained)

**You will use Cursor AI to generate this dataset** (detailed guide provided separately)

## Week Timeline & Required Tasks

### **Day 1: Data Acquisition & Initial EDA**

#### MUST DO:

1. Download Netflix reviews from Kaggle
2. Load data into pandas and explore structure
3. Generate banking app reviews using Cursor AI (matching Netflix structure exactly)
4. Document basic statistics for both datasets

**Deliverable**: `01_netflix_eda.ipynb` + `banking_reviews.csv`

**Minimal Code Structure**:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('netflix_reviews.csv')

# TODO: Explore shape, columns, data types
# TODO: Check missing values
# TODO: Visualize rating distribution
# TODO: Analyze review lengths
# TODO: Check time range of reviews
```

### **Day 2: Data Cleaning & Feature Engineering**

#### MUST DO:

1. Handle missing values in both datasets
2. Remove duplicates
3. Convert data types (timestamps, etc.)
4. Create text-based features (review length, word count, punctuation)
5. Create time-based features (year, month, day of week)
6. Create domain-specific features

**Deliverable**: `02_feature_engineering.ipynb`

**Minimal Code Structure**:

```python
# Handle missing values
# TODO: Fill or drop missing content/userName

# Remove duplicates
# TODO: Check and remove duplicate reviewIds

# Feature engineering
# TODO: Create review_length, word_count
# TODO: Extract time features from 'at' column
# TODO: Create sentiment indicator features (exclamation, question marks)
# TODO: Parse app version numbers
```

### **Day 3: Exploratory Data Analysis (EDA)**

#### MUST DO:

1. Analyze rating distributions for both datasets
2. Find most common words in positive vs negative reviews
3. Identify time-based trends
4. Compare Netflix vs Banking app reviews
5. Create word clouds for different rating categories
6. Analyze correlation between features and ratings

**Deliverable**: `03_comprehensive_eda.ipynb` with visualizations

**Minimal Code Structure**:

```python
# Rating analysis
# TODO: Create bar charts for rating distribution
# TODO: Analyze rating trends over time

# Text analysis
# TODO: Create word clouds for 1-star vs 5-star reviews
# TODO: Find most common words by sentiment

# Comparison
# TODO: Compare Netflix vs Banking datasets side-by-side
# TODO: Identify key differences in review patterns
```

### **Day 4: Machine Learning - Sentiment Classification**

#### MUST DO:

1. Create sentiment labels (Positive/Neutral/Negative) from ratings
2. Split data into train/test sets
3. Vectorize text using TF-IDF
4. Train classification model(s)
5. Evaluate model performance
6. Create confusion matrix
7. Test on both Netflix and Banking datasets

**Deliverable**: `04_sentiment_classification.ipynb`

**Minimal Code Structure**:

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create labels
# TODO: Map scores to sentiment (1-2=Negative, 3=Neutral, 4-5=Positive)

# Prepare data
# TODO: Split into X (features) and y (labels)
# TODO: Create train/test split

# Vectorize text
# TODO: Initialize TfidfVectorizer
# TODO: Fit on training data, transform both train and test

# Train model
# TODO: Initialize classifier
# TODO: Fit on training data

# Evaluate
# TODO: Make predictions on test set
# TODO: Calculate accuracy, precision, recall, F1
# TODO: Create confusion matrix
```

### **Day 5: Machine Learning - Rating Prediction**

#### MUST DO:

1. Predict exact star rating (1-5) from review text and features
2. Combine text features (TF-IDF) with numeric features
3. Train multi-class classification model
4. Evaluate using accuracy and MAE (Mean Absolute Error)
5. Analyze feature importance
6. Compare model performance on Netflix vs Banking data

**Deliverable**: `05_rating_prediction.ipynb`

**Minimal Code Structure**:

```python
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# Prepare features
# TODO: Get TF-IDF vectors from text
# TODO: Get numeric features (review_length, word_count, etc.)
# TODO: Combine text and numeric features

# Train model
# TODO: Initialize multi-class classifier
# TODO: Fit on combined features

# Evaluate
# TODO: Calculate accuracy
# TODO: Calculate MAE (Mean Absolute Error)
# TODO: Analyze which features are most important
```

### **Day 6: Dashboard Creation**

#### MUST DO:

1. Create Streamlit dashboard with following components:
   - Dataset overview (total reviews, avg rating, date range)
   - Interactive filters (dataset selection, rating filter, date range)
   - Rating distribution charts
   - Time-based trend analysis
   - Model performance metrics display
   - Word clouds for different sentiments
   - Side-by-side comparison of Netflix vs Banking
2. Make dashboard interactive with user inputs
3. Take screenshots of dashboard

**Deliverable**: `dashboard.py` + screenshots in `06_dashboard_screenshots/`

**Minimal Code Structure**:

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("App Reviews Analysis Dashboard")

# Load data
# TODO: Load Netflix and Banking datasets

# Sidebar filters
# TODO: Add dataset selector
# TODO: Add rating filter slider
# TODO: Add date range selector

# Main content
# TODO: Display key metrics (total reviews, avg rating, etc.)
# TODO: Create rating distribution chart
# TODO: Create time-based trend chart
# TODO: Display model performance metrics
# TODO: Show word clouds
# TODO: Add comparison view
```

**Run Dashboard**:

```bash
streamlit run dashboard.py
```

### **Day 7: Final Report**

#### MUST DO:

1. Write comprehensive report documenting entire project
2. Include executive summary
3. Explain data generation process
4. Present key EDA findings
5. Document ML models and performance
6. Provide business insights and recommendations
7. Discuss challenges faced and lessons learned

**Deliverable**: `FINAL_REPORT.md` (3-5 pages)

**Report Structure**:

```markdown
# App Reviews ML Analysis - Final Report

## 1. Executive Summary

- Project overview
- Key findings
- Business recommendations

## 2. Data Generation Process

- How you used Cursor AI
- Challenges faced
- Data quality validation

## 3. Exploratory Data Analysis

- Key insights from both datasets
- Interesting patterns discovered
- Visualizations

## 4. Machine Learning Models

- Models built (sentiment, rating prediction)
- Performance metrics
- Feature importance

## 5. Dashboard

- Features implemented
- User guide

## 6. Business Insights & Recommendations

- What did you learn from the data?
- What actions should the app company take?
- Trends to watch

## 7. Challenges & Learnings

- Technical challenges
- What you learned
- Future improvements
```

## Required Project Structure

Your final submission **MUST** have this structure:

```
app-reviews-ml-project/
│
├── data/
│   ├── netflix_reviews.csv
│   ├── banking_reviews.csv
│   └── processed/
│       ├── netflix_processed.csv
│       └── banking_processed.csv
│
├── notebooks/
│   ├── 01_netflix_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_comprehensive_eda.ipynb
│   ├── 04_sentiment_classification.ipynb
│   └── 05_rating_prediction.ipynb
│
├── models/
│   ├── sentiment_model.pkl
│   ├── rating_model.pkl
│   └── vectorizer.pkl
│
├── dashboard/
│   ├── dashboard.py
│   └── screenshots/
│       ├── overview.png
│       ├── trends.png
│       └── models.png
│
├── FINAL_REPORT.md
├── requirements.txt
└── README.md
```

## 🛠️ Technical Requirements

### Required Libraries

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.17.0
wordcloud>=1.9.0
nltk>=3.8.0
jupyter>=1.0.0
```

### Installation

```bash
# Create conda environment
conda create -n ml-reviews python=3.10 -y
conda activate ml-reviews

# Install dependencies
pip install -r requirements.txt
```

## Important Guidelines

### 1. **Data Generation with Cursor**

- Banking reviews **MUST** have same 8 columns as Netflix
- Rating distribution should be similar (39/9/10/11/31)
- Reviews must be banking-specific (not Netflix content)
- See `Data_Generation_Guide.md` for detailed instructions

### 2. **Feature Engineering**

- Don't just use raw text - extract meaningful features
- Combine text features (TF-IDF) with numeric features
- Create domain-specific features based on your EDA

### 3. **Model Building**

- Always use train/test split (no data leakage!)
- Evaluate on test set, not training set
- Handle class imbalance appropriately
- Try multiple models and compare

### 4. **Dashboard**

- Must be functional and interactive
- Focus on telling a story with your data
- Include filters and user controls
- Test before submitting

### 5. **Code Quality**

- Write clean, documented code
- Add comments explaining your logic
- Make notebooks reproducible
- Follow Python best practices

## Common Pitfalls to Avoid

- ❌ Not handling missing values properly
- ❌ Training on test data (data leakage)
- ❌ Ignoring class imbalance
- ❌ Not documenting your code
- ❌ Waiting until last day to start
- ❌ Generated banking data doesn't match Netflix structure
- ❌ Using only text features (ignore numeric features)
- ❌ Not testing dashboard before submission

## Minimum Performance Expectations

### Sentiment Classification

- **Accuracy**: >70%
- **Must**: Handle 3-class problem (Positive/Neutral/Negative)
- **Must**: Show confusion matrix

### Rating Prediction

- **MAE**: <1.0
- **Must**: Predict all 5 rating levels (1-5)
- **Must**: Combine text + numeric features

### Dashboard

- **Must**: Be functional (no errors)
- **Must**: Include all required components
- **Must**: Be interactive (filters work)
