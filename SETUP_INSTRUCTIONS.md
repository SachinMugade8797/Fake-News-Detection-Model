# 📦 Fake News Detector - Setup Instructions

## For You (Sharing the Project)

### Step 1: Create the ZIP File

**Files to Include in ZIP:**
- ✅ `app.py` (main application)
- ✅ `model.pkl` (trained ML model)
- ✅ `vectorizer.pkl` (text vectorizer)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `data/` folder (with `fake.csv` and `true.csv`)
- ✅ `train_model.py` (optional - for retraining)
- ✅ `download_nltk.py` (optional - for NLTK setup)

**Files to EXCLUDE:**
- ❌ `venv/` folder (virtual environment - too large, will be recreated)
- ❌ `__pycache__/` folders
- ❌ `.streamlit/secrets.toml` (contains API keys - don't share!)

**How to Create ZIP:**
1. Select all the files above (except venv and secrets)
2. Right-click → "Send to" → "Compressed (zipped) folder"
3. Name it: `fake_news_detector.zip`

---

## For Your Friend (Setting Up on Their Laptop)

### Prerequisites
- Python 3.8 or higher installed
- Internet connection (for downloading packages)

### Step 1: Extract the ZIP File
1. Download `fake_news_detector.zip` from you
2. Right-click on the ZIP file
3. Select "Extract All..."
4. Choose a location (e.g., `C:\Users\YourName\Desktop\`)
5. Click "Extract"
6. You should see a folder named `fake_news_detector`

### Step 2: Open Command Prompt/Terminal
**Windows:**
- Press `Win + R`
- Type `cmd` and press Enter
- OR search for "Command Prompt" in Start menu

**Mac/Linux:**
- Open Terminal

### Step 3: Navigate to Project Folder
```bash
cd path\to\fake_news_detector
```

**Example:**
```bash
cd C:\Users\YourName\Desktop\fake_news_detector
```

### Step 4: Create Virtual Environment
```bash
python -m venv venv
```

**Note:** If `python` doesn't work, try `python3` or `py`

### Step 5: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Success indicator:** You should see `(venv)` at the start of your command line

### Step 6: Install Dependencies
```bash
pip install -r requirements.txt
```

**This will install:**
- streamlit
- scikit-learn
- pandas
- nltk
- google-generativeai
- numpy
- scipy
- joblib

**Wait for installation to complete** (may take 2-5 minutes)

### Step 7: Download NLTK Data (Optional but Recommended)
```bash
python download_nltk.py
```

**OR manually:**
```python
python -c "import nltk; nltk.download('stopwords')"
```

### Step 8: Set Up API Keys (Optional - for Gemini API features)

**Option A: Using Environment Variable (Windows)**
```bash
set GEMINI_API_KEY=your_api_key_here
```

**Option B: Using Environment Variable (Mac/Linux)**
```bash
export GEMINI_API_KEY=your_api_key_here
```

**Option C: Create Secrets File**
1. Create a folder named `.streamlit` in the project directory
2. Create a file named `secrets.toml` inside `.streamlit`
3. Add this line:
```toml
GEMINI_API_KEY = "your_api_key_here"
```

**Note:** The app will work without API keys, but API features won't be available.

### Step 9: Run the Application
```bash
streamlit run app.py
```

**What happens:**
- Streamlit will start the server
- A browser window will open automatically
- If not, look for a URL like: `http://localhost:8501`
- Copy and paste that URL into your browser

### Step 10: Use the App
1. Enter news text in the text area
2. Click "🔍 Detect News"
3. View the prediction results!

### To Stop the App
- Press `Ctrl + C` in the terminal/command prompt

---

## Troubleshooting

### Problem: "python is not recognized"
**Solution:** 
- Install Python from python.org
- Make sure to check "Add Python to PATH" during installation

### Problem: "pip is not recognized"
**Solution:**
- Use `python -m pip` instead of just `pip`
- Or reinstall Python with PATH option

### Problem: "ModuleNotFoundError"
**Solution:**
- Make sure virtual environment is activated (see `(venv)` in prompt)
- Run `pip install -r requirements.txt` again

### Problem: "Model not found"
**Solution:**
- Make sure `model.pkl` and `vectorizer.pkl` are in the same folder as `app.py`

### Problem: Port already in use
**Solution:**
- Close other Streamlit apps
- Or use: `streamlit run app.py --server.port 8502`

### Problem: API features not working
**Solution:**
- Check if API keys are set correctly
- The app will still work with ML model only

---

## Quick Start (Summary)
```bash
# 1. Extract ZIP
# 2. Open terminal in project folder
cd fake_news_detector

# 3. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run app
streamlit run app.py
```

---

## File Structure After Setup
```
fake_news_detector/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── data/
│   ├── fake.csv
│   └── true.csv
├── venv/          (created after setup)
└── .streamlit/    (optional - for API keys)
    └── secrets.toml
```

---

**Need Help?** Make sure all files are extracted correctly and virtual environment is activated!



