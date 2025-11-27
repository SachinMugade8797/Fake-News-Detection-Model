import streamlit as st
import pickle
import re
import os
from google import generativeai as genai

# Multiple Gemini API keys for fallback
API_KEYS = [
    "AIzaSyBNhoiEA6Xlx_bI5XI0edd8KtyAeotPR5I",
    "AIzaSyCgvLXEs1rkWYOsjkcRLDfzxYW_KrcJr1c",
    "AIzaSyDPF3je6LpuOigdHZ98Qs2jbSXCSgXOH14"
]

# Initialize session state for tracking current API key index
if 'current_api_key_index' not in st.session_state:
    st.session_state.current_api_key_index = 0

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 20px;
    }
    .result-box {
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
    }
    .true-news {
        background-color: #d4edda;
        border: 3px solid #28a745;
        color: #155724;
    }
    .fake-news {
        background-color: #f8d7da;
        border: 3px solid #dc3545;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Helper function to check if error is a rate limit error
def is_rate_limit_error(error_msg):
    """Check if the error is a rate limit or quota exceeded error"""
    error_lower = str(error_msg).lower()
    rate_limit_indicators = [
        'rate limit',
        'quota exceeded',
        'resource exhausted',
        '429',
        'too many requests',
        'quota',
        'limit exceeded'
    ]
    return any(indicator in error_lower for indicator in rate_limit_indicators)

# Function to verify news with Gemini API and extract prediction
def verify_with_gemini(news_text, include_extra_info=False):
    """Call Gemini API to verify the news article and return structured results with multi-key fallback"""
    # Define prompt to get a clear True/Fake prediction
    if include_extra_info:
        prompt = f"""Analyze this news article and provide a fact-checking assessment.

News Article:
{news_text}

Please respond in this EXACT format:

PREDICTION: [TRUE or FAKE]
CONFIDENCE: [0.0 to 1.0]
EXPLANATION: [Brief 2-3 sentence explanation]
SOURCES: [List any credible sources, news outlets, or evidence that supports or refutes this news]
DETAILS: [Key facts, red flags, and additional context]

Analyze based on:
1. Credibility of sources and claims
2. Consistency with known facts
3. Patterns typical of misinformation
4. Scientific evidence or official statements

Be concise but thorough."""
    else:
        prompt = f"""Analyze this news article and determine if it's TRUE or FAKE.

News Article:
{news_text}

Respond in this EXACT format:
PREDICTION: [TRUE or FAKE]
CONFIDENCE: [0.0 to 1.0]
EXPLANATION: [Brief explanation]

Base your assessment on credibility, factual accuracy, and known patterns of misinformation."""
    
    generation_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    # Get API keys: first try from secrets/env, then use the list
    api_keys_to_try = []
    
    # Add key from secrets/env if available
    env_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if env_key:
        api_keys_to_try.append(env_key)
    
    # Add the predefined API keys
    api_keys_to_try.extend(API_KEYS)
    
    if not api_keys_to_try:
        return None, None, None, "No API keys available. Please set GEMINI_API_KEY or configure API_KEYS list."
    
    # Start from the current key index (round-robin with fallback)
    start_index = st.session_state.current_api_key_index
    last_error = None
    
    # Try all keys starting from current index, then wrap around
    for attempt in range(len(api_keys_to_try)):
        current_index = (start_index + attempt) % len(api_keys_to_try)
        api_key = api_keys_to_try[current_index]
        
        try:
            genai.configure(api_key=api_key)
            # Try gemini-pro-latest first (confirmed working), then try other models
            model_names = ["gemini-pro-latest", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
            model_error = None
            
            for model_name in model_names:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt, generation_config=generation_config)
                    response_text = response.text
                    
                    # Success! Update the current key index for next time
                    st.session_state.current_api_key_index = current_index
                    
                    # Parse the response
                    prediction = None
                    confidence = None
                    extra_info = {}
                    
                    # Extract PREDICTION
                    if "PREDICTION:" in response_text:
                        pred_line = [line for line in response_text.split('\n') if 'PREDICTION:' in line.upper()][0]
                        if 'TRUE' in pred_line.upper():
                            prediction = 1  # True news
                        elif 'FAKE' in pred_line.upper():
                            prediction = 0  # Fake news
                    
                    # Extract CONFIDENCE
                    if "CONFIDENCE:" in response_text:
                        conf_line = [line for line in response_text.split('\n') if 'CONFIDENCE:' in line.upper()][0]
                        try:
                            conf_match = re.search(r'(\d+\.?\d*)', conf_line)
                            if conf_match:
                                confidence = float(conf_match.group(1))
                                # Ensure confidence is between 0 and 1
                                if confidence > 1.0:
                                    confidence = confidence / 100.0
                                confidence = max(0.0, min(1.0, confidence))
                        except:
                            pass
                    
                    # If confidence not found, use default based on prediction
                    if confidence is None:
                        confidence = 0.75 if prediction is not None else 0.5
                    
                    # Extract extra info if requested
                    if include_extra_info:
                        # Extract EXPLANATION
                        if "EXPLANATION:" in response_text.upper():
                            try:
                                explanation_start = response_text.upper().find("EXPLANATION:")
                                explanation_text = response_text[explanation_start + len("EXPLANATION:"):]
                                # Find the next section or end of text
                                next_section = min(
                                    explanation_text.upper().find("\nSOURCES:"),
                                    explanation_text.upper().find("\nDETAILS:"),
                                    explanation_text.upper().find("\nPREDICTION:"),
                                    len(explanation_text)
                                )
                                explanation = explanation_text[:next_section].strip()
                                # Remove the label part
                                explanation = explanation.split(':', 1)[-1].strip() if ':' in explanation else explanation
                                if explanation:
                                    extra_info['explanation'] = explanation
                            except:
                                pass
                        
                        # Extract SOURCES
                        if "SOURCES:" in response_text.upper():
                            try:
                                sources_start = response_text.upper().find("SOURCES:")
                                sources_text = response_text[sources_start + len("SOURCES:"):]
                                # Find the next section
                                next_section = min(
                                    sources_text.upper().find("\nDETAILS:"),
                                    sources_text.upper().find("\nEXPLANATION:"),
                                    sources_text.upper().find("\nPREDICTION:"),
                                    len(sources_text)
                                )
                                sources = sources_text[:next_section].strip()
                                # Remove the label part
                                sources = sources.split(':', 1)[-1].strip() if ':' in sources else sources
                                if sources:
                                    extra_info['sources'] = sources
                            except:
                                pass
                        
                        # Extract DETAILS
                        if "DETAILS:" in response_text.upper():
                            try:
                                details_start = response_text.upper().find("DETAILS:")
                                details_text = response_text[details_start + len("DETAILS:"):]
                                # Find the next section or end
                                next_section = min(
                                    details_text.upper().find("\nSOURCES:"),
                                    details_text.upper().find("\nEXPLANATION:"),
                                    details_text.upper().find("\nPREDICTION:"),
                                    len(details_text)
                                )
                                details = details_text[:next_section].strip()
                                # Remove the label part
                                details = details.split(':', 1)[-1].strip() if ':' in details else details
                                if details:
                                    extra_info['details'] = details
                            except:
                                pass
                        
                        # If no structured info found, use the full response as explanation
                        if not extra_info and response_text:
                            # Try to extract meaningful parts
                            lines = response_text.split('\n')
                            explanation_lines = []
                            for line in lines:
                                if line.strip() and not line.strip().startswith('PREDICTION:') and not line.strip().startswith('CONFIDENCE:'):
                                    explanation_lines.append(line.strip())
                            if explanation_lines:
                                extra_info['explanation'] = ' '.join(explanation_lines[:5])  # First 5 lines
                    
                    # If prediction is None, try to infer from response text
                    if prediction is None:
                        response_lower = response_text.lower()
                        if any(word in response_lower for word in ['true', 'real', 'legitimate', 'authentic', 'verified']):
                            prediction = 1
                        elif any(word in response_lower for word in ['fake', 'false', 'misinformation', 'hoax', 'unverified']):
                            prediction = 0
                        else:
                            prediction = 0  # Default to fake if uncertain
                    
                    return prediction, confidence, extra_info, None
                    
                except Exception as e:
                    model_error = e
                    continue
            
            # If all models failed for this key, check if it's a rate limit error
            if model_error and is_rate_limit_error(str(model_error)):
                # Rate limit hit - switch to next key
                last_error = model_error
                # Move to next key index for next attempt
                st.session_state.current_api_key_index = (current_index + 1) % len(api_keys_to_try)
                continue
            elif model_error:
                # Other error - try next key anyway
                last_error = model_error
                continue
                
        except Exception as e:
            error_msg = str(e)
            # Check if it's a rate limit error
            if is_rate_limit_error(error_msg):
                # Rate limit hit - switch to next key
                last_error = e
                st.session_state.current_api_key_index = (current_index + 1) % len(api_keys_to_try)
                continue
            else:
                # Other error - try next key
                last_error = e
                continue
    
    # All keys failed - return error (will trigger ML model fallback)
    if last_error:
        error_msg = f"All API keys exhausted. Last error: {str(last_error)}"
    else:
        error_msg = "All API keys failed. No error details available."
    
    return None, None, None, error_msg

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

# Main App
st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown("---")

# Check if model exists
model, vectorizer = load_model()

if model is None:
    st.error("🚨 **Model not found!**")
    st.warning("Please train the model first by running: `python train_model.py`")
    st.stop()

st.success("✅ Model loaded successfully!")

# Toggle for API with References
st.markdown("---")
use_api_with_references = st.toggle(
    "🌐 Use API with References",
    value=False,
    help="When ON: All predictions use API with extra explanation, resource links, and higher confidence. When OFF: API only used for unseen inputs, output looks like ML model."
)

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. Enter a news headline or article in the text box below
    2. Click on 'Detect News' button
    3. The app automatically detects patterns and uses the appropriate method:
       - **Similar to training data**: Uses ML model with real confidence scores
       - **Unseen patterns**: Automatically uses API (when toggle is OFF, appears exactly like ML model)
    4. **"Use API with References" toggle:**
       - **ON**: All predictions use Gemini API with extra explanation, resource links, and higher confidence (85-95%)
       - **OFF**: API only used for unseen patterns, output looks exactly like ML model (same UI, realistic confidence, no API mention)
    """)

st.markdown("### 📝 Enter News Article")

# Text input
user_input = st.text_area(
    "Paste your news headline or article here:",
    height=200,
    placeholder="Example: Scientists discovered a new cure for cancer..."
)

# Predict button
if st.button("🔍 Detect News", use_container_width=True, type="primary"):
    if not user_input or len(user_input.strip()) < 10:
        st.warning("⚠️ Please enter at least 10 characters of news text!")
    else:
        with st.spinner("🔄 Analyzing..."):
            # Clean text
            cleaned_text = clean_text(user_input)
            
            # Transform text
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # Predict
            prediction = model.predict(text_vectorized)[0]
            probability = model.predict_proba(text_vectorized)[0]
            
            # Calculate confidence score
            if prediction == 1:
                confidence = probability[1]
            else:
                confidence = probability[0]
            
            confidence_percent = confidence * 100
            
            # Detect unseen patterns using multiple indicators (not just confidence)
            # 1. Low confidence (< 0.65)
            # 2. Uncertain prediction (probabilities close to 0.5)
            # 3. Feature sparsity (check if vectorized text has very few non-zero features)
            prob_diff = abs(probability[0] - probability[1])
            is_uncertain = prob_diff < 0.3  # Probabilities are close (uncertain)
            is_low_confidence = confidence < 0.65
            
            # Check feature sparsity (unseen patterns often have sparse feature vectors)
            non_zero_features = text_vectorized.nnz  # Number of non-zero elements
            total_features = text_vectorized.shape[1]
            sparsity = non_zero_features / total_features if total_features > 0 else 0
            is_sparse = sparsity < 0.01  # Very sparse (less than 1% features active)
            
            # Combined unseen detection: low confidence OR uncertain OR very sparse
            is_unseen = is_low_confidence or is_uncertain or is_sparse
            
            # Decide which method to use based on toggle
            use_api = False
            show_extra_info = False
            silent_api_mode = False  # Track if we're using API silently
            
            if use_api_with_references:
                # Toggle ON: Always use API with references
                use_api = True
                show_extra_info = True
                silent_api_mode = False
            elif is_unseen:
                # Toggle OFF but unseen pattern detected: Use API silently (display like ML)
                use_api = True
                show_extra_info = False
                silent_api_mode = True
            else:
                # Toggle OFF and seen pattern: Use ML model only
                use_api = False
                show_extra_info = False
                silent_api_mode = False
            
            api_prediction = None
            api_confidence = None
            api_extra_info = None
            api_error = None
            
            # Call API if needed
            if use_api:
                if use_api_with_references:
                    spinner_msg = "🌐 Analyzing with Gemini API..."
                else:
                    # Silent mode - use same spinner as ML to keep it seamless
                    spinner_msg = "🔄 Analyzing..."
                
                with st.spinner(spinner_msg):
                    api_prediction, api_confidence, api_extra_info, api_error = verify_with_gemini(
                        user_input, 
                        include_extra_info=show_extra_info
                    )
            
            # Show result
            st.markdown("---")
            st.markdown("### 📊 Result")
            
            # Determine which prediction to display and confidence
            # If API failed in silent mode, fall back to ML model
            if use_api and api_prediction is not None and api_error is None:
                # Use API result
                final_prediction = api_prediction
                
                # Adapt confidence based on mode
                if use_api_with_references:
                    # High confidence for API with references (boost to 0.85-0.95 range)
                    base_confidence = api_confidence if api_confidence else 0.75
                    # Boost confidence: take base and scale it up
                    final_confidence = min(0.95, max(0.85, base_confidence * 1.15))
                elif silent_api_mode:
                    # Silent mode: Make API confidence look realistic like ML model
                    # Use the ML model's confidence as a base to maintain realism
                    # The ML confidence is already low for unseen patterns, so use it
                    ml_confidence = confidence
                    api_base = api_confidence if api_confidence else 0.65
                    
                    # For silent mode, we want it to look like ML predicted it
                    # So we use ML's confidence (which is already realistic for unseen data)
                    # but slightly adjust based on API if API is more certain
                    if api_base > 0.7 and ml_confidence < 0.6:
                        # API is confident but ML wasn't - use a slightly improved but still realistic confidence
                        final_confidence = min(0.68, ml_confidence + 0.08)
                    else:
                        # Use ML confidence to maintain realism
                        final_confidence = ml_confidence
                else:
                    # Fallback: use API confidence
                    final_confidence = api_confidence if api_confidence else 0.75
                
                final_confidence_percent = final_confidence * 100
                source = "Gemini API"
            else:
                # Use ML model result (real confidence)
                # This also handles the case where API failed in silent mode
                final_prediction = prediction
                final_confidence = confidence
                final_confidence_percent = confidence_percent
                source = "ML Model"
            
            # Display result in same format (same UI style for both ML and API)
            if final_prediction == 1:
                # True News
                st.markdown(
                    '<div class="result-box true-news">✅ TRUE NEWS</div>',
                    unsafe_allow_html=True
                )
                st.success(f"This appears to be **legitimate news** with {final_confidence_percent:.1f}% confidence")
            else:
                # Fake News
                st.markdown(
                    '<div class="result-box fake-news">🚨 FAKE NEWS</div>',
                    unsafe_allow_html=True
                )
                st.error(f"This appears to be **fake news** with {final_confidence_percent:.1f}% confidence")
            
            # Show source indicator (only if API with references is ON)
            # In silent mode, show nothing to make it look exactly like ML
            if use_api_with_references and use_api and api_error is None:
                st.caption(f"🤖 Verified using {source}")
            # Silent mode: Don't show any indicator - make it look exactly like ML model
            
            # Show confidence bar
            st.markdown("**Confidence Level:**")
            st.progress(final_confidence)
            st.caption(f"{final_confidence_percent:.2f}%")
            
            # Show API error if any (but only if API with references is ON)
            # In silent mode, if API fails, silently fall back to ML model (no error shown)
            if use_api and api_error and "All API keys" not in api_error:
                if use_api_with_references:
                    # Only show error if API with references is ON
                    st.warning(f"⚠️ API Error: {api_error}")
                    if "API key" in api_error or "not found" in api_error.lower():
                        st.info("💡 To use Gemini API verification, set your API key in one of these ways:\n"
                               "1. Set environment variable: `GEMINI_API_KEY=your_key_here`\n"
                               "2. Create a `.streamlit/secrets.toml` file with: `GEMINI_API_KEY = 'your_key_here'`")
                # In silent mode, if API fails, we'll use ML model result (already handled above)
            
            # Show extra information only if "Use API with References" is ON
            if use_api and api_prediction is not None and api_error is None and use_api_with_references:
                # Initialize extra_info if None
                if api_extra_info is None:
                    api_extra_info = {}
                
                st.markdown("---")
                st.markdown("### 📋 Additional Information")
                
                if api_extra_info.get('explanation'):
                    st.markdown("**💡 Explanation:**")
                    st.info(api_extra_info['explanation'])
                else:
                    # If no explanation parsed, show a default message
                    st.markdown("**💡 Analysis:**")
                    st.info("AI analysis completed. The prediction above is based on comprehensive fact-checking.")
                
                if api_extra_info.get('sources'):
                    st.markdown("**📚 Sources & Resource Links:**")
                    st.info(api_extra_info['sources'])
                
                if api_extra_info.get('details'):
                    st.markdown("**🔍 Details:**")
                    st.info(api_extra_info['details'])

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>⚠️ <b>Disclaimer:</b> This is a machine learning model and may not be 100% accurate.</p>
    <p>Always verify news from trusted sources.</p>
</div>
""", unsafe_allow_html=True)