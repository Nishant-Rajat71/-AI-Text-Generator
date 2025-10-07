import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2LMHeadModel,
    GPT2Tokenizer
)

# Page configuration
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="✍️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">✍️ AI Text Generator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate sentiment-aligned text using AI</p>', unsafe_allow_html=True)

# Caching models to avoid reloading
@st.cache_resource
def load_sentiment_model():
    """Load sentiment analysis model"""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_text_generator():
    """Load text generation model"""
    model_name = "distilgpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

# Load models with progress
with st.spinner('Loading AI models... This may take a minute on first run.'):
    sentiment_tokenizer, sentiment_model = load_sentiment_model()
    gen_tokenizer, gen_model = load_text_generator()

st.success('✅ Models loaded successfully!')

# Sentiment analysis function
def analyze_sentiment(text):
    """Analyze sentiment of input text"""
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        sentiment_id = torch.argmax(predictions).item()
        confidence = predictions[0][sentiment_id].item()
    
    sentiment_map = {0: 'negative', 1: 'positive'}
    return sentiment_map[sentiment_id], confidence

# Text generation function
def generate_text(prompt, sentiment, max_length, temperature):
    """Generate sentiment-aligned text"""
    sentiment_prompts = {
        'positive': [
            "This is wonderful because",
            "I am delighted to share that",
            "The amazing thing is",
            "Fortunately,",
            "What a fantastic"
        ],
        'negative': [
            "Unfortunately,",
            "This is disappointing because",
            "Sadly,",
            "The problem is that",
            "It's frustrating that"
        ],
        'neutral': [
            "It should be noted that",
            "The fact is that",
            "Generally speaking,",
            "According to reports,",
            "In this case,"
        ]
    }
    
    import random
    sentiment_prefix = random.choice(sentiment_prompts.get(sentiment, ['']))
    full_prompt = f"{sentiment_prefix} {prompt}"
    
    inputs = gen_tokenizer.encode(full_prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)
    
    outputs = gen_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        pad_token_id=gen_tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sidebar for settings
st.sidebar.header("⚙️ Generation Settings")
sentiment_option = st.sidebar.selectbox(
    "Sentiment Control",
    ["Auto-detect", "Positive", "Negative", "Neutral"],
    help="Choose to detect sentiment automatically or override manually"
)

text_length = st.sidebar.slider(
    "Text Length",
    min_value=50,
    max_value=300,
    value=150,
    step=10,
    help="Adjust the length of generated text"
)

creativity = st.sidebar.slider(
    "Creativity Level",
    min_value=0.5,
    max_value=1.5,
    value=0.8,
    step=0.1,
    help="Higher values = more creative (but less coherent)"
)

# Main input area
st.markdown("### 📝 Enter Your Prompt")
user_prompt = st.text_area(
    "Type your prompt here:",
    placeholder="e.g., 'The future of artificial intelligence'",
    height=100,
    label_visibility="collapsed"
)

# Generate button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button("🚀 Generate Text", use_container_width=True, type="primary")

# Generation logic
if generate_button:
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt first!")
    else:
        with st.spinner('🤖 Analyzing sentiment and generating text...'):
            # Determine sentiment
            if sentiment_option == "Auto-detect":
                sentiment, confidence = analyze_sentiment(user_prompt)
            else:
                sentiment = sentiment_option.lower()
                confidence = 1.0
            
            # Generate text
            generated_text = generate_text(
                user_prompt,
                sentiment,
                text_length,
                creativity
            )
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Detected Sentiment",
                    value=sentiment.upper(),
                    delta=f"{confidence:.1%} confidence"
                )
            
            with col2:
                st.metric(
                    label="Generated Words",
                    value=len(generated_text.split())
                )
            
            st.markdown("### 📄 Generated Text")
            st.info(generated_text)
            
            # Download button
            st.download_button(
                label="💾 Download Text",
                data=generated_text,
                file_name="generated_text.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit | Powered by Hugging Face Transformers</p>
    </div>
""", unsafe_allow_html=True)

# Add examples in expander
with st.expander("💡 See Example Prompts"):
    st.markdown("""
    - "Artificial intelligence is transforming healthcare"
    - "The economic situation is challenging"
    - "Climate change requires immediate action"
    - "Remote work has changed how we collaborate"
    - "Space exploration opens new possibilities"
    """)
