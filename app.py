import streamlit as st
import pandas as pd
import os
import io
import uuid
from datetime import datetime

# Must be first Streamlit command
st.set_page_config(
    page_title="Multilingual Sacred Text NLP Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# Session ID (Fix 4: Authentication)
# ============================================================
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# ============================================================
# Dark / Light Mode Toggle (Feature 10)
# ============================================================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    st.session_state.dark_mode = dark_mode

# Apply theme CSS
if st.session_state.dark_mode:
    bg_color = "#0E1117"
    card_bg = "#1E1E1E"
    text_color = "#FAFAFA"
    sub_text = "#aaa"
    border_color = "#333"
else:
    bg_color = "#FFFFFF"
    card_bg = "#F8F9FA"
    text_color = "#1a1a1a"
    sub_text = "#555"
    border_color = "#ddd"

st.markdown(f"""
<style>
    .main {{ background-color: {bg_color}; color: {text_color}; }}
    h1 {{ color: #4CAF50; text-align: center; font-family: 'Inter', sans-serif; }}
    .stTextArea textarea {{
        background-color: {card_bg}; color: {text_color};
        border: 1px solid {border_color}; border-radius: 10px;
    }}
    .result-card {{
        background-color: {card_bg}; color: {text_color};
        padding: 20px; border-radius: 10px; margin-top: 10px;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }}
    .metric-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
    .sample-btn {{ margin: 2px; }}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar: History Tab (Feature 6)
# ============================================================
with st.sidebar:
    if st.button("🧹 Clear Server Cache (Fix Errors)"):
        st.cache_resource.clear()
        st.success("Cache cleared! Please refresh the page.")
    st.markdown("---")
    st.markdown("### 📜 Analysis History")
    
    try:
        from database.db_connection import get_sqlalchemy_engine
        engine = get_sqlalchemy_engine()
        if engine:
            query = f"SELECT * FROM history WHERE user_id = '{st.session_state.user_id}' ORDER BY timestamp DESC LIMIT 20"
            df_hist = pd.read_sql(query, con=engine)
            
            if len(df_hist) > 0:
                st.dataframe(df_hist[['timestamp', 'input_text', 'language', 'sentiment', 'emotion']], 
                           use_container_width=True, height=300)
                st.caption(f"Recent analyses shown")
            else:
                st.info("No history yet. Analyze some text first!")
        else:
            st.warning("Database not connected.")
    except Exception as e:
        st.info("No history yet.")
        
    # Show Raag Confusion Matrix if exists
    matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raag_confusion_matrix.png')
    if os.path.exists(matrix_path):
        st.markdown("---")
        st.markdown("### 📊 Raag Classifier Performance")
        st.image(matrix_path, caption="SVM Model Confusion Matrix", use_container_width=True)

# ============================================================
# Main Title
# ============================================================
st.title("🧠 Multilingual Sacred Text NLP Analyzer")
st.markdown(f"<p style='text-align: center; color: {sub_text};'>Analyze Language, Sentiment, Emotion, Author, Raag & Cross-Scripture Similarity</p>", unsafe_allow_html=True)

# ============================================================
# Tabs: Single Analysis & Batch Analysis
# ============================================================
tab1, tab2 = st.tabs(["📝 Single Analysis", "📁 Batch Analysis"])

# ==============================================================
# TAB 1: Single Analysis
# ==============================================================
with tab1:
    
    # Sample Text Buttons (Feature 8)
    st.markdown("#### 🔖 Quick Sample Texts")
    sample_cols = st.columns(4)
    
    sample_texts = {
        "Try Gurbani": "ਜਪੁ ਤਪੁ ਸੰਜਮੁ ਧਰਮੁ ਨ ਕਮਾਇਆ ॥ ਸੇਵਾ ਸਾਧ ਨ ਜਾਣਿਆ ਹਰਿ ਰਾਇਆ ॥",
        "Try Quran": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "Try Bible": "For God so loved the world that he gave his only begotten Son that whosoever believeth in him should not perish but have everlasting life.",
        "Try Hindi": "भगवान सबसे प्यार करता है और हमें एक दूसरे की मदद करनी चाहिए।"
    }
    
    if 'sample_text' not in st.session_state:
        st.session_state.sample_text = ""
    
    for i, (label, sample) in enumerate(sample_texts.items()):
        with sample_cols[i]:
            if st.button(label, use_container_width=True):
                st.session_state.sample_text = sample
    
    # Text Input
    user_input = st.text_area(
        "Enter your text here (English, Punjabi, Hindi, Arabic, etc.):",
        value=st.session_state.sample_text,
        height=150,
        placeholder="Type or click a sample button above..."
    )
    
    if st.button("🚀 Analyze Text", use_container_width=True):
        if not user_input or not user_input.strip():
            st.error("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing your text, please wait."):
                from main_pipeline import analyze_text
                results = analyze_text(user_input, user_id=st.session_state.user_id)
            
            if results.get('error'):
                st.error(results['error'])
            else:
                st.success("✅ Analysis Complete!")
                
                # Store results in session for export
                st.session_state.last_results = results
                
                st.markdown("### 📊 Analysis Results")
                
                # ---- Language & Translation ----
                st.markdown(f"""
                <div class="result-card">
                    <h4>🌐 Language</h4>
                    <p>Detected Language: <b>{results['language']}</b></p>
                """, unsafe_allow_html=True)
                if results['translated_text'] != results['original_text']:
                    st.markdown(f"<p>📝 Translated to English: <i>{results['translated_text']}</i></p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ---- Sentiment & Emotion (with Confidence Warning - Feature 7) ----
                c1, c2 = st.columns(2)
                
                with c1:
                    label_up = results['sentiment_label'].upper() if results['sentiment_label'] else ''
                    if label_up == "POSITIVE":
                        sent_color = "#4CAF50"
                    elif label_up == "NEUTRAL":
                        sent_color = "#FFC107"
                    else:
                        sent_color = "#F44336"
                    
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color: {sent_color};">
                        <h4>💭 Sentiment</h4>
                        <p class="metric-value" style="color: {sent_color};">{results['sentiment_label'].title() if results['sentiment_label'] else 'N/A'}</p>
                        <p>Confidence: {results['sentiment_score']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if results['sentiment_score'] < 60:
                        st.warning("⚠️ Low confidence result, please verify manually.")
                
                with c2:
                    st.markdown(f"""
                    <div class="result-card" style="border-left-color: #2196F3;">
                        <h4>💫 Emotion</h4>
                        <p class="metric-value" style="color: #2196F3;">{results['emotion_label'].title() if results['emotion_label'] else 'N/A'}</p>
                        <p>Confidence: {results['emotion_score']}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if results['emotion_score'] < 60:
                        st.warning("⚠️ Low confidence result, please verify manually.")
                
                # ---- Author & Raag Prediction (Features 2 & 3) ----
                c3, c4 = st.columns(2)
                
                with c3:
                    author_data = results.get('predicted_author', {})
                    if author_data.get('author') != 'N/A':
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: #9C27B0;">
                            <h4>👤 Predicted Author / Guru</h4>
                            <p class="metric-value" style="color: #9C27B0;">{author_data.get('author', 'Unknown')}</p>
                            <p>Confidence: {author_data.get('confidence', 0)}%</p>
                            {f'<p>📖 SGGS Ang (Page): <b>{author_data.get("ang", "")}</b></p>' if author_data.get('ang') else ''}
                            {f'<p style="font-size: 0.8em; color: gray;"><i>Approx. match: "{author_data.get("matched_text", "")[:60]}..."</i></p>' if author_data.get("matched_text") and author_data.get('confidence', 0) < 100 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if author_data.get('confidence', 0) < 60:
                            st.warning("⚠️ Low confidence result, please verify manually.")
                    else:
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: #9C27B0; opacity: 0.6;">
                            <h4>👤 Author / Guru</h4>
                            <p><i>Not applicable (Gurbani only)</i></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with c4:
                    raag_data = results.get('predicted_raag', {})
                    if raag_data.get('raag') != 'N/A':
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: #FF5722;">
                            <h4>🎵 Predicted Raag</h4>
                            <p class="metric-value" style="color: #FF5722;">{raag_data.get('raag', 'Unknown')}</p>
                            <p>Confidence: {raag_data.get('confidence', 0)}%</p>
                            {f'<p style="font-size: 0.8em; color: gray;"><i>Approx. match: "{raag_data.get("matched_text", "")}"</i></p>' if raag_data.get("matched_text") and raag_data.get('confidence', 0) < 100 else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if raag_data.get('confidence', 0) < 60:
                            st.warning("⚠️ Low confidence result, please verify manually.")
                    else:
                        st.markdown(f"""
                        <div class="result-card" style="border-left-color: #FF5722; opacity: 0.6;">
                            <h4>🎵 Predicted Raag</h4>
                            <p><i>Not applicable (Gurbani only)</i></p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # ---- Themes / Entities ----
                st.markdown(f"""
                <div class="result-card" style="border-left-color: #FFC107;">
                    <h4>📌 Themes / Entities</h4>
                """, unsafe_allow_html=True)
                if results['themes']:
                    for ent_text, ent_label in results['themes']:
                        st.markdown(f"- **{ent_text}** : `{ent_label}`")
                else:
                    st.markdown("<i>No specific entities found.</i>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ---- Cross-Scripture Similarity (Feature 1) ----
                similar = results.get('similar_verses', {})
                input_text_clean = results.get('original_text', '').strip()
                
                if similar:
                    st.markdown("### 📖 Cross-Scripture Similar Verses")
                    
                    # Always show all 3 tabs with debug lengths
                    g_len = len(similar.get('Gurbani', []))
                    q_len = len(similar.get('Quran', []))
                    b_len = len(similar.get('Bible', []))
                    sim_tabs = st.tabs([f"🕉️ Gurbani ({g_len})", f"☪️ Quran ({q_len})", f"✝️ Bible ({b_len})"])
                    
                    tab_colors = {'Gurbani': '#FF9800', 'Quran': '#4CAF50', 'Bible': '#2196F3'}
                    tab_icons  = {'Gurbani': '🕉️', 'Quran': '☪️', 'Bible': '✝️'}
                    
                    for tab, scripture_name in zip(sim_tabs, ['Gurbani', 'Quran', 'Bible']):
                        with tab:
                            all_verses = similar.get(scripture_name, [])
                            
                            # For Gurbani tab: filter out the exact verse the user entered
                            if scripture_name == 'Gurbani':
                                verses = [
                                    v for v in all_verses
                                    if v.get('text', '').strip() != input_text_clean
                                    and input_text_clean not in v.get('text', '')
                                ]
                                if not verses and all_verses:
                                    # All were the same verse — show from index 1 onwards
                                    verses = all_verses[1:]
                            else:
                                verses = all_verses
                            
                            color = tab_colors[scripture_name]
                            icon  = tab_icons[scripture_name]
                            
                            if verses:
                                for j, v in enumerate(verses, 1):
                                    gurmukhi_line = v.get('text', '')
                                    english_line  = v.get('english', v.get('text', ''))
                                    
                                    if scripture_name == 'Gurbani':
                                        ang_ref    = v.get('ang', '')
                                        author_val = v.get('author', '')
                                        raag_val   = v.get('raag', '')
                                        
                                        ang_display  = f'&nbsp;|&nbsp; 📖 Ang <b>{ang_ref}</b>' if ang_ref else ''
                                        raag_display = f'&nbsp;|&nbsp; 🎵 <b>{raag_val}</b>' if raag_val and raag_val.lower() not in ('unknown', '', 'none') else ''
                                        show_author  = author_val and author_val.lower() not in ('unknown', '', 'none')
                                        
                                        if show_author or ang_ref:
                                            meta_line = f'<p>{icon} <b>{author_val}</b>{raag_display}{ang_display}</p>'
                                        elif raag_display:
                                            meta_line = f'<p>{icon} {raag_display}</p>'
                                        else:
                                            meta_line = ''
                                        
                                        st.markdown(f"""
                                        <div class="result-card" style="border-left-color: {color};">
                                            <p><b>#{j}</b> — Similarity: <b>{v['score']}%</b></p>
                                            <p style="font-size:1.1em; font-family: serif;">{gurmukhi_line}</p>
                                            {f'<p><i>"{english_line}"</i></p>' if english_line and english_line != gurmukhi_line else ''}
                                            {meta_line}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="result-card" style="border-left-color: {color};">
                                            <p><b>#{j}</b> — Similarity: <b>{v['score']}%</b></p>
                                            <p><i>"{v.get('text', '')}"</i></p>
                                            <p>{icon} {v.get('author', '')}</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                st.info(f"No other similar {scripture_name} verses found.")
                
                # ---- Export Single Result (Feature 9) ----
                st.markdown("---")
                export_data = {
                    'Language': [results['language']],
                    'Original Text': [results['original_text'][:500]],
                    'Translated Text': [results['translated_text'][:500]],
                    'Sentiment': [results['sentiment_label']],
                    'Sentiment Score': [results['sentiment_score']],
                    'Emotion': [results['emotion_label']],
                    'Emotion Score': [results['emotion_score']],
                    'Predicted Author': [results['predicted_author']['author']],
                    'Author Confidence': [results['predicted_author']['confidence']],
                    'Predicted Raag': [results['predicted_raag']['raag']],
                    'Raag Confidence': [results['predicted_raag']['confidence']],
                    'Entities': [str(results['themes'])]
                }
                
                # Add similar verses to export
                for scripture_name in ['Gurbani', 'Quran', 'Bible']:
                    verses = results.get('similar_verses', {}).get(scripture_name, [])
                    for k, v in enumerate(verses, 1):
                        export_data[f'Similar {scripture_name} {k}'] = [v['text'][:200]]
                        export_data[f'Similar {scripture_name} {k} Score'] = [v['score']]
                
                df_export = pd.DataFrame(export_data)
                csv_buffer = df_export.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_buffer,
                    file_name=f"nlp_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ==============================================================
# TAB 2: Batch Analysis (Feature 4)
# ==============================================================
with tab2:
    st.markdown("### 📁 Batch Analysis — Upload CSV")
    st.markdown("Upload a CSV file with a `text` column. The system will run the full NLP pipeline on every row.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df_uploaded = pd.read_csv(uploaded_file, encoding='latin-1')
        
        if 'text' not in df_uploaded.columns:
            st.error("CSV must have a 'text' column! Found columns: " + str(df_uploaded.columns.tolist()))
        else:
            st.success(f"Loaded {len(df_uploaded)} rows. Click below to start batch analysis.")
            
            if st.button("🚀 Run Batch Analysis", use_container_width=True):
                from main_pipeline import analyze_text
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                batch_results = []
                total = len(df_uploaded)
                
                for i, row in df_uploaded.iterrows():
                    text = str(row['text'])
                    status_text.text(f"Analyzing row {i+1}/{total}...")
                    progress_bar.progress((i + 1) / total)
                    
                    r = analyze_text(text, user_id=st.session_state.user_id)
                    
                    batch_results.append({
                        'Input Text': text[:200],
                        'Language': r.get('language', ''),
                        'Translation': r.get('translated_text', '')[:200],
                        'Sentiment': r.get('sentiment_label', ''),
                        'Sentiment Score': r.get('sentiment_score', ''),
                        'Emotion': r.get('emotion_label', ''),
                        'Emotion Score': r.get('emotion_score', ''),
                        'Author': r.get('predicted_author', {}).get('author', ''),
                        'Author Confidence': r.get('predicted_author', {}).get('confidence', ''),
                        'Raag': r.get('predicted_raag', {}).get('raag', ''),
                        'Raag Confidence': r.get('predicted_raag', {}).get('confidence', ''),
                        'Entities': str(r.get('themes', []))
                    })
                
                status_text.text("✅ Batch analysis complete!")
                
                df_batch = pd.DataFrame(batch_results)
                st.dataframe(df_batch, use_container_width=True)
                
                # Download batch results
                batch_csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Results as CSV",
                    data=batch_csv,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

# ============================================================
# Footer
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; color:{sub_text};'>© 2026 Multilingual Sacred Text NLP Analyzer. Developed by Darshan Lal.</p>", unsafe_allow_html=True)
