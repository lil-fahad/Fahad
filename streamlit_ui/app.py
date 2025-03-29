import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os
import subprocess
import logging
import re

from core.auto_predictor import run_prediction
from core.recommender import recommend_options
from core.explainable_ai import explain_prediction

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title='OmniMarket Prophet', layout='wide')
st.title('📊 OmniMarket Prophet - التنبؤ والتحليل الذكي')

# Upload CSV file
uploaded_file = st.file_uploader('📂 حمّل ملف بيانات (CSV)', type=['csv'])
symbol = st.text_input('🔍 أدخل رمز السهم للعرض', value='AAPL')
model_choice = st.selectbox('اختر النموذج', ['xgb', 'lstm'])

# Handle uploaded file
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            st.error('❌ تنسيق الملف غير صالح. تأكد من وجود الأعمدة: Open, High, Low, Close, Volume')
        else:
            file_path = f'data/historical/{symbol}.csv'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            st.success(f'✅ تم حفظ البيانات كـ {file_path}')
    except Exception as e:
        st.error(f'❌ حدث خطأ أثناء تحميل الملف: {e}')

# Model training
if uploaded_file is not None and st.button('تدريب النماذج'):
    try:
        with st.spinner('🧠 جاري تدريب النماذج...'):
            log_path = 'models/training_log.txt'
            os.makedirs('models', exist_ok=True)
            result = subprocess.run(['python3', 'training/trainer.py', '--csv', file_path, '--model', 'all'], capture_output=True)
            if result.returncode == 0:
                st.success('✅ تم تدريب النماذج بنجاح!')
                log_content = result.stdout.decode('utf-8')
                st.markdown('### 📝 سجل التدريب:')
                st.code(log_content)
            else:
                st.error('❌ حدث خطأ أثناء التدريب')
                st.code(result.stderr.decode('utf-8'))
    except Exception as e:
        st.error(f'❌ خطأ أثناء تدريب النماذج: {e}')

# Prediction and Analysis
if st.button('ابدأ التحليل'):
    try:
        with st.spinner('📈 يتم الآن تحليل البيانات...'):
            results = run_prediction(symbol, model_choice=model_choice)
            if not results:
                st.error('❌ لم يتم الحصول على نتائج.')
            for r in results:
                st.write(f'النموذج: {r["model"]}')
                st.write(f'MAE: {r["mae"]:.2f}')
                st.write(f'دقة الاتجاه: {r["direction_accuracy"]:.2f}%')
                recommendation = recommend_options(100, 105)
                explanation = explain_prediction(symbol, f'تغير متوقع بنسبة {recommendation["expected_change_pct"]}%', r['model'])
                st.json(recommendation)
                st.info(explanation)
    except Exception as e:
        st.error(f'❌ خطأ أثناء التحليل: {e}')
