import streamlit as st
import websocket
import threading
import json
import time
import pandas as pd
import plotly.graph_objs as go
import logging

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title='OmniMarket Live Chart', layout='centered')
st.title('📈 OmniMarket Live Chart')

API_KEY = st.secrets.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
symbol = st.text_input('أدخل رمز السهم', value='AAPL').upper()

price_display = st.empty()
status_display = st.empty()
chart_display = st.empty()
latest_data = []

def on_open(ws):
    try:
        status_display.success('✅ الاتصال مفتوح')
        ws.send(json.dumps({'action': 'auth', 'params': API_KEY}))
        ws.send(json.dumps({'action': 'subscribe', 'params': f'T.{symbol}'}))
    except Exception as e:
        logging.exception('Error during WebSocket connection setup')

def on_message(ws, message):
    try:
        data = json.loads(message)
        if isinstance(data, list) and 'p' in data[0]:
            timestamp = pd.to_datetime(data[0]['t'], unit='ms')
            price = data[0]['p']
            latest_data.append((timestamp, price))
            if len(latest_data) > 100:
                latest_data.pop(0)
    except (KeyError, ValueError, IndexError) as e:
        logging.error(f'Error processing message: {e}')

def on_error(ws, error):
    status_display.error(f'❌ WebSocket Error: {error}')
    logging.error(f'WebSocket Error: {error}')

def on_close(ws):
    status_display.warning('🔌 تم إغلاق الاتصال')
    logging.info('WebSocket connection closed')

def run_ws():
    try:
        ws = websocket.WebSocketApp(
            'wss://socket.polygon.io/stocks',
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        ws.run_forever()
    except Exception as e:
        logging.exception('WebSocket encountered a critical failure')

ws_thread = threading.Thread(target=run_ws)
ws_thread.daemon = True
ws_thread.start()

while True:
    if latest_data:
        df = pd.DataFrame(latest_data, columns=['time', 'price'])
        fig = go.Figure(go.Scatter(x=df['time'], y=df['price'], mode='lines+markers'))
        fig.update_layout(title=f'السعر اللحظي لـ {symbol}', xaxis_title='الوقت', yaxis_title='السعر')
        chart_display.plotly_chart(fig, use_container_width=True)
        price_display.metric(label='أحدث سعر', value=f'${df["price"].iloc[-1]:.2f}')
    time.sleep(1)
