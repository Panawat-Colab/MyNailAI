import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# 1. ตั้งค่าหน้าเว็บ (ต้องอยู่บนสุดเสมอ)
st.set_page_config(
    page_title="MyNailAI - ตรวจโรคเล็บ",
    page_icon="💅",
    layout="wide"
)

# 2. ฟังก์ชันสำหรับโหลดโมเดลตัวจบ (V3 Fine-Tuned)
@st.cache_resource
def load_model():
    # เปลี่ยนมาใช้ชื่อไฟล์ล่าสุดที่เซฟมาจาก Colab Cell 5
    model = tf.keras.models.load_model('nail_disease_model_final.h5')
    return model

with st.spinner('กำลังเตรียมระบบ AI... 🤖'):
    model = load_model()

# 3. ปรับแต่งส่วนหัวหน้าเว็บ
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>💅 MyNailAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>ระบบผู้ช่วย AI ประเมินความเสี่ยงโรคเล็บเบื้องต้น</p>", unsafe_allow_html=True)
st.divider()

# 4. จัดเลย์เอาต์
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 1. อัปโหลดรูปภาพเล็บ")
    uploaded_file = st.file_uploader("เลือกรูปภาพของคุณ (แนะนำถ่ายในที่สว่างให้เห็นหน้าเล็บชัดเจน)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='ภาพที่ระบบรับเข้ามา', use_container_width=True)

with col2:
    st.subheader("🩺 2. ผลการวิเคราะห์จาก AI")
    
    if uploaded_file is None:
        st.info("👈 รอรับรูปภาพจากคุณทางฝั่งซ้ายอยู่ครับ...")
    
    if uploaded_file is not None:
        with st.spinner('กำลังสแกนและประเมินผล...'):
            # แปลงโหมดภาพเพื่อป้องกัน Error จากไฟล์ PNG บางประเภท
            image = image.convert('RGB') 
            
            # --- ไฮไลต์สำคัญ: ย่อรูปโดยรักษาสัดส่วนเดิม 100% แล้วเติมขอบสีดำ (Padding) ---
            image_padded = ImageOps.pad(image, (224, 224), color=(0, 0, 0))
            
            # แปลงเป็น Array และปรับสเกลสี
            img_array = np.array(image_padded)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0) 
            
            # สั่งให้ AI ทำนายผล
            predictions = model.predict(img_array)
            
            class_names = [
                'โรคมะเร็งผิวหนัง (Acral Lentiginous Melanoma)',
                'เล็บปกติ (Healthy Nail)',                      
                'โรคเล็บหนางุ้ม (Onychogryphosis)',             
                'ภาวะเล็บเขียวคล้ำ (Blue Finger)',               
                'โรคนิ้วปุ้ม/เล็บโค้ง (Clubbing)',                 
                'เล็บเป็นหลุมสะเก็ดเงิน (Pitting)'                 
            ]   
            
            confidence = float(np.max(predictions) * 100)
            predicted_class = class_names[np.argmax(predictions)]
            
            # --- ระบบคัดกรองรูปภาพ (Threshold 75%) ---
            if confidence < 75.0:
                st.warning("⚠️ ภาพไม่ชัดเจน หรืออาจไม่ใช่ภาพเล็บ")
                st.write(f"ระบบมีความมั่นใจเพียง **{confidence:.2f}%** (ต่ำกว่าเกณฑ์ 75%) กรุณาถ่ายรูปในมุมที่ชัดเจนขึ้นแล้วลองใหม่อีกครั้งครับ")
                
                # โชว์รูปที่โดนเติมขอบดำให้ผู้ใช้เห็นว่า AI มองเห็นภาพแบบไหน
                st.image(image_padded, caption='ภาพที่ AI มองเห็น (มีขอบดำเพื่อรักษาสัดส่วน)', width=150)
            else:
                if np.argmax(predictions) == 1: 
                    st.success(f"ผลการวิเคราะห์: **{predicted_class}** 🌟")
                else:
                    st.error(f"พบความเสี่ยง: **{predicted_class}**")
                
                st.write(f"ความมั่นใจของโมเดล: **{confidence:.2f}%**")
                st.progress(int(confidence))
            
            st.warning("⚠️ หมายเหตุ: นี่เป็นการประเมินเบื้องต้นโดย AI เท่านั้น ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการรักษานะครับ")
            
            with st.expander("ดูข้อมูลเชิงลึก (สำหรับนักพัฒนา)"):
                st.write("ค่าความน่าจะเป็นของแต่ละหมวดหมู่:", predictions)
