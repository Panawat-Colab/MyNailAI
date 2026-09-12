import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. ตั้งค่าหน้าเว็บ (ต้องอยู่บนสุดเสมอ)
st.set_page_config(
    page_title="MyNailAI - ตรวจโรคเล็บ",
    page_icon="💅",
    layout="wide" # ขยายหน้าเว็บให้กว้างขึ้น
)

# ฟังก์ชันสำหรับโหลดโมเดล 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('nail_disease_model_v1.h5')
    return model

with st.spinner('กำลังเตรียมระบบ AI... 🤖'):
    model = load_model()

# 2. ปรับแต่งส่วนหัวด้วย HTML/Markdown ให้ดูโดดเด่น
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>💅 MyNailAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>ระบบผู้ช่วย AI ประเมินความเสี่ยงโรคเล็บเบื้องต้น</p>", unsafe_allow_html=True)
st.divider() # เส้นคั่นเพื่อความสวยงาม

# 3. จัดเลย์เอาต์แบบ 2 คอลัมน์ (ซ้ายอัปโหลดรูป - ขวาแสดงผล)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 1. อัปโหลดรูปภาพเล็บ")
    uploaded_file = st.file_uploader("เลือกรูปภาพของคุณ (แนะนำให้ถ่ายในที่สว่างและเห็นเล็บชัดเจน)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='ภาพที่ระบบกำลังวิเคราะห์', use_container_width=True)

with col2:
    st.subheader("🩺 2. ผลการวิเคราะห์จาก AI")
    
    if uploaded_file is None:
        st.info("👈 รอรับรูปภาพจากคุณทางฝั่งซ้ายอยู่ครับ...")
    
    if uploaded_file is not None:
        with st.spinner('กำลังสแกนและประเมินผล...'):
            # แปลงรูปภาพให้เข้ากับโมเดล
# --- อัปเดตใหม่: ตัดรูปภาพให้สัดส่วนไม่เพี้ยน ---
            image = image.convert('RGB') 
            
            # 1. หาขนาดที่สั้นที่สุดเพื่อตัดรูปเป็นสี่เหลี่ยมจัตุรัส (Center Crop)
            width, height = image.size
            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2
            image_cropped = image.crop((left, top, right, bottom))
            
            # 2. ค่อยย่อขนาดเป็น 128x128 ตามที่ AI ต้องการ
            image_resized = image_cropped.resize((128, 128)) 
            
            img_array = np.array(image_resized)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0) 
            # --------------------------------------------
            # ให้ AI ทำนายผล
            predictions = model.predict(img_array)
            
         # --- แก้ไขชื่อโรค 6 คลาสให้ตรงกับที่เทรนใน Colab เป๊ะๆ ---
            class_names = [
                'โรคมะเร็งผิวหนัง (Acral Lentiginous Melanoma)', # คลาส 0
                'เล็บปกติ (Healthy Nail)',                      # คลาส 1
                'โรคเล็บหนางุ้ม (Onychogryphosis)',             # คลาส 2
                'ภาวะเล็บเขียวคล้ำ (Blue Finger)',               # คลาส 3
                'โรคนิ้วปุ้ม/เล็บโค้ง (Clubbing)',                 # คลาส 4
                'เล็บเป็นหลุมสะเก็ดเงิน (Pitting)'                 # คลาส 5
            ]   
            
            predicted_class = class_names[np.argmax(predictions)]
            confidence = float(np.max(predictions) * 100)
            
            # แสดงผลลัพธ์
            st.success(f"พบความเสี่ยง: **{predicted_class}**")
            
            # แสดงหลอดความมั่นใจ
            st.write(f"ความมั่นใจของโมเดล: **{confidence:.2f}%**")
            st.progress(int(confidence)) # หลอดสีแสดงเปอร์เซ็นต์
            
            st.warning("⚠️ หมายเหตุ: นี่เป็นการประเมินเบื้องต้นโดย AI เท่านั้น ควรปรึกษาแพทย์ผู้เชี่ยวชาญเพื่อการรักษานะครับ")
            
            # ซ่อนข้อมูลดิบไว้เพื่อไม่ให้เกะกะ
            with st.expander("ดูข้อมูลเชิงลึก (สำหรับนักพัฒนา)"):
                st.write("ค่าความน่าจะเป็นของแต่ละหมวดหมู่:", predictions)    
