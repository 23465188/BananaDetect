import streamlit as st
from zhipuai import ZhipuAI
import base64
from PIL import Image
import io
import cv2
import numpy as np
import plotly.graph_objects as go
from rembg import remove

# ================= 1. 产品级配置 (全端通用设置) =================
st.set_page_config(
    page_title="香蕉智能分选 V60 Universal",
    page_icon="🍌",
    layout="wide",  # 电脑端铺满全屏，手机端自动适应
    initial_sidebar_state="auto"  # 智能判断：电脑展开，手机收起
)

# ================= V60 CSS: 深色工业风 (适配 PC & Mobile) =================
st.markdown("""
<style>
    /* 1. 全局深色背景 */
    .stApp { background-color: #262730 !important; }

    /* 2. 字体适配 */
    p, h1, h2, h3, h4, h5, h6, span, label, div[data-testid="stMetricLabel"], .stTable {
        color: #E0E0E0 !important;
    }
    div[data-testid="stMetricValue"] { color: #FFD700 !important; }

    /* 3. 按钮优化：兼顾鼠标点击和手指触摸 */
    .stButton>button {
        background-color: #F4D03F; 
        color: #1F2026 !important; 
        border-radius: 10px; 
        width: 100%; 
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02); /* 电脑端悬停特效 */
    }

    /* 4. 结果卡片 */
    .result-card {
        background: #363940; 
        border: 1px solid #444;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }

    /* 5. 侧边栏背景 */
    section[data-testid="stSidebar"] { background-color: #1F2026 !important; }

    /* 6. 图片圆角 */
    [data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# API 初始化 (这里从 secrets 读取 key)
try:
    # 兼容两种写法，防止报错
    if "zhipu_api_key" in st.secrets:
        API_KEY = st.secrets["zhipu_api_key"]
    elif "ZHIPU_API_KEY" in st.secrets:
        API_KEY = st.secrets["ZHIPU_API_KEY"]
    else:
        API_KEY = None

    if API_KEY:
        client = ZhipuAI(api_key=API_KEY)
    else:
        client = None
except Exception:
    client = None


# ================= 2. 核心算法 (V50 固化内核) =================
@st.cache_data(show_spinner=False)
def opencv_engine(pil_image):
    max_width = 800
    if pil_image.width > max_width:
        ratio = max_width / pil_image.width
        new_height = int(pil_image.height * ratio)
        pil_image = pil_image.resize((max_width, new_height))

    try:
        nobg_pil = remove(pil_image)
    except Exception:
        return pil_image, 0, 0, 0, 0, 0, 0.0

    img_rgba = np.array(nobg_pil)
    # 防止空图报错
    if img_rgba.ndim != 3 or img_rgba.shape[2] != 4:
        return pil_image, 0, 0, 0, 0, 0, 0.0

    base_mask = (img_rgba[:, :, 3] > 20).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    banana_mask = cv2.erode(base_mask, kernel, iterations=2)
    total_pixels = cv2.countNonZero(banana_mask)
    if total_pixels == 0: return pil_image, 0, 0, 0, 0, 0, 0.0

    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b_channel))
    img_corrected = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2HSV)

    mask_brown = cv2.inRange(hsv, np.array([0, 40, 0]), np.array([25, 255, 140]))
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    mask_muddy = cv2.inRange(hsv, np.array([0, 5, 0]), np.array([180, 60, 110]))

    mask_rot_all = cv2.bitwise_or(mask_brown, mask_black)
    mask_rot_all = cv2.bitwise_or(mask_rot_all, mask_muddy)
    mask_rot_final = cv2.bitwise_and(mask_rot_all, mask_rot_all, mask=banana_mask)

    num_raw, _, stats_raw, _ = cv2.connectedComponentsWithStats(mask_rot_final, connectivity=4)
    max_blob_raw = 0.0
    if num_raw > 1:
        max_blob_raw = round((np.max(stats_raw[1:, 4]) / total_pixels) * 100, 2)

    mask_rot_eroded = cv2.erode(mask_rot_final, np.ones((3, 3), np.uint8), iterations=2)
    num_split, _, stats_split, _ = cv2.connectedComponentsWithStats(mask_rot_eroded, connectivity=4)
    max_blob_eroded = 0.0
    if num_split > 1:
        max_blob_eroded = round((np.max(stats_split[1:, 4]) / total_pixels) * 100, 2)

    survival = 0.0
    if max_blob_raw > 0.01:
        survival = round(max_blob_eroded / max_blob_raw, 2)

    lower_green = np.array([36, 40, 40]);
    upper_green = np.array([90, 255, 255])
    lower_yellow = np.array([20, 40, 46]);
    upper_yellow = np.array([35, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.bitwise_and(mask_green, cv2.bitwise_not(mask_rot_final), mask=banana_mask)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_yellow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_rot_final), mask=banana_mask)
    mask_yellow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_green), mask=banana_mask)

    g = round((cv2.countNonZero(mask_green) / total_pixels) * 100, 2)
    y = round((cv2.countNonZero(mask_yellow) / total_pixels) * 100, 2)
    b = round((cv2.countNonZero(mask_rot_final) / total_pixels) * 100, 2)

    res_img = img_corrected.copy()
    contours, _ = cv2.findContours(mask_rot_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res_img, contours, -1, (0, 0, 255), 1)

    final_show = cv2.cvtColor(res_img, cv2.COLOR_BGR2BGRA)
    final_show[:, :, 3] = base_mask
    white_bg = np.ones_like(final_show, dtype=np.uint8) * 255
    alpha = final_show[:, :, 3] / 255.0
    for c in range(3):
        white_bg[:, :, c] = (1.0 - alpha) * white_bg[:, :, c] + alpha * final_show[:, :, c]

    return Image.fromarray(cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB)), g, y, b, max_blob_raw, max_blob_eroded, survival


# ================= 3. 业务逻辑层 =================
def analyze_data(g, y, b, max_blob, max_eroded, survival, feel, smell):
    if "酒精" in smell or "发酵" in smell:
        return "严重腐烂 (内部变质)", 1, True, 0, "闻到酒精/发酵味，说明内部已发生厌氧腐烂！"
    if "软烂" in feel:
        return "严重腐烂 (结构崩解)", 1, True, 0, "手感软烂，细胞壁已破裂，不可食用。"

    if max_blob > 15.0:
        return "严重腐烂/压伤", 1, True, 0, f"检测到巨型坏死区域(占比{max_blob}%)，触发熔断。"

    if max_blob > 10.0:
        if max_eroded > 5.0 or survival > 0.15:
            return "局部压伤", 2, True, 0, "检测到深层损伤，切除后谨慎食用。"
        return "特级芝麻蕉", 10, False, 1, "高密度糖心斑点，熟度极佳，立即食用。"

    if b > 10.0:
        return "优选芝麻蕉", 9, False, 2, "均匀芝麻斑，口感软糯，赏味期最佳。"

    if g > 15.0:
        return "生鲜香蕉", 4, False, 5, "尚未完全成熟，建议催熟 3-5 天。"

    return "标准好果", 9, False, 4, "色泽金黄，果体饱满。常温存放 3-4 天。"


def get_radar_chart(visual, touch, smell, score, safety):
    fig = go.Figure(data=go.Scatterpolar(
        r=[visual, touch, smell, score, safety],
        theta=['外观', '触感', '气味', '综合', '安全'],
        fill='toself', line_color='#F4D03F', opacity=0.8
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=False, linecolor='#555'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False, height=220, margin=dict(t=20, b=20, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0', size=12)
    )
    return fig


def encode_img(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# ================= 4. UI 交互层 =================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/banana.png", width=90)
    st.markdown("### 🍌 智能分选 Universal")
    mode = st.radio("模式", ["🛒 生活精选", "🏭 工业分选", "👓 无障碍"])
    st.markdown("---")
    with st.expander("🖐️ 传感器校准"):
        feel_opt = st.radio("触感", ["未知", "硬实", "有弹性", "软烂"], index=0)
        smell_opt = st.radio("气味", ["未知", "无味", "浓郁", "酒精味"], index=0)

st.markdown(f"#### {mode}")

# 灵活布局：提供多种输入方式
# st.camera_input 电脑上有摄像头也能用，没有就隐藏
camera_input = st.camera_input("📸 拍照检测")
upload_input = st.file_uploader("📂 上传图片", type=["jpg", "png", "jpeg"])

target_file = camera_input if camera_input else upload_input

col1, col2 = st.columns([1, 1])

if target_file:
    img = Image.open(target_file).convert('RGB')

    if 'last_id' not in st.session_state or st.session_state.last_id != target_file.file_id:
        st.session_state.last_id = target_file.file_id
        with st.spinner("⚡ 正在分析..."):
            # 防止 RGBA 错误
            if img.mode != 'RGB':
                img = img.convert('RGB')
            cv_img, g, y, b, max_b, max_e, surv = opencv_engine(img)
            st.session_state.data = (g, y, b, max_b, max_e, surv)
            st.session_state.res_img = cv_img
            # 每次新图片也清空一下旧的 AI 评价，强制刷新
            if 'ai_comment' in st.session_state:
                del st.session_state.ai_comment

    with col1:
        # 电脑端并排，手机端自动变上面
        if 'res_img' in st.session_state:
            st.image(st.session_state.res_img if mode == "🏭 工业分选" else img,
                     caption="AI 分析视图", use_container_width=True)

    with col2:
        if 'data' in st.session_state:
            g, y, b, max_b, max_e, surv = st.session_state.data
            grade, score, is_fatal, days, advice = analyze_data(g, y, b, max_b, max_e, surv, feel_opt, smell_opt)

            visual_score = 1 if is_fatal else (8 if b > 30 else 9)
            touch_score = 1 if "软烂" in feel_opt or is_fatal else 10
            smell_score = 1 if "酒精" in smell_opt or is_fatal else 10
            safety_score = 1 if is_fatal else 10

            st.markdown("---")
            color = "#FF4B4B" if is_fatal else "#28a745"

            if mode == "👓 无障碍":
                if is_fatal:
                    st.error("🛑 坏果！不可食用")
                else:
                    st.success(f"✅ 好果！{grade}")

            elif mode == "🏭 工业分选":
                c1, c2 = st.columns(2)
                c1.metric("评分", score)
                c2.metric("硬度", f"{int(surv * 100)}%")
                st.table({"维度": ["瑕疵率", "判定"], "数值": [f"{b}%", grade]})

            else:
                st.markdown(f"""
                <div class="result-card" style="border-left: 5px solid {color};">
                    <h2 style="color:{color}; margin:0;">{grade}</h2>
                    <h3 style="color:#FFD700;">综合评分: {score}</h3>
                    <div style="margin-top:10px; color:#ccc;">
                        📅 <strong>保质期:</strong> {days} 天<br>
                        💡 <strong>建议:</strong> {advice}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(get_radar_chart(visual_score, touch_score, smell_score, score, safety_score),
                                use_container_width=True)

            # ================= 5. AI 鉴赏师模块 (接在雷达图后面) =================
            st.markdown("### 🎩 AI 鉴赏师点评")

            if client:
                # 只有当没有缓存的评论时才请求，节省Token
                if 'ai_comment' not in st.session_state:
                    try:
                        img_b64 = encode_img(st.session_state.res_img)  # 使用去背景后的图给AI看
                        prompt = f"""
                        你是一位幽默毒舌但专业的水果鉴赏家。
                        OpenCV检测数据：【{grade}】，评分【{score}分】。
                        请根据图片和数据，用一两句风趣的话点评。
                        如果是好香蕉就浮夸地夸，如果是烂香蕉就幽默警示，如果是青香蕉就调侃。
                        """
                        with st.spinner("🤖 AI 鉴赏师正在整理毒舌语录..."):
                            response = client.chat.completions.create(
                                model="glm-4v",
                                messages=[
                                    {"role": "user", "content": [
                                        {"type": "text", "text": prompt},
                                        {"type": "image_url", "image_url": {"url": img_b64}}
                                    ]}
                                ]
                            )
                            st.session_state.ai_comment = response.choices[0].message.content
                    except Exception as e:
                        st.caption(f"AI 连接波动: {e}")

                # 显示评论 (金边黑底 V60 样式)
                if 'ai_comment' in st.session_state:
                    st.markdown(
                        f"""
                        <div style="background-color:#2b2b2b;padding:20px;border-radius:10px;border-left:5px solid #FFC107;">
                            <p style="font-size:16px;font-style:italic;color:#E0E0E0">“{st.session_state.ai_comment}”</p>
                            <p style="text-align:right;font-size:12px;color:#888;">—— 智谱 GLM-4V</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.caption("🔒 鉴赏师未上线 (请配置 Secrets: zhipu_api_key)")