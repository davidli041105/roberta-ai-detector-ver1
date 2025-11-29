import streamlit as st
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

# ================= 页面配置 =================
st.set_page_config(page_title="AI 文本检测器", page_icon="🕵️")
st.title("🕵️ AI 文本生成检测 (RoBERTa + LoRA)")
st.markdown("基于 RoBERTa-base 微调模型，判断文本是 **人类撰写** 还是 **AI 生成**。")

# ================= 模型加载逻辑 =================

# 你的 LoRA 权重文件夹名 (必须和上传到 GitHub 的文件夹名一致)
LORA_PATH = "final_lora_model" 

@st.cache_resource
def load_model():
    """
    加载模型函数，使用缓存避免每次预测都重新下载
    """
    print("正在加载配置...")
    
    # 1. 加载 LoRA 配置
    # 注意：这里我们只读取配置，不直接加载模型
    if not os.path.exists(LORA_PATH):
        st.error(f"找不到文件夹: {LORA_PATH}，请检查 GitHub 仓库结构")
        return None, None
        
    config = PeftConfig.from_pretrained(LORA_PATH)
    
    # 2. 确定基座模型名称
    # 关键修改：云端没有 '/root/autodl-tmp/...' 这种路径。
    # 我们强制将其指向 Hugging Face 官方模型 ID。
    # 如果你用的是 roberta-base，这里写 "roberta-base"
    # 如果是中文 roberta，可能是 "hfl/chinese-roberta-wwm-ext"
    base_model_name = "roberta-base" 
    
    print(f"正在从 HuggingFace 下载基座模型: {base_model_name}...")
    
    # 3. 加载基座模型 (从网络下载)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2, # 保持和你训练时一致
        ignore_mismatched_sizes=True 
    )
    
    # 4. 加载分词器 (Tokenizer)
    # 优先尝试从 LoRA 文件夹加载，如果没有，则从基座加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 5. 合并 LoRA 权重
    print("正在合并 LoRA 权重...")
    inference_model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # 6. 设备配置 (Streamlit Cloud 只有 CPU，所以这里自动判断)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_model.to(device)
    inference_model.eval()
    
    return inference_model, tokenizer, device

# ================= UI 交互与推理 =================

# 显示加载状态
with st.spinner('正在初始化模型，初次运行可能需要下载基座模型 (约500MB)...'):
    try:
        model, tokenizer, device = load_model()
        if model:
            st.success("模型加载完毕！")
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        st.stop()

# 输入区域
text_input = st.text_area("请输入要检测的英文文本：", height=200, placeholder="Type something here...")

if st.button("开始检测", type="primary"):
    if not text_input.strip():
        st.warning("请先输入内容！")
    else:
        # 数据预处理
        inputs = tokenizer(
            text_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            # 获取概率
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # 获取最大概率的标签索引
            pred_label = torch.argmax(probs, dim=-1).item()
            
            # 获取 AI (标签1) 的概率
            ai_probability = probs[0][1].item()
            human_probability = probs[0][0].item()

        # 结果展示
        st.divider()
        
        # 逻辑：标签 1 = AI, 标签 0 = 人类
        if pred_label == 1:
            st.error("🤖 检测结果：AI 生成")
            st.progress(ai_probability)
            st.write(f"**AI 概率:** {ai_probability:.2%}")
        else:
            st.success("🧑 检测结果：人类撰写")
            st.progress(human_probability)
            st.write(f"**人类概率:** {human_probability:.2%}")

# debug 信息 (可选)
with st.expander("查看详细概率"):
    if 'probs' in locals():
        st.json({
            "Human_Label_0": f"{probs[0][0].item():.4f}",
            "AI_Label_1": f"{probs[0][1].item():.4f}"
        })