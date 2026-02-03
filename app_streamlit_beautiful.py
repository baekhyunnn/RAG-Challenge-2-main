import os
import streamlit as st
from streamlit_option_menu import option_menu
from app_streamlit import StreamlitRAG

# 页面配置
st.set_page_config(page_title="AI知识库助手", page_icon="🤖", layout="wide")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

class BeautifulRAGApp:
    def __init__(self):
        self.rag = StreamlitRAG()
    
    def run(self):
        with st.sidebar:
            st.title("🤖 AI 知识库")
            selected = option_menu(
                menu_title=None,
                options=["智能问答", "系统设置"],
                icons=["chat", "gear"],
                default_index=0
            )
            st.divider()
            st.metric("索引文档块", self.rag.doc_count)
            st.write("模型状态:", "✅ 就绪" if self.rag.llm else "❌ 未配置")
            if st.button("🗑️ 清空历史"):
                st.session_state.chat_history = []
                st.rerun()

        if selected == "智能问答":
            self.chat_page()
        else:
            self.settings_page()

    def chat_page(self):
        st.markdown("### 💬 智能问答")
        for msg in st.session_state.chat_history:
            st.chat_message("user").write(msg[0])
            st.chat_message("assistant").write(msg[1])

        if prompt := st.chat_input("请输入您的问题..."):
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("检索中..."):
                    result = self.rag.ask_question(prompt)
                    if result:
                        st.write(result["answer"])
                        st.caption(f"⏱️ 耗时: {result['time']:.2f}s")
                        st.session_state.chat_history.append((prompt, result["answer"]))
                    else:
                        st.error("无法回答，请检查数据库或 API 配置。")

    def settings_page(self):
        st.markdown("### ⚙️ 系统设置")
        with st.form("settings"):
            key = st.text_input("API Key", value=os.getenv("DASHSCOPE_API_KEY", ""), type="password")
            if st.form_submit_button("保存"):
                os.environ["DASHSCOPE_API_KEY"] = key
                self.rag.api_key = key
                self.rag.init_llm()
                st.success("配置已更新")

if __name__ == "__main__":
    BeautifulRAGApp().run()