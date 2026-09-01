import streamlit as st
import whisper
import tempfile
import os
from openai import OpenAI

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Voice Notes → Action Items",
    page_icon="🎙️"
)

st.title("🎙️ Voice Notes → Action Items")

st.write(
    "Upload a voice note and automatically generate a transcript, "
    "summary, and action items."
)

# -----------------------------
# Load Whisper
# -----------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


model = load_whisper_model()

# -----------------------------
# Upload Audio
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your audio file",
    type=["mp3", "wav", "m4a", "mp4"]
)

# -----------------------------
# Process Audio
# -----------------------------
if uploaded_file:
    st.audio(uploaded_file)

    if st.button("🚀 Process Voice Note"):

        # Save audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())
            audio_path = temp_file.name

        try:

            # -----------------------------
            # Speech to Text
            # -----------------------------
            with st.spinner("🎙️ Converting speech to text..."):

                result = model.transcribe(audio_path)
                transcript = result["text"].strip()

            st.success("✅ Transcription completed!")

            # -----------------------------
            # Display Transcript
            # -----------------------------
            st.subheader("📝 Transcript")
            st.write(transcript)

            # -----------------------------
            # AI Analysis
            # -----------------------------
            with st.spinner("🤖 Generating summary and action items..."):

                prompt = f"""
You are an assistant that analyzes voice notes.

Analyze the following transcript:

{transcript}

Return your answer using EXACTLY this structure:

SUMMARY:

Write a short 1-2 sentence summary.

ACTION ITEMS:

1. First action item

2. Second action item

3. Third action item

DEADLINES:

Mention any deadlines found in the transcript.
If there is no deadline, write "No specific deadline mentioned."
"""

                client = OpenAI(
                    api_key=st.secrets["OPENAI_API_KEY"]
                )

                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=prompt
                )

                ai_result = response.output_text

            # -----------------------------
            # Display AI Result
            # -----------------------------
            st.subheader("🤖 AI Analysis")
            st.write(ai_result)

        except Exception as e:

            st.error(f"Something went wrong: {e}")

        finally:

            # Delete temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)