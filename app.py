import os
import PyPDF2
import streamlit as st
from autogen import ConversableAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Please check your .env file.")
    st.stop()

# Streamlit UI Setup
st.set_page_config(page_title="AI Courtroom", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #D72638;'>⚖️ AI Courtroom Debate ⚖️</h1>
    <p style='text-align: center; font-size:18px; color: #5755FE;'>
    Upload a Legal Case PDF and Witness the Full Legal Debate!
    </p>
    <hr style="border: 1px solid #5755FE;">
    """,
    unsafe_allow_html=True,
)

# File Upload
uploaded_file = st.file_uploader("📄 Upload Legal Case PDF", type=["pdf"])

def extract_text_from_pdf(uploaded_file):
    """Extract text from an uploaded PDF file."""
    text = []
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def format_chat_history(chat_result):
    """Format the entire chat history with speaker labels."""
    if not hasattr(chat_result, 'chat_history'):
        return "No conversation history available."
    
    formatted_history = []
    for entry in chat_result.chat_history:
        speaker = entry.get('name', 'Unknown')
        content = entry.get('content', '')
        formatted_history.append(f"🎤 {speaker}: {content}")
    
    return "\n\n".join(formatted_history)

def run_courtroom_simulation(case_text):
    """Runs the AI courtroom debate and displays detailed conversation."""
    # Defense Attorney
    defense_agent = ConversableAgent(
        name="Defense_Attorney",
        system_message="""You are a meticulous defense attorney representing the accused. Your objectives:
        - Carefully analyze all evidence presented
        - Identify and highlight potential weaknesses in the prosecution's case
        - Protect the client's legal rights
        - Construct a strong, logical defense strategy
        - Challenge the reliability and completeness of evidence
        
        Approach the case with precision, empathy, and unwavering commitment to justice.""",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": api_key}]},
        human_input_mode="NEVER"
    )

    # Prosecution Attorney
    prosecution_agent = ConversableAgent(
        name="Prosecution_Attorney",
        system_message="""You are a determined prosecutor seeking to establish the truth. Your mission:
        - Compile and present compelling, irrefutable evidence
        - Construct a clear narrative of the alleged crime
        - Demonstrate the defendant's culpability beyond reasonable doubt
        - Address and refute potential defense arguments
        - Emphasize the legal and societal implications of the case
        
        Pursue justice with intellectual rigor and moral conviction.""",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": api_key}]},
        human_input_mode="NEVER"
    )

    # Judge (non-conversational)
    judge_agent = ConversableAgent(
        name="Presiding_Judge",
        system_message="""You are a seasoned, impartial judge with a responsibility to:
        - Carefully evaluate all presented evidence and arguments
        - Maintain judicial objectivity
        - Apply legal principles and precedents
        - Consider both factual and contextual elements of the case
        - Render a verdict that upholds justice and legal integrity
        
        Your decision must be comprehensive, well-reasoned, and grounded in law.""",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": api_key}]},
        human_input_mode="NEVER"
    )

    # Display Courtroom Debate
    st.subheader("🔍 Detailed Legal Proceedings")

    # Defense Initial Argument
    st.write("### 🛡️ Defense's Initial Argument")
    defense_initial = defense_agent.initiate_chat(
        prosecution_agent, 
        message=f"Case Overview: {case_text[:1000]}\n\nPresent a comprehensive initial defense strategy, carefully analyzing the available evidence and identifying key points that create reasonable doubt.",
        max_turns=1
    )
    st.markdown(format_chat_history(defense_initial), unsafe_allow_html=True)

    # Prosecution Counter-Argument
    st.write("### ⚔️ Prosecution's Counter-Argument")
    prosecution_response = prosecution_agent.initiate_chat(
        defense_agent, 
        message=f"Respond to the defense's arguments: {defense_initial.chat_history[-1]['content'][:500]}\n\nSystematically address each point, reinforce the strength of the prosecution's evidence, and demonstrate why the defendant's guilt is clear and convincing.",
        max_turns=1
    )
    st.markdown(format_chat_history(prosecution_response), unsafe_allow_html=True)

    # Defense Rebuttal
    st.write("### 🛡️ Defense's Rebuttal")
    defense_rebuttal = defense_agent.initiate_chat(
        prosecution_agent, 
        message=f"Provide a detailed rebuttal to the prosecution's arguments: {prosecution_response.chat_history[-1]['content'][:500]}\n\nHighlight inconsistencies, challenge the interpretation of evidence, and reinforce the principle of reasonable doubt.",
        max_turns=1
    )
    st.markdown(format_chat_history(defense_rebuttal), unsafe_allow_html=True)

    # Judge's Final Verdict
    st.write("### ⚖️ Judge's Final Verdict")
    final_verdict = judge_agent.initiate_chat(
        defense_agent,
        message=f"""Comprehensive Case Review:
        Defense Arguments: {defense_initial.chat_history[-1]['content'][:500]}
        Prosecution Arguments: {prosecution_response.chat_history[-1]['content'][:500]}
        Defense Rebuttal: {defense_rebuttal.chat_history[-1]['content'][:500]}

        Carefully analyze the entire case record. Consider the arguments from both the defense and prosecution. Evaluate the evidence presented, assess its credibility, and determine whether the prosecution has proven guilt beyond a reasonable doubt. Provide a clear, well-reasoned verdict that explains your judicial reasoning.""",
        max_turns=1
    )
    
    # Display the final verdict
    st.markdown(format_chat_history(final_verdict), unsafe_allow_html=True)

if __name__ == "__main__":
    if uploaded_file:
        with st.spinner("Extracting Case Details..."):
            case_text = extract_text_from_pdf(uploaded_file)

        if not case_text.strip():
            st.error("The extracted case text is empty. Please upload a valid PDF.")
        else:
            st.subheader("📜 Case Overview:")
            st.info(case_text[:800] + "...")  # Show first 800 characters

            if st.button("⚖️ Start Legal Debate"):
                run_courtroom_simulation(case_text)
