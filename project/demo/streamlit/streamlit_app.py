import streamlit as st

st.set_page_config(layout="wide")
st.header("1. Input game data")
upload, sample = st.columns(2, gap="large")

sample_games = {
    "Sample 1": {
        "file": "project/demo/data/5-9_13:39.txt",
        "description": "A game with 8 players, 5 rounds, and 9 cards.",
    }
}

with upload:
    st.write("Upload a text file with game proceedings:")
    st.write()
    text_file = st.file_uploader("Upload game proceedings", type=["txt"], label_visibility="collapsed")
with sample:
    st.write("Try one of our sample files:")
    use_sample = st.pills("Use sample data", sample_games.keys(), label_visibility="collapsed")
    if use_sample:
        st.write(sample_games[use_sample]["description"])
        text_file = None
if text_file or use_sample:
    st.write("File preview:")
    filelines = text_file.getvalue().decode("utf-8") if text_file else "".join(open(sample_games[use_sample]["file"], "r").readlines())
    st.code(filelines, language=None, height=200, wrap_lines=True, line_numbers=True)
    
    st.header("2. Run the agent")
    st.button("🏃 Run")
