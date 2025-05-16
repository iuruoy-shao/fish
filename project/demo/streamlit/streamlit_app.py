import streamlit as st
import sys
sys.path.append(st.secrets["PYTHONPATH"])

from project.verify_data import SimulatedFishGame
from project.agent import QLearningAgent
import copy
import numpy as np
import huggingface_hub
import torch

st.set_page_config(layout="wide")
st.header("1. Input game data")
upload, sample = st.columns(2, gap="large")

@st.cache_resource
def load_model():
    torch.classes.__path__ = []
    model_path = huggingface_hub.hf_hub_download(repo_id="iuruoy-shao/fish", filename="model.pth")
    model = QLearningAgent()
    model.load_model(model_path)
    return model

# Load the model when the app starts
model = load_model()
model.epsilon = 0

def get_players(first_line):
    return [player.split(":")[0] for player in first_line.split()]

def get_print_hands(hands):
    print(hands)
    return {card:{p:round(float(v),4) for p,v in hands[card].items()} for card in hands}

def run_model(lines, player, call):
    game = SimulatedFishGame(8)
    game.help_threshold = 0
    game.datarows = lines
    game.init_hands = {player.split(":")[0]:set(player.split(":")[1][1:-1].split(',')) 
                            for player in game.datarows[0].split()}
    game.hands = [game.init_hands]
    game.players = list(game.init_hands.keys())
    game.verify()
    game.rotate(player)
    game_state = game.to_state()
    state = model.tensor(np.stack([game.get_state(i, game_state) for i in range(len(game.hands))]))
    mask = model.action_masks(*model.unpack_memory([game.mask_dep(len(game.hands)-1, player)]).values())
    pred_hands, action = model.act(model.condense_state(state), mask)
    new_hands = copy.deepcopy(game.hands[-1])
    turn = game.datarows[-2][3:5]
    
    if call or (action['call'][0] > action['call'][1] and not turn == player):
        act = game.handle_call(action, new_hands, player)
    else:
        act = game.handle_ask(action, new_hands, player, help_call=False)
    return game.decode_pred_hands(pred_hands.cpu().detach().numpy()), act

sample_games = {
    "Sample 1": {
        "file": "project/demo/data/5-9_13:39.txt",
        "description": "Asking for a card.",
        "player_id": "P6",
        "call": False
    },
    "Sample 2": {
        "file": "project/demo/data/12-4_11:11.txt",
        "description": "Asking for a card.",
        "player_id": "P1",
        "call": False
    },
    "Sample 3": {
        "file": "project/demo/data/1-15_11:15.txt",
        "description": "Attempting to call set.",
        "player_id": "P7",
        "call": True
    }
}

with upload:
    st.write("Upload a text file with game proceedings:")
    st.write()
    text_file = st.file_uploader("Upload game proceedings", type=["txt"], label_visibility="collapsed")
    if text_file:
        try:
            players = get_players(text_file.getvalue().decode("utf-8").splitlines()[0])
            player = st.selectbox("Select player to simulate", players)
        except Exception as e:
            st.error("Invalid file format.")
with sample:
    st.write("Try one of our sample files:")
    use_sample = st.pills("Use sample data", sample_games.keys(), label_visibility="collapsed")
    if use_sample:
        st.write("Player ID:", sample_games[use_sample]["player_id"])
        st.write(sample_games[use_sample]["description"])
        text_file = None
if text_file or use_sample:
    st.write("File preview:")
    filelines = text_file.getvalue().decode("utf-8") if text_file else "".join(open(sample_games[use_sample]["file"], "r").readlines())
    st.code(filelines, language=None, height=200, wrap_lines=True, line_numbers=True)
    
    st.header("2. Run the agent")
    if st.button("🏃 Run"):
        if text_file:
            lines = text_file.getvalue().decode("utf-8").splitlines()
            call = False
        else:
            lines = open(sample_games[use_sample]["file"], "r").readlines()
            player = sample_games[use_sample]["player_id"]
            call = sample_games[use_sample]["call"]
        pred_hands, act = run_model(lines, player, call)
        
        st.write(f"**Chosen Action:** `{act}`")
        st.write("**Predicted Hands:**", get_print_hands(pred_hands))