import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json

@st.cache_resource()
def get_data(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file)) as f:
                data.append(json.load(f))
    return pd.DataFrame(data)

@st.cache_resource()
def scater_chart(df):
    fig = px.scatter(
        df,
        x="model_name",
        y="model_performance",
        color="model_name",
        hover_data=[
                "model_name",
                "learning_rate",
                "batch_size",
                "reset",
                "box_near_goal",
                "box_close_goal",
                "box_move_reward",
                "box_goal_reward",
                "box_player_reward",
                "final_player_reward",
                "preform_step",
                "win_reward",
                "invalid_move_reward",
                "no_win_reward",
                "model_type",
                "model_performance",
        ]
    )
    fig.update_layout(
        title="Model Performance",
        # xaxis_title="Model Name",
        yaxis_title="Model Performance",
        # legend_title_text='Model Name',
    )
    return fig

def calculate_optimal_parameters(df , top_n=10):
    # return top 10 models
    return df.nlargest(top_n, 'model_performance')

def averaged_best_model_performance(df, top_n=15):
    df = df.nlargest(top_n, 'model_performance')
    df.drop(columns=['model_name','map_size', 'net_arch', 'net_arch_dqn', 'model_type' , 'policy' , 'folder_path_for_models'], inplace=True)
    # average every parameter
    return df.mean()

df = get_data('models-tone')

st.text('Data')
st.plotly_chart(scater_chart(df))

st.text('Show Best models parameters')
top_models = st.slider('Select the top N models', 1, 100, 10)
st.write(calculate_optimal_parameters(df , top_models))

st.text('Average Best Models Parameters')
top_n = st.slider('Select the top N models', 1, 100, 10 , key='top_n')
st.write(averaged_best_model_performance(df , top_n))