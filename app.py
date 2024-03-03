import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json

def get_data(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file)) as f:
                data.append(json.load(f))
    return pd.DataFrame(data)
    
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
                "max_invalid_move_reset",
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

def create_corr_matrix(df):
    corr = df.corr()["model_performance"].sort_values(ascending=False).to_frame()
    fig = px.imshow(corr)
    fig.update_layout(
        title="Correlation Matrix with Model Performance",
        coloraxis_colorbar=dict(title="Correlation"),
    )
    return fig


# df = get_data('models-tone')
# new_df = df[[
#     "learning_rate",
#     "batch_size",
#     "reset",
#     "box_near_goal",
#     "box_close_goal",
#     "box_move_reward",
#     "box_goal_reward",
#     "box_player_reward",
#     "final_player_reward",
#     "preform_step",
#     "win_reward",
#     "invalid_move_reward",
#     "max_invalid_move_reset",
#     "model_performance",
# ]]


# st.plotly_chart(scater_chart(df))
# st.plotly_chart(create_corr_matrix(new_df))
# st.write(calculate_optimal_parameters(df))