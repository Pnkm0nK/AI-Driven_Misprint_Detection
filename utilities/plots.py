import sklearn as skl
import json

from sklearn.metrics import roc_curve, auc
import utilities.config as config
import numpy as np

import pandas as pd
import plotly.express as px

def generate_roc_curves():
    trace_folder = config.RESULTS_DIR / "final_traces"
    models = ["HOG", "CNN", "CLIP", "PatchCore", "Dinomaly", "AnomalyDino"]
    df_list = []
    for model in models:
            with open(trace_folder / f"{model.lower()}_trace.json", "r") as f:
                trace = json.load(f)
            
            y_t = trace["y_true"]
            y_s = trace["y_scores"]
            
            fpr, tpr, _ = roc_curve(y_t, y_s)
            roc_auc = auc(fpr, tpr)
            
            model_df = pd.DataFrame({
                "FPR": fpr,
                "TPR": tpr,
                "Model": f"{model} (AUC = {roc_auc:.3f})"
            })
            df_list.append(model_df)

    # Combine all models into one DataFrame
    df_roc = pd.concat(df_list, ignore_index=True)

    import plotly.graph_objects as go

    fig = go.Figure()
    for model_name in models:
        model_df = df_roc[df_roc["Model"].str.contains(model_name)]
        legend_label = model_df["Model"].iloc[0]

        fig.add_trace(go.Scatter(
            x=model_df["FPR"],
            y=model_df["TPR"],
            mode="lines",
            name=legend_label,
            hoverinfo="x+y"
        ))

    fig.add_shape(
        type="line", line=dict(dash="dash", color="#718096", width=1.5),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_xaxes(
        title_text="False Positive Rate (FPR)",
        range=[0, 1.01],
        gridcolor="#E2E8F0",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True
    )
    fig.update_yaxes(
        title_text="True Positive Rate (TPR)",
        range=[0, 1.01],
        gridcolor="#E2E8F0",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        scaleanchor="x",
        scaleratio=1
    )

    fig.update_layout(
        width=700,
        height=700,
        plot_bgcolor="white",
        font=dict(family="Libertinus Serif", color="black", size=18),
        legend=dict(
            title_text="",
            bordercolor="#CBD5E0",
            borderwidth=1,
            x=0.4,
            y=0.05,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=24),
            bgcolor="rgba(255, 255, 255, 0.85)"
        ),
        margin=dict(l=60, r=40, t=40, b=60)
    )
    fig.write_image("roc_curves.pdf", engine="kaleido")

def plot_anomaly_map(anomaly_map: np.ndarray):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.imshow(anomaly_map, cmap="inferno")
    plt.colorbar(label="Anomaly Score")
    plt.title("Anomaly Map")
    plt.axis("off")
    return plt.gcf()



def generate_roc_curves_cv():
    np.random.seed(42)
    trace_folder = config.RESULTS_DIR / "crossval_traces" / "cnn"
    models = ["SVM", "GMM", "KNN"]
    traces = [trace_folder / f"best_{model.lower()}_trace.json" for model in models]
    folds = 5


    # --- 2. Process Traces and Compute Average ROC ---
    mean_fpr = np.linspace(0, 1, 100)
    plot_data = []

    for model_name, trace_file in zip(models, traces):
        fold_data = json.load(open(trace_file, "r"))
        tprs = []
        aucs = []
        
        for fold_idx in range(folds): 
            normal_key = f"fold_{fold_idx+1}_val_normal_scores"
            anomaly_key = f"fold_{fold_idx+1}_val_anomaly_scores"
            
            normal_scores = fold_data[normal_key]
            anomaly_scores = fold_data[anomaly_key]
            
            scores = np.concatenate([normal_scores, anomaly_scores])
            labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
            
            fpr, tpr, _ = roc_curve(labels, scores)
            
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
            aucs.append(auc(fpr, tpr))
            
        # Average TPR and AUC across all 5 folds for the current model
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        
        # Append to dataframe rows
        for f, t in zip(mean_fpr, mean_tpr):
            plot_data.append({
                "FPR": f,
                "TPR": t,
                "Model": f"{model_name} (Mean AUC = {mean_auc:.3f})"
            })

    df_plot = pd.DataFrame(plot_data)

    import plotly.graph_objects as go

    # --- 3. Plot with Plotly Graph Objects (for filled areas) ---
    fig = go.Figure()

    # Define your academic colors (RGBA format to control opacity/transparency)
    # Format: (Line Color, Fill Color with alpha/transparency)

    # Group the data by model to plot them one by one
    for model_name in models:
        model_df = df_plot[df_plot["Model"].str.contains(model_name)]
        
        # Extract the exact string used for the legend (which includes the AUC score)
        legend_label = model_df["Model"].iloc[0]
        
        
        fig.add_trace(go.Scatter(
            x=model_df["FPR"],
            y=model_df["TPR"],
            mode="lines",
            name=legend_label,
            fill="tozeroy",              # Fills the area down to the X-axis (y=0)
            hoverinfo="x+y"
        ))

    # Add reference line for random classifier
    fig.add_shape(
        type="line", line=dict(dash="dash", color="#718096", width=1.5),
        x0=0, x1=1, y0=0, y1=1
    )

    # Update axes to fix the negative values issue and enforce a strict 0 to 1 scale
    fig.update_xaxes(
        title_text="False Positive Rate (FPR)",
        range=[0, 1.01], 
        gridcolor="#E2E8F0", 
        showline=True, 
        linewidth=1, 
        linecolor="black", 
        mirror=True
    )
    fig.update_yaxes(
        title_text="True Positive Rate (TPR)",
        range=[0, 1.01], 
        gridcolor="#E2E8F0", 
        showline=True, 
        linewidth=1, 
        linecolor="black", 
        mirror=True,
        scaleanchor="x", 
        scaleratio=1
    )

    # Clean layout with an embedded legend in the bottom-right corner
    fig.update_layout(
        width=700,   
        height=700,
        plot_bgcolor="white", 
        font=dict(family="Libertinus Serif", color="black", size=18),
        legend=dict(
            title_text="",
            bordercolor="#CBD5E0",
            borderwidth=1,
            x=0.3,       
            y=0.05,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=24),
            bgcolor="rgba(255, 255, 255, 0.85)" 
        ),
        margin=dict(l=60, r=40, t=40, b=60)
    )

    fig.show()

def generate_ocr_violin_plot():
    # Load data
    tess_results = json.load(open(config.RESULTS_DIR / "ocr_trials_tesseract" / "results.json", "r"))
    paddle_results = json.load(open(config.RESULTS_DIR / "ocr_trials_paddleocr" / "results.json", "r"))
    easyocr_results = json.load(open(config.RESULTS_DIR / "ocr_trials_easyocr" / "results.json", "r"))
    
    t_scores = tess_results["cer_array"]
    p_scores = paddle_results["cer_array"]
    e_scores = easyocr_results["cer_array"]

    # Create long-format DataFrame
    df = pd.DataFrame({
        "CER": t_scores + p_scores + e_scores,
        "Engine": ["Tesseract"] * len(t_scores) + ["PaddleOCR"] * len(p_scores) + ["EasyOCR"] * len(e_scores)
    })

    fig = px.violin(
        df, 
        x="Engine", 
        y="CER", 
        color="Engine",
        box=True, 
        points="all", 
    )
    fig.update_layout(
    font=dict(family="Libertinus Serif", color="black",size=24
    )
)
    
    fig.show()
    return fig

def generate_roc_plot(fpr, tpr):
    roc_auc = skl.metrics.auc(fpr, tpr)
    fig = px.area(
        x=fpr,
        y=tpr,
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
    )
    fig.update_layout(
        font=dict(family="Libertinus Serif", color="black",size=24)
    )
    return fig

def generate_pr_plot(y, scores):
    precision, recall, thresholds = skl.metrics.precision_recall_curve(y, scores)
    pr_auc = skl.metrics.auc(recall, precision)
    fig = px.area(
        x=recall,
        y=precision,
        color_discrete_sequence=["#fc5a28"],
        
        labels={"x": "Recall", "y": "Precision"},
    )
    fig.update_layout(
        font=dict(family="Libertinus Serif", color="black",size=24)
    )
    return fig

if __name__ == "__main__":
    # Example usage
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    fig = generate_pr_plot(y_true, scores)
    fig.show()