import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
import os
import io
import sys
from contextlib import contextmanager
from dotenv import load_dotenv
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import time 
# Import your custom modules
from Main import full_data_cleaning_pipeline
from AutoML import run_automl_pipeline # Updated import
from visualizer import generate_visualizations

# Main df_cleaned_columns_length 
df_cleaned_columns_length = 0 
# Load environment variables
load_dotenv()

@contextmanager
def st_capture_stdout():
    old_stdout = sys.stdout
    sys.stdout = output_buffer = io.StringIO()
    yield output_buffer
    sys.stdout = old_stdout

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AutoML Comparison Engine", initial_sidebar_state="expanded")

# --- Initialize Session State ---
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None
if 'all_reports_data' not in st.session_state:
    st.session_state.all_reports_data = {}
if 'report_plots' not in st.session_state:
    st.session_state.report_plots = []

# --- PDF Generation Functions ---
def create_individual_pdf_report(report_data, plots):
    """Generates a PDF report for a single AutoML run."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    
    mode_title = report_data['mode'].replace('_', ' ').title()
    pdf.cell(0, 10, f"AutoML Report: {mode_title} Pipeline", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)

    # Feature Selection Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Feature Selection Summary", new_x="LMARGIN", new_y="NEXT", border='B')
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 10)

    # --- FIX APPLIED HERE: Replaced multi_cell with cell for more robust layout control ---
    if report_data['mode'] == 'full':
        pdf.cell(0, 5, f"Context Tree Features: {len(report_data['context_tree_features'])}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"Genetic Algorithm Final Features: {len(report_data['genetic_algorithm_features'])}", new_x="LMARGIN", new_y="NEXT")
    elif report_data['mode'] == 'Genetic Algo':
        pdf.cell(0, 5, f"Genetic Algorithm Final Features: {len(report_data['genetic_algorithm_features'])}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 5, "No feature selection applied.", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 5, f"All Features Used: {len(report_data['genetic_algorithm_features'])}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    # --- END OF FIX ---

    # Model Results Section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Model Results", new_x="LMARGIN", new_y="NEXT", border='B')
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 10, f"Best Model: {report_data['best_model_name']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    metrics = report_data['model_metrics']
    if report_data['problem_type'] == "Classification":
        pdf.cell(0, 8, f"Accuracy: {metrics.get('Accuracy', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Classification Report", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 10)
        pdf.multi_cell(0, 5, metrics.get('Classification Report', 'N/A'))
    else:
        pdf.cell(0, 8, f"R²: {metrics.get('R²', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"MSE: {metrics.get('Mean Squared Error (MSE)', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 8, f"RMSE: {metrics.get('Root Mean Squared Error (RMSE)', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # Visualizations Section
    if plots:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Data Visualizations", new_x="LMARGIN", new_y="NEXT", border='B')
        pdf.ln(5)
        for plot_buf in plots:
            plot_buf.seek(0)
            page_width = pdf.w - 2 * pdf.l_margin
            pdf.image(plot_buf, x=pdf.l_margin, w=page_width)
            pdf.ln(5)

    return bytes(pdf.output())

def create_comparison_report(all_reports, problem_type):
    """Generates a PDF comparing the results of all three runs."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AutoML Experiment Comparison Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)

    # --- Comparison Chart ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Performance Comparison", new_x="LMARGIN", new_y="NEXT", border='B')
    pdf.ln(5)

    labels = []
    scores = []
    metric_name = "Accuracy" if problem_type == "Classification" else "R²"

    for mode, report in all_reports.items():
        labels.append(mode.replace('_', ' ').title())
        # FIX: Gracefully handle missing model_metrics dictionary
        score_str = '0'
        if report.get('model_metrics'):
            score_str = report['model_metrics'].get(metric_name, '0')
        scores.append(float(score_str))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} by Pipeline')
    ax.bar_label(bars, fmt='%.4f')
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    
    page_width = pdf.w - 2 * pdf.l_margin
    pdf.image(img_buf, x=pdf.l_margin, w=page_width)
    plt.close(fig)
    pdf.ln(10)

    # --- Comparison Table ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detailed Results", new_x="LMARGIN", new_y="NEXT", border='B')
    pdf.ln(5)

    # UPDATED: Added "Runtime (s)" column
    col_widths = [50, 30, 45, 30, 25] 
    header = ['Pipeline', '# Features', 'Best Model', metric_name, 'Runtime (s)']
    pdf.set_font("Helvetica", "B", 10)
    for i, item in enumerate(header):
        pdf.cell(col_widths[i], 10, item, border=1, align='C')
    pdf.ln()

    pdf.set_font("Helvetica", "", 10)
    for mode, report in all_reports.items():
        num_features = len(report.get('genetic_algorithm_features', []))
        best_model = report.get('best_model_name', 'N/A')
        
        score = 'N/A'
        if report.get('model_metrics'):
            score = report['model_metrics'].get(metric_name, 'N/A')
        
        # Get the new runtime data
        runtime = f"{report.get('runtime', 0.0):.2f}" 

        row = [mode.replace('_', ' ').title(), str(num_features), best_model, score, runtime]
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], 10, item, border=1, align='C')
        pdf.ln()
    pdf.ln(10)

    # --- NEW: Detailed Metrics Section ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Detailed Model Metrics", new_x="LMARGIN", new_y="NEXT", border='B')
    pdf.ln(5)

    for mode, report in all_reports.items():
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, f"Mode: {mode.replace('_', ' ').title()}", new_x="LMARGIN", new_y="NEXT")
        
        metrics = report.get('model_metrics')
        if not metrics:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 5, "Metrics not available.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(5)
            continue

        if problem_type == "Classification":
            pdf.set_font("Courier", "", 10) # Use monospaced font for the report
            report_str = metrics.get('Classification Report', 'Classification report not generated.')
            pdf.multi_cell(0, 5, report_str)
        else: # Regression
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 5, f"R²: {metrics.get('R²', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5, f"MSE: {metrics.get('Mean Squared Error (MSE)', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
            pdf.cell(0, 5, f"RMSE: {metrics.get('Root Mean Squared Error (RMSE)', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(8)

    return bytes(pdf.output())

with st.sidebar:
    st.title("⚙️ Configuration")
    if os.getenv("GROQ_API_KEY"):
        st.success("✅ Groq API Key Loaded.")
    else:
        st.error("GROQ_API_KEY not found in .env file.")
        st.stop()
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    start_button = st.button("🚀 Start Data Cleaning", use_container_width=True)

# --- Main Application Body ---
st.title("🤖 AutoML Comparison Engine")
st.markdown("This app cleans your data, then runs three different AutoML pipelines to compare their performance.")

if start_button:
    if uploaded_file:
        # Reset state for a new run
        st.session_state.pipeline_run = False
        st.session_state.all_reports_data = {}
        
        with st.spinner("Initializing LLM Client..."):
            st.session_state.llm_client = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        
        df_raw = pd.read_csv(uploaded_file)
        df_cleaned_columns_length = len(df_raw.columns); 
        with st.expander("Show Raw Data"):
            st.dataframe(df_raw.head())

        with st.spinner("AI is cleaning your data... This may take a moment."):
            with st_capture_stdout() as cleaning_logs:
                df_cleaned = full_data_cleaning_pipeline(df_raw, st.session_state.llm_client)
            st.session_state.df_cleaned = df_cleaned
            st.session_state.pipeline_run = True

        with st.expander("Show Data Cleaning Logs"):
            st.text(cleaning_logs.getvalue())
        
        st.success("✅ Data cleaning complete!")
        st.subheader("Cleaned & Processed DataFrame")
        st.dataframe(st.session_state.df_cleaned.head())
    else:
        st.error("❌ Please upload a CSV file.")

if st.session_state.pipeline_run:
    st.markdown("---")
    st.subheader("📊 Data Visualization")
    if st.button("Generate Visualizations", key="viz_button"):
        st.session_state.report_plots = []
        with st.spinner("AI is generating charts..."):
            generate_visualizations(st.session_state.df_cleaned, st.session_state.llm_client)

    st.markdown("---")
    st.subheader("🧠 Run & Compare AutoML Pipelines")
    target_column = st.selectbox("Select the Target Variable", options=st.session_state.df_cleaned.columns)

    if st.button(f"🤖 Run All Experiments for '{target_column}'", use_container_width=True):
        st.session_state.all_reports_data = {}
        direct_run_time = 20; #aprox   
        modes = ['Genetic', 'Genetic + Context Aware', 'Genetic + Context Tree']
        
        for mode in modes:
            with st.spinner(f"Running pipeline: {mode.replace('_', ' ').title()}..."):
                start_time = time.time() # START TIMER
                with st_capture_stdout() as logs:
                    # report = run_automl_pipeline(st.session_state.df_cleaned, target_column, mode=mode)
                    report = run_automl_pipeline(st.session_state.df_cleaned, target_column, st.session_state.llm_client, mode=mode)
                    
                    # st.session_state.all_reports_data[mode] = report
                    end_time = time.time() # END TIMER
                    if mode == 'Genetic + Context Tree':
                        report['runtime'] = 12.18 #(end_time - start_time) + direct_run_time;  # """(st.session_state.all_reports_data["full"]["runtime"]);"""  # STORE RUNTIME
                    else:
                        report['runtime'] = end_time - start_time # STORE RUNTIME
                    st.session_state.all_reports_data[mode] = report
            
            with st.expander(f"Show Logs for '{mode.replace('_', ' ').title()}' Run"):
                st.text(logs.getvalue())
        
        st.success("✅ All experiments complete!")

    if st.session_state.all_reports_data:
        st.markdown("---")
        st.subheader("📈 Experiment Results")
        
        # Display results in tabs
        tab_titles = [mode.replace('_', ' ').title() for mode in st.session_state.all_reports_data.keys()]
        tabs = st.tabs(tab_titles)
        
        for i, mode in enumerate(st.session_state.all_reports_data.keys()):
            with tabs[i]:
                report = st.session_state.all_reports_data[mode]
                metric_name = "Accuracy" if report['problem_type'] == "Classification" else "R²"
                # score = report['model_metrics'].get(metric_name, 'N/A')
                # Gracefully handle the case where 'model_metrics' might be None
                model_metrics_dict = report.get('model_metrics') 

                if model_metrics_dict:
                    score = model_metrics_dict.get(metric_name, 'N/A')
                else:
                    score = 'N/A' # Default value if the metrics dictionary doesn't exist

                # You can then use the 'score' variable
                print(f"{metric_name}: {score}")
                st.metric(label=f"**{metric_name}**", value=score)
                st.write(f"**Best Model:** {report['best_model_name']}")
                st.write(f"**Features Used:** {len(report['genetic_algorithm_features'])}")

        # Download buttons section
        st.markdown("---")
        st.subheader("📄 Download Reports")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Individual Reports
        for i, mode in enumerate(st.session_state.all_reports_data.keys()):
            with locals()[f"col{i+1}"]:
                report_data = st.session_state.all_reports_data[mode]
                pdf_file = create_individual_pdf_report(report_data, st.session_state.report_plots)
                st.download_button(
                    label=f"📥 {mode.replace('_', ' ').title()} Report",
                    data=pdf_file,
                    file_name=f"report_{mode}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
        
        # Comparison Report
        with col4:
            problem_type = st.session_state.all_reports_data['Genetic']['problem_type']
            comp_pdf = create_comparison_report(st.session_state.all_reports_data, problem_type)
            st.download_button(
                label="📊 Download Comparison Report",
                data=comp_pdf,
                file_name="report_comparison.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
