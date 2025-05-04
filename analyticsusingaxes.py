import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import os
import tempfile
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate API key
if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set it in your .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')


def clean_data(df):
    """Clean the dataframe by handling missing values and removing duplicates"""
    df = df.drop_duplicates()
    df = df.ffill().bfill()
    return df

def get_recommended_chart_types(data, x_axis, y_axis, current_chart_type):
    """Suggest alternative chart types based on data characteristics"""
    recommendations = []
    x_numeric = pd.api.types.is_numeric_dtype(data[x_axis])
    y_numeric = pd.api.types.is_numeric_dtype(data[y_axis])
    
    base_recommendations = {
        ("Line Chart", "Trends over time/ordered categories"),
        ("Bar Chart", "Compare categories"),
        ("Scatter Plot", "Relationships between numeric variables"),
        ("Histogram", "Distribution of numeric data"),
        ("Box Plot", "Distribution and outliers"),
        ("Pie Chart", "Composition (max 10 categories)")
    }
    
    if current_chart_type == "Line Chart":
        if not x_numeric:
            recommendations.append(("Bar Chart", "for categorical X-axis"))
        if x_numeric and y_numeric:
            recommendations.append(("Scatter Plot", "for numeric-numeric relationship"))
    
    elif current_chart_type == "Bar Chart":
        if x_numeric and y_numeric:
            recommendations.append(("Scatter Plot", "better for numeric-numeric"))
        if len(data[x_axis].unique()) > 20:
            recommendations.append(("Histogram", "for high-cardinality data"))
    
    elif current_chart_type == "Pie Chart":
        if len(data[x_axis].unique()) > 10:
            recommendations.append(("Bar Chart", "better for many categories"))
    
    # Add base recommendations if not already present
    for chart, desc in base_recommendations:
        if chart != current_chart_type and chart not in [r[0] for r in recommendations]:
            if (chart == "Scatter Plot" and x_numeric and y_numeric) or \
               (chart == "Bar Chart" and not x_numeric and y_numeric) or \
               (chart not in ["Scatter Plot", "Bar Chart"]):
                recommendations.append((chart, desc))
    
    return recommendations

def validate_chart_selection(data, x_axis, y_axis, chart_type):
    """Validate if the selected axes are appropriate for the chart type"""
    errors = []
    recommendations = []
    x_numeric = pd.api.types.is_numeric_dtype(data[x_axis])
    y_numeric = pd.api.types.is_numeric_dtype(data[y_axis])
    
    validation_rules = {
        "Line Chart": [
            (not x_numeric, "Line charts typically need ordered/numeric X-axis"),
            (y_numeric and not x_numeric, "For trends, put time/ordered data on X-axis")
        ],
        "Bar Chart": [
            (len(data[x_axis].unique()) > 20, "Too many categories for clear bars")
        ],
        "Scatter Plot": [
            (not (x_numeric and y_numeric), "Scatter plots need both axes to be numeric")
        ],
        "Histogram": [
            (not x_numeric, "Histograms require numeric data")
        ],
        "Box Plot": [
            (not y_numeric, "Box plots require numeric Y-axis")
        ],
        "Pie Chart": [
            (len(data[x_axis].unique()) > 10, "Too many slices for clear pie chart"),
            (x_numeric, "Pie charts typically use categorical data")
        ]
    }
    
    for condition, error_msg in validation_rules.get(chart_type, []):
        if condition:
            errors.append(error_msg)
    
    if errors:
        recommendations = get_recommended_chart_types(data, x_axis, y_axis, chart_type)
    
    return errors, recommendations

def generate_plotly_chart(data, x_axis, y_axis, chart_type, color=None):
    """Generate interactive Plotly chart based on selections"""
    try:
        if chart_type == "Line Chart":
            fig = px.line(data, x=x_axis, y=y_axis, color=color)
        elif chart_type == "Bar Chart":
            fig = px.bar(data, x=x_axis, y=y_axis, color=color)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(data, x=x_axis, y=y_axis, color=color)
        elif chart_type == "Histogram":
            fig = px.histogram(data, x=x_axis)
        elif chart_type == "Box Plot":
            fig = px.box(data, y=y_axis)
        elif chart_type == "Pie Chart":
            fig = px.pie(data, names=x_axis, values=y_axis)
        else:
            return None
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        return fig
    except Exception as e:
        st.error(f"Error generating {chart_type}: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Advanced Data Visualizer", layout="wide")
    st.title("📊 Advanced Data Visualizer with AI Recommendations")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=['csv', 'xlsx'])
    
    data = None
    if uploaded_file is not None:
        try:
            # Read the file based on extension
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Clean and edit data
            data = clean_data(data)
            st.subheader("Data Preview (Editable)")
            edited_data = st.data_editor(data, num_rows="dynamic", height=300)
            
            if st.button("Save Data for Analysis", type="primary"):
                data = clean_data(edited_data)
                st.session_state['analysis_data'] = data
                st.success("Data saved for analysis!")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    # Analysis section
    if 'analysis_data' in st.session_state:
        data = st.session_state['analysis_data']
        
        st.subheader("Chart Configuration")
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            x_axis = st.selectbox("X-axis column", data.columns)
        with col2:
            y_axis = st.selectbox("Y-axis column", data.columns)
        
        with col3:
            color = st.selectbox(
                "Color by (optional)", 
                ["None"] + [col for col in data.columns if col not in [x_axis, y_axis]]
            )
        
        chart_type = st.selectbox(
            "Select Chart Type",
            options=[
                ("Line Chart", "Trends over time/ordered categories"),
                ("Bar Chart", "Compare categories"),
                ("Scatter Plot", "Relationships between numeric variables"),
                ("Histogram", "Distribution of numeric data"),
                ("Box Plot", "Distribution and outliers"),
                ("Pie Chart", "Composition (max 10 categories)")
            ],
            format_func=lambda x: f"{x[0]} - {x[1]}",
            index=0
        )[0]  # Get just the chart type name
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner("Creating visualization..."):
                errors, recommendations = validate_chart_selection(data, x_axis, y_axis, chart_type)
                
                if errors:
                    for error in errors:
                        st.error(error)
                    
                    if recommendations:
                        st.warning("Recommended alternatives:")
                        cols = st.columns(len(recommendations))
                        for idx, (rec, desc) in enumerate(recommendations):
                            if cols[idx].button(f"{rec}: {desc}"):
                                if "Original Line Chart" in rec:
                                    st.session_state['x_axis'], st.session_state['y_axis'] = y_axis, x_axis
                                else:
                                    st.session_state['chart_type'] = rec
                                st.rerun()
                else:
                    color_col = None if color == "None" else color
                    fig = generate_plotly_chart(data, x_axis, y_axis, chart_type, color_col)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                                    try:
                                        fig.write_image(tmpfile.name, engine="kaleido")
                                        st.download_button(
                                            label="Download as PNG",
                                            data=open(tmpfile.name, "rb"),
                                            file_name=f"{chart_type}_{x_axis}_vs_{y_axis}.png",
                                            mime="image/png"
                                        )
                                    except Exception as e:
                                        st.warning(f"Couldn't export PNG: {str(e)}")
                                        st.info("Please install kaleido: pip install -U kaleido")
                                os.unlink(tmpfile.name)
                            except Exception as e:
                                st.error(f"PNG export failed: {str(e)}")
                        
                        with col2:
                            try:
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
                                    fig.write_html(tmpfile.name)
                                    st.download_button(
                                        label="Download as Interactive HTML",
                                        data=open(tmpfile.name, "rb"),
                                        file_name=f"{chart_type}_{x_axis}_vs_{y_axis}.html",
                                        mime="text/html"
                                    )
                                os.unlink(tmpfile.name)
                            except Exception as e:
                                st.error(f"HTML export failed: {str(e)}")

if __name__ == "__main__":
    main()
