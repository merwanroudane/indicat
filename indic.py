import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import zscore, bartlett, chi2
from scipy.linalg import det
import io
import base64
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Economic & Social Indicator Builder — by Dr Merwan Roudane",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c5282;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        background-color: #f7fafc;
        border-left: 4px solid #4299e1;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .test-result {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .test-pass {
        background-color: #f0fff4;
        border-left-color: #38a169;
        color: #2f855a;
    }
    .test-fail {
        background-color: #fff5f5;
        border-left-color: #e53e3e;
        color: #c53030;
    }
    .test-warning {
        background-color: #fffbeb;
        border-left-color: #d69e2e;
        color: #b7791f;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f7fafc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'indicator_results' not in st.session_state:
    st.session_state.indicator_results = {}
if 'pre_tests_results' not in st.session_state:
    st.session_state.pre_tests_results = {}


# Advanced Helper Functions
@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload Excel or CSV files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def kaiser_meyer_olkin(data):
    """Calculate Kaiser-Meyer-Olkin (KMO) test"""
    corr_matrix = np.corrcoef(data.T)
    inv_corr_matrix = np.linalg.pinv(corr_matrix)

    # Partial correlations
    partial_corr = np.zeros_like(corr_matrix)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            if i != j:
                partial_corr[i, j] = -inv_corr_matrix[i, j] / np.sqrt(inv_corr_matrix[i, i] * inv_corr_matrix[j, j])

    # KMO calculation
    sum_sq_corr = np.sum(corr_matrix ** 2) - np.sum(np.diag(corr_matrix) ** 2)
    sum_sq_partial = np.sum(partial_corr ** 2)

    kmo = sum_sq_corr / (sum_sq_corr + sum_sq_partial)

    # Individual KMO values
    individual_kmo = {}
    for i in range(len(corr_matrix)):
        sum_sq_corr_i = np.sum(corr_matrix[i] ** 2) - corr_matrix[i, i] ** 2
        sum_sq_partial_i = np.sum(partial_corr[i] ** 2)
        individual_kmo[i] = sum_sq_corr_i / (sum_sq_corr_i + sum_sq_partial_i)

    return kmo, individual_kmo


def bartlett_sphericity_test(data):
    """Perform Bartlett's test of sphericity"""
    n, p = data.shape
    corr_matrix = np.corrcoef(data.T)

    # Chi-square statistic
    chi_square = -((n - 1) - (2 * p + 5) / 6) * np.log(det(corr_matrix))

    # Degrees of freedom
    df = p * (p - 1) / 2

    # P-value
    p_value = 1 - chi2.cdf(chi_square, df)

    return chi_square, p_value, df


def adequacy_tests(data, variable_names):
    """Perform comprehensive adequacy tests"""
    results = {}

    # KMO Test
    overall_kmo, individual_kmo = kaiser_meyer_olkin(data)
    results['kmo'] = {
        'overall': overall_kmo,
        'individual': {variable_names[i]: individual_kmo[i] for i in range(len(variable_names))}
    }

    # Bartlett's Test
    chi_square, p_value, df = bartlett_sphericity_test(data)
    results['bartlett'] = {
        'chi_square': chi_square,
        'p_value': p_value,
        'df': df
    }

    # Determinant of correlation matrix
    corr_matrix = np.corrcoef(data.T)
    results['determinant'] = det(corr_matrix)

    # Anti-image correlation matrix
    inv_corr = np.linalg.pinv(corr_matrix)
    anti_image = np.zeros_like(corr_matrix)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            anti_image[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
    np.fill_diagonal(anti_image, 1)
    results['anti_image'] = anti_image

    return results


def create_advanced_pca(data, method='varimax', n_components=None):
    """Advanced PCA with rotation options"""
    if n_components is None:
        n_components = min(data.shape[1], data.shape[0])

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Principal Component Analysis
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)

    # Rotation (simplified varimax)
    if method == 'varimax':
        loadings = pca.components_.T
        rotated_loadings = varimax_rotation(loadings[:, :min(n_components, 3)])
    else:
        rotated_loadings = pca.components_.T

    return pca, pca_result, scaler, rotated_loadings


def varimax_rotation(loadings, max_iter=1000, tolerance=1e-6):
    """Varimax rotation for factor loadings"""
    n_vars, n_factors = loadings.shape
    rotation_matrix = np.eye(n_factors)

    for _ in range(max_iter):
        old_rotation = rotation_matrix.copy()

        # Simplified varimax implementation
        loadings_rotated = loadings @ rotation_matrix

        # Update rotation matrix (simplified)
        u, s, vt = np.linalg.svd(loadings.T @ (
                loadings_rotated ** 3 - loadings_rotated @ np.diag(np.sum(loadings_rotated ** 2, axis=0)) / n_vars))
        rotation_matrix = u @ vt

        if np.allclose(old_rotation, rotation_matrix, atol=tolerance):
            break

    return loadings @ rotation_matrix


def factor_analysis_method(data, n_factors):
    """Factor Analysis implementation"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa_result = fa.fit_transform(scaled_data)

    return fa, fa_result, scaler


def create_composite_indicators(data, method, **kwargs):
    """Create composite indicators using various methods"""
    results = {}

    if method == "PCA":
        pca, pca_result, scaler, rotated_loadings = create_advanced_pca(
            data,
            method=kwargs.get('rotation', 'none'),
            n_components=kwargs.get('n_components', None)
        )

        indicator = None  # Initialize indicator
        # Different PCA-based indicators
        if kwargs.get('pca_method') == 'first_component':
            indicator = pca_result[:, 0]
        elif kwargs.get('pca_method') == 'weighted':
            weights = pca.explained_variance_ratio_
            indicator = np.average(pca_result, axis=1, weights=weights)
        elif kwargs.get('pca_method') == 'cumulative_80':
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_comp_80 = np.argmax(cumsum >= 0.8) + 1
            weights = pca.explained_variance_ratio_[:n_comp_80]
            indicator = np.average(pca_result[:, :n_comp_80], axis=1, weights=weights)
        elif kwargs.get('pca_method') == 'kaiser':
            # Kaiser criterion: keep components with eigenvalues > 1
            # For standardized data, eigenvalues correspond to explained_variance_
            n_kaiser = np.sum(pca.explained_variance_ > 1.0)
            if n_kaiser == 0:
                n_kaiser = 1  # Fallback to first component if none are > 1
            weights = pca.explained_variance_ratio_[:n_kaiser]
            indicator = np.average(pca_result[:, :n_kaiser], axis=1, weights=weights)

        results = {
            'indicator': indicator,
            'method': 'PCA',
            'pca_object': pca,
            'scaler': scaler,
            'explained_variance': pca.explained_variance_ratio_,
            'components': pca.components_,
            'rotated_loadings': rotated_loadings,
            'pca_scores': pca_result
        }

    elif method == "Factor_Analysis":
        fa, fa_result, scaler = factor_analysis_method(data, kwargs.get('n_factors', 2))
        indicator = np.mean(fa_result, axis=1)  # Simple average of factors

        results = {
            'indicator': indicator,
            'method': 'Factor Analysis',
            'fa_object': fa,
            'scaler': scaler,
            'factor_scores': fa_result,
            'loadings': fa.components_
        }

    elif method == "Distance_to_Reference":
        # Distance to reference point (e.g., best performer)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)

        reference_point = kwargs.get('reference', np.max(scaled_data, axis=0))
        distances = np.sqrt(np.sum((scaled_data - reference_point) ** 2, axis=1))
        indicator = -distances  # Negative so higher is better

        results = {
            'indicator': indicator,
            'method': 'Distance to Reference',
            'reference_point': reference_point,
            'scaler': scaler
        }

    elif method == "TOPSIS":
        # TOPSIS method
        indicator = topsis_method(data, kwargs.get('weights'), kwargs.get('criteria_types'))

        results = {
            'indicator': indicator,
            'method': 'TOPSIS',
            'weights': kwargs.get('weights'),
            'criteria_types': kwargs.get('criteria_types')
        }

    elif method == "DEA":
        # Data Envelopment Analysis (simplified)
        indicator = simplified_dea(data)

        results = {
            'indicator': indicator,
            'method': 'DEA'
        }

    return results


def topsis_method(data, weights=None, criteria_types=None):
    """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)"""
    if weights is None:
        weights = np.ones(data.shape[1]) / data.shape[1]
    if criteria_types is None:
        criteria_types = ['max'] * data.shape[1]  # Assume all criteria are benefit type

    # Normalize the decision matrix
    normalized_data = data / np.sqrt(np.sum(data ** 2, axis=0))

    # Weight the normalized matrix
    weighted_data = normalized_data * weights

    # Determine ideal and negative-ideal solutions
    ideal_solution = np.zeros(data.shape[1])
    negative_ideal = np.zeros(data.shape[1])

    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 'max':
            ideal_solution[i] = np.max(weighted_data[:, i])
            negative_ideal[i] = np.min(weighted_data[:, i])
        else:  # 'min'
            ideal_solution[i] = np.min(weighted_data[:, i])
            negative_ideal[i] = np.max(weighted_data[:, i])

    # Calculate distances
    distance_to_ideal = np.sqrt(np.sum((weighted_data - ideal_solution) ** 2, axis=1))
    distance_to_negative = np.sqrt(np.sum((weighted_data - negative_ideal) ** 2, axis=1))

    # Calculate TOPSIS score
    topsis_score = distance_to_negative / (distance_to_ideal + distance_to_negative)

    return topsis_score


def simplified_dea(data):
    """Simplified DEA (Data Envelopment Analysis)"""
    # This is a very simplified version - in practice, DEA requires linear programming
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Simple efficiency score based on distance from frontier
    max_values = np.max(scaled_data, axis=0)
    efficiency_scores = np.mean(scaled_data / max_values, axis=1)

    return efficiency_scores


def detect_advanced_outliers(df, methods=['IQR', 'Z-Score', 'Isolation_Forest', 'DBSCAN']):
    """Advanced outlier detection using multiple methods"""
    outliers_info = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for method in methods:
        outliers_info[method] = {}

        if method == 'IQR':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
                outliers_info[method][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'indices': outliers.tolist()
                }

        elif method == 'Z-Score':
            for col in numeric_cols:
                z_scores = np.abs(zscore(df[col].dropna()))
                outliers = df[z_scores > 3].index
                outliers_info[method][col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df)) * 100,
                    'indices': outliers.tolist()
                }

        elif method == 'Isolation_Forest':
            iso_forest = IsolationForest(contamination='auto',
                                         random_state=42)  # Changed contamination to auto for better general performance
            outlier_labels = iso_forest.fit_predict(df[numeric_cols].dropna())
            outlier_indices = df[numeric_cols].dropna().index[outlier_labels == -1]
            outliers_info[method]['all_variables'] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(df)) * 100,
                'indices': outlier_indices.tolist()
            }

        elif method == 'DBSCAN':
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(StandardScaler().fit_transform(df[numeric_cols].dropna()))
            outlier_indices = df[numeric_cols].dropna().index[cluster_labels == -1]
            outliers_info[method]['all_variables'] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(df)) * 100,
                'indices': outlier_indices.tolist()
            }

    return outliers_info


def create_advanced_visualizations(data, indicator, method_results):
    """Create advanced visualizations for indicator analysis"""
    figures = {}

    # 1. Biplot for PCA
    if method_results.get('method') == 'PCA':
        fig = create_pca_biplot(
            method_results['pca_scores'],
            method_results['components'],
            data.columns,
            method_results['explained_variance']
        )
        figures['biplot'] = fig

    # 2. Parallel coordinates plot
    fig = create_parallel_coordinates(data, indicator)
    figures['parallel_coordinates'] = fig

    # 3. Radar chart for top performers
    fig = create_radar_chart(data, indicator)
    figures['radar_chart'] = fig

    # 4. Sensitivity analysis heatmap
    fig = create_sensitivity_heatmap(data, indicator)
    figures['sensitivity_heatmap'] = fig

    return figures


def create_pca_biplot(scores, components, feature_names, explained_variance):
    """Create PCA biplot"""
    fig = go.Figure()

    # Add score points
    fig.add_trace(go.Scatter(
        x=scores[:, 0],
        y=scores[:, 1],
        mode='markers',
        marker=dict(size=8, opacity=0.6),
        name='Observations',
        text=[f'Obs {i + 1}' for i in range(len(scores))]
    ))

    # Add loading vectors
    loadings = components[:2].T
    for i, feature in enumerate(feature_names):
        fig.add_trace(go.Scatter(
            x=[0, loadings[i, 0] * 3],
            y=[0, loadings[i, 1] * 3],
            mode='lines+text',
            line=dict(color='red', width=2),
            text=['', feature],
            textposition="top center",
            showlegend=False,
            name=f'Loading {feature}'
        ))

    fig.update_layout(
        title='PCA Biplot',
        xaxis_title=f'PC1 ({explained_variance[0]:.1%} variance)',
        yaxis_title=f'PC2 ({explained_variance[1]:.1%} variance)',
        width=700,
        height=600
    )

    return fig


def create_parallel_coordinates(data, indicator):
    """Create parallel coordinates plot"""
    df_plot = data.copy()
    df_plot['Indicator'] = indicator
    df_plot['Indicator_Quartile'] = pd.qcut(indicator, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    fig = go.Figure(data=go.Parcoords(
        line=dict(color=indicator, colorscale='Viridis', showscale=True),
        dimensions=[dict(label=col, values=df_plot[col]) for col in data.columns]
    ))

    fig.update_layout(
        title='Parallel Coordinates Plot',
        width=800,
        height=500
    )

    return fig


def create_radar_chart(data, indicator):
    """Create radar chart for top performers"""
    # Get top 5 performers
    top_indices = np.argsort(indicator)[-5:]

    # Normalize data for radar chart
    normalized_data = (data - data.min()) / (data.max() - data.min())

    fig = go.Figure()

    for i, idx in enumerate(top_indices):
        fig.add_trace(go.Scatterpolar(
            r=normalized_data.iloc[idx].values,
            theta=data.columns,
            fill='toself',
            name=f'Rank {len(indicator) - i} (Obs {idx + 1})',
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title='Top 5 Performers - Radar Chart',
        width=600,
        height=600
    )

    return fig


def create_sensitivity_heatmap(data, indicator):
    """Create sensitivity analysis heatmap"""
    correlations = data.corrwith(pd.Series(indicator))

    # Create correlation matrix with indicator
    extended_data = data.copy()
    extended_data['Composite_Indicator'] = indicator
    corr_matrix = extended_data.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title='Correlation Matrix with Composite Indicator',
        width=600,
        height=600
    )

    return fig


# Main Application
def main():
    # Header
    st.markdown('<div class="main-header">🎯 Advanced Economic & Social Indicator Builder</div>',
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🧭 Navigation")
    tabs = [
        "📁 Data Upload",
        "🔍 Data Exploration",
        "🧹 Data Cleaning",
        "🧪 Pre-Tests & Adequacy",
        "📊 Indicator Construction",
        "📈 Advanced Analysis",
        "🔬 Validation & Robustness",
        "💾 Export & Documentation"
    ]

    selected_tab = st.sidebar.radio("Select Step:", tabs)

    # Progress tracking
    progress_dict = {
        "📁 Data Upload": 0,
        "🔍 Data Exploration": 15,
        "🧹 Data Cleaning": 30,
        "🧪 Pre-Tests & Adequacy": 45,
        "📊 Indicator Construction": 60,
        "📈 Advanced Analysis": 75,
        "🔬 Validation & Robustness": 85,
        "💾 Export & Documentation": 100
    }

    progress = progress_dict[selected_tab]
    st.sidebar.progress(progress / 100)
    st.sidebar.text(f"Progress: {progress}%")

    # Display current status
    if st.session_state.data is not None:
        st.sidebar.success("✅ Data Loaded")
    if st.session_state.processed_data is not None:
        st.sidebar.success("✅ Data Processed")
    if st.session_state.pre_tests_results:
        st.sidebar.success("✅ Pre-tests Completed")
    if st.session_state.indicator_results:
        st.sidebar.success("✅ Indicator Constructed")

    # Tab 1: Data Upload
    if selected_tab == "📁 Data Upload":
        st.markdown('<div class="section-header">📁 Data Upload & Initial Setup</div>',
                    unsafe_allow_html=True)

        # File upload section
        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Choose your data file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload Excel or CSV file containing your economic/social indicators"
            )

        with col2:
            st.info("📋 **Supported Formats:**\n- Excel (.xlsx, .xls)\n- CSV (.csv)")

        if uploaded_file is not None:
            with st.spinner('🔄 Loading and validating data...'):
                df = load_data(uploaded_file)

            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Data loaded successfully!")

                # Enhanced data summary
                st.subheader("📊 Data Summary")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📝 Total Rows", f"{df.shape[0]:,}")
                with col2:
                    st.metric("📊 Total Columns", f"{df.shape[1]:,}")
                with col3:
                    st.metric("🔢 Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
                with col4:
                    st.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")

                # Data quality indicators
                st.subheader("🎯 Data Quality Assessment")

                quality_metrics = {}
                quality_metrics['Completeness'] = (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                quality_metrics['Numeric_Ratio'] = (len(df.select_dtypes(include=[np.number]).columns) / df.shape[
                    1]) * 100

                # Detect duplicate rows
                duplicates = df.duplicated().sum()
                quality_metrics['Uniqueness'] = (1 - duplicates / len(df)) * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    completeness = quality_metrics['Completeness']
                    st.metric("📈 Completeness", f"{completeness:.1f}%",
                              delta=f"{'Good' if completeness > 90 else 'Needs attention'}")

                with col2:
                    numeric_ratio = quality_metrics['Numeric_Ratio']
                    st.metric("🔢 Numeric Ratio", f"{numeric_ratio:.1f}%",
                              delta=f"{'Good' if numeric_ratio > 70 else 'Check data types'}")

                with col3:
                    uniqueness = quality_metrics['Uniqueness']
                    st.metric("🎯 Uniqueness", f"{uniqueness:.1f}%",
                              delta=f"{duplicates} duplicates found" if duplicates > 0 else "No duplicates")

                # Enhanced data preview
                st.subheader("👀 Data Preview")

                preview_options = st.columns(3)
                with preview_options[0]:
                    show_head = st.checkbox("Show first 10 rows", value=True)
                with preview_options[1]:
                    show_tail = st.checkbox("Show last 5 rows")
                with preview_options[2]:
                    show_sample = st.checkbox("Show random sample")

                if show_head:
                    st.write("**First 10 rows:**")
                    st.dataframe(df.head(10), use_container_width=True)

                if show_tail:
                    st.write("**Last 5 rows:**")
                    st.dataframe(df.tail(5), use_container_width=True)

                if show_sample:
                    st.write("**Random sample (5 rows):**")
                    st.dataframe(df.sample(min(5, len(df))), use_container_width=True)

                # Enhanced data types analysis
                st.subheader("🔍 Data Types Analysis")

                dtype_analysis = pd.DataFrame({
                    'Column': df.columns,
                    'Data_Type': df.dtypes.astype(str),
                    'Non_Null_Count': df.count(),
                    'Null_Count': df.isnull().sum(),
                    'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
                    'Unique_Values': [df[col].nunique() for col in df.columns],
                    'Memory_Usage_KB': [df[col].memory_usage(deep=True) / 1024 for col in df.columns]
                })

                st.dataframe(dtype_analysis, use_container_width=True)

        else:
            st.info("👆 Please upload a data file to begin the analysis")

            # Enhanced sample data option
            st.subheader("🧪 Try with Sample Data")

            sample_options = st.columns(2)

            with sample_options[0]:
                if st.button("📊 Economic Development Dataset"):
                    np.random.seed(42)
                    countries = [f"Country_{chr(65 + i)}" for i in range(50)]
                    sample_data = pd.DataFrame({
                        'Country': countries,
                        'GDP_per_capita': np.random.lognormal(10, 0.5, 50),
                        'Life_Expectancy': np.random.normal(75, 8, 50),
                        'Education_Index': np.random.beta(8, 2, 50),
                        'Healthcare_Spending_GDP': np.random.normal(8, 3, 50),
                        'Unemployment_Rate': np.random.exponential(2, 50),
                        'Gini_Coefficient': np.random.normal(35, 10, 50),
                        'Infrastructure_Quality': np.random.normal(4, 1.5, 50),
                        'Innovation_Index': np.random.gamma(2, 20, 50),
                        'Environmental_Performance': np.random.normal(60, 15, 50)
                    })

                    # Add some correlations
                    sample_data['Life_Expectancy'] += sample_data['GDP_per_capita'] * 0.0001 + np.random.normal(0, 2,
                                                                                                                50)
                    sample_data['Education_Index'] = np.clip(sample_data['Education_Index'], 0, 1)

                    # Add missing values strategically
                    missing_indices = np.random.choice(50, 8, replace=False)
                    sample_data.loc[missing_indices[:4], 'Education_Index'] = np.nan
                    sample_data.loc[missing_indices[4:], 'Environmental_Performance'] = np.nan

                    st.session_state.data = sample_data
                    st.success("✅ Economic Development sample data loaded!")
                    st.rerun()

            with sample_options[1]:
                if st.button("🏛️ Social Progress Dataset"):
                    np.random.seed(123)
                    cities = [f"City_{i + 1}" for i in range(40)]
                    sample_data = pd.DataFrame({
                        'City': cities,
                        'Safety_Index': np.random.normal(70, 15, 40),
                        'Healthcare_Quality': np.random.normal(75, 12, 40),
                        'Education_Access': np.random.normal(80, 10, 40),
                        'Income_Level': np.random.lognormal(10.5, 0.3, 40),
                        'Social_Cohesion': np.random.normal(65, 18, 40),
                        'Cultural_Diversity': np.random.normal(60, 20, 40),
                        'Environmental_Quality': np.random.normal(55, 25, 40),
                        'Digital_Access': np.random.normal(75, 20, 40),
                        'Transportation_Quality': np.random.normal(65, 15, 40)
                    })

                    # Add some missing values
                    missing_indices = np.random.choice(40, 6, replace=False)
                    sample_data.loc[missing_indices[:3], 'Cultural_Diversity'] = np.nan
                    sample_data.loc[missing_indices[3:], 'Digital_Access'] = np.nan

                    st.session_state.data = sample_data
                    st.success("✅ Social Progress sample data loaded!")
                    st.rerun()

    # Tab 2: Data Exploration
    elif selected_tab == "🔍 Data Exploration":
        if st.session_state.data is None:
            st.warning("⚠️ Please upload data first!")
            return

        df = st.session_state.data
        st.markdown('<div class="section-header">🔍 Comprehensive Data Exploration</div>',
                    unsafe_allow_html=True)

        # Enhanced summary statistics
        st.subheader("📊 Advanced Statistical Summary")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:

            # Comprehensive statistics
            stats_df = df[numeric_cols].describe()

            # Add additional statistics
            additional_stats = pd.DataFrame({
                'skewness': df[numeric_cols].skew(),
                'kurtosis': df[numeric_cols].kurtosis(),
                'variance': df[numeric_cols].var(),
                'coeff_variation': df[numeric_cols].std() / df[numeric_cols].mean() * 100
            }).T

            comprehensive_stats = pd.concat([stats_df, additional_stats])
            st.dataframe(comprehensive_stats, use_container_width=True)

            # Distribution analysis
            st.subheader("📈 Distribution Analysis")

            dist_col1, dist_col2 = st.columns(2)

            with dist_col1:
                selected_var = st.selectbox("Select variable for detailed analysis:", numeric_cols)

            with dist_col2:
                show_outliers = st.checkbox("Highlight outliers", value=True)

            if selected_var:
                # Create distribution plot with statistics
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Histogram', 'Box Plot', 'Q-Q Plot', 'Statistics'],
                    specs=[[{'type': 'histogram'}, {'type': 'box'}],
                           [{'type': 'scatter'}, {'type': 'table'}]]
                )

                # Histogram
                fig.add_trace(
                    go.Histogram(x=df[selected_var], nbinsx=30, name='Distribution'),
                    row=1, col=1
                )

                # Box plot
                fig.add_trace(
                    go.Box(y=df[selected_var], name='Box Plot'),
                    row=1, col=2
                )

                # Q-Q plot
                from scipy.stats import probplot
                qq_data = probplot(df[selected_var].dropna(), dist="norm")
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q Plot'),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=qq_data[0][0], y=qq_data[1][1] + qq_data[1][0] * qq_data[0][0],
                               mode='lines', name='Normal line'),
                    row=2, col=1
                )

                # Statistics table
                var_stats = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max'],
                    'Value': [
                        df[selected_var].mean(),
                        df[selected_var].median(),
                        df[selected_var].std(),
                        df[selected_var].skew(),
                        df[selected_var].kurtosis(),
                        df[selected_var].min(),
                        df[selected_var].max()
                    ]
                })

                fig.add_trace(
                    go.Table(
                        header=dict(values=['Statistic', 'Value']),
                        cells=dict(values=[var_stats['Statistic'],
                                           [f"{val:.3f}" for val in var_stats['Value']]])
                    ),
                    row=2, col=2
                )

                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Advanced correlation analysis
        st.subheader("🔗 Advanced Correlation Analysis")

        if len(numeric_cols) > 1:

            corr_method = st.selectbox("Correlation method:", ["Pearson", "Spearman", "Kendall"])

            if corr_method == "Pearson":
                corr_matrix = df[numeric_cols].corr(method='pearson')
            elif corr_method == "Spearman":
                corr_matrix = df[numeric_cols].corr(method='spearman')
            else:
                corr_matrix = df[numeric_cols].corr(method='kendall')

            # Enhanced correlation heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))

            fig.update_layout(
                title=f'{corr_method} Correlation Matrix',
                width=700,
                height=700
            )
            st.plotly_chart(fig)

            # Correlation strength analysis
            st.subheader("💪 Correlation Strength Analysis")

            # Flatten correlation matrix and analyze
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable_1': corr_matrix.columns[i],
                        'Variable_2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j],
                        'Abs_Correlation': abs(corr_matrix.iloc[i, j]),
                        'Strength': 'Strong' if abs(corr_matrix.iloc[i, j]) > 0.7 else
                        'Moderate' if abs(corr_matrix.iloc[i, j]) > 0.3 else 'Weak'
                    })

            corr_df = pd.DataFrame(corr_pairs).sort_values('Abs_Correlation', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Strongest Correlations:**")
                st.dataframe(corr_df.head(10), use_container_width=True)

            with col2:
                strength_counts = corr_df['Strength'].value_counts()
                fig = px.pie(values=strength_counts.values, names=strength_counts.index,
                             title="Distribution of Correlation Strengths")
                st.plotly_chart(fig)

        # Advanced missing values analysis
        st.subheader("❌ Missing Values Pattern Analysis")

        if df.isnull().sum().sum() > 0:
            # Missing values heatmap
            missing_matrix = df.isnull()

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(missing_matrix, cbar=True, ax=ax, cmap='viridis')
            plt.title('Missing Values Pattern')
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Missing values statistics
            missing_stats = pd.DataFrame({
                'Column': df.columns,
                'Missing_Count': df.isnull().sum(),
                'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
                'Data_Type': df.dtypes
            }).sort_values('Missing_Percentage', ascending=False)

            missing_stats = missing_stats[missing_stats['Missing_Count'] > 0]

            if len(missing_stats) > 0:
                st.dataframe(missing_stats, use_container_width=True)

                # Missing values by patterns
                st.write("**Missing Value Patterns:**")
                missing_patterns = df.isnull().value_counts().head(10)
                st.write(missing_patterns)

        else:
            st.success("✅ No missing values found in the dataset!")

        # Multivariate analysis
        st.subheader("🔀 Multivariate Analysis")

        if len(numeric_cols) >= 3:

            analysis_type = st.selectbox(
                "Select multivariate analysis:",
                ["Scatter Plot Matrix", "3D Scatter Plot", "Principal Component Preview"]
            )

            if analysis_type == "Scatter Plot Matrix":
                selected_vars = st.multiselect(
                    "Select variables (max 5):",
                    numeric_cols,
                    default=list(numeric_cols[:4])
                )

                if len(selected_vars) >= 2:
                    fig = px.scatter_matrix(df, dimensions=selected_vars, height=600)
                    st.plotly_chart(fig, use_container_width=True)

            elif analysis_type == "3D Scatter Plot":
                col1, col2, col3 = st.columns(3)

                with col1:
                    x_var = st.selectbox("X-axis:", numeric_cols, index=0)
                with col2:
                    y_var = st.selectbox("Y-axis:", numeric_cols, index=1)
                with col3:
                    z_var = st.selectbox("Z-axis:", numeric_cols, index=2)

                fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var,
                                    title=f'3D Scatter: {x_var} vs {y_var} vs {z_var}')
                st.plotly_chart(fig)

            elif analysis_type == "Principal Component Preview":
                # Quick PCA preview
                # CORRECTED: Local imports were removed from here to fix the UnboundLocalError
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[numeric_cols].dropna())

                pca_preview = PCA()
                pca_preview.fit(scaled_data)

                # Explained variance plot
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f'PC{i + 1}' for i in range(len(pca_preview.explained_variance_ratio_))],
                    y=pca_preview.explained_variance_ratio_,
                    name='Individual'
                ))
                fig.add_trace(go.Scatter(
                    x=[f'PC{i + 1}' for i in range(len(pca_preview.explained_variance_ratio_))],
                    y=np.cumsum(pca_preview.explained_variance_ratio_),
                    mode='lines+markers',
                    name='Cumulative'
                ))

                fig.update_layout(
                    title='PCA Preview - Explained Variance',
                    xaxis_title='Principal Components',
                    yaxis_title='Explained Variance Ratio'
                )
                st.plotly_chart(fig)

                st.info(
                    f"💡 First 3 components explain {np.sum(pca_preview.explained_variance_ratio_[:3]):.1%} of total variance")

    # Tab 3: Data Cleaning
    elif selected_tab == "🧹 Data Cleaning":
        if st.session_state.data is None:
            st.warning("⚠️ Please upload data first!")
            return

        df = st.session_state.data.copy()
        st.markdown('<div class="section-header">🧹 Advanced Data Cleaning & Preprocessing</div>',
                    unsafe_allow_html=True)

        # Data cleaning overview
        st.subheader("📋 Cleaning Plan Overview")

        cleaning_steps = []
        if df.isnull().sum().sum() > 0:
            cleaning_steps.append("❌ Handle missing values")
        if df.duplicated().sum() > 0:
            cleaning_steps.append("🔄 Remove duplicate rows")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            cleaning_steps.extend([
                "🎯 Detect and treat outliers",
                "⚖️ Scale/normalize data",
                "📊 Transform distributions"
            ])

        if cleaning_steps:
            for step in cleaning_steps:
                st.write(f"• {step}")
        else:
            st.success("✅ Data appears to be clean!")

        # Missing values treatment (Enhanced)
        st.subheader("❌ Advanced Missing Values Treatment")

        if df.isnull().sum().sum() > 0:

            missing_analysis = pd.DataFrame({
                'Column': df.columns,
                'Missing_Count': df.isnull().sum(),
                'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            missing_analysis = missing_analysis[missing_analysis['Missing_Count'] > 0]

            st.dataframe(missing_analysis, use_container_width=True)

            # Missing value treatment strategy
            col1, col2 = st.columns(2)

            with col1:
                missing_strategy = st.selectbox(
                    "Global missing values strategy:",
                    ["Custom per column", "Drop rows with any missing", "Drop rows with >50% missing",
                     "Mean imputation (numeric)", "Median imputation (numeric)",
                     "Mode imputation (all)", "KNN imputation", "Iterative imputation"]
                )

            with col2:
                if missing_strategy == "Custom per column":
                    st.info("👇 Configure treatment for each column below")
                else:
                    missing_threshold = st.slider(
                        "Missing threshold (%)",
                        0, 100, 20,
                        help="Columns with more missing values will be dropped"
                    )

            # Apply missing value treatment
            df_cleaned = df.copy()

            if missing_strategy == "Custom per column":
                st.write("**Configure treatment for each column:**")

                treatments = {}
                for col in missing_analysis['Column']:
                    col_treatment = st.selectbox(
                        f"Treatment for {col}:",
                        ["Drop column", "Drop rows", "Mean", "Median", "Mode", "Forward fill", "Backward fill",
                         "Interpolate"],
                        key=f"treatment_{col}"
                    )
                    treatments[col] = col_treatment

                # Apply treatments
                for col, treatment in treatments.items():
                    if treatment == "Drop column":
                        df_cleaned = df_cleaned.drop(columns=[col])
                    elif treatment == "Drop rows":
                        df_cleaned = df_cleaned.dropna(subset=[col])
                    elif treatment == "Mean" and col in numeric_cols:
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                    elif treatment == "Median" and col in numeric_cols:
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                    elif treatment == "Mode":
                        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                    elif treatment == "Forward fill":
                        df_cleaned[col].fillna(method='ffill', inplace=True)
                    elif treatment == "Backward fill":
                        df_cleaned[col].fillna(method='bfill', inplace=True)
                    elif treatment == "Interpolate" and col in numeric_cols:
                        df_cleaned[col].interpolate(inplace=True)

            else:
                # Apply global strategy
                if missing_strategy == "Drop rows with any missing":
                    df_cleaned = df_cleaned.dropna()
                elif missing_strategy == "Drop rows with >50% missing":
                    threshold = len(df_cleaned.columns) * 0.5
                    df_cleaned = df_cleaned.dropna(thresh=threshold)
                elif missing_strategy == "Mean imputation (numeric)":
                    for col in numeric_cols:
                        df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                elif missing_strategy == "Median imputation (numeric)":
                    for col in numeric_cols:
                        df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                elif missing_strategy == "Mode imputation (all)":
                    for col in df_cleaned.columns:
                        if df_cleaned[col].isnull().sum() > 0:
                            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
                elif missing_strategy == "KNN imputation":
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])
                elif missing_strategy == "Iterative imputation":
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    imputer = IterativeImputer(random_state=42)
                    df_cleaned[numeric_cols] = imputer.fit_transform(df_cleaned[numeric_cols])

            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Missing values before", df.isnull().sum().sum())
            with col2:
                st.metric("Missing values after", df_cleaned.isnull().sum().sum())

        else:
            df_cleaned = df.copy()
            st.success("✅ No missing values to handle!")

        # Advanced outlier detection and treatment
        st.subheader("🎯 Advanced Outlier Detection & Treatment")

        if len(numeric_cols) > 0:

            outlier_methods = st.multiselect(
                "Select outlier detection methods:",
                ["IQR", "Z-Score", "Modified Z-Score", "Isolation_Forest", "DBSCAN", "Local Outlier Factor"],
                default=["IQR", "Isolation_Forest"]
            )

            if outlier_methods:
                outliers_info = detect_advanced_outliers(df_cleaned[numeric_cols], outlier_methods)

                # Display outlier information
                for method, method_results in outliers_info.items():
                    st.write(f"**{method} Results:**")

                    if 'all_variables' in method_results:
                        # For methods that detect outliers across all variables
                        st.write(f"Outliers detected: {method_results['all_variables']['count']} "
                                 f"({method_results['all_variables']['percentage']:.2f}%)")
                    else:
                        # For methods that detect outliers per variable
                        method_df = pd.DataFrame({
                            'Variable': method_results.keys(),
                            'Outlier_Count': [info['count'] for info in method_results.values()],
                            'Outlier_Percentage': [f"{info['percentage']:.2f}%" for info in method_results.values()]
                        })
                        st.dataframe(method_df, use_container_width=True)

                # Outlier treatment options
                outlier_treatment = st.selectbox(
                    "Select outlier treatment:",
                    ["None", "Remove outliers", "Cap outliers (Winsorizing)",
                     "Transform (Log)", "Transform (Square root)", "Transform (Box-Cox)",
                     "Robust scaling only"]
                )

                if outlier_treatment != "None":
                    if outlier_treatment == "Remove outliers":
                        # Combine outliers from all methods
                        all_outlier_indices = set()
                        for method_results in outliers_info.values():
                            if 'all_variables' in method_results:
                                all_outlier_indices.update(method_results['all_variables']['indices'])
                            else:
                                for var_results in method_results.values():
                                    all_outlier_indices.update(var_results['indices'])

                        df_cleaned = df_cleaned.drop(all_outlier_indices)
                        st.success(f"✅ Removed {len(all_outlier_indices)} outlier rows")

                    elif outlier_treatment == "Cap outliers (Winsorizing)":
                        winsorize_percentile = st.slider("Winsorization percentile:", 1, 10, 5)
                        lower_perc = winsorize_percentile / 100
                        upper_perc = 1 - lower_perc

                        for col in numeric_cols:
                            if col in df_cleaned.columns:
                                lower_bound = df_cleaned[col].quantile(lower_perc)
                                upper_bound = df_cleaned[col].quantile(upper_perc)
                                df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

                        st.success(f"✅ Applied {winsorize_percentile}% winsorization")

                    elif outlier_treatment == "Transform (Log)":
                        for col in numeric_cols:
                            if col in df_cleaned.columns and (df_cleaned[col] > 0).all():
                                df_cleaned[col] = np.log1p(df_cleaned[col])
                        st.success("✅ Applied log transformation")

                    elif outlier_treatment == "Transform (Square root)":
                        for col in numeric_cols:
                            if col in df_cleaned.columns and (df_cleaned[col] >= 0).all():
                                df_cleaned[col] = np.sqrt(df_cleaned[col])
                        st.success("✅ Applied square root transformation")

                    elif outlier_treatment == "Transform (Box-Cox)":
                        from scipy.stats import boxcox
                        for col in numeric_cols:
                            if col in df_cleaned.columns and (df_cleaned[col] > 0).all():
                                df_cleaned[col], _ = boxcox(df_cleaned[col])
                        st.success("✅ Applied Box-Cox transformation")

        # Data normalization and scaling
        st.subheader("⚖️ Data Normalization & Scaling")

        if len(numeric_cols) > 0:

            scaling_method = st.selectbox(
                "Select scaling method:",
                ["None", "StandardScaler (Z-score)", "MinMaxScaler (0-1)",
                 "RobustScaler", "Quantile Uniform", "Quantile Normal", "Power Transformer"]
            )

            if scaling_method != "None":
                numeric_cols_available = [col for col in numeric_cols if col in df_cleaned.columns]

                if scaling_method == "StandardScaler (Z-score)":
                    scaler = StandardScaler()
                    df_cleaned[numeric_cols_available] = scaler.fit_transform(df_cleaned[numeric_cols_available])

                elif scaling_method == "MinMaxScaler (0-1)":
                    scaler = MinMaxScaler()
                    df_cleaned[numeric_cols_available] = scaler.fit_transform(df_cleaned[numeric_cols_available])

                elif scaling_method == "RobustScaler":
                    scaler = RobustScaler()
                    df_cleaned[numeric_cols_available] = scaler.fit_transform(df_cleaned[numeric_cols_available])

                elif scaling_method == "Quantile Uniform":
                    from sklearn.preprocessing import QuantileTransformer
                    scaler = QuantileTransformer(output_distribution='uniform')
                    df_cleaned[numeric_cols_available] = scaler.fit_transform(df_cleaned[numeric_cols_available])

                elif scaling_method == "Quantile Normal":
                    from sklearn.preprocessing import QuantileTransformer
                    scaler = QuantileTransformer(output_distribution='normal')
                    df_cleaned[numeric_cols_available] = scaler.fit_transform(df_cleaned[numeric_cols_available])

                elif scaling_method == "Power Transformer":
                    from sklearn.preprocessing import PowerTransformer
                    scaler = PowerTransformer(method='yeo-johnson')
                    df_cleaned[numeric_cols_available] = scaler.fit_transform(df_cleaned[numeric_cols_available])

                st.success(f"✅ Applied {scaling_method}")

        # Duplicate removal
        st.subheader("🔄 Duplicate Removal")

        duplicates = df_cleaned.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates} duplicate rows")

            if st.button("Remove duplicates"):
                df_cleaned = df_cleaned.drop_duplicates()
                st.success(f"✅ Removed {duplicates} duplicate rows")
        else:
            st.success("✅ No duplicate rows found!")

        # Data validation
        st.subheader("✅ Data Validation")

        validation_results = []

        # Check for infinite values
        inf_count = np.isinf(df_cleaned.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            validation_results.append(f"❌ Found {inf_count} infinite values")
        else:
            validation_results.append("✅ No infinite values")

        # Check for extreme values
        for col in df_cleaned.select_dtypes(include=[np.number]).columns:
            if df_cleaned[col].std() > 0:  # Avoid division by zero
                z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / df_cleaned[col].std())
                extreme_count = (z_scores > 4).sum()
                if extreme_count > 0:
                    validation_results.append(f"⚠️ {col}: {extreme_count} extreme values (|z| > 4)")

        for result in validation_results:
            if result.startswith("✅"):
                st.success(result)
            elif result.startswith("⚠️"):
                st.warning(result)
            else:
                st.error(result)

        # Cleaning summary
        st.subheader("📊 Cleaning Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows before", df.shape[0])
            st.metric("Rows after", df_cleaned.shape[0])

        with col2:
            st.metric("Columns before", df.shape[1])
            st.metric("Columns after", df_cleaned.shape[1])

        with col3:
            st.metric("Missing before", df.isnull().sum().sum())
            st.metric("Missing after", df_cleaned.isnull().sum().sum())

        with col4:
            data_retention = (df_cleaned.shape[0] / df.shape[0]) * 100
            st.metric("Data retention", f"{data_retention:.1f}%")

        # Save cleaned data
        if st.button("💾 Save Cleaned Data", type="primary"):
            st.session_state.processed_data = df_cleaned
            st.success("✅ Cleaned data saved! You can now proceed to pre-tests.")

        # Preview cleaned data
        st.subheader("👀 Cleaned Data Preview")
        st.dataframe(df_cleaned.head(10), use_container_width=True)

    # Tab 4: Pre-Tests & Adequacy
    elif selected_tab == "🧪 Pre-Tests & Adequacy":
        if st.session_state.processed_data is None:
            st.warning("⚠️ Please clean data first!")
            return

        df = st.session_state.processed_data
        st.markdown('<div class="section-header">🧪 Pre-Tests & Data Adequacy Assessment</div>',
                    unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            st.error("❌ Need at least 2 numeric variables for adequacy tests!")
            return

        # Variable selection for tests
        st.subheader("🎯 Variable Selection for Analysis")

        selected_variables = st.multiselect(
            "Select variables for adequacy testing:",
            numeric_cols,
            default=numeric_cols[:min(8, len(numeric_cols))],
            help="Select variables you plan to use for indicator construction"
        )

        if len(selected_variables) < 2:
            st.warning("⚠️ Please select at least 2 variables!")
            return

        # Prepare data for tests
        test_data = df[selected_variables].dropna()

        if len(test_data) < 3:
            st.error("❌ Insufficient data for testing after removing missing values!")
            return

        st.info(f"📊 Running tests on {len(test_data)} observations and {len(selected_variables)} variables")

        # Run adequacy tests
        if st.button("🚀 Run Adequacy Tests", type="primary"):
            with st.spinner("Running comprehensive adequacy tests..."):
                # Perform adequacy tests
                adequacy_results = adequacy_tests(test_data.values, selected_variables)

                # Store results
                st.session_state.pre_tests_results = {
                    'adequacy': adequacy_results,
                    'variables': selected_variables,
                    'data': test_data
                }

        # Display results if available
        if st.session_state.pre_tests_results:
            results = st.session_state.pre_tests_results['adequacy']

            # KMO Test Results
            st.subheader("🎯 Kaiser-Meyer-Olkin (KMO) Test")

            overall_kmo = results['kmo']['overall']

            # Overall KMO interpretation
            if overall_kmo >= 0.9:
                kmo_class = "test-pass"
                kmo_interpretation = "Excellent"
            elif overall_kmo >= 0.8:
                kmo_class = "test-pass"
                kmo_interpretation = "Very Good"
            elif overall_kmo >= 0.7:
                kmo_class = "test-pass"
                kmo_interpretation = "Good"
            elif overall_kmo >= 0.6:
                kmo_class = "test-warning"
                kmo_interpretation = "Mediocre"
            elif overall_kmo >= 0.5:
                kmo_class = "test-warning"
                kmo_interpretation = "Poor"
            else:
                kmo_class = "test-fail"
                kmo_interpretation = "Unacceptable"

            st.markdown(f"""
            <div class="test-result {kmo_class}">
                <strong>Overall KMO: {overall_kmo:.4f}</strong><br>
                Interpretation: {kmo_interpretation}<br>
                {'✅ Data is suitable for factor analysis' if overall_kmo >= 0.6 else '❌ Data may not be suitable for factor analysis'}
            </div>
            """, unsafe_allow_html=True)

            # Individual KMO values
            st.write("**Individual Variable KMO Values:**")
            individual_kmo_df = pd.DataFrame({
                'Variable': selected_variables,
                'KMO_Value': [results['kmo']['individual'][var] for var in selected_variables],
                'Assessment': [
                    'Excellent' if kmo >= 0.9 else
                    'Very Good' if kmo >= 0.8 else
                    'Good' if kmo >= 0.7 else
                    'Mediocre' if kmo >= 0.6 else
                    'Poor' if kmo >= 0.5 else 'Unacceptable'
                    for kmo in [results['kmo']['individual'][var] for var in selected_variables]
                ]
            }).sort_values('KMO_Value', ascending=False)

            st.dataframe(individual_kmo_df, use_container_width=True)

            # Bartlett's Test Results
            st.subheader("🔮 Bartlett's Test of Sphericity")

            chi_square = results['bartlett']['chi_square']
            p_value = results['bartlett']['p_value']
            df_bartlett = results['bartlett']['df']

            if p_value < 0.05:
                bartlett_class = "test-pass"
                bartlett_interpretation = "Reject null hypothesis - Variables are sufficiently correlated"
            else:
                bartlett_class = "test-fail"
                bartlett_interpretation = "Fail to reject null hypothesis - Variables may be too independent"

            st.markdown(f"""
            <div class="test-result {bartlett_class}">
                <strong>Bartlett's Test Results:</strong><br>
                Chi-square statistic: {chi_square:.4f}<br>
                Degrees of freedom: {df_bartlett:.0f}<br>
                P-value: {p_value:.2e}<br>
                Interpretation: {bartlett_interpretation}
            </div>
            """, unsafe_allow_html=True)

            # Correlation Matrix Determinant
            st.subheader("🔢 Correlation Matrix Determinant")

            determinant = results['determinant']

            if determinant > 0.00001:
                det_class = "test-pass"
                det_interpretation = "Good - No perfect multicollinearity"
            elif determinant > 0.000001:
                det_class = "test-warning"
                det_interpretation = "Acceptable - Some multicollinearity present"
            else:
                det_class = "test-fail"
                det_interpretation = "Poor - High multicollinearity detected"

            st.markdown(f"""
            <div class="test-result {det_class}">
                <strong>Determinant: {determinant:.2e}</strong><br>
                Interpretation: {det_interpretation}
            </div>
            """, unsafe_allow_html=True)

            # Additional Tests
            st.subheader("🔬 Additional Statistical Tests")

            # Normality tests
            st.write("**Normality Tests (Shapiro-Wilk):**")
            normality_results = []

            for var in selected_variables:
                if len(test_data[var]) <= 5000:  # Shapiro-Wilk limit
                    stat, p_val = stats.shapiro(test_data[var])
                    normality_results.append({
                        'Variable': var,
                        'Statistic': stat,
                        'P_value': p_val,
                        'Normal': 'Yes' if p_val > 0.05 else 'No'
                    })

            if normality_results:
                normality_df = pd.DataFrame(normality_results)
                st.dataframe(normality_df, use_container_width=True)

            # Multicollinearity assessment
            st.write("**Variance Inflation Factor (VIF):**")

            try:
                from statsmodels.stats.outliers_influence import variance_inflation_factor

                vif_data = pd.DataFrame()
                vif_data["Variable"] = selected_variables
                vif_data["VIF"] = [variance_inflation_factor(test_data.values, i)
                                   for i in range(len(selected_variables))]
                vif_data["Assessment"] = vif_data["VIF"].apply(
                    lambda x: "Low" if x < 5 else "Moderate" if x < 10 else "High"
                )

                st.dataframe(vif_data, use_container_width=True)

                high_vif = vif_data[vif_data["VIF"] > 10]
                if len(high_vif) > 0:
                    st.warning(f"⚠️ High multicollinearity detected in: {', '.join(high_vif['Variable'].tolist())}")

            except ImportError:
                st.info("📝 Install statsmodels for VIF calculation: pip install statsmodels")

            # Sample size adequacy
            st.subheader("📏 Sample Size Adequacy")

            n_obs = len(test_data)
            n_vars = len(selected_variables)

            # Rule of thumb: at least 5-10 observations per variable
            min_recommended = n_vars * 5
            good_recommended = n_vars * 10

            if n_obs >= good_recommended:
                sample_class = "test-pass"
                sample_interpretation = f"Excellent - {n_obs} observations for {n_vars} variables"
            elif n_obs >= min_recommended:
                sample_class = "test-warning"
                sample_interpretation = f"Adequate - {n_obs} observations for {n_vars} variables"
            else:
                sample_class = "test-fail"
                sample_interpretation = f"Insufficient - {n_obs} observations for {n_vars} variables"

            st.markdown(f"""
            <div class="test-result {sample_class}">
                <strong>Sample Size Assessment:</strong><br>
                Observations: {n_obs}<br>
                Variables: {n_vars}<br>
                Ratio: {n_obs / n_vars:.1f}:1<br>
                {sample_interpretation}<br>
                Recommended minimum: {min_recommended} observations
            </div>
            """, unsafe_allow_html=True)

            # Overall Assessment
            st.subheader("📋 Overall Adequacy Assessment")

            adequacy_score = 0
            max_score = 4

            # KMO score
            if overall_kmo >= 0.7:
                adequacy_score += 1
                kmo_status = "✅"
            elif overall_kmo >= 0.6:
                adequacy_score += 0.5
                kmo_status = "⚠️"
            else:
                kmo_status = "❌"

            # Bartlett's score
            if p_value < 0.05:
                adequacy_score += 1
                bartlett_status = "✅"
            else:
                bartlett_status = "❌"

            # Determinant score
            if determinant > 0.00001:
                adequacy_score += 1
                det_status = "✅"
            elif determinant > 0.000001:
                adequacy_score += 0.5
                det_status = "⚠️"
            else:
                det_status = "❌"

            # Sample size score
            if n_obs >= good_recommended:
                adequacy_score += 1
                sample_status = "✅"
            elif n_obs >= min_recommended:
                adequacy_score += 0.5
                sample_status = "⚠️"
            else:
                sample_status = "❌"

            adequacy_percentage = (adequacy_score / max_score) * 100

            if adequacy_percentage >= 85:
                overall_class = "test-pass"
                overall_recommendation = "Proceed with indicator construction"
            elif adequacy_percentage >= 60:
                overall_class = "test-warning"
                overall_recommendation = "Proceed with caution - consider variable selection"
            else:
                overall_class = "test-fail"
                overall_recommendation = "Not recommended - improve data quality first"

            st.markdown(f"""
            <div class="test-result {overall_class}">
                <strong>Overall Adequacy Score: {adequacy_score:.1f}/{max_score} ({adequacy_percentage:.0f}%)</strong><br><br>
                <strong>Test Summary:</strong><br>
                {kmo_status} KMO Test: {overall_kmo:.3f}<br>
                {bartlett_status} Bartlett's Test: p={p_value:.3e}<br>
                {det_status} Determinant: {determinant:.2e}<br>
                {sample_status} Sample Size: {n_obs}/{min_recommended} minimum<br><br>
                <strong>Recommendation: {overall_recommendation}</strong>
            </div>
            """, unsafe_allow_html=True)

            # Visualization of correlation matrix
            st.subheader("🔗 Correlation Matrix Visualization")

            corr_matrix = test_data.corr()

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 3),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))

            fig.update_layout(
                title='Correlation Matrix of Selected Variables',
                width=600,
                height=600
            )
            st.plotly_chart(fig)

    # Tab 5: Indicator Construction
    elif selected_tab == "📊 Indicator Construction":
        if not st.session_state.pre_tests_results:
            st.warning("⚠️ Please complete pre-tests first!")
            return

        st.markdown('<div class="section-header">📊 Advanced Indicator Construction</div>',
                    unsafe_allow_html=True)

        test_data = st.session_state.pre_tests_results['data']
        available_variables = st.session_state.pre_tests_results['variables']

        # Method selection
        st.subheader("🔧 Construction Method Selection")

        method_categories = {
            "Statistical Methods": ["Principal Component Analysis (PCA)", "Factor Analysis",
                                    "Independent Component Analysis"],
            "Mathematical Methods": ["Equal Weights", "Custom Weights", "Geometric Mean"],
            "Optimization Methods": ["TOPSIS", "Distance to Reference", "Data Envelopment Analysis"],
            "Machine Learning": ["Clustering-based", "Supervised Learning Weights"]
        }

        selected_category = st.selectbox("Select method category:", list(method_categories.keys()))
        method_type = st.selectbox("Select specific method:", method_categories[selected_category])

        # Variable selection for indicator
        st.subheader("🎯 Variable Selection & Configuration")

        selected_variables = st.multiselect(
            "Select variables for indicator construction:",
            available_variables,
            default=available_variables[:min(6, len(available_variables))]
        )

        if len(selected_variables) < 2:
            st.warning("⚠️ Please select at least 2 variables!")
            return

        # Method-specific parameters
        st.subheader("⚙️ Method Configuration")

        # Initialize variables to prevent UnboundLocalError
        show_loadings = False
        show_scores = False

        if method_type == "Principal Component Analysis (PCA)":
            col1, col2 = st.columns(2)

            with col1:
                pca_approach = st.selectbox(
                    "PCA Approach:",
                    ["First Component Only", "Weighted by Variance", "Cumulative 80%", "Kaiser Criterion"]
                )

                n_components = st.slider(
                    "Max components to consider:",
                    min_value=1,
                    max_value=min(len(selected_variables), len(test_data)),
                    value=min(4, len(selected_variables))
                )

            with col2:
                rotation_method = st.selectbox("Rotation method:", ["None", "Varimax"])

                show_loadings = st.checkbox("Show factor loadings", value=True)
                show_scores = st.checkbox("Show component scores", value=True)

        elif method_type == "Factor Analysis":
            col1, col2 = st.columns(2)

            with col1:
                n_factors = st.slider(
                    "Number of factors:",
                    min_value=1,
                    max_value=min(len(selected_variables) - 1, 5),
                    value=2
                )

            with col2:
                fa_method = st.selectbox("Extraction method:", ["Maximum Likelihood", "Principal Factors"])

        elif method_type == "Custom Weights":
            st.write("**Define custom weights for each variable:**")

            weights = {}
            weight_sum = 0

            # Create weight inputs
            cols = st.columns(min(3, len(selected_variables)))
            for i, var in enumerate(selected_variables):
                with cols[i % len(cols)]:
                    weight = st.number_input(
                        f"Weight for {var}:",
                        min_value=0.0,
                        max_value=1.0,
                        value=1.0 / len(selected_variables),
                        step=0.01,
                        key=f"weight_{var}"
                    )
                    weights[var] = weight
                    weight_sum += weight

            # Show weight summary
            if weight_sum > 0:
                normalized_weights = {k: v / weight_sum for k, v in weights.items()}
                st.write("**Normalized weights:**")
                weight_df = pd.DataFrame({
                    'Variable': normalized_weights.keys(),
                    'Weight': normalized_weights.values(),
                    'Percentage': [f"{v * 100:.1f}%" for v in normalized_weights.values()]
                })
                st.dataframe(weight_df, use_container_width=True)

        elif method_type == "TOPSIS":
            st.write("**TOPSIS Configuration:**")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Criteria Types:**")
                criteria_types = {}
                for var in selected_variables:
                    criteria_types[var] = st.selectbox(
                        f"{var}:",
                        ["Benefit (higher is better)", "Cost (lower is better)"],
                        key=f"criteria_{var}"
                    )

            with col2:
                st.write("**Weights:**")
                topsis_weights = {}
                for var in selected_variables:
                    topsis_weights[var] = st.number_input(
                        f"Weight for {var}:",
                        min_value=0.1,
                        max_value=1.0,
                        value=1.0 / len(selected_variables),
                        step=0.05,
                        key=f"topsis_weight_{var}"
                    )

        # Construct indicator
        if st.button("🚀 Construct Indicator", type="primary"):

            with st.spinner("Constructing indicator using advanced methods..."):

                selected_data = test_data[selected_variables].dropna()

                if len(selected_data) == 0:
                    st.error("❌ No data available after removing missing values!")
                    return

                # Initialize results dictionary
                results = {}

                # Construct indicator based on method
                if method_type == "Principal Component Analysis (PCA)":

                    # Map approach to method parameter
                    pca_method_map = {
                        "First Component Only": "first_component",
                        "Weighted by Variance": "weighted",
                        "Cumulative 80%": "cumulative_80",
                        "Kaiser Criterion": "kaiser"
                    }

                    results = create_composite_indicators(
                        selected_data,
                        "PCA",
                        pca_method=pca_method_map.get(pca_approach, "first_component"),
                        n_components=n_components,
                        rotation=rotation_method.lower()
                    )

                elif method_type == "Factor Analysis":
                    results = create_composite_indicators(
                        selected_data,
                        "Factor_Analysis",
                        n_factors=n_factors
                    )

                elif method_type == "Equal Weights":
                    # Simple equal weighting
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(selected_data)
                    indicator = np.mean(scaled_data, axis=1)

                    results = {
                        'indicator': indicator,
                        'method': 'Equal Weights',
                        'weights': {var: 1 / len(selected_variables) for var in selected_variables}
                    }

                elif method_type == "Custom Weights":
                    # Normalize weights
                    total_weight = sum(weights.values())
                    if total_weight > 0:
                        normalized_weights = {k: v / total_weight for k, v in weights.items()}
                    else:
                        normalized_weights = weights

                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(selected_data)

                    # Apply weights
                    weighted_data = scaled_data * np.array(list(normalized_weights.values()))
                    indicator = np.sum(weighted_data, axis=1)

                    results = {
                        'indicator': indicator,
                        'method': 'Custom Weights',
                        'weights': normalized_weights
                    }

                elif method_type == "TOPSIS":
                    # Prepare TOPSIS parameters
                    topsis_criteria = [
                        'max' if criteria_types[var] == "Benefit (higher is better)" else 'min'
                        for var in selected_variables
                    ]

                    # Normalize weights
                    total_weight = sum(topsis_weights.values())
                    normalized_topsis_weights = [topsis_weights[var] / total_weight for var in selected_variables]

                    results = create_composite_indicators(
                        selected_data,
                        "TOPSIS",
                        weights=normalized_topsis_weights,
                        criteria_types=topsis_criteria
                    )

                elif method_type == "Distance to Reference":
                    results = create_composite_indicators(
                        selected_data,
                        "Distance_to_Reference",
                        reference='best'  # Use best performer as reference
                    )

                # Ensure 'data' and 'variables' keys are always present to prevent KeyError
                results['data'] = selected_data
                results['variables'] = selected_variables

                # Store results
                st.session_state.indicator_results = results

                st.success("✅ Indicator constructed successfully!")

        # Display results if available
        if st.session_state.indicator_results:
            results = st.session_state.indicator_results
            indicator = results['indicator']

            st.subheader("📊 Indicator Construction Results")

            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📊 Mean", f"{np.mean(indicator):.4f}")
            with col2:
                st.metric("📏 Std Dev", f"{np.std(indicator):.4f}")
            with col3:
                st.metric("📉 Minimum", f"{np.min(indicator):.4f}")
            with col4:
                st.metric("📈 Maximum", f"{np.max(indicator):.4f}")

            # Method-specific results
            if results['method'] == 'PCA':

                st.subheader("🔍 PCA Analysis Results")

                # Explained variance
                explained_var = results['explained_variance']

                col1, col2 = st.columns(2)

                with col1:
                    # Scree plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, len(explained_var) + 1)),
                        y=results['pca_object'].explained_variance_,  # Use eigenvalues for scree plot
                        mode='lines+markers',
                        name='Eigenvalues',
                        line=dict(width=3)
                    ))
                    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                                  annotation_text="Kaiser Criterion")
                    fig.update_layout(
                        title="Scree Plot",
                        xaxis_title="Component Number",
                        yaxis_title="Eigenvalue"
                    )
                    st.plotly_chart(fig)

                with col2:
                    # Cumulative variance
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=[f"PC{i + 1}" for i in range(len(explained_var))],
                        y=explained_var,
                        name="Individual"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[f"PC{i + 1}" for i in range(len(explained_var))],
                        y=np.cumsum(explained_var),
                        mode='lines+markers',
                        name="Cumulative",
                        yaxis='y2'
                    ))

                    fig.update_layout(
                        title="Explained Variance",
                        yaxis=dict(title="Individual Variance"),
                        yaxis2=dict(title="Cumulative Variance", overlaying='y', side='right')
                    )
                    st.plotly_chart(fig)

                # Component loadings table
                if show_loadings:
                    st.subheader("📋 Component Loadings")

                    n_components_show = min(4, len(results['components']))
                    loadings_df = pd.DataFrame(
                        results['components'][:n_components_show].T,
                        columns=[f"PC{i + 1}" for i in range(n_components_show)],
                        index=results['variables']
                    )

                    # Add communalities
                    communalities = np.sum(loadings_df.values ** 2, axis=1)
                    loadings_df['Communality'] = communalities

                    st.dataframe(loadings_df, use_container_width=True)

                # Component scores
                if show_scores and 'pca_scores' in results:
                    st.subheader("📊 Component Scores Distribution")

                    scores_df = pd.DataFrame(
                        results['pca_scores'][:, :min(3, results['pca_scores'].shape[1])],
                        columns=[f"PC{i + 1}" for i in range(min(3, results['pca_scores'].shape[1]))]
                    )

                    fig = px.histogram(scores_df, title="Distribution of Component Scores")
                    st.plotly_chart(fig, use_container_width=True)

            elif results['method'] == 'Factor Analysis':

                st.subheader("🔬 Factor Analysis Results")

                # Factor loadings
                loadings_df = pd.DataFrame(
                    results['loadings'].T,
                    columns=[f"Factor{i + 1}" for i in range(results['loadings'].shape[0])],
                    index=results['variables']
                )

                st.write("**Factor Loadings:**")
                st.dataframe(loadings_df, use_container_width=True)

                # Factor scores distribution
                if 'factor_scores' in results:
                    scores_df = pd.DataFrame(
                        results['factor_scores'],
                        columns=[f"Factor{i + 1}" for i in range(results['factor_scores'].shape[1])]
                    )

                    fig = px.histogram(scores_df, title="Distribution of Factor Scores")
                    st.plotly_chart(fig, use_container_width=True)

            elif results['method'] in ['Equal Weights', 'Custom Weights']:

                st.subheader("⚖️ Weighting Scheme")

                weights_df = pd.DataFrame({
                    'Variable': results['weights'].keys(),
                    'Weight': results['weights'].values(),
                    'Percentage': [f"{v * 100:.1f}%" for v in results['weights'].values()]
                })

                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(weights_df, use_container_width=True)

                with col2:
                    fig = px.pie(weights_df, values='Weight', names='Variable',
                                 title='Variable Weights Distribution')
                    st.plotly_chart(fig)

            elif results['method'] == 'TOPSIS':

                st.subheader("🎯 TOPSIS Results")

                st.write("**TOPSIS Score Interpretation:**")
                st.write("- Higher scores indicate better performance")
                st.write("- Scores range from 0 to 1")
                st.write("- Scores closer to 1 indicate proximity to ideal solution")

                # Show TOPSIS configuration
                if 'weights' in results and 'criteria_types' in results:
                    config_df = pd.DataFrame({
                        'Variable': results['variables'],
                        'Weight': results['weights'],
                        'Criteria_Type': results['criteria_types']
                    })
                    st.dataframe(config_df, use_container_width=True)

            # Indicator distribution
            st.subheader("📊 Indicator Distribution Analysis")

            col1, col2 = st.columns(2)

            with col1:
                # Histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=indicator,
                    nbinsx=30,
                    name="Indicator Distribution",
                    opacity=0.7
                ))
                fig.update_layout(title="Composite Indicator Distribution")
                st.plotly_chart(fig)

            with col2:
                # Box plot
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=indicator,
                    name="Indicator",
                    boxmean='sd'
                ))
                fig.update_layout(title="Indicator Box Plot")
                st.plotly_chart(fig)

            # Statistical properties
            st.subheader("📈 Statistical Properties")

            col1, col2, col3 = st.columns(3)

            with col1:
                skewness = stats.skew(indicator)
                st.metric("Skewness", f"{skewness:.4f}")
                if abs(skewness) < 0.5:
                    st.success("✅ Approximately symmetric")
                elif abs(skewness) < 1:
                    st.warning("⚠️ Moderately skewed")
                else:
                    st.error("❌ Highly skewed")

            with col2:
                kurtosis = stats.kurtosis(indicator)
                st.metric("Kurtosis", f"{kurtosis:.4f}")
                if abs(kurtosis) < 1:
                    st.success("✅ Normal-like tails")
                else:
                    st.warning("⚠️ Non-normal tails")

            with col3:
                # Normality test
                if len(indicator) <= 5000:
                    stat, p_value = stats.shapiro(indicator)
                    st.metric("Shapiro-Wilk p-value", f"{p_value:.4f}")
                    if p_value > 0.05:
                        st.success("✅ Normally distributed")
                    else:
                        st.warning("⚠️ Not normally distributed")

    # Tab 6: Advanced Analysis
    elif selected_tab == "📈 Advanced Analysis":
        if not st.session_state.indicator_results:
            st.warning("⚠️ Please construct an indicator first!")
            return

        st.markdown('<div class="section-header">📈 Advanced Analysis & Insights</div>',
                    unsafe_allow_html=True)

        results = st.session_state.indicator_results
        indicator = results['indicator']

        # Ensure 'data' key exists and is a DataFrame
        if 'data' not in results or not isinstance(results['data'], pd.DataFrame):
            st.error("Error: Processed data for the indicator is not available.")
            return

        analysis_data = results['data']

        # Create advanced visualizations
        st.subheader("🎨 Advanced Visualizations")

        # Generate advanced plots
        advanced_plots = create_advanced_visualizations(
            analysis_data,
            indicator,
            results
        )

        # Display plots
        viz_tabs = st.tabs(["📊 Biplot", "🔗 Parallel Coordinates", "🎯 Radar Chart", "🔥 Sensitivity Heatmap"])

        with viz_tabs[0]:
            if 'biplot' in advanced_plots:
                st.plotly_chart(advanced_plots['biplot'], use_container_width=True)
            else:
                st.info("Biplot available only for PCA method")

        with viz_tabs[1]:
            st.plotly_chart(advanced_plots['parallel_coordinates'], use_container_width=True)

        with viz_tabs[2]:
            st.plotly_chart(advanced_plots['radar_chart'], use_container_width=True)

        with viz_tabs[3]:
            st.plotly_chart(advanced_plots['sensitivity_heatmap'], use_container_width=True)

        # Ranking analysis with clustering
        st.subheader("🏆 Advanced Ranking Analysis")

        # Create comprehensive ranking
        ranking_df = pd.DataFrame({
            'Index': analysis_data.index,
            'Indicator_Value': indicator,
            'Rank': stats.rankdata(-indicator, method='ordinal'),
            'Percentile': stats.rankdata(indicator, method='average') / len(indicator) * 100
        })

        # Add performance categories
        ranking_df['Performance_Category'] = pd.cut(
            ranking_df['Percentile'],
            bins=[0, 25, 50, 75, 100.1],  # Extend bin to include 100
            labels=['Bottom Quartile', 'Below Average', 'Above Average', 'Top Quartile'],
            right=True,
            include_lowest=True
        )

        # Add entity names if available
        entity_col = None
        original_df = st.session_state.data
        for col in original_df.columns:
            if col.lower() in ['country', 'city', 'region', 'entity', 'name'] and not pd.api.types.is_numeric_dtype(
                    original_df[col]):
                entity_col = col
                break

        if entity_col:
            # Use original index to map entity names correctly
            ranking_df = ranking_df.merge(original_df[[entity_col]], left_on='Index', right_index=True, how='left')
            ranking_df.rename(columns={entity_col: 'Entity'}, inplace=True)
        else:
            ranking_df['Entity'] = [f"Entity_{i}" for i in ranking_df['Index']]

        ranking_df = ranking_df.sort_values('Rank')

        # Performance distribution
        col1, col2 = st.columns(2)

        with col1:
            category_counts = ranking_df['Performance_Category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Performance Distribution"
            )
            st.plotly_chart(fig)

        with col2:
            fig = px.box(
                ranking_df,
                x='Performance_Category',
                y='Indicator_Value',
                title="Indicator Values by Performance Category"
            )
            st.plotly_chart(fig)

        # Top and bottom performers
        col1, col2 = st.columns(2)

        with col1:
            st.write("**🥇 Top 10 Performers:**")
            top_performers = ranking_df.head(10)[['Entity', 'Indicator_Value', 'Rank', 'Percentile']]
            st.dataframe(top_performers, use_container_width=True)

        with col2:
            st.write("**🔻 Bottom 10 Performers:**")
            bottom_performers = ranking_df.tail(10)[['Entity', 'Indicator_Value', 'Rank', 'Percentile']].sort_values(
                'Rank')
            st.dataframe(bottom_performers, use_container_width=True)

        # Clustering analysis
        st.subheader("🎯 Performance Clustering Analysis")

        clustering_method = st.selectbox(
            "Select clustering method:",
            ["K-Means", "Hierarchical", "DBSCAN"]
        )

        if clustering_method == "K-Means":
            n_clusters = st.slider("Number of clusters:", 2, 8, 4)

            # Prepare data for clustering
            cluster_data = analysis_data.copy()
            cluster_data['Indicator'] = indicator

            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(StandardScaler().fit_transform(cluster_data))

            # Add clusters to ranking
            ranking_df['Cluster'] = clusters

            # Visualize clusters
            fig = px.scatter(
                ranking_df,
                x='Index',
                y='Indicator_Value',
                color='Cluster',
                title=f"K-Means Clustering (k={n_clusters})",
                hover_data=['Entity', 'Rank']
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster statistics
            cluster_stats = ranking_df.groupby('Cluster').agg({
                'Indicator_Value': ['mean', 'std', 'count'],
                'Rank': ['mean', 'min', 'max']
            }).round(3)

            st.write("**Cluster Statistics:**")
            st.dataframe(cluster_stats, use_container_width=True)

    # Tab 7: Validation & Robustness
    elif selected_tab == "🔬 Validation & Robustness":
        if not st.session_state.indicator_results:
            st.warning("⚠️ Please construct an indicator first!")
            return

        st.markdown('<div class="section-header">🔬 Validation & Robustness Testing</div>',
                    unsafe_allow_html=True)

        results = st.session_state.indicator_results
        indicator = results['indicator']

        # Robustness tests
        st.subheader("💪 Robustness Analysis")

        robustness_tests = st.multiselect(
            "Select robustness tests to perform:",
            [
                "Bootstrap Confidence Intervals",
                "Leave-One-Out Validation",
                "Monte Carlo Simulation",
                "Sensitivity to Outliers",
                "Alternative Weighting Schemes",
                "Subsample Stability"
            ],
            default=["Bootstrap Confidence Intervals", "Leave-One-Out Validation"]
        )

        if st.button("🚀 Run Robustness Tests", type="primary"):

            robustness_results = {}

            with st.spinner("Running robustness tests..."):

                # Bootstrap Confidence Intervals
                if "Bootstrap Confidence Intervals" in robustness_tests:
                    st.write("**Running Bootstrap Analysis...**")

                    n_bootstrap = 1000
                    bootstrap_indicators = []

                    for i in range(n_bootstrap):
                        # Resample data
                        bootstrap_indices = np.random.choice(
                            len(results['data']),
                            size=len(results['data']),
                            replace=True
                        )
                        bootstrap_data = results['data'].iloc[bootstrap_indices]

                        # Reconstruct indicator
                        if results['method'] == 'PCA':
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(bootstrap_data)
                            pca = PCA(n_components=min(3, bootstrap_data.shape[1]))
                            pca_result = pca.fit_transform(scaled_data)
                            bootstrap_indicator = pca_result[:, 0]  # First component
                        else:
                            # For other methods, use simple weighted average
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(bootstrap_data)
                            if 'weights' in results and results['weights'] is not None:
                                weights_array = np.array(list(results['weights'].values()))
                                bootstrap_indicator = np.average(scaled_data, axis=1, weights=weights_array)
                            else:
                                bootstrap_indicator = np.mean(scaled_data, axis=1)

                        bootstrap_indicators.append(bootstrap_indicator)

                    # Calculate confidence intervals
                    bootstrap_means = [np.mean(bi) for bi in bootstrap_indicators]
                    ci_lower = np.percentile(bootstrap_means, 2.5)
                    ci_upper = np.percentile(bootstrap_means, 97.5)

                    robustness_results['bootstrap'] = {
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'original_mean': np.mean(indicator),
                        'bootstrap_std': np.std(bootstrap_means)
                    }

                # Leave-One-Out Validation
                if "Leave-One-Out Validation" in robustness_tests:
                    st.write("**Running Leave-One-Out Analysis...**")

                    loo_indicators = []
                    original_data = results['data']

                    for i in range(len(original_data)):
                        # Remove one observation
                        loo_data = original_data.drop(original_data.index[i])

                        # Reconstruct indicator
                        if results['method'] == 'PCA':
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(loo_data)
                            pca = PCA(n_components=min(3, loo_data.shape[1]))
                            pca_result = pca.fit_transform(scaled_data)
                            loo_indicator = pca_result[:, 0]
                        else:
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(loo_data)
                            if 'weights' in results and results['weights'] is not None:
                                weights_array = np.array(list(results['weights'].values()))
                                loo_indicator = np.average(scaled_data, axis=1, weights=weights_array)
                            else:
                                loo_indicator = np.mean(scaled_data, axis=1)

                        loo_indicators.append(loo_indicator)

                    # Calculate stability metrics
                    correlations = []
                    original_ranks = stats.rankdata(-indicator)

                    for i, loo_ind in enumerate(loo_indicators):
                        if len(loo_ind) > 1:
                            # Compute correlation with original (excluding the left-out observation)
                            loo_ranks = stats.rankdata(-loo_ind)
                            # Align ranks (remove corresponding position)
                            aligned_original = np.delete(original_ranks, i)
                            if len(aligned_original) == len(loo_ranks):
                                corr = np.corrcoef(aligned_original, loo_ranks)[0, 1]
                                correlations.append(corr)

                    robustness_results['loo'] = {
                        'mean_correlation': np.mean(correlations) if correlations else 0,
                        'std_correlation': np.std(correlations) if correlations else 0,
                        'min_correlation': np.min(correlations) if correlations else 0
                    }

                # Monte Carlo Simulation
                if "Monte Carlo Simulation" in robustness_tests:
                    st.write("**Running Monte Carlo Simulation...**")

                    n_simulations = 500
                    mc_indicators = []

                    for i in range(n_simulations):
                        # Add noise to original data
                        noise_level = 0.05  # 5% noise
                        noisy_data = results['data'] + np.random.normal(
                            0,
                            noise_level * results['data'].std(),
                            results['data'].shape
                        )

                        # Reconstruct indicator
                        if results['method'] == 'PCA':
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(noisy_data)
                            pca = PCA(n_components=min(3, noisy_data.shape[1]))
                            pca_result = pca.fit_transform(scaled_data)
                            mc_indicator = pca_result[:, 0]
                        else:
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(noisy_data)
                            if 'weights' in results and results['weights'] is not None:
                                weights_array = np.array(list(results['weights'].values()))
                                mc_indicator = np.average(scaled_data, axis=1, weights=weights_array)
                            else:
                                mc_indicator = np.mean(scaled_data, axis=1)

                        mc_indicators.append(mc_indicator)

                    # Calculate stability
                    mc_correlations = []
                    for mc_ind in mc_indicators:
                        corr = np.corrcoef(indicator, mc_ind)[0, 1]
                        mc_correlations.append(corr)

                    robustness_results['monte_carlo'] = {
                        'mean_correlation': np.mean(mc_correlations),
                        'std_correlation': np.std(mc_correlations),
                        'stability_95': np.percentile(mc_correlations, 5)  # 95% of correlations above this
                    }

            # Display robustness results
            st.subheader("📊 Robustness Test Results")

            if 'bootstrap' in robustness_results:
                bootstrap_res = robustness_results['bootstrap']

                st.markdown(f"""
                <div class="test-result test-pass">
                    <strong>Bootstrap Confidence Intervals (95%):</strong><br>
                    Original Mean: {bootstrap_res['original_mean']:.4f}<br>
                    CI Lower: {bootstrap_res['ci_lower']:.4f}<br>
                    CI Upper: {bootstrap_res['ci_upper']:.4f}<br>
                    Bootstrap Std: {bootstrap_res['bootstrap_std']:.4f}
                </div>
                """, unsafe_allow_html=True)

            if 'loo' in robustness_results:
                loo_res = robustness_results['loo']

                stability_class = "test-pass" if loo_res['mean_correlation'] > 0.9 else \
                    "test-warning" if loo_res['mean_correlation'] > 0.8 else "test-fail"

                st.markdown(f"""
                <div class="test-result {stability_class}">
                    <strong>Leave-One-Out Validation:</strong><br>
                    Mean Correlation: {loo_res['mean_correlation']:.4f}<br>
                    Std Correlation: {loo_res['std_correlation']:.4f}<br>
                    Min Correlation: {loo_res['min_correlation']:.4f}<br>
                    {'✅ High stability' if loo_res['mean_correlation'] > 0.9 else
                '⚠️ Moderate stability' if loo_res['mean_correlation'] > 0.8 else
                '❌ Low stability'}
                </div>
                """, unsafe_allow_html=True)

            if 'monte_carlo' in robustness_results:
                mc_res = robustness_results['monte_carlo']

                noise_class = "test-pass" if mc_res['stability_95'] > 0.8 else \
                    "test-warning" if mc_res['stability_95'] > 0.6 else "test-fail"

                st.markdown(f"""
                <div class="test-result {noise_class}">
                    <strong>Monte Carlo Noise Sensitivity:</strong><br>
                    Mean Correlation: {mc_res['mean_correlation']:.4f}<br>
                    Std Correlation: {mc_res['std_correlation']:.4f}<br>
                    95% Stability Level: {mc_res['stability_95']:.4f}<br>
                    {'✅ Robust to noise' if mc_res['stability_95'] > 0.8 else
                '⚠️ Moderately sensitive' if mc_res['stability_95'] > 0.6 else
                '❌ Highly sensitive to noise'}
                </div>
                """, unsafe_allow_html=True)

        # Cross-validation with alternative methods
        st.subheader("🔄 Cross-Method Validation")

        if st.button("🔍 Compare with Alternative Methods"):

            with st.spinner("Comparing with alternative construction methods..."):

                # Compare current method with alternatives
                alternative_results = {}

                # Always compare with equal weights
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(results['data'])
                equal_weights_indicator = np.mean(scaled_data, axis=1)
                alternative_results['Equal Weights'] = equal_weights_indicator

                # If current method is not PCA, compare with PCA
                if results['method'] != 'PCA':
                    pca = PCA(n_components=1)
                    pca_result = pca.fit_transform(scaled_data)
                    alternative_results['PCA (1st component)'] = pca_result[:, 0]

                # If current method is not TOPSIS, add simple distance method
                if results['method'] != 'TOPSIS':
                    # Distance to best performer
                    best_point = np.max(scaled_data, axis=0)
                    distances = np.sqrt(np.sum((scaled_data - best_point) ** 2, axis=1))
                    alternative_results['Distance to Best'] = -distances

                # Calculate correlations with original indicator
                comparison_df = pd.DataFrame({
                    'Method': list(alternative_results.keys()),
                    'Correlation': [
                        np.corrcoef(indicator, alt_ind)[0, 1]
                        for alt_ind in alternative_results.values()
                    ],
                    'Rank_Correlation': [
                        stats.spearmanr(indicator, alt_ind)[0]
                        for alt_ind in alternative_results.values()
                    ]
                })

                st.write("**Cross-Method Validation Results:**")
                st.dataframe(comparison_df, use_container_width=True)

                # Visualization
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=comparison_df['Method'],
                    y=comparison_df['Correlation'],
                    name='Pearson Correlation',
                    opacity=0.7
                ))

                fig.add_trace(go.Bar(
                    x=comparison_df['Method'],
                    y=comparison_df['Rank_Correlation'],
                    name='Spearman Correlation',
                    opacity=0.7
                ))

                fig.update_layout(
                    title='Correlation with Alternative Methods',
                    xaxis_title='Alternative Method',
                    yaxis_title='Correlation Coefficient',
                    barmode='group'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Interpretation
                min_correlation = comparison_df['Correlation'].min()
                if min_correlation > 0.8:
                    st.success("✅ High agreement across methods - robust indicator")
                elif min_correlation > 0.6:
                    st.warning("⚠️ Moderate agreement - consider method sensitivity")
                else:
                    st.error("❌ Low agreement - method choice significantly affects results")

    # Tab 8: Export & Documentation
    elif selected_tab == "💾 Export & Documentation":
        if not st.session_state.indicator_results:
            st.warning("⚠️ Please construct an indicator first!")
            return

        st.markdown('<div class="section-header">💾 Export Results & Comprehensive Documentation</div>',
                    unsafe_allow_html=True)

        results = st.session_state.indicator_results
        indicator = results['indicator']
        indicator_data = results['data']

        # Prepare comprehensive export data
        st.subheader("📊 Data Preparation for Export")

        # Start with a fresh copy of the data used for the indicator
        export_df = st.session_state.data.loc[indicator_data.index].copy()

        export_df['Composite_Indicator'] = indicator

        # Add rankings and percentiles
        ranks = stats.rankdata(-indicator, method='ordinal')
        percentiles = stats.rankdata(indicator, method='average') / len(indicator) * 100

        export_df['Indicator_Rank'] = ranks
        export_df['Indicator_Percentile'] = percentiles

        # Add performance categories
        performance_cats = pd.cut(
            percentiles,
            bins=[0, 25, 50, 75, 100.1],
            labels=['Bottom Quartile', 'Below Average', 'Above Average', 'Top Quartile'],
            right=True, include_lowest=True
        )
        export_df['Performance_Category'] = performance_cats

        # Export format selection
        st.subheader("📁 Export Format & Options")

        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox(
                "Select export format:",
                ["Excel (Comprehensive)", "CSV (Data Only)", "JSON (Full Structure)", "PDF Report"]
            )

        with col2:
            include_options = st.multiselect(
                "Include additional data:",
                ["Method Details", "Statistical Tests", "Robustness Results", "Visualizations", "Full Documentation"],
                default=["Method Details", "Statistical Tests", "Full Documentation"]
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate comprehensive documentation
        def generate_comprehensive_documentation():
            doc = f"""
# COMPREHENSIVE ECONOMIC/SOCIAL INDICATOR CONSTRUCTION REPORT

**Report Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## EXECUTIVE SUMMARY

This report documents the construction of a composite indicator using advanced statistical methods. The indicator was constructed using {results['method']} methodology on {len(results.get('variables', []))} selected variables from a dataset of {len(indicator)} observations.

### Key Findings:
- **Method Used:** {results['method']}
- **Final Indicator Mean:** {np.mean(indicator):.4f}
- **Standard Deviation:** {np.std(indicator):.4f}
- **Skewness:** {stats.skew(indicator):.4f}
- **Kurtosis:** {stats.kurtosis(indicator):.4f}

## METHODOLOGY

### Data Adequacy Tests Performed:
"""

            if st.session_state.pre_tests_results:
                adequacy = st.session_state.pre_tests_results['adequacy']
                doc += f"""
- **Kaiser-Meyer-Olkin (KMO) Test:** {adequacy['kmo']['overall']:.4f}
- **Bartlett's Test of Sphericity:** p-value = {adequacy['bartlett']['p_value']:.2e}
- **Correlation Matrix Determinant:** {adequacy['determinant']:.2e}
"""

            doc += f"""

### Variables Included:
{chr(10).join([f"- {var}" for var in results.get('variables', [])])}

### Construction Method Details:
**Primary Method:** {results['method']}

"""

            if results['method'] == 'PCA':
                doc += f"""
**PCA Specifications:**
- Components used: {len(results['explained_variance'])}
- Total variance explained by first 3 components: {np.sum(results['explained_variance'][:3]):.1%}
- First component variance: {results['explained_variance'][0]:.1%}

**Component Loadings (First 3):**
{pd.DataFrame(results['components'][:3].T, columns=[f"PC{i + 1}" for i in range(min(3, len(results['components'])))], index=results.get('variables', [])).to_string()}
"""

            elif 'weights' in results and results['weights'] is not None:
                doc += f"""
**Variable Weights:**
{pd.DataFrame({'Variable': results['weights'].keys(), 'Weight': results['weights'].values()}).to_string(index=False)}
"""

            doc += f"""

## DATA QUALITY ASSESSMENT

### Original Dataset:
- **Total Observations:** {len(st.session_state.data)}
- **Total Variables:** {len(st.session_state.data.columns)}
- **Missing Values:** {st.session_state.data.isnull().sum().sum()}

### Processed Dataset:
- **Final Observations:** {len(indicator_data)}
- **Variables Used:** {len(results.get('variables', []))}
- **Data Retention Rate:** {(len(indicator_data) / len(st.session_state.data) * 100):.1f}%

## STATISTICAL PROPERTIES

### Indicator Distribution:
- **Mean:** {np.mean(indicator):.6f}
- **Median:** {np.median(indicator):.6f}
- **Standard Deviation:** {np.std(indicator):.6f}
- **Minimum:** {np.min(indicator):.6f}
- **Maximum:** {np.max(indicator):.6f}
- **Range:** {np.max(indicator) - np.min(indicator):.6f}

### Distribution Characteristics:
- **Skewness:** {stats.skew(indicator):.4f} ({'Right-skewed' if stats.skew(indicator) > 0.5 else 'Left-skewed' if stats.skew(indicator) < -0.5 else 'Approximately Symmetric'})
- **Kurtosis:** {stats.kurtosis(indicator):.4f} ({'Heavy-tailed (Leptokurtic)' if stats.kurtosis(indicator) > 0.5 else 'Light-tailed (Platykurtic)' if stats.kurtosis(indicator) < -0.5 else 'Normal-tailed (Mesokurtic)'})

### Normality Assessment:
"""

            if len(indicator) <= 5000:
                stat, p_val = stats.shapiro(indicator)
                doc += f"- **Shapiro-Wilk Test:** statistic = {stat:.4f}, p-value = {p_val:.4f}"
                doc += f"\n- **Normally Distributed (alpha=0.05):** {'Yes' if p_val > 0.05 else 'No'}"
            else:
                doc += "- **Shapiro-Wilk Test:** Not performed (sample size > 5000)"

            doc += f"""

## RANKING ANALYSIS

### Performance Distribution:
- **Top Quartile (75-100%):** {len(export_df[export_df['Performance_Category'] == 'Top Quartile'])} entities
- **Above Average (50-75%):** {len(export_df[export_df['Performance_Category'] == 'Above Average'])} entities  
- **Below Average (25-50%):** {len(export_df[export_df['Performance_Category'] == 'Below Average'])} entities
- **Bottom Quartile (0-25%):** {len(export_df[export_df['Performance_Category'] == 'Bottom Quartile'])} entities

### Top 10 Performers:
"""

            # FIX: Robustly find the entity column and handle cases where none is found
            # This prevents the KeyError: "['Index'] not in index"

            # Find a suitable entity column (e.g., 'Country', 'City')
            entity_col = None
            for col in st.session_state.data.columns:
                if col.lower() in ['country', 'city', 'region', 'entity', 'name'] and not pd.api.types.is_numeric_dtype(
                        st.session_state.data[col]):
                    entity_col = col
                    break

            # Get the top 10 performing rows
            top_10_df = export_df.nsmallest(10, 'Indicator_Rank')

            # Define the columns we always want to show
            display_cols = ['Composite_Indicator', 'Indicator_Rank', 'Indicator_Percentile']

            # Prepare the final DataFrame for display
            if entity_col:
                # If we found an entity column, use it
                display_cols.insert(0, entity_col)
                top_10 = top_10_df[display_cols]
            else:
                # If no entity column, use the DataFrame's index
                top_10 = top_10_df[display_cols]
                top_10.index.name = 'Entity_Index'  # Name the index
                top_10 = top_10.reset_index()  # Turn the index into a column

            if not top_10.empty:
                doc += top_10.to_string()

            doc += f"""

## TECHNICAL SPECIFICATIONS

### Software Environment:
- **Platform:** Streamlit-based Advanced Indicator Builder
- **Statistical Libraries:** scikit-learn, scipy, pandas
- **Visualization Libraries:** plotly, matplotlib

### Reproducibility:
- **Random Seed:** 42 (where applicable)
- **Scaling Method:** StandardScaler (default for PCA/FA)
- **Missing Value Treatment:** {'Applied' if st.session_state.processed_data.isnull().sum().sum() != st.session_state.data.isnull().sum().sum() else 'Not required/applied'}

## LIMITATIONS AND CONSIDERATIONS

1. **Data Quality:** Results are contingent on the quality, accuracy, and representativeness of the input data.
2. **Method Selection:** The choice of construction methodology can influence the final indicator scores and rankings.
3. **Variable Selection:** The indicator is a reflection of the selected variables and may not capture all dimensions of the concept being measured.
4. **Temporal Validity:** The indicator reflects the state of affairs for the specific time period of the source data.
5. **Interpretation:** Scores should be interpreted relatively within the sample, not as absolute measures of performance.

## RECOMMENDATIONS

### For Users:
1. Validate the indicator's results against external benchmarks or expert opinions where possible.
2. Be mindful of the stated limitations when interpreting and communicating the indicator scores.
3. Consider updating the indicator periodically with new data to maintain its relevance.
4. Test the robustness of the findings by exploring alternative construction methods or variable sets.

### For Further Analysis:
1. Conduct a sensitivity analysis by altering variable weights or combinations.
2. Explore the temporal stability of the indicator if time-series data becomes available.
3. Validate the indicator against external criteria (criterion validity).
4. Consider developing sub-indicators to provide a more granular view of specific performance domains.

## CONCLUSION

The composite indicator was successfully constructed using the {results['method']} methodology. The indicator exhibits {'robust' if abs(stats.skew(indicator)) < 1 else 'acceptable'} statistical properties and provides a comprehensive ranking of entities based on the selected variables.

---

**Report Prepared by:** Advanced Economic & Social Indicator Builder
**Version:** 1.0
"""

            return doc

        # Export functionality
        if export_format == "Excel (Comprehensive)":
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Main data sheet
                export_df.to_excel(writer, sheet_name='Indicator_Data', index=False)

                # Method details
                method_info = pd.DataFrame({
                    'Parameter': [
                        'Construction_Method',
                        'Variables_Used',
                        'Number_of_Variables',
                        'Number_of_Observations',
                        'Construction_Date',
                        'Data_Retention_Rate',
                        'Indicator_Mean',
                        'Indicator_StdDev',
                        'Indicator_Skewness',
                        'Indicator_Kurtosis'
                    ],
                    'Value': [
                        results['method'],
                        ', '.join(results.get('variables', [])),
                        len(results.get('variables', [])),
                        len(indicator),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{(len(indicator_data) / len(st.session_state.data) * 100):.1f}%",
                        f"{np.mean(indicator):.6f}",
                        f"{np.std(indicator):.6f}",
                        f"{stats.skew(indicator):.4f}",
                        f"{stats.kurtosis(indicator):.4f}"
                    ]
                })

                if "Method Details" in include_options:
                    method_info.to_excel(writer, sheet_name='Method_Details', index=False)

                if "Statistical Tests" in include_options and st.session_state.pre_tests_results:
                    adequacy = st.session_state.pre_tests_results['adequacy']
                    tests_summary = {
                        "KMO Overall": adequacy['kmo']['overall'],
                        "Bartlett Chi-Square": adequacy['bartlett']['chi_square'],
                        "Bartlett P-Value": adequacy['bartlett']['p_value'],
                        "Determinant": adequacy['determinant']
                    }
                    pd.DataFrame(list(tests_summary.items()), columns=['Test', 'Value']).to_excel(
                        writer, sheet_name='Adequacy_Tests', index=False
                    )

                if "Full Documentation" in include_options:
                    doc_text = generate_comprehensive_documentation()
                    doc_df = pd.DataFrame(doc_text.split('\n'), columns=['Documentation'])
                    doc_df.to_excel(writer, sheet_name='Documentation', index=False)

            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name=f"indicator_report_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        elif export_format == "CSV (Data Only)":
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV Data",
                data=csv_data,
                file_name=f"indicator_data_{timestamp}.csv",
                mime="text/csv",
            )

        elif export_format == "JSON (Full Structure)":
            # Custom encoder to handle numpy types which are not JSON serializable by default
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if pd.api.types.is_period(obj):
                        return str(obj)
                    if isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    return super(NpEncoder, self).default(obj)

            export_data_json = {
                "metadata": {
                    "report_generated": datetime.now().isoformat(),
                    "construction_method": results.get('method'),
                    "variables_used": results.get('variables')
                },
                "indicator_summary": {k: v for k, v in results.items() if
                                      isinstance(v, (str, int, float, list, dict)) and k not in ['pca_object',
                                                                                                 'fa_object', 'scaler',
                                                                                                 'data']},
                "pre_test_results": st.session_state.pre_tests_results,
                "final_data_with_indicator": export_df.to_dict(orient='records')
            }

            # Clean up large/non-serializable items before final dump
            if 'data' in export_data_json.get('pre_test_results', {}):
                del export_data_json['pre_test_results']['data']
            if 'adequacy' in export_data_json.get('pre_test_results', {}) and 'anti_image' in \
                    export_data_json['pre_test_results']['adequacy']:
                del export_data_json['pre_test_results']['adequacy']['anti_image']

            json_string = json.dumps(export_data_json, indent=4, cls=NpEncoder)

            st.download_button(
                label="📥 Download JSON Report",
                data=json_string,
                file_name=f"indicator_report_{timestamp}.json",
                mime="application/json"
            )

        elif export_format == "PDF Report":
            st.info(
                "💡 Full PDF generation requires specialized libraries. This will download a comprehensive markdown text file that can be easily converted to PDF using common tools.")

            documentation = generate_comprehensive_documentation()

            st.download_button(
                label="📥 Download Report (Markdown)",
                data=documentation.encode('utf-8'),
                file_name=f"indicator_report_{timestamp}.md",
                mime="text/markdown",
            )


if __name__ == "__main__":

    main()
