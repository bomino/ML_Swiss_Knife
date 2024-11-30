import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import joblib
import base64
import io

def main():
    st.title("ML Swiss Army Knife")
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Tutorial", "Data Upload & Analysis", "Time Series Analysis", "Model Training", "Predictions"]
    )
    
    if page == "Data Upload & Analysis":
        data_analysis_page()
    elif page == "Time Series Analysis":
        time_series_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Predictions":
        prediction_page()
    elif page == "Tutorial":
        tutorial_page()

def data_analysis_page():
    st.header("Data Upload & Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            
            # Data Preview Section
            st.subheader("Data Preview")
            if st.checkbox("Show raw data"):
                st.write(df)
            else:
                st.write(df.head())
            
            # Basic Information Section
            st.subheader("Basic Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Dataset Dimensions:")
                st.write(f"- Rows: {df.shape[0]:,}")
                st.write(f"- Columns: {df.shape[1]:,}")
                
                memory_usage = df.memory_usage(deep=True).sum()
                st.write(f"Memory Usage: {memory_usage / 1024 / 1024:.2f} MB")
            
            with col2:
                st.write("Quick Statistics:")
                st.write(f"- Duplicate Rows: {df.duplicated().sum():,}")
                total_missing = df.isnull().sum().sum()
                st.write(f"- Total Missing Values: {total_missing:,}")
                st.write(f"- Missing Cells: {(total_missing/(df.shape[0]*df.shape[1])*100):.2f}%")
            
            # Data Types and Missing Values Section
            st.subheader("Columns Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Data Types:")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index.tolist(),
                    'Type': [str(dtype) for dtype in df.dtypes.values]  # Convert dtype objects to strings
                })
                st.dataframe(dtype_df.astype(str))  # Ensure all values are strings
            
            with col2:
                st.write("Missing Values:")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Values': df.isnull().sum(),
                    'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Missing Values'] > 0]
                if not missing_df.empty:
                    st.dataframe(missing_df)
                else:
                    st.write("No missing values found!")
            
            # Enhanced Data Analysis Section
            st.subheader("Enhanced Data Analysis")
            
            # Identify column types
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

            # Analysis Options
            analysis_type = st.radio(
                "Select Analysis Type",
                ["Numerical Analysis", "Categorical Analysis", "Temporal Analysis"],
                key="main_analysis_type"
            )

            if analysis_type == "Numerical Analysis" and numeric_cols:
                selected_num_col = st.selectbox("Select numerical column", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Basic stats
                    stats = df[selected_num_col].describe()
                    st.write("Basic Statistics:")
                    st.write(stats)
                
                with col2:
                    # Distribution plot
                    fig = px.histogram(df, x=selected_num_col, 
                                     title=f'Distribution of {selected_num_col}',
                                     marginal="box")  # Added box plot on the margin
                    st.plotly_chart(fig)

                # Add outlier analysis
                Q1 = df[selected_num_col].quantile(0.25)
                Q3 = df[selected_num_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[selected_num_col] < (Q1 - 1.5 * IQR)) | 
                            (df[selected_num_col] > (Q3 + 1.5 * IQR))][selected_num_col]
                
                st.write(f"Outliers detected: {len(outliers)} ({(len(outliers)/len(df)*100):.2f}% of data)")

            elif analysis_type == "Categorical Analysis" and categorical_cols:
                selected_cat_col = st.selectbox("Select categorical column", categorical_cols)
                
                # Add analysis type selection
                cat_analysis_type = st.radio(
                    "Select Category Analysis Type",
                    ["Basic Analysis", "Pareto Analysis"]
                )
                
                if cat_analysis_type == "Basic Analysis":
                    col1, col2 = st.columns(2)
                    with col1:
                        # Category statistics
                        unique_vals = df[selected_cat_col].nunique()
                        mode_val = df[selected_cat_col].mode()[0]
                        null_count = df[selected_cat_col].isnull().sum()
                        
                        st.write("Category Statistics:")
                        st.write(f"- Unique values: {unique_vals}")
                        st.write(f"- Most common: {mode_val}")
                        st.write(f"- Missing values: {null_count}")
                    
                    with col2:
                        # Top categories bar chart
                        value_counts = df[selected_cat_col].value_counts().head(10)
                        fig = px.bar(
                            x=value_counts.index, 
                            y=value_counts.values,
                            title=f'Top 10 Categories in {selected_cat_col}'
                        )
                        st.plotly_chart(fig)
                    
                    # Detailed value counts
                    st.write("Detailed Category Breakdown:")
                    value_counts_df = pd.DataFrame({
                        'Category': df[selected_cat_col].value_counts().index,
                        'Count': df[selected_cat_col].value_counts().values,
                        'Percentage': (df[selected_cat_col].value_counts(normalize=True) * 100).round(2)
                    })
                    st.dataframe(value_counts_df)
                
                else:  # Pareto Analysis
                    st.write("### Pareto Analysis")
                    st.write("The Pareto principle states that roughly 80% of effects come from 20% of causes.")
                    
                    # Add analysis method selection
                    analysis_method = st.radio(
                        "Select Analysis Method",
                        ["Count", "Sum by Value"],
                        key="pareto_analysis_method"
                    )
                    
                    if analysis_method == "Count":
                        # Calculate value counts
                        value_counts = df[selected_cat_col].value_counts()
                        total = value_counts.sum()
                        
                        pareto_df = pd.DataFrame({
                            'Category': value_counts.index,
                            'Count': value_counts.values,
                            'Percentage': (value_counts.values / total * 100).round(2)
                        })
                        
                        y_axis_title = 'Count'
                        
                    else:  # Sum by Value
                        # Let user select numerical column for sum
                        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
                        value_col = st.selectbox(
                            "Select numerical column for sum",
                            num_cols,
                            key="pareto_value_column"
                        )
                        
                        # Calculate sums by category
                        value_sums = df.groupby(selected_cat_col)[value_col].sum().sort_values(ascending=False)
                        total = value_sums.sum()
                        
                        pareto_df = pd.DataFrame({
                            'Category': value_sums.index,
                            'Sum': value_sums.values,
                            'Percentage': (value_sums.values / total * 100).round(2)
                        })
                        
                        y_axis_title = f'Sum of {value_col}'
                    
                    # Calculate cumulative percentage
                    pareto_df['Cumulative_Percentage'] = pareto_df['Percentage'].cumsum()
                    
                    # Create Pareto chart
                    fig = go.Figure()
                    
                    # Add bars
                    fig.add_trace(go.Bar(
                        x=pareto_df['Category'],
                        y=pareto_df['Count' if analysis_method == "Count" else 'Sum'],
                        name=y_axis_title,
                        marker_color='blue'
                    ))
                    
                    # Add cumulative line
                    fig.add_trace(go.Scatter(
                        x=pareto_df['Category'],
                        y=pareto_df['Cumulative_Percentage'],
                        name='Cumulative %',
                        marker_color='red',
                        yaxis='y2'
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'Pareto Chart - {selected_cat_col}',
                        yaxis=dict(title=y_axis_title),
                        yaxis2=dict(
                            title='Cumulative %',
                            overlaying='y',
                            side='right',
                            range=[0, 100]
                        ),
                        showlegend=True
                    )
                    
                    # Add 80% reference line
                    fig.add_hline(
                        y=80, line_dash="dash", line_color="green",
                        annotation_text="80% Reference Line",
                        yref='y2'
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Calculate and display key insights
                    categories_80 = len(pareto_df[pareto_df['Cumulative_Percentage'] <= 80])
                    total_categories = len(pareto_df)
                    
                    st.write("### Key Insights")
                    st.write(f"- Total number of categories: {total_categories}")
                    st.write(f"- Number of categories accounting for 80% of total: {categories_80}")
                    st.write(f"- Percentage of categories accounting for 80% of total: {(categories_80/total_categories*100):.1f}%")
                    
                    # Display detailed Pareto table
                    st.write("### Detailed Pareto Analysis")
                    st.dataframe(pareto_df)
                    
                    # Download section
                    st.write("### Download Analysis Results")
                    
                    # Create Excel buffer
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Write main analysis
                        pareto_df.to_excel(writer, sheet_name='Pareto Analysis', index=False)
                        
                        # Write insights to second sheet
                        insights_df = pd.DataFrame({
                            'Metric': [
                                'Total Categories',
                                'Categories for 80%',
                                'Percentage of Categories for 80%',
                                'Analysis Method',
                                'Analysis Date'
                            ],
                            'Value': [
                                total_categories,
                                categories_80,
                                f"{(categories_80/total_categories*100):.1f}%",
                                analysis_method,
                                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                            ]
                        })
                        insights_df.to_excel(writer, sheet_name='Insights', index=False)
                        
                        # Get the xlsxwriter workbook and worksheet objects
                        workbook = writer.book
                        worksheet = writer.sheets['Pareto Analysis']
                        
                        # Add formats
                        percent_format = workbook.add_format({'num_format': '0.00%'})
                        number_format = workbook.add_format({'num_format': '#,##0'})
                        
                        # Apply formats to columns
                        worksheet.set_column('C:C', 12, percent_format)  # Percentage column
                        worksheet.set_column('D:D', 12, percent_format)  # Cumulative Percentage
                        if analysis_method == "Sum by Value":
                            worksheet.set_column('B:B', 15, number_format)  # Sum column
                    
                    # Create download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv = pareto_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f'pareto_analysis_{selected_cat_col}.csv',
                            mime='text/csv'
                        )
                    
                    with col2:
                        # Excel download
                        st.download_button(
                            label="Download Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f'pareto_analysis_{selected_cat_col}.xlsx',
                            mime='application/vnd.ms-excel'
                        )

            elif analysis_type == "Temporal Analysis" and datetime_cols:
                if datetime_cols:
                    selected_date_col = st.selectbox("Select datetime column", datetime_cols)
                    
                    # Convert to datetime if not already
                    df[selected_date_col] = pd.to_datetime(df[selected_date_col])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Temporal Statistics:")
                        st.write(f"- Date range: {df[selected_date_col].min()} to {df[selected_date_col].max()}")
                        st.write(f"- Time span: {(df[selected_date_col].max() - df[selected_date_col].min()).days} days")
                    
                    with col2:
                        # Time series plot if there's a numeric column to plot
                        if numeric_cols:
                            selected_value_col = st.selectbox("Select value column for time series", numeric_cols)
                            fig = px.line(
                                df, 
                                x=selected_date_col, 
                                y=selected_value_col,
                                title=f'{selected_value_col} over Time'
                            )
                            st.plotly_chart(fig)
                else:
                    st.write("No datetime columns found in the dataset.")
            
            # Statistical Summary Section
            if numeric_cols:
                st.subheader("Statistical Summary")
                st.write(df[numeric_cols].describe())
            
            # Distribution Analysis Section
            st.subheader("Distribution Analysis")
            
            if numeric_cols:
                st.write("Numerical Distributions")
                selected_num_col = st.selectbox("Select numerical column", 
                                              numeric_cols,
                                              key="dist_num_select")
                fig = px.histogram(df, x=selected_num_col, 
                                 title=f'Distribution of {selected_num_col}')
                st.plotly_chart(fig)
            
            if categorical_cols:
                st.write("Categorical Distributions")
                selected_cat_col = st.selectbox("Select categorical column", 
                                              categorical_cols,
                                              key="dist_cat_select")
                value_counts = df[selected_cat_col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f'Distribution of {selected_cat_col}')
                st.plotly_chart(fig)
            
            # Correlation Analysis Section
            if len(numeric_cols) > 1:
                st.subheader("Correlation Analysis")
                correlation_matrix = df[numeric_cols].corr()
                fig = px.imshow(correlation_matrix,
                               labels=dict(color="Correlation"),
                               x=correlation_matrix.columns,
                               y=correlation_matrix.columns,
                               title="Correlation Matrix")
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error analyzing data: {str(e)}")
            st.write("Please check your data format and try again.")
    else:
        st.info("Please upload a CSV file to begin analysis.")


def time_series_page():
    """
    Comprehensive time series analysis page with visualizations and statistical analysis.
    """
    if 'data' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    st.header("Time Series Analysis")
    df = st.session_state['data']
    
    # Date column selection
    date_columns = df.select_dtypes(include=['datetime64', 'object']).columns
    if len(date_columns) == 0:
        st.error("No date columns found in the dataset!")
        return
        
    date_col = st.selectbox("Select date column", date_columns, key='ts_date_select')
    
    # Convert to datetime if not already
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        st.error(f"Error converting {date_col} to datetime: {str(e)}")
        return
    
    # Select target variable
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) == 0:
        st.error("No numeric columns found for analysis!")
        return
        
    target_col = st.selectbox("Select target variable", numeric_cols, key='ts_target_select')
    
    # Create time series DataFrame
    ts_df = df[[date_col, target_col]].copy()
    ts_df.set_index(date_col, inplace=True)
    ts_df.sort_index(inplace=True)
    
    # Basic Time Series Statistics
    st.subheader("Basic Time Series Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Date Range", f"{ts_df.index.min().strftime('%Y-%m-%d')} to {ts_df.index.max().strftime('%Y-%m-%d')}")
    with col2:
        st.metric("Time Span", f"{(ts_df.index.max() - ts_df.index.min()).days} days")
    with col3:
        st.metric("Number of Observations", len(ts_df))
    
    # Time Series Visualization Options
    viz_type = st.radio(
        "Select Visualization Type",
        ["Raw Data", "Resampled Data", "Rolling Statistics"],
        key='ts_viz_type'
    )
    
    if viz_type == "Raw Data":
        fig = px.line(ts_df, y=target_col, title=f'{target_col} Over Time')
        st.plotly_chart(fig)
        
    elif viz_type == "Resampled Data":
        freq = st.selectbox(
            "Select Resampling Frequency",
            ['D', 'W', 'M', 'Q', 'Y'],
            format_func=lambda x: {
                'D': 'Daily',
                'W': 'Weekly',
                'M': 'Monthly',
                'Q': 'Quarterly',
                'Y': 'Yearly'
            }[x],
            key='ts_resample_freq'
        )
        
        resampled = ts_df[target_col].resample(freq).agg(['mean', 'min', 'max'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=resampled.index, y=resampled['mean'], name='Mean'))
        fig.add_trace(go.Scatter(x=resampled.index, y=resampled['min'], name='Min', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=resampled.index, y=resampled['max'], name='Max', line=dict(dash='dash')))
        fig.update_layout(title=f'{target_col} - Resampled to {freq} Frequency')
        st.plotly_chart(fig)
        
    else:  # Rolling Statistics
        window = st.slider("Select Rolling Window", 2, 100, 7, key='ts_rolling_window')
        rolling = ts_df[target_col].rolling(window=window)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_df.index, y=ts_df[target_col], name='Raw Data'))
        fig.add_trace(go.Scatter(x=ts_df.index, y=rolling.mean(), name=f'{window}-period Moving Average'))
        fig.add_trace(go.Scatter(x=ts_df.index, y=rolling.std(), name=f'{window}-period Standard Deviation'))
        fig.update_layout(title=f'{target_col} - Rolling Statistics (Window: {window})')
        st.plotly_chart(fig)
    
    # Advanced Analysis Section
    st.subheader("Advanced Time Series Analysis")
    
    if len(ts_df) < 2:
        st.warning("Not enough data points for advanced analysis!")
        return
        
    try:
        # Decomposition Period Selection
        st.info("Select the decomposition period based on your data frequency (e.g., 12 for monthly data, 4 for quarterly, 7 for daily)")
        period = st.slider(
            "Decomposition Period",
            min_value=2,
            max_value=52,
            value=12,
            help="Period for seasonal decomposition. Choose based on your data frequency.",
            key='decomp_period_slider'
        )
        
        # Perform decomposition
        decomposition = seasonal_decompose(
            ts_df[target_col].fillna(method='ffill'),  # Fill NA values
            period=period,
            extrapolate_trend='freq'  # Handle missing values in trend
        )
        
        # Create decomposition plot
        fig = make_subplots(
            rows=4, 
            cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Add components to plot
        fig.add_trace(
            go.Scatter(x=ts_df.index, y=ts_df[target_col], name='Original'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=ts_df.index, y=decomposition.trend, name='Trend'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=ts_df.index, y=decomposition.seasonal, name='Seasonal'),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=ts_df.index, y=decomposition.resid, name='Residual'),
            row=4, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Time Series Decomposition Analysis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Component Analysis
        st.subheader("Component Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Trend Analysis")
            trend_data = decomposition.trend.dropna()
            trend_changes = trend_data.diff().dropna()
            trend_direction = "Upward" if trend_changes.mean() > 0 else "Downward"
            
            st.write(f"Overall Trend Direction: {trend_direction}")
            st.write(f"Average Trend Change: {trend_changes.mean():.2f}")
            st.write(f"Trend Volatility (std): {trend_changes.std():.2f}")
        
        with col2:
            st.write("Seasonality Analysis")
            seasonal_data = decomposition.seasonal.dropna()
            st.write(f"Seasonal Range: {seasonal_data.max():.2f} to {seasonal_data.min():.2f}")
            st.write(f"Seasonal Strength: {(seasonal_data.std() / ts_df[target_col].std() * 100):.2f}%")
        
        # Component Statistics
        st.subheader("Component Statistics")
        summary_stats = pd.DataFrame({
            'Original': ts_df[target_col].describe(),
            'Trend': decomposition.trend.describe(),
            'Seasonal': decomposition.seasonal.describe(),
            'Residual': decomposition.resid.describe()
        }).round(2)
        
        st.dataframe(summary_stats)
        
        # Stationarity Analysis
        st.subheader("Stationarity Analysis")
        result = adfuller(ts_df[target_col].dropna())
        
        st.write("Augmented Dickey-Fuller Test Results:")
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
        is_stationary = result[1] < 0.05
        st.write(f"Series is {'stationary' if is_stationary else 'non-stationary'} with 95% confidence")
        
        if not is_stationary:
            st.info("Consider differencing the series to achieve stationarity.")
            
    except Exception as e:
        st.error(f"Error during advanced analysis: {str(e)}")
        st.write("Troubleshooting tips:")
        st.write("1. Check if your data has a clear seasonal pattern")
        st.write("2. Try adjusting the decomposition period")
        st.write("3. Ensure your data is regularly spaced in time")
        st.write("4. Make sure you have enough data points for the selected period")

def model_training_page():
    if 'data' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    st.header("Model Training")
    
    df = st.session_state['data']
    
    # Data preprocessing
    st.subheader("Data Preprocessing")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Display column types
    st.write("Numeric columns:", list(numeric_cols))
    st.write("Categorical columns:", list(categorical_cols))
    
    # Let user select categorical columns to encode
    selected_categorical = st.multiselect(
        "Select categorical columns to encode",
        options=list(categorical_cols),
        default=list(categorical_cols)[:2]  # Default select first two categorical columns
    )
    
    # Create dummy variables for selected categorical columns
    df_encoded = df.copy()
    if selected_categorical:
        df_encoded = pd.get_dummies(df_encoded, columns=selected_categorical, drop_first=True)
    
    # Update available features with both numeric and encoded columns
    available_features = df_encoded.columns
    
    if len(available_features) < 2:
        st.error("Not enough features available for modeling (minimum 2 required)!")
        return
    
    # Feature and target selection
    st.subheader("Feature Selection")
    target = st.selectbox("Select target variable", available_features)
    
    # Create list of features excluding target
    potential_features = [col for col in available_features if col != target]
    
    # Calculate default number of features (minimum of 5 or available features)
    default_n_features = min(5, len(potential_features))
    
    features = st.multiselect(
        "Select features",
        options=potential_features,
        default=potential_features[:default_n_features],
        key="feature_selector"
    )
    
    if not features:
        st.warning("Please select at least one feature!")
        return
    
    # Prepare data
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Show correlation matrix
    st.subheader("Feature Correlations")
    corr_matrix = X.corr(numeric_only=True)
    fig = px.imshow(corr_matrix, 
                    title='Feature Correlation Matrix',
                    labels=dict(color="Correlation"))
    st.plotly_chart(fig)
    
    # Optional feature selection based on correlation
    correlation_threshold = st.slider(
        "Remove highly correlated features above threshold", 
        0.0, 1.0, 0.9, 
        help="Features with correlation above this threshold will be removed to prevent multicollinearity"
    )
    
    # Remove highly correlated features
    features_to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                colname = corr_matrix.columns[i]
                features_to_drop.add(colname)
    
    X = X.drop(columns=list(features_to_drop))
    if len(features_to_drop) > 0:
        st.write(f"Removed {len(features_to_drop)} highly correlated features: {', '.join(features_to_drop)}")
    
    # Model training section
    st.subheader("Train-Test Split")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2)
    
    # Model selection
    st.subheader("Model Selection")
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel='rbf')
    }
    
    selected_models = st.multiselect(
        "Select models to train",
        options=list(models.keys()),
        default=["Ridge Regression"]  # Set a default model
    )
    
    if not selected_models:
        st.warning("Please select at least one model!")
        return

    if st.button("Train Models"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            results = {}
            
            with st.spinner('Training models...'):
                for model_name in selected_models:
                    model = models[model_name]
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results[model_name] = {
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2
                    }
                    
                    # Save model and scaler
                    st.session_state[f'model_{model_name}'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['features'] = features
            
            # Display results
            st.subheader("Model Comparison")
            results_df = pd.DataFrame(results).round(4)
            st.write(results_df)
            
            # Plot actual vs predicted
            best_model_name = max(results.keys(), key=lambda x: results[x]["R2"])
            best_model = models[best_model_name]
            y_pred = best_model.predict(X_test_scaled)
            
            fig = go.Figure()
            fig.add_scatter(x=y_test, y=y_pred, mode='markers', name='Predictions')
            fig.add_scatter(x=y_test, y=y_test, mode='lines', name='Perfect Prediction')
            fig.update_layout(
                title=f'Actual vs Predicted ({best_model_name})',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values'
            )
            st.plotly_chart(fig)
            
            # Feature importance for tree-based models
            if best_model_name in ["Random Forest", "Gradient Boosting"]:
                st.subheader("Feature Importance")
                importances = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_model.feature_importances_
                })
                importances = importances.sort_values('importance', ascending=False)
                
                fig = px.bar(importances, x='feature', y='importance',
                           title=f'Feature Importance ({best_model_name})')
                st.plotly_chart(fig)
                
            st.success('Models trained successfully!')
            
        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
            st.write("Try adjusting the feature selection or preprocessing parameters.")

def prediction_page():
    if 'data' not in st.session_state:
        st.warning("Please train models first!")
        return
    
    st.header("Make Predictions")
    
    prediction_type = st.radio("Select prediction type", ["Single Prediction", "Batch Prediction"])
    
    if prediction_type == "Single Prediction":
        make_single_prediction()
    else:
        make_batch_prediction()

def make_single_prediction():
    features = st.session_state.get('features')
    if not features:
        st.warning("No trained models available!")
        return
    
    # Create input fields for each feature
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Enter value for {feature}")
    
    if st.button("Predict"):
        # Prepare input data
        input_df = pd.DataFrame([input_data])
        input_scaled = st.session_state['scaler'].transform(input_df)
        
        # Make predictions with all available models
        st.subheader("Predictions")
        for model_name in [key.replace('model_', '') for key in st.session_state.keys() if key.startswith('model_')]:
            model = st.session_state[f'model_{model_name}']
            prediction = model.predict(input_scaled)[0]
            st.write(f"{model_name}: {prediction:.2f}")

def make_batch_prediction():
    uploaded_file = st.file_uploader("Upload CSV file for predictions", type="csv")
    
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        if st.button("Make Predictions"):
            # Verify features
            missing_features = set(st.session_state['features']) - set(input_df.columns)
            if missing_features:
                st.error(f"Missing features in input data: {missing_features}")
                return
            
            # Prepare input data
            input_scaled = st.session_state['scaler'].transform(input_df[st.session_state['features']])
            
            # Make predictions with all available models
            predictions = {}
            for model_name in [key.replace('model_', '') for key in st.session_state.keys() if key.startswith('model_')]:
                model = st.session_state[f'model_{model_name}']
                predictions[f'{model_name}_prediction'] = model.predict(input_scaled)
            
            # Create results DataFrame
            results_df = pd.concat([input_df, pd.DataFrame(predictions)], axis=1)
            
            # Display results
            st.write(results_df)
            
            # Download link
            csv = results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download predictions CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)


def tutorial_page():
    st.title("Model Training Tutorial")
    
    # Sidebar for navigation through tutorial sections
    section = st.sidebar.radio(
        "Tutorial Sections",
        ["Introduction",
         "Understanding Your Data",
         "Data Preprocessing",
         "Feature Selection",
         "Model Selection Guide",
         "Forecasting Models",
         "Best Practices",
         "Troubleshooting"]
    )
    
    if section == "Introduction":
        #st.header("Welcome to the Model Training Tutorial")
        
        st.markdown("""
        This tutorial will guide you through the process of using the Model Training page effectively.
        Use the sidebar to navigate through different sections of the tutorial.
        
        ### What You'll Learn
        - How to understand and preprocess your data
        - How to select the right features for your model
        - How to choose and train appropriate models
        - Best practices and troubleshooting tips
        """)
        
        # Show sample dataset info if available
        if 'data' in st.session_state:
            st.subheader("Your Current Dataset Overview")
            df = st.session_state['data']
            st.write("Number of rows:", len(df))
            st.write("Number of columns:", len(df.columns))
            st.write("Columns:", list(df.columns))
        else:
            st.info("💡 Upload your data in the Data Upload page to see a personalized overview.")
            
    elif section == "Understanding Your Data":
        st.header("Understanding Your Data")
        
        st.markdown("""
        ### Types of Data Columns
        
        📊 **Numeric Columns**
        - Contain numerical values (e.g., 'Spend')
        - Can be used directly in models
        - Example: Transaction amounts, counts, measurements
        
        📝 **Categorical Columns**
        - Contain text or categories (e.g., 'CategoryLevel1', 'TransactionGroup')
        - Need to be encoded before using in models
        - Example: Categories, status, groups
        """)
        
        # Display actual data types if data is available
        if 'data' in st.session_state:
            df = st.session_state['data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Numeric Columns")
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                for col in numeric_cols:
                    st.write(f"- {col}")
                    
            with col2:
                st.subheader("Categorical Columns")
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    st.write(f"- {col}")
                    
            # Show sample of unique values
            st.subheader("Sample Categories")
            selected_col = st.selectbox("Select a categorical column to see unique values:", categorical_cols)
            st.write(f"Unique values in {selected_col}:", df[selected_col].nunique())
            st.write("Sample values:", list(df[selected_col].unique()[:5]))
            
        else:
            st.info("💡 Upload your data to see specific information about your columns.")
            
    elif section == "Data Preprocessing":
        st.header("Data Preprocessing")
        
        st.markdown("""
        ### Categorical Column Encoding
        
        #### What to Encode
        
        ✅ **Good Candidates for Encoding:**
        - Columns with meaningful categories (e.g., 'CategoryLevel1')
        - Columns with reasonable number of unique values
        - Columns that could influence your target variable
        
        ❌ **Think Twice Before Encoding:**
        - Columns with too many unique values (e.g., IDs, names)
        - Columns with single value (no variation)
        - Free text columns
        
        #### Encoding Process
        1. Select categorical columns to encode
        2. System creates binary columns (one-hot encoding)
        3. Original categorical column is replaced with binary columns
        
        #### Example of One-Hot Encoding:
        """)
        
        # Create a sample dataframe for demonstration
        import pandas as pd
        sample_df = pd.DataFrame({
            'Category': ['Facilities', 'Marketing', 'HR', 'Facilities'],
            'Spend': [1000, 2000, 1500, 1200]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before Encoding")
            st.write(sample_df)
            
        with col2:
            st.subheader("After Encoding")
            st.write(pd.get_dummies(sample_df, columns=['Category']))
            
        st.markdown("""
        ### 💡 Tips for Preprocessing
        1. Start with fewer categories to test the model
        2. Monitor the number of features created
        3. Consider combining rare categories
        """)
        
    elif section == "Feature Selection":
        st.header("Feature Selection")
        
        st.markdown("""
        ### Selecting Target Variable
        
        Your target variable is what you want to predict. Choose based on your business goal:
        
        #### For Spend Analysis
        - Target: 'Spend'
        - Goal: Predict spending based on categories
        
        #### For Category Prediction
        - Target: 'CategoryLevel1' or 'Final Category'
        - Goal: Automatically categorize transactions
        
        ### Selecting Features
        
        #### Best Practices for Feature Selection:
        1. **Start Small**: Begin with 3-5 most relevant features
        2. **Use Domain Knowledge**: Choose features that logically influence your target
        3. **Check Correlations**: Use the correlation matrix to identify:
           - Strongly correlated features (potential redundancy)
           - Features with strong correlation to target
        """)
        
        # Interactive feature selection example
        st.subheader("Interactive Feature Selection Example")
        
        if 'data' in st.session_state:
            df = st.session_state['data']
            
            # Create correlation matrix
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                st.write("Correlation Matrix for Numeric Columns:")
                st.write(corr_matrix)
                
                st.markdown("""
                #### How to Read the Correlation Matrix:
                - Values close to 1: Strong positive correlation
                - Values close to -1: Strong negative correlation
                - Values close to 0: Little to no correlation
                """)
            else:
                st.info("Need more numeric columns to show correlations.")
                
        else:
            st.info("💡 Upload your data to see correlation analysis.")

    elif section == "Model Selection Guide":
        st.header("Model Selection Guide")
        
        st.markdown("""
        ### Types of Models
        
        #### 1. Traditional Machine Learning Models
        
        👉 **Ridge Regression**
        - Good baseline model
        - Works well with encoded categorical variables
        - Best for linear relationships
        
        👉 **Lasso Regression**
        - Similar to Ridge, but can eliminate irrelevant features
        - Good for feature selection
        
        👉 **Random Forest**
        - Handles non-linear relationships
        - Can capture complex patterns
        - Provides feature importance
        
        👉 **Gradient Boosting**
        - Often best performance
        - Can overfit if not tuned properly
        - Good for both regression and classification
        
        #### 2. Advanced Models
        
        👉 **XGBoost**
        - Very fast and efficient
        - Handles missing values well
        - Great for large datasets
        - Built-in regularization
        
        👉 **LightGBM**
        - Faster than XGBoost
        - Lower memory usage
        - Great for large datasets
        - Good with categorical features
        
        ### When to Use Each Model
        
        Choose based on your:
        1. Data size
        2. Feature types
        3. Performance requirements
        4. Training time constraints
        """)
        
        # Interactive model selector
        st.subheader("Model Selection Helper")
        
        data_size = st.radio("How much data do you have?", 
                            ["Small (< 1000 rows)", 
                             "Medium (1000-10000 rows)", 
                             "Large (> 10000 rows)"])
        
        feature_type = st.radio("What type of features do you have?",
                               ["Mostly numeric",
                                "Mostly categorical",
                                "Mixed"])
        
        priority = st.radio("What's your main priority?",
                           ["Accuracy",
                            "Training speed",
                            "Model interpretability"])
        
        # Show recommendation based on selections
        st.subheader("Recommended Model")
        if data_size == "Small (< 1000 rows)":
            if priority == "Accuracy":
                st.write("➡️ Try Random Forest or XGBoost")
            elif priority == "Training speed":
                st.write("➡️ Try Ridge Regression")
            else:  # interpretability
                st.write("➡️ Try Lasso Regression")
        elif data_size == "Large (> 10000 rows)":
            if priority == "Accuracy":
                st.write("➡️ Try LightGBM or XGBoost")
            elif priority == "Training speed":
                st.write("➡️ Try LightGBM")
            else:  # interpretability
                st.write("➡️ Try Random Forest")
                
    elif section == "Forecasting Models":
        st.header("Forecasting Models")
        
        st.markdown("""
        ### Specialized Forecasting Models
        
        #### 1. Prophet (by Facebook)
        👉 **Best for:**
        - Daily/weekly data with seasonal patterns
        - Data with missing values
        - Data with outliers
        - Multiple seasonality patterns
        
        👉 **Features:**
        - Handles holidays automatically
        - Detects seasonality
        - Provides uncertainty intervals
        - Very robust to missing data
        
        #### 2. SARIMA
        👉 **Best for:**
        - Clear seasonal patterns
        - Regular time intervals
        - Stationary data (or easily made stationary)
        
        👉 **Features:**
        - Explicit modeling of trends
        - Handles seasonal components
        - Good for short-term forecasts
        
        ### Choosing the Right Forecasting Model
        
        #### When to Use Prophet
        - You have daily/weekly data
        - Your data has multiple seasonal patterns
        - You have missing values or outliers
        - You need automatic handling of holidays
        
        #### When to Use SARIMA
        - Your data is already stationary
        - You have clear, single seasonality
        - You have regular time intervals
        - You need interpretable components
        
        #### When to Use XGBoost/LightGBM for Forecasting
        - You have many external features
        - Your patterns are complex
        - You need very high accuracy
        """)
        
        # Interactive forecasting example
        st.subheader("Forecasting Model Selection Helper")
        
        data_frequency = st.selectbox("What's your data frequency?",
                                    ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"])
        
        seasonality = st.multiselect("What types of seasonality do you have?",
                                   ["Daily", "Weekly", "Monthly", "Yearly", "None"],
                                   default=["None"])
        
        missing_values = st.radio("Do you have missing values?",
                                ["Yes", "No"])
        
        external_features = st.radio("Do you have external features (like weather, promotions)?",
                                   ["Yes", "No"])
        
        # Show recommendation
        st.subheader("Recommended Forecasting Model")
        
        if len(seasonality) > 1 or missing_values == "Yes":
            st.write("➡️ Prophet is recommended because:")
            if len(seasonality) > 1:
                st.write("- Can handle multiple seasonality patterns")
            if missing_values == "Yes":
                st.write("- Robust to missing values")
                
        elif external_features == "Yes":
            st.write("➡️ XGBoost/LightGBM is recommended because:")
            st.write("- Can incorporate external features")
            st.write("- Handles complex patterns well")
            
        elif "None" not in seasonality and data_frequency in ["Monthly", "Quarterly"]:
            st.write("➡️ SARIMA is recommended because:")
            st.write("- Good for regular seasonal patterns")
            st.write("- Works well with monthly/quarterly data")
            
        # Add example configuration for each model
        st.subheader("Model Configuration Examples")
        
        model_example = st.selectbox("Select model to see example configuration",
                                   ["Prophet", "SARIMA", "XGBoost"])
        
        if model_example == "Prophet":
            st.code("""
# Prophet Configuration Example
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)
            """)
        elif model_example == "SARIMA":
            st.code("""
# SARIMA Configuration Example
model = SARIMAX(
    order=(1, 1, 1),           # (p, d, q)
    seasonal_order=(1, 1, 1, 12)  # (P, D, Q, s)
)
            """)
        elif model_example == "XGBoost":
            st.code("""
# XGBoost Configuration Example
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
            """)
            
        # Add performance metrics explanation
        st.subheader("Understanding Forecasting Metrics")
        
        metrics = {
            "RMSE (Root Mean Square Error)": "Average prediction error (lower is better)",
            "MAPE (Mean Absolute Percentage Error)": "Average percentage error (lower is better)",
            "MAE (Mean Absolute Error)": "Average absolute error (lower is better)",
            "R² Score": "Proportion of variance explained (higher is better)"
        }
        
        for metric, description in metrics.items():
            st.markdown(f"**{metric}**")
            st.write(description)       

        
    elif section == "Best Practices":
        st.header("Best Practices")
        
        st.markdown("""
        ### 1. Data Preparation
        - Clean your data before training
        - Handle missing values appropriately
        - Remove or fix outliers if necessary
        
        ### 2. Feature Engineering
        - Start with basic features
        - Add engineered features gradually
        - Document feature importance
        
        ### 3. Model Training Process
        - Start simple, add complexity as needed
        - Use cross-validation for robust results
        - Monitor for overfitting
        
        ### 4. Model Evaluation
        - Compare multiple models
        - Use appropriate metrics
        - Consider business context
        
        ### 5. Iteration Tips
        1. Make one change at a time
        2. Document changes and results
        3. Save best performing models
        """)
        
        # Add interactive examples if data is available
        if 'data' in st.session_state:
            st.subheader("Your Data Best Practices")
            df = st.session_state['data']
            
            # Show some basic statistics
            st.write("Dataset Statistics:")
            st.write(df.describe())
            
            # Show missing values if any
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.write("Missing Values:")
                st.write(missing[missing > 0])
                
    elif section == "Troubleshooting":
        st.header("Troubleshooting Common Issues")
        
        st.markdown("""
        ### Common Problems and Solutions
        
        #### 1. "Not Enough Features" Error
        **Problem**: Model training fails due to insufficient features
        **Solutions**:
        - Encode more categorical columns
        - Add relevant features
        - Check for data loading issues
        
        #### 2. Poor Model Performance
        **Problem**: Model predictions are not accurate enough
        **Solutions**:
        - Add more relevant features
        - Try different model types
        - Check for data quality issues
        - Consider feature engineering
        
        #### 3. Overfitting
        **Problem**: Model performs well on training data but poorly on test data
        **Solutions**:
        - Reduce number of features
        - Use simpler model
        - Increase training data
        - Add regularization
        
        #### 4. Long Training Time
        **Problem**: Model takes too long to train
        **Solutions**:
        - Reduce number of features
        - Use smaller data sample for testing
        - Choose simpler model
        - Optimize feature selection
        
        ### Getting Help
        - Check error messages carefully
        - Review data preprocessing steps
        - Verify feature engineering steps
        - Document changes and results
        """)
        
        # Add interactive troubleshooting tool
        st.subheader("Interactive Troubleshooting")
        
        issue = st.selectbox(
            "What issue are you experiencing?",
            ["Select an issue...",
             "Not enough features",
             "Poor model performance",
             "Overfitting",
             "Long training time"]
        )
        
        if issue == "Not enough features":
            st.markdown("""
            ### Steps to Resolve:
            1. Check your categorical column selection
            2. Verify encoding process
            3. Consider adding more features
            """)
        elif issue == "Poor model performance":
            st.markdown("""
            ### Steps to Resolve:
            1. Review feature selection
            2. Try different models
            3. Check data quality
            """)
        elif issue == "Overfitting":
            st.markdown("""
            ### Steps to Resolve:
            1. Reduce feature count
            2. Increase regularization
            3. Get more training data
            """)
        elif issue == "Long training time":
            st.markdown("""
            ### Steps to Resolve:
            1. Reduce feature count
            2. Use data sample
            3. Choose simpler model
            """)

    # Add a help button in the sidebar
    with st.sidebar:
        st.markdown("---")
        if st.button("Need Help?"):
            st.info("""
            📧 Contact support for:
            - Technical issues
            - Feature requests
            - Bug reports
            """)            

if __name__ == "__main__":
    main()