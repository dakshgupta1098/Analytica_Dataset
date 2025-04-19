import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from statsmodels.tools.sm_exceptions import PerfectSeparationError # For specific Logit error
import warnings
import traceback # For detailed error logging if needed

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Mental Health Analytics", page_icon="")

# Suppress common warnings (use cautiously)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='seaborn')
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")
warnings.filterwarnings("ignore", message="The figure layout has changed to tight") # Matplotlib/Seaborn

# --- Custom CSS ---
st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #1f2937;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #d1d5db; /* Light gray text */
        }
        .stTabs [aria-selected="true"] {
            background-color: #7979f8; /* Purple for selected tab */
            color: white;
            font-weight: bold;
        }
        .metric-card {
            background-color: #1f2937;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px; /* Add space below cards */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            border: 1px solid #374151; /* Subtle border */
            height: 100px; /* Fixed height for alignment */
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-label { font-size: 14px; color: #9ca3af; /* Slightly darker gray */ margin-bottom: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #7979f8; line-height: 1.2; }
        .subtitle { font-size: 18px; color: #d1d5db; margin-top: -5px; margin-bottom: 15px; }
        hr { border-top: 1px solid #374151; margin: 20px 0px; } /* Style horizontal rule */
        h1, h2, h3, h4, h5, h6 { color: #e5e7eb; } /* Lighter headings */
        .stPlotlyChart { border-radius: 8px; overflow: hidden; } /* Style Plotly charts */
    </style>
""", unsafe_allow_html=True)

# --- Robust Data Loading and Initial Cleaning ---
@st.cache_data
def load_and_prepare_data(file_path="Analytica Dataset.csv"):
    """Loads, cleans, and prepares the dataset."""
    st.write(f"Attempting to load data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        st.success(f"Successfully loaded data. Found {len(df)} rows and {len(df.columns)} columns initially.")
    except FileNotFoundError:
        st.error(f"Fatal Error: File not found at '{file_path}'. Please ensure the file exists in the correct location.")
        return None
    except pd.errors.EmptyDataError:
        st.error(f"Fatal Error: The file '{file_path}' is empty.")
        return None
    except Exception as e:
        st.error(f"Fatal Error during file loading: {e}")
        return None

    # --- Basic Cleaning ---
    df.columns = df.columns.str.strip() # Clean column names first
    st.write("Cleaned Column Names:", df.columns.tolist())

    # Drop ID column if it exists (common variations)
    id_cols_to_drop = [col for col in df.columns if col.lower() == 'id']
    if id_cols_to_drop:
        df.drop(columns=id_cols_to_drop, inplace=True)
        st.write(f"Dropped ID column(s): {id_cols_to_drop}")

    # Remove specific unwanted values in 'City' (More robustly)
    values_to_remove_city = [
        "'Less Delhi'", "3", "'Less than 5 Kalyan'", 'Saanvi', 'M.Tech', 'Bhavna', 'Mira',
        'Harsha', 'Vaanya', 'Gaurav', 'Harsh', 'Reyansh', 'Kibara', 'Rashi', 'ME',
        'M.Com', 'Nalyan', 'Mihir', 'Nalini', 'Nandini', 'Khaziabad', "0"
    ]
    if 'City' in df.columns:
        initial_rows = len(df)
        df['City_str'] = df['City'].astype(str).str.strip() # Work on string version
        df = df[~df['City_str'].isin(values_to_remove_city)]
        df.drop(columns=['City_str'], inplace=True) # Remove temporary column
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            st.write(f"Removed {rows_removed} rows based on specific 'City' values.")
    else:
        st.warning("Column 'City' not found for specific value removal.")

    # --- Define Column Types (Based on your descriptions) ---
    # IMPORTANT: Verify these lists match your actual data columns after cleaning
    expected_numeric_cols = [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction',
        'Work/Study Hours', 'Financial Stress'
    ]
    expected_categorical_cols = [
        'Gender', 'City', 'Profession', 'Sleep Duration',
        'Dietary Habits', 'Degree'
    ]
    expected_yes_no_cols = [
        'Have you ever had suicidal thoughts ?',
        'Family History of Mental Illness',
        'Depression' # Target variable
    ]
    target_col = 'Depression' # Explicitly define target

    # --- Type Conversion and Imputation ---
    st.write("--- Data Type Conversion and Imputation ---")
    actual_numeric_cols = []
    for col in expected_numeric_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                # st.write(f"Numeric '{col}': Converted (originally {original_dtype}). Imputed {nan_count} NaNs with median ({median_val:.2f}).")
            else:
                # st.write(f"Numeric '{col}': Converted (originally {original_dtype}). No NaNs found.")
                actual_numeric_cols.append(col)
        else:
            st.warning(f"Expected numeric column '{col}' not found.")

    actual_yes_no_cols = []
    for col in expected_yes_no_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = df[col].astype(str).str.strip().str.lower()
            mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'true': 1, 'false': 0}
            df[col] = df[col].map(mapping) # Apply mapping
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                try:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                except IndexError:
                    mode_val = 0 # Fallback if mode calculation fails
                df[col].fillna(mode_val, inplace=True)
                # st.write(f"Yes/No '{col}': Mapped & Int Converted (originally {original_dtype}). Imputed {nan_count} NaNs/unmapped with mode ({mode_val}).")
            else:
                # st.write(f"Yes/No '{col}': Mapped & Int Converted (originally {original_dtype}). No NaNs/unmapped found.")
                df[col] = df[col].astype(int) # Final conversion to integer
                actual_yes_no_cols.append(col)
        else:
            st.warning(f"Expected Yes/No column '{col}' not found.")

    # Target Variable Check (Crucial!)
    if target_col in actual_yes_no_cols:
        if not df[target_col].isin([0, 1]).all():
            st.error(f"Fatal Error: Target column '{target_col}' contains values other than 0 or 1 after cleaning. Cannot proceed with modeling.")
            st.dataframe(df[~df[target_col].isin([0, 1])]) # Show problematic rows
            return None
        else:
            st.write(f"Target column '{target_col}' successfully validated as binary (0/1).")
    else:
         st.error(f"Fatal Error: Target column '{target_col}' not found or not processed correctly. Check column names and types.")
         return None


    actual_categorical_cols = []
    for col in expected_categorical_cols:
        if col in df.columns:
            original_dtype = df[col].dtype
            nan_count = df[col].isnull().sum()
            # Convert to string first for consistent imputation/categorization
            df[col] = df[col].astype(str)
            if nan_count > 0:
                df[col].replace(['nan', 'NaN', 'None', '', ' '], 'Unknown', inplace=True) # Handle various forms of missing explicitly
                df[col].fillna('Unknown', inplace=True) # Catch any remaining standard NaNs
                # st.write(f"Categorical '{col}': Type Str (originally {original_dtype}). Imputed {nan_count} missing values with 'Unknown'.")
            # Convert to category type after imputation
            df[col] = df[col].astype('category')
            actual_categorical_cols.append(col)
        else:
            st.warning(f"Expected categorical column '{col}' not found.")

    # --- Feature Engineering: Age Range ---
    if 'Age' in actual_numeric_cols:
        max_age = df['Age'].max()
        # Adjusted bins to be more inclusive, handle potential float ages
        bins = [0, 17, 24, 34, 44, 54, 64, 100]
        labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        try:
            df['AgeRange'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True) # Use right=True typically
            if df['AgeRange'].isnull().any():
                df['AgeRange'].fillna('Unknown', inplace=True)
            df['AgeRange'] = df['AgeRange'].astype('category')
            actual_categorical_cols.append('AgeRange') # Add to list if created
            st.write("Created 'AgeRange' category from 'Age'.")
        except Exception as e:
            st.warning(f"Could not create 'AgeRange': {e}")
    else:
        st.warning("Numeric column 'Age' not found or not processed, cannot create 'AgeRange'.")


    st.success("--- Data Loading and Preprocessing Complete ---")
    # st.write("Final Data Info:")
    # st.write(df.info())
    # st.write("Sample Data:")
    # st.dataframe(df.head())

    # Final Check: Ensure dataframe is not empty
    if df.empty:
        st.error("Fatal Error: Dataframe became empty after cleaning and preprocessing.")
        return None

    return df, actual_numeric_cols, actual_categorical_cols, actual_yes_no_cols, target_col

# --- Load Data ---
loaded_data = load_and_prepare_data() # Use default path

# --- Main App Logic ---
if loaded_data:
    df, actual_numeric_cols, actual_categorical_cols, actual_yes_no_cols, target_col = loaded_data
    st.write(f"Proceeding with dashboard using {len(df)} processed rows.")

    # --- Sidebar Filters ---
    st.sidebar.header(" Filter Dashboard Views")
    # Use only categorical columns that actually exist for filtering
    filter_cols = [col for col in ['Gender', 'City', 'Profession', 'AgeRange', 'Dietary Habits', 'Degree', 'Sleep Duration'] if col in actual_categorical_cols]

    filters = {}
    if not filter_cols:
        st.sidebar.warning("No categorical columns available for filtering.")
    else:
        for col in filter_cols:
            # Convert unique values to string, sort, and handle potential errors
            try:
                unique_values = df[col].dropna().unique().astype(str)
                options = sorted(list(unique_values))
                if options:
                    filters[col] = st.sidebar.multiselect(f"Filter by {col}", options, default=options)
                else:
                     st.sidebar.info(f"No unique values found for '{col}' to filter.")
            except Exception as e:
                st.sidebar.error(f"Error setting up filter for '{col}': {e}")


    # --- Filter Data for Visualizations (Carefully) ---
    df_filtered = df.copy()
    if filters:
        for col, selected_options in filters.items():
            if selected_options and col in df_filtered.columns:
                # Ensure we compare string representations if options are strings
                try:
                    df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected_options)]
                except Exception as e:
                     st.warning(f"Could not apply filter for '{col}': {e}")
                     # Continue without this filter if it fails
            # else: # Handle case where filter is defined but no options selected (means select all)
            #     pass # Keep all data for this filter if nothing is selected

    if df_filtered.empty and not df.empty: # Only warn if filters caused emptiness
        st.warning("Warning: The current filter combination results in no data for visualizations.")
        # Optionally, reset to full data for viewing: df_filtered = df.copy()

    # --- Create Tabs ---
    tab_titles = ["Overview", " Depression Analysis", " Sleep Patterns", " Dietary Analysis", " Advanced Analytics"]
    tabs = st.tabs(tab_titles)

    # ------------------ OVERVIEW TAB ------------------
    with tabs[0]:
        st.markdown("<h1 style='text-align: center; color: #7979f8;'>Mental Health & Lifestyle Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align: center;'>Exploring relationships between lifestyle and mental well-being</p><hr>", unsafe_allow_html=True)

        # Use df_filtered for overview KPIs
        display_df = df if df_filtered.empty else df_filtered # Use original df if filtering leads to empty

        col1, col2, col3, col4 = st.columns(4)

        # --- Calculate KPIs ---
        total_responses = len(display_df)
        depression_rate = 100 * display_df[target_col].mean() if target_col in display_df.columns else 0
        most_common_sleep = "N/A"
        if 'Sleep Duration' in display_df.columns and not display_df['Sleep Duration'].empty:
            try:
                most_common_sleep = display_df['Sleep Duration'].mode()[0]
            except IndexError: pass # Handle empty mode
        avg_financial_stress = display_df['Financial Stress'].mean() if 'Financial Stress' in display_df.columns else None
        stress_display = f"{avg_financial_stress:.1f}" if avg_financial_stress is not None and not np.isnan(avg_financial_stress) else "N/A"

        # --- Display KPIs ---
        with col1: st.markdown(f'<div class="metric-card"><p class="metric-label">Total Responses</p><p class="metric-value">{total_responses}</p></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><p class="metric-label">Depression Rate</p><p class="metric-value">{depression_rate:.1f}%</p></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><p class="metric-label">Most Common Sleep</p><p class="metric-value" style="font-size: 18px;">{most_common_sleep}</p></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><p class="metric-label">Avg Financial Stress</p><p class="metric-value">{stress_display}</p></div>', unsafe_allow_html=True)

        st.markdown("### Depression Rate by Demographics")
        col_d1, col_d2 = st.columns(2)

        # Gender Plot
        with col_d1:
            st.markdown("##### By Gender")
            if not display_df.empty and 'Gender' in display_df.columns:
                try:
                    gender_dep = display_df.groupby('Gender', observed=True)[target_col].mean().reset_index()
                    gender_dep['Depression Rate (%)'] = gender_dep[target_col] * 100
                    if not gender_dep.empty:
                        fig = px.bar(gender_dep, x='Gender', y='Depression Rate (%)', text_auto='.1f',
                                     color='Depression Rate (%)', color_continuous_scale='Bluered',
                                     labels={'Gender': 'Gender', 'Depression Rate (%)': 'Depression Rate (%)'})
                        fig.update_layout(title="Depression Rate by Gender", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No aggregated data for Gender vs Depression.")
                except Exception as e: st.warning(f"Could not plot Gender vs Depression: {e}")
            else: st.info("Required columns ('Gender', 'Depression') missing or no data.")

        # AgeRange Plot
        with col_d2:
            st.markdown("##### By Age Range")
            if not display_df.empty and 'AgeRange' in display_df.columns:
                try:
                    age_order = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Unknown']
                    display_df['AgeRange'] = pd.Categorical(display_df['AgeRange'].astype(str), categories=age_order, ordered=True)
                    age_dep = display_df.groupby('AgeRange', observed=False)[target_col].mean().reset_index()
                    age_dep['Depression Rate (%)'] = age_dep[target_col] * 100
                    if not age_dep.empty:
                        fig = px.bar(age_dep, x='AgeRange', y='Depression Rate (%)', text_auto='.1f',
                                     color='Depression Rate (%)', color_continuous_scale='Bluered',
                                     labels={'AgeRange': 'Age Range', 'Depression Rate (%)': 'Depression Rate (%)'})
                        fig.update_layout(title="Depression Rate by Age Range", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No aggregated data for Age Range vs Depression.")
                except Exception as e: st.warning(f"Could not plot Age Range vs Depression: {e}")
            else: st.info("Required columns ('AgeRange', 'Depression') missing or no data.")

        # Profession Treemap
        st.markdown("### Depression & Profession Breakdown")
        if not display_df.empty and 'Profession' in display_df.columns:
            try:
                dep_prof = display_df.groupby(['Profession', target_col], observed=True).size().reset_index(name='count')
                dep_prof['Depression_Status'] = dep_prof[target_col].map({1: 'Depressed', 0: 'Not Depressed'})
                if not dep_prof.empty:
                    fig = px.treemap(dep_prof, path=['Profession', 'Depression_Status'], values='count',
                                     color='Depression_Status', color_discrete_map={'Depressed': '#e11d48', 'Not Depressed': '#22c55e'}, # Red/Green
                                     title="Depression Distribution by Profession")
                    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), paper_bgcolor='#0e1117', font_color='white')
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("No aggregated data for Profession Treemap.")
            except Exception as e: st.warning(f"Could not plot Profession Treemap: {e}")
        else: st.info("Required columns ('Profession', 'Depression') missing or no data.")


    # ----------------- DEPRESSION ANALYSIS TAB (MODELING) -----------------
    with tabs[1]:
        st.markdown("<h2 style='text-align: center; color: #7979f8;'>Depression Analysis & Predictors</h2>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align: center;'>Modeling using the <strong>full dataset</strong> to identify factors associated with depression</p><hr>", unsafe_allow_html=True)

        # --- Prepare Full Dataset for Modeling (Use original 'df') ---
        st.markdown("### Predictor Analysis Setup")

        # Define potential predictors from ALL actual columns found
        potential_predictors = [col for col in df.columns if col != target_col]
        st.write(f"Using target: '{target_col}'")
        st.write(f"Potential predictors ({len(potential_predictors)}): {potential_predictors}")

        # Ensure target exists in the main df (already checked in load_and_prepare_data)
        if target_col not in df.columns:
            st.error(f"Critical Error: Target '{target_col}' missing from main dataframe. Cannot proceed.")
        elif not potential_predictors:
            st.error("Critical Error: No potential predictor columns found. Cannot proceed.")
        else:
            # Use the full, cleaned dataframe for modeling
            df_model = df[[target_col] + potential_predictors].copy()
            # Final check: Drop rows where target is NA (shouldn't happen if cleaning worked)
            df_model.dropna(subset=[target_col], inplace=True)

            if df_model.empty:
                 st.error("Critical Error: Modeling dataframe is empty after selecting target and predictors.")
            else:
                st.write(f"Modeling with {len(df_model)} rows.")
                y = df_model[target_col].astype(int) # Ensure target is integer
                X = df_model[potential_predictors]

                # --- Dynamic Preprocessing Setup ---
                st.markdown("#### Preprocessing Steps")
                try:
                    # Identify column types in the feature set X
                    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
                    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

                    # Define which categoricals *should* be ordinal (use actual numeric columns for this)
                    # These were likely converted to numeric earlier if they were like 1,2,3,4,5
                    # Let's treat them as numeric unless explicitly defined otherwise.
                    # We *will* encode non-numeric categoricals.
                    ordinal_features_candidates = [
                        'Academic Pressure', 'Work Pressure', 'Study Satisfaction',
                        'Job Satisfaction', 'Work/Study Hours', 'Financial Stress'
                    ]
                    # Actual ordinal features are those candidates that are *still categorical* in X
                    # (e.g., if they were 'Low', 'Medium', 'High' originally and converted to category)
                    ordinal_features = [col for col in ordinal_features_candidates if col in categorical_features]
                    nominal_features = [col for col in categorical_features if col not in ordinal_features]

                    

                    # --- Build Preprocessing Pipelines (More Robustly) ---
                    # --- [Locate this part in your code, within the 'Depression Analysis' tab] ---

                        # --- Build Preprocessing Pipelines (More Robustly) ---
                    transformers = []

                        # Numerical Pipeline *** MODIFIED: StandardScaler uncommented ***
                    if numerical_features:
                        num_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
                            ('scaler', StandardScaler()) # Scaling is often needed for Logit & helps stability
                        ])
                        transformers.append(('num', num_pipeline, numerical_features))
                        # --- [Rest of your Ordinal and Nominal Pipelines remain the same] ---
                    # Ordinal Pipeline (Auto-detect categories)
                    if ordinal_features:
                        # Get categories directly from data for each ordinal column
                        ordinal_categories = []
                        valid_ordinal_features = [] # Keep track of ones we can process
                        for col in ordinal_features:
                            try:
                                # Get unique non-NA values and sort them if possible (numerically or alphabetically)
                                cats = sorted(X[col].dropna().unique())
                                ordinal_categories.append(cats)
                                valid_ordinal_features.append(col)
                            except TypeError: # Cannot sort mixed types
                                cats = X[col].dropna().unique().tolist()
                                ordinal_categories.append(cats)
                                valid_ordinal_features.append(col)
                            except Exception as e:
                                 st.warning(f"Could not auto-detect categories for ordinal feature '{col}': {e}. Skipping.")

                        if valid_ordinal_features: # Only proceed if we have valid features/categories
                            ord_pipeline = Pipeline(steps=[
                                 ('imputer', SimpleImputer(strategy='most_frequent')), # Impute before encoding
                                 ('encoder', OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1)) # Encode known, map unknown to -1
                             ])
                            transformers.append(('ord', ord_pipeline, valid_ordinal_features))

                    # Nominal Pipeline
                    if nominal_features:
                        nom_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')), # Or use 'constant', fill_value='Unknown'
                            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
                        ])
                        transformers.append(('nom', nom_pipeline, nominal_features))

                    if not transformers:
                        st.error("No valid features selected for preprocessing. Check data types and column lists.")
                    else:
                        # --- Create and Apply Column Transformer ---
                        preprocessor = ColumnTransformer(
                            transformers=transformers,
                            remainder='passthrough' # Keep columns not specified (should be none if lists are correct)
                        )
                        st.write("Fitting preprocessor...")
                        X_processed = preprocessor.fit_transform(X)
                        st.write(f"Preprocessing complete. Processed data shape: {X_processed.shape}")

                        # --- Get Feature Names After Transformation ---
                        # [Your existing code for getting feature names and creating X_processed_df]
                        try:
                            feature_names_out = preprocessor.get_feature_names_out()
                            feature_names_cleaned = [name.split('__')[-1] for name in feature_names_out]
                        except Exception as e:
                            st.warning(f"Could not automatically get feature names: {e}. Using generic names.")
                            feature_names_cleaned = [f"feature_{i}" for i in range(X_processed.shape[1])]

                        # Convert processed data back to DataFrame
                        X_processed_df = pd.DataFrame(X_processed, columns=feature_names_cleaned, index=X.index)

                        # --- Final Checks on Processed Data ---
                        # [Your existing code for checking/imputing NaNs/Infs]
                        nan_check = X_processed_df.isnull().sum()
                        inf_check = np.isinf(X_processed_df).sum()
                        if nan_check.sum() > 0 or inf_check.sum() > 0:
                             st.warning("NaNs or Infs detected *after* initial preprocessing pipelines.")
                             st.write("NaN counts per column:", nan_check[nan_check > 0])
                             st.write("Inf counts per column:", inf_check[inf_check > 0])
                             # Impute remaining NaNs/Infs robustly (e.g., with median/0)
                             final_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
                             X_processed_df = pd.DataFrame(final_imputer.fit_transform(X_processed_df), columns=X_processed_df.columns, index=X_processed_df.index)
                             final_inf_imputer = SimpleImputer(missing_values=np.inf, strategy='constant', fill_value=np.finfo(np.float64).max) # Replace inf with large number
                             X_processed_df = pd.DataFrame(final_inf_imputer.fit_transform(X_processed_df), columns=X_processed_df.columns, index=X_processed_df.index)
                             final_neg_inf_imputer = SimpleImputer(missing_values=-np.inf, strategy='constant', fill_value=np.finfo(np.float64).min) # Replace -inf
                             X_processed_df = pd.DataFrame(final_neg_inf_imputer.fit_transform(X_processed_df), columns=X_processed_df.columns, index=X_processed_df.index)
                             st.write("Attempted final imputation for remaining NaNs/Infs.")
                             if X_processed_df.isnull().values.any() or np.isinf(X_processed_df).values.any():
                                 st.error("Fatal Error: NaNs or Infs persist even after final imputation. Cannot fit models.")
                                 X_processed_df = None # Prevent model fitting
                        else:
                             st.write("Data successfully preprocessed with no NaNs or Infs detected.")


                        # ============ Logistic Regression (Statsmodels) ============
                        if X_processed_df is not None:
                            st.markdown("#### Logistic Regression Coefficients")
                            try:
                                X_logit_final = X_processed_df.copy()
                                # Ensure all columns are float for statsmodels
                                X_logit_final = X_logit_final.astype(float)
                                # Drop constant columns (low variance)
                                cols_to_drop_logit = [col for col in X_logit_final.columns if X_logit_final[col].nunique() <= 1]
                                if cols_to_drop_logit:
                                    st.write(f"Note: Removing constant columns before fitting Logit: {cols_to_drop_logit}")
                                    X_logit_final = X_logit_final.drop(columns=cols_to_drop_logit)

                                if X_logit_final.empty:
                                    st.warning("No valid predictor columns remained after removing constant ones for Logistic Regression.")
                                else:
                                    # Add constant (intercept)
                                    X_const = sm.add_constant(X_logit_final, has_constant='raise') # Raise error if constant already exists

                                # --- Fit Logit Model *** MODIFIED: Added cov_type *** ---
                                st.write(f"Fitting Logistic Regression model on {X_const.shape[0]} samples and {X_const.shape[1]-1} predictors (using robust covariance)...")
                                # Using cov_type='HC1' for robustness against heteroscedasticity, might help with numerical issues
                                log_reg = sm.Logit(y, X_const).fit(method='bfgs', cov_type='HC1', disp=0) # Added cov_type, method='bfgs' can sometimes help convergence

                                # --- Display Results ---
                                # [Your existing code for displaying results]
                                coef_df = log_reg.summary2().tables[1]
                                if 'const' in coef_df.index:
                                    coef_df = coef_df.drop('const')
                                if not coef_df.empty:
                                            coef_df['abs_coef'] = coef_df['Coef.'].abs()
                                            coef_df_sorted = coef_df.sort_values(by='abs_coef', ascending=True)
                                            fig_logit = px.bar(
                                                coef_df_sorted, x='Coef.', y=coef_df_sorted.index, orientation='h',
                                                labels={'Coef.': 'Coefficient (Log-Odds)', 'index': 'Predictor Variable'},
                                                title="Logistic Regression Coefficients (Significant Predictors)",
                                                color='Coef.', color_continuous_scale='Bluered_r',
                                                text='Coef.',
                                            )
                                            fig_logit.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                                            fig_logit.update_layout(
                                                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                                font_color='white', yaxis_title=None,
                                                height=max(400, len(coef_df_sorted)*25) # Adjust height dynamically
                                            )
                                            st.plotly_chart(fig_logit, use_container_width=True)

                                            with st.expander("View Full Regression Summary (Statsmodels)"):
                                                 st.text(log_reg.summary())
                                            with st.expander("View Coefficient Table"):
                                                 st.dataframe(coef_df.sort_values(by='abs_coef', ascending=False))
                                else:
                                            st.warning("Logistic Regression completed, but no predictor coefficients were generated (excluding constant).")

                                # --- Specific Error Handling for Logit ---
                            except PerfectSeparationError as e:
                                    st.error(f"Logistic Regression Error: Perfect separation detected. {e}")
                                    st.error("This means one or more predictors perfectly predict the outcome for a subset of data. Consider removing predictors or simplifying the model.")
                            except np.linalg.LinAlgError as e:
                                     st.error(f"Logistic Regression Error: Linear Algebra Error (Likely Multicollinearity / Singular Matrix). {e}")
                                     st.error("Scaling and robust covariance applied, but the issue might persist. Consider checking VIF or removing highly correlated predictors manually if this error continues.")
                            except Exception as e:
                                     st.error(f"An unexpected error occurred during Logistic Regression: {e}")
                                     st.error("Check data types, NaN/infinite values (should be handled), and predictor selection.")
                                     # traceback.print_exc() # Uncomment for detailed traceback in console/logs

# --- [Rest of your code, including Random Forest section] ---

                        # ============ Feature Importance (Random Forest) ============
                        if X_processed_df is not None:
                            st.markdown("#### Random Forest Feature Importance")
                            try:
                                X_rf_final = X_processed_df.copy()

                                # RF handles NaNs implicitly, but cleaning names is good practice
                                # Clean column names for RF (less strict than Logit but good practice)
                                X_rf_final.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X_rf_final.columns]
                                cleaned_feature_names_rf = X_rf_final.columns.tolist()

                                if X_rf_final.empty:
                                     st.warning("No valid features remaining for Random Forest.")
                                else:
                                    st.write(f"Fitting Random Forest model on {X_rf_final.shape[0]} samples and {X_rf_final.shape[1]} predictors...")
                                    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1) # Use available cores
                                    rf.fit(X_rf_final, y)

                                    importances = rf.feature_importances_
                                    forest_importances = pd.Series(importances, index=cleaned_feature_names_rf)
                                    forest_imp_df = pd.DataFrame({'Feature': forest_importances.index, 'Importance': forest_importances.values})
                                    forest_imp_df = forest_imp_df.sort_values('Importance', ascending=True)

                                    if not forest_imp_df.empty:
                                        fig_rf = px.bar(forest_imp_df.tail(20), x='Importance', y='Feature', orientation='h', # Show top 20
                                                        title="Top 20 Feature Importances from Random Forest",
                                                        color='Importance', color_continuous_scale='Viridis')
                                        fig_rf.update_layout(
                                            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                                            font_color='white', yaxis_title=None,
                                            height=max(400, len(forest_imp_df.tail(20))*25)
                                        )
                                        st.plotly_chart(fig_rf, use_container_width=True)

                                        with st.expander("View All Importance Values"):
                                            st.dataframe(forest_imp_df.sort_values('Importance', ascending=False))
                                    else:
                                        st.warning("Could not calculate Random Forest feature importances.")

                            except Exception as e:
                                st.error(f"An unexpected error occurred during Random Forest analysis: {e}")
                                st.error("Check the processed data and feature names.")
                                # traceback.print_exc() # Uncomment for detailed traceback

                # --- General Error Handling for Preprocessing ---
                except KeyError as e:
                    st.error(f"Data Preparation Error: A specified column is missing: {e}. Check column definitions.")
                except ValueError as e:
                    st.error(f"Data Preparation Error: Issue with data types or encoding. Error: {e}")
                    st.error("Check category definitions (now auto-detected) and ensure data consistency.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during data preparation/modeling setup: {e}")
                    # traceback.print_exc() # Uncomment for detailed traceback

    # ----------------- SLEEP PATTERNS TAB -----------------
    with tabs[2]:
        st.markdown("<h2 style='text-align: center; color: #7979f8;'>Sleep Patterns Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align: center;'>Analyzing sleep duration categories (using filtered data)</p><hr>", unsafe_allow_html=True)
        sleep_col = 'Sleep Duration'
        display_df_sleep = df if df_filtered.empty else df_filtered

        if sleep_col not in display_df_sleep.columns:
             st.warning(f"Column '{sleep_col}' not found in the data.")
        elif display_df_sleep.empty:
             st.info("No data available for sleep analysis based on filters.")
        else:
            col_s1, col_s2 = st.columns(2)
            # Sleep Distribution
            with col_s1:
                st.markdown("### Sleep Category Distribution")
                try:
                    sleep_counts = display_df_sleep[sleep_col].value_counts().reset_index()
                    sleep_counts.columns = [sleep_col, 'Count'] # Rename columns
                    if not sleep_counts.empty:
                        fig = px.bar(sleep_counts, x=sleep_col, y='Count',
                                     title=f"Distribution of {sleep_col}",
                                     color='Count', color_continuous_scale=px.colors.sequential.Viridis)
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No data for Sleep Duration distribution.")
                except Exception as e: st.warning(f"Could not plot sleep distribution: {e}")

            # Depression by Sleep
            with col_s2:
                st.markdown(f"### Depression Rate by {sleep_col}")
                if target_col in display_df_sleep.columns:
                    try:
                        sleep_dep = display_df_sleep.groupby(sleep_col, observed=True)[target_col].mean().reset_index()
                        sleep_dep['Depression Rate (%)'] = sleep_dep[target_col] * 100
                        sleep_dep = sleep_dep.sort_values('Depression Rate (%)', ascending=False)
                        if not sleep_dep.empty:
                            fig = px.bar(sleep_dep, x=sleep_col, y='Depression Rate (%)', text_auto='.1f',
                                         title=f"Depression Rate by {sleep_col}",
                                         color='Depression Rate (%)', color_continuous_scale='Bluered')
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis_title='Depression Rate (%)')
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.info(f"No aggregated data for Depression by {sleep_col}.")
                    except Exception as e: st.warning(f"Could not plot Depression by {sleep_col}: {e}")
                else: st.info(f"Target column '{target_col}' needed for this plot.")

            # Sleep vs Other Numeric Factors
            st.markdown(f"### {sleep_col} vs Other Factors")
            try:
                available_numeric_sleep = display_df_sleep.select_dtypes(include=np.number).columns.tolist()
                if target_col in available_numeric_sleep: available_numeric_sleep.remove(target_col) # Exclude target
                if available_numeric_sleep:
                    selected_numeric_sleep = st.selectbox(f"Select Numeric Factor to Compare with {sleep_col}:", available_numeric_sleep, key="sleep_compare_adv")
                    if selected_numeric_sleep:
                        fig = px.box(display_df_sleep, x=sleep_col, y=selected_numeric_sleep, points="all",
                                     color=target_col if target_col in display_df_sleep.columns else None,
                                     color_discrete_map={0: '#22c55e', 1: '#e11d48'}, # Green/Red
                                     title=f"{selected_numeric_sleep} by {sleep_col} Category")
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No suitable numeric factors found to compare with Sleep Duration.")
            except Exception as e: st.warning(f"Could not create Sleep vs Factor plot: {e}")


    # ----------------- DIETARY ANALYSIS TAB -----------------
    with tabs[3]:
        st.markdown("<h2 style='text-align: center; color: #7979f8;'>Dietary Habits Analysis</h2>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align: center;'>Analyzing dietary habits and relationships (using filtered data)</p><hr>", unsafe_allow_html=True)
        diet_col = 'Dietary Habits'
        display_df_diet = df if df_filtered.empty else df_filtered

        if diet_col not in display_df_diet.columns:
             st.warning(f"Column '{diet_col}' not found in the data.")
        elif display_df_diet.empty:
             st.info("No data available for diet analysis based on filters.")
        else:
            col_d1, col_d2 = st.columns(2)
            # Diet Distribution
            with col_d1:
                st.markdown(f"### {diet_col} Distribution")
                try:
                    diet_counts = display_df_diet[diet_col].value_counts().reset_index()
                    diet_counts.columns = [diet_col, 'Count']
                    if not diet_counts.empty:
                        fig = px.pie(diet_counts, values='Count', names=diet_col,
                                     title=f"Distribution of {diet_col}", hole=0.4,
                                     color_discrete_sequence=px.colors.sequential.Plasma)
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                    else: st.info("No data for Dietary Habits distribution.")
                except Exception as e: st.warning(f"Could not plot diet distribution: {e}")

            # Depression by Diet
            with col_d2:
                st.markdown(f"### Depression Rate by {diet_col}")
                if target_col in display_df_diet.columns:
                    try:
                        diet_dep = display_df_diet.groupby(diet_col, observed=True)[target_col].mean().reset_index()
                        diet_dep['Depression Rate (%)'] = diet_dep[target_col] * 100
                        diet_dep = diet_dep.sort_values('Depression Rate (%)', ascending=False)
                        if not diet_dep.empty:
                            fig = px.bar(diet_dep, x=diet_col, y='Depression Rate (%)', text_auto='.1f',
                                         title=f"Depression Rate by {diet_col}",
                                         color='Depression Rate (%)', color_continuous_scale='Bluered')
                            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis_title='Depression Rate (%)')
                            st.plotly_chart(fig, use_container_width=True)
                        else: st.info(f"No aggregated data for Depression by {diet_col}.")
                    except Exception as e: st.warning(f"Could not plot Depression by {diet_col}: {e}")
                else: st.info(f"Target column '{target_col}' needed for this plot.")

            # Diet vs Other Numeric Factors
            st.markdown(f"### {diet_col} vs Other Factors")
            try:
                available_numeric_diet = display_df_diet.select_dtypes(include=np.number).columns.tolist()
                if target_col in available_numeric_diet: available_numeric_diet.remove(target_col)
                if available_numeric_diet:
                    selected_numeric_diet = st.selectbox(f"Select Numeric Factor to Compare with {diet_col}:", available_numeric_diet, key="diet_compare_adv")
                    if selected_numeric_diet:
                        fig = px.box(display_df_diet, x=diet_col, y=selected_numeric_diet, points="all",
                                     color=target_col if target_col in display_df_diet.columns else None,
                                     color_discrete_map={0: '#22c55e', 1: '#e11d48'}, # Green/Red
                                     title=f"{selected_numeric_diet} by {diet_col}")
                        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("No suitable numeric factors found to compare with Dietary Habits.")
            except Exception as e: st.warning(f"Could not create Diet vs Factor plot: {e}")


    # ----------------- ADVANCED ANALYTICS TAB -----------------
    with tabs[4]:
        st.markdown("<h2 style='text-align: center; color: #7979f8;'>Advanced Analytics</h2>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle' style='text-align: center;'>Correlations and multivariate views (using filtered data)</p><hr>", unsafe_allow_html=True)
        display_df_adv = df if df_filtered.empty else df_filtered

        # Correlation Matrix
        st.markdown("### Correlation Matrix (Numeric Variables)")
        if not display_df_adv.empty:
            numeric_cols_corr = display_df_adv.select_dtypes(include=np.number).columns.tolist()
            # Include target for correlation, but maybe remove constant/binary later if needed
            if len(numeric_cols_corr) > 1:
                try:
                    corr_matrix = display_df_adv[numeric_cols_corr].corr()
                    fig_sns, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', ax=ax, linewidths=.5, annot_kws={"size": 8})
                    ax.set_title('Correlation Matrix (Filtered Numeric Data)', fontsize=14, color='white')
                    plt.xticks(rotation=45, ha='right', color='white', fontsize=8)
                    plt.yticks(rotation=0, color='white', fontsize=8)
                    fig_sns.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    plt.tight_layout()
                    st.pyplot(fig_sns)
                except Exception as e: st.warning(f"Could not generate correlation matrix: {e}")
            else:
                st.info(f"Not enough numeric columns ({len(numeric_cols_corr)}) for correlation matrix.")
        else: st.info("No data available for correlation matrix based on filters.")

        # 3D Scatter Plot
        st.markdown("### Multivariate Analysis (3D Scatter)")
        if not display_df_adv.empty:
            scatter_cols_3d = display_df_adv.select_dtypes(include=np.number).columns.tolist()
            if len(scatter_cols_3d) >= 3:
                try:
                    col1, col2, col3 = st.columns(3)
                    with col1: x_var = st.selectbox("X-axis:", scatter_cols_3d, index=0, key="3d_x_adv_tab")
                    with col2: y_var = st.selectbox("Y-axis:", scatter_cols_3d, index=min(1, len(scatter_cols_3d)-1), key="3d_y_adv_tab")
                    with col3: z_var = st.selectbox("Z-axis:", scatter_cols_3d, index=min(2, len(scatter_cols_3d)-1), key="3d_z_adv_tab")

                    color_var_3d = target_col if target_col in display_df_adv.columns else None

                    if x_var and y_var and z_var and len(set([x_var, y_var, z_var])) == 3:
                        hover_cols_3d = [col for col in ['Profession', 'Degree', 'City', 'AgeRange'] if col in display_df_adv.columns]
                        fig = px.scatter_3d(display_df_adv, x=x_var, y=y_var, z=z_var,
                                            color=color_var_3d,
                                            color_discrete_map={0: '#22c55e', 1: '#e11d48'}, # Green/Red
                                            opacity=0.7, title=f"3D Scatter: {x_var} vs {y_var} vs {z_var}",
                                            hover_data=hover_cols_3d)

                        fig.update_layout(
                            scene=dict(
                                xaxis_title=x_var, yaxis_title=y_var, zaxis_title=z_var,
                                bgcolor='#0e1117',
                                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False, zerolinecolor="gray", color='white'),
                                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False, zerolinecolor="gray", color='white'),
                                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False, zerolinecolor="gray", color='white')),
                            margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor='#0e1117', font_color='white'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(set([x_var, y_var, z_var])) < 3:
                         st.warning("Please select three *different* variables for the axes.")
                except Exception as e: st.warning(f"Could not generate 3D scatter plot: {e}")
            else:
                st.info(f"Need at least 3 numeric columns for 3D scatter plot. Found: {len(scatter_cols_3d)}")
        else: st.info("No data available for 3D scatter plot based on filters.")

    # --- Footer ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af;'>End of Dashboard</p>", unsafe_allow_html=True)

    # --- Sidebar Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info(" Dashboard views reflect filtered data. \n\n Depression Analysis tab uses the full, preprocessed dataset for modeling.")

else:
    # This block runs if load_and_prepare_data returned None
    st.error("Dashboard cannot be displayed: Data loading or initial preprocessing failed. Please check the data file and script configuration.")
