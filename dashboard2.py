import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # If needed elsewhere
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Suppress specific warnings if needed (e.g., from statsmodels)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='seaborn')
warnings.filterwarnings("ignore", message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names")


# --- Configuration ---
st.set_page_config(layout="wide", page_title="Mental Health Analytics", page_icon="")
st.markdown("""
    <style>
        /* Your CSS remains the same */
        .main {background-color: #0e1117;}
        /* ... other styles ... */
         .metric-card {background-color: #1f2937; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}
        .metric-value {font-size: 24px; font-weight: bold; color: #7979f8;}
        .metric-label {font-size: 14px; color: #d1d5db;}
        .subtitle {font-size: 18px; color: #d1d5db; margin-top: -5px; margin-bottom: 15px;}
        hr {margin: 15px 0px;}
    </style>
""", unsafe_allow_html=True)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data(file_path="Analytica Dataset.csv"):
    try:
        df = pd.read_csv(file_path)
        st.success(f"Successfully loaded data from {file_path}")
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please ensure 'Analytica Dataset.csv' is in the same directory or provide the correct path.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None

    # Clean column names FIRST - Just strip whitespace
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip()
    cleaned_columns = df.columns.tolist()

    # Debug: Show original vs cleaned names if they changed
    if original_columns != cleaned_columns:
         st.write("Original Columns:", original_columns)
         st.write("Cleaned Columns:", cleaned_columns)
    else:
         st.write("Cleaned Column Names:", cleaned_columns) # Still useful to see


    # --- Define Columns based *only* on the provided image/list ---
    # !! IMPORTANT: Verify these names exactly match your CSV header after stripping whitespace !!
    numeric_cols = [
        'Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
        'Study Satisfaction', 'Job Satisfaction',
        'Work/Study Hours', 'Financial Stress'
    ]
    categorical_cols = [
        'Gender', 'City', 'Profession', 'Sleep Duration', # Sleep Duration is CATEGORICAL
        'Dietary Habits', 'Degree'
    ]
    # Ensure correct casing and spacing from your actual file
    yes_no_cols = [
        'Have you ever had suicidal thoughts ?',
        'Family History of Mental Illness',
        'Depression' # Target variable
    ]

    for col in numeric_cols:
        if col in df.columns: # Check if column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # Drop rows where *any* of the specified columns became NaN after conversion
    df.dropna(subset=numeric_cols, inplace=True)


    # Drop ID column if it exists (use cleaned name)
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    elif 'Id' in df.columns: # Check common variations
         df.drop(columns=['Id'], inplace=True)

    values_to_remove = [
    "'Less Delhi'",
    3,
    "'Less than 5 Kalyan'",
    'Saanvi',
    'M.Tech',
    'Bhavna',
    'Mira',
    'Harsha',
    'Vaanya',
    'Gaurav',
    'Harsh',
    'Reyansh',
    'Kibara',
    'Rashi',
    'ME',
    'M.Com',
    'Nalyan',
    'Mihir',
    'Nalini',
    'Nandini',
    'Khaziabad',
    0
    ]

    df = df[~df['City'].astype(str).isin([str(v) for v in values_to_remove])]
    
    # --- Handle Data Types and Initial NaNs ---
    # Convert potential numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                # st.write(f"Imputed NaNs in numeric '{col}' with median ({median_val}).")
        else:
            st.warning(f"Expected numeric column '{col}' not found in data.")

    # Convert Yes/No columns to 0/1 (Robustly)
    for col in yes_no_cols:
        if col in df.columns:
            # First, convert to string to allow .str methods and handle mixed types
            df[col] = df[col].astype(str)
            # Apply string cleaning
            df[col] = df[col].str.strip().str.lower()
            # Define mapping for common variations + already numeric/boolean
            mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'true': 1, 'false': 0}
            # Map values, keep track of original NaNs (now 'nan' string)
            original_nan_mask = df[col] == 'nan'
            df[col] = df[col].map(mapping)
            # Impute NaNs (both original and those from failed mapping) with mode
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 0 # Default to 0 if mode fails
                df[col].fillna(mode_val, inplace=True)
                # st.write(f"Imputed NaNs/unmapped in Yes/No '{col}' with mode ({mode_val}).")
            df[col] = df[col].astype(int) # Ensure final type is integer
        else:
            st.warning(f"Expected Yes/No column '{col}' not found in data.")

    # Convert other categorical columns to 'category' type and impute NaNs
    for col in categorical_cols:
        if col in df.columns:
            if df[col].isnull().any():
                 # Impute categorical NaNs (replace with 'Unknown')
                 df[col].fillna('Unknown', inplace=True)
                 # st.write(f"Imputed NaNs in categorical '{col}' with 'Unknown'.")
            # Convert to string first to ensure consistency before category conversion
            df[col] = df[col].astype(str).astype('category')
        else:
             st.warning(f"Expected categorical column '{col}' not found in data.")

    # Define age bins and labels - Use the cleaned numeric 'Age'
    if 'Age' in df.columns and pd.api.types.is_numeric_dtype(df['Age']):
        # Make bins slightly wider to catch edges if needed
        max_age = df['Age'].max()
        bins = [0, 17, 24, 34, 44, 54, 64, 100]
        labels = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        df['AgeRange'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        if df['AgeRange'].isnull().any():
            df['AgeRange'].fillna('Unknown', inplace=True)
            # st.write("Imputed NaNs in 'AgeRange' with 'Unknown'.")
        df['AgeRange'] = df['AgeRange'].astype('category')
        # Add 'AgeRange' to categorical list if created successfully
        if 'AgeRange' not in categorical_cols:
            categorical_cols.append('AgeRange')
    else:
        st.warning("Numeric column 'Age' not found or not numeric, cannot create 'AgeRange'.")


    # Ensure target variable 'Depression' is integer type after mapping/filling
    if 'Depression' in df.columns:
         df['Depression'] = df['Depression'].astype(int)

    st.write("--- Data Loading and Preprocessing Complete ---")
    # st.write(df.head()) # Optional: Display first few rows after cleaning
    # st.write(df.info()) # Optional: Display data types and non-null counts
    return df

# --- Load Data ---
df = load_data() # Use default path or specify another

# --- Main App Logic ---
if df is not None and not df.empty:

    # --- Sidebar Filters ---
    st.sidebar.header(" Filter the Data")

    filter_options = {}
    # Define columns available for filtering (use cleaned names)
    # Use only columns confirmed to exist after loading/cleaning
    possible_filter_cols = ['Gender', 'City', 'Profession', 'AgeRange',
                            'Dietary Habits', 'Degree', 'Sleep Duration'] # Sleep Duration is categorical

    for col in possible_filter_cols:
        if col in df.columns:
            unique_values = df[col].dropna().unique()
            # Ensure options are strings for multiselect
            options = sorted([str(item) for item in unique_values])
            if options:
                 filter_options[col] = options

    filters = {}
    for col, options in filter_options.items():
         # Default to selecting all options
        filters[col] = st.sidebar.multiselect(f"Filter by {col}", options, default=options)

    # Filter the data - Apply filters cumulatively
    df_filtered = df.copy()
    for col, selected_str_options in filters.items():
         if selected_str_options: # Only filter if options are selected
            # Filter using string representation for robustness as options are strings
            df_filtered = df_filtered[df_filtered[col].astype(str).isin(selected_str_options)]

    # Display warning if filters result in empty data
    if df_filtered.empty:
        st.warning("Warning: The current filter combination results in no data.")

    # --- Create Tabs ---
    tabs = st.tabs([" Overview", " Depression Analysis", " Sleep Patterns", " Dietary Analysis", " Advanced Analytics"])


    # ------------------ OVERVIEW TAB ------------------
    with tabs[0]:
        st.markdown("""
            <h1 style='text-align: center; color: #7979f8;'> Mental Health & Lifestyle Dashboard</h1>
            <p class='subtitle' style='text-align: center;'>Exploring the relationship between lifestyle factors and mental well-being</p>
            <hr>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        # --- Calculate KPIs (Adapt for available columns) ---
        total_responses = len(df_filtered) if not df_filtered.empty else 0

        # Use the cleaned target column name 'Depression'
        depression_col = 'Depression'
        depression_rate = 0
        if depression_col in df_filtered.columns and not df_filtered.empty:
             depression_rate = 100 * df_filtered[depression_col].mean()

        # Sleep Duration KPI - Mode or Count Plot
        sleep_col = 'Sleep Duration'
        most_common_sleep = "N/A"
        if sleep_col in df_filtered.columns and not df_filtered.empty:
             try:
                  most_common_sleep = df_filtered[sleep_col].mode()[0]
             except IndexError: # Handle empty Series case after filtering
                  most_common_sleep = "N/A"


        # Use 'Financial Stress' as a proxy for stress/wellbeing KPI
        stress_col = 'Financial Stress' # Example using an existing numeric col
        avg_financial_stress = 0
        if stress_col in df_filtered.columns and not df_filtered.empty:
              avg_financial_stress = df_filtered[stress_col].mean()


        # --- Display KPIs ---
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Total Responses</p>
                    <p class="metric-value">{total_responses}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
             st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Depression Rate</p>
                    <p class="metric-value">{depression_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Most Common Sleep</p>
                    <p class="metric-value" style="font-size: 18px;">{most_common_sleep}</p>
                </div>
            """, unsafe_allow_html=True)

        with col4:
             stress_display = f"{avg_financial_stress:.1f}" if avg_financial_stress is not None and not np.isnan(avg_financial_stress) else "N/A"
             st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Avg Financial Stress</p>
                     <p class="metric-value">{stress_display}</p>
                </div>
            """, unsafe_allow_html=True)


        st.markdown("###  Depression Rate by Demographics")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### By Gender")
            gender_col = 'Gender'
            if not df_filtered.empty and gender_col in df_filtered.columns and depression_col in df_filtered.columns:
                # Ensure observed=False if 'Gender' is categorical
                gender_dep = df_filtered.groupby(gender_col, observed=True)[depression_col].mean().reset_index()
                gender_dep[depression_col] = gender_dep[depression_col] * 100
                if not gender_dep.empty:
                    fig = px.bar(gender_dep, x=gender_col, y=depression_col, text_auto='.1f',
                                 title="Depression Rate by Gender (%)",
                                 color=depression_col, color_continuous_scale='Bluered',
                                 labels={depression_col: 'Depression Rate (%)', gender_col: 'Gender'})
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("No data to plot for Gender vs Depression after filtering.")
            else: st.info("Gender or Depression column missing or no data after filtering.")

        with col_d2:
            st.markdown("##### By Age Range")
            age_range_col = 'AgeRange'
            if not df_filtered.empty and age_range_col in df_filtered.columns and depression_col in df_filtered.columns:
                age_order = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Unknown']
                # Ensure the column is categorical with the correct order
                df_filtered[age_range_col] = pd.Categorical(df_filtered[age_range_col].astype(str), categories=age_order, ordered=True)
                age_dep = df_filtered.groupby(age_range_col, observed=False)[depression_col].mean().reset_index()
                age_dep[depression_col] = age_dep[depression_col] * 100
                if not age_dep.empty:
                    fig = px.bar(age_dep, x=age_range_col, y=depression_col, text_auto='.1f',
                                 title="Depression Rate by Age Range (%)",
                                 color=depression_col, color_continuous_scale='Bluered',
                                 labels={depression_col: 'Depression Rate (%)', age_range_col: 'Age Range'})
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("No data to plot for Age Range vs Depression after filtering.")
            else: st.info("AgeRange or Depression column missing or no data after filtering.")

        st.markdown("###  Depression & Profession Breakdown")
        profession_col = 'Profession'
        if not df_filtered.empty and profession_col in df_filtered.columns and depression_col in df_filtered.columns:
            dep_prof = df_filtered.groupby([profession_col, depression_col], observed=True).size().reset_index(name='count')
            dep_prof['Depression_Status'] = dep_prof[depression_col].map({1: 'Depressed', 0: 'Not Depressed'})
            if not dep_prof.empty:
                fig = px.treemap(dep_prof, path=[profession_col, 'Depression_Status'], values='count',
                                 color='Depression_Status', color_discrete_map={'Depressed': 'red', 'Not Depressed': 'green'},
                                 title="Depression Distribution by Profession")
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("No data to plot for Profession Treemap after filtering.")
        else: st.info("Profession or Depression column missing or no data after filtering.")

    #depression tabbbbbbbbb
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
                    transformers = []

                    # Numerical Pipeline
                    if numerical_features:
                        num_pipeline = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
                            #('scaler', StandardScaler()) # Scaling is often needed for Logit, optional for RF
                        ])
                        transformers.append(('num', num_pipeline, numerical_features))

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
                                # st.write(f"Auto-detected categories for '{col}': {cats}")
                            except TypeError: # Cannot sort mixed types
                                cats = X[col].dropna().unique().tolist()
                                ordinal_categories.append(cats)
                                valid_ordinal_features.append(col)
                                # st.write(f"Auto-detected (unsorted) categories for '{col}': {cats}")
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
                        try:
                            feature_names_out = preprocessor.get_feature_names_out()
                            feature_names_cleaned = [name.split('__')[-1] for name in feature_names_out]
                        except Exception as e:
                            st.warning(f"Could not automatically get feature names: {e}. Using generic names.")
                            feature_names_cleaned = [f"feature_{i}" for i in range(X_processed.shape[1])]

                        # Convert processed data back to DataFrame
                        X_processed_df = pd.DataFrame(X_processed, columns=feature_names_cleaned, index=X.index)

                        # --- Final Checks on Processed Data ---
                        # Check for NaNs introduced (e.g., by OrdinalEncoder unknown_value if set to np.nan)
                        nan_check = X_processed_df.isnull().sum()
                        inf_check = np.isinf(X_processed_df).sum()

                        if nan_check.sum() > 0 or inf_check.sum() > 0:
                            st.warning("NaNs or Infs detected *after* initial preprocessing pipelines.")
                            st.write("NaN counts per column:", nan_check[nan_check > 0])
                            st.write("Inf counts per column:", inf_check[inf_check > 0])

                            # Impute remaining NaNs/Infs robustly (e.g., with median/0)
                            # Use SimpleImputer that handles both
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

                                    # --- Fit Logit Model ---
                                    st.write(f"Fitting Logistic Regression model on {X_const.shape[0]} samples and {X_const.shape[1]-1} predictors...")
                                    log_reg = sm.Logit(y, X_const).fit(disp=0) # disp=0 suppresses convergence messages

                                    # --- Display Results ---
                                    coef_df = log_reg.summary2().tables[1]
                                    if 'const' in coef_df.index:
                                        coef_df = coef_df.drop('const')

                                    if not coef_df.empty:
                                        coef_df['abs_coef'] = coef_df['Coef.'].abs()
                                        # Filter out insignificant coefficients if desired (e.g., P>|z| > 0.05)
                                        # coef_df_significant = coef_df[coef_df['P>|z|'] <= 0.05]
                                        coef_df_sorted = coef_df.sort_values(by='abs_coef', ascending=True)

                                        fig_logit = px.bar(
                                            coef_df_sorted, x='Coef.', y=coef_df_sorted.index, orientation='h',
                                            labels={'Coef.': 'Coefficient (Log-Odds)', 'index': 'Predictor Variable'},
                                            title="Logistic Regression Coefficients (Significant Predictors)",
                                            color='Coef.', color_continuous_scale='Bluered_r', # Red for positive, Blue for negative
                                            text='Coef.', # Show coefficient value on bars
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
                                st.error(f"Logistic Regression Error: Linear Algebra Error (Likely Multicollinearity). {e}")
                                st.error("Check for highly correlated predictors using the correlation matrix in 'Advanced Analytics'. Consider removing some predictors.")
                                # Optional: Add VIF calculation here
                            except Exception as e:
                                st.error(f"An unexpected error occurred during Logistic Regression: {e}")
                                st.error("Check data types, NaN/infinite values (should be handled), and predictor selection.")
                                # traceback.print_exc() # Uncomment for detailed traceback in console/logs

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
        st.markdown("""
            <h2 style='text-align: center; color: #7979f8;'> Sleep Patterns (Categorical)</h2>
            <p class='subtitle' style='text-align: center;'>Analyzing sleep duration categories and relationships</p>
            <hr>
        """, unsafe_allow_html=True)

        sleep_col = 'Sleep Duration' # Use cleaned name

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("###  Sleep Category Distribution")
            if not df_filtered.empty and sleep_col in df_filtered.columns:
                sleep_counts = df_filtered[sleep_col].value_counts().reset_index()
                sleep_counts.columns = [sleep_col, 'Count'] # Rename columns after reset_index

                if not sleep_counts.empty:
                     # Use a bar chart for distribution as pie might get crowded
                     fig = px.bar(sleep_counts, x=sleep_col, y='Count',
                                  title=f"Distribution of {sleep_col} (Filtered)",
                                  color='Count', color_continuous_scale=px.colors.sequential.Viridis)
                     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                     st.plotly_chart(fig, use_container_width=True)
                else: st.info("No data for Sleep Duration distribution after filtering.")
            else: st.info("Sleep Duration column missing or no data after filtering.")


        with col_s2:
             st.markdown(f"###  Depression Rate by {sleep_col}")
             if not df_filtered.empty and sleep_col in df_filtered.columns and depression_col in df_filtered.columns:
                 sleep_dep = df_filtered.groupby(sleep_col, observed=True)[depression_col].mean().reset_index()
                 sleep_dep['Depression Rate'] = sleep_dep[depression_col] * 100
                 sleep_dep = sleep_dep.sort_values('Depression Rate', ascending=False)
                 if not sleep_dep.empty:
                      fig = px.bar(sleep_dep, x=sleep_col, y='Depression Rate',
                                    title=f"Depression Rate by {sleep_col} (%)", text_auto='.1f',
                                    color='Depression Rate', color_continuous_scale='Bluered')
                      fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                        yaxis=dict(title='Depression Rate (%)'))
                      st.plotly_chart(fig, use_container_width=True)
                 else: st.info(f"No data for Depression by {sleep_col} after filtering.")
             else: st.info("Sleep Duration or Depression column missing or no data after filtering.")


        st.markdown(f"###  {sleep_col} vs Other Factors")
        if not df_filtered.empty and sleep_col in df_filtered.columns:
             # Compare sleep categories against numeric variables using box plots
             potential_comp_cols = ['Age', 'CGPA', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
             available_numeric = [col for col in potential_comp_cols if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col])]

             if available_numeric:
                  selected_numeric = st.selectbox(f"Select Numeric Factor to Compare with {sleep_col}:", available_numeric, key="sleep_compare")
                  if selected_numeric:
                       fig = px.box(df_filtered, x=sleep_col, y=selected_numeric,
                                    color=depression_col if depression_col in df_filtered.columns else None,
                                    color_discrete_map={0: 'lightgreen', 1: 'crimson'},
                                    title=f"{selected_numeric} by {sleep_col} Category (Filtered)",
                                    points="all")
                       fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                       st.plotly_chart(fig, use_container_width=True)
             else:
                  st.warning("No suitable numeric factors found to compare with Sleep Duration in the filtered data.")
        else: st.info("Sleep Duration column missing or no data after filtering.")


    # ----------------- DIETARY ANALYSIS TAB -----------------
    with tabs[3]:
        st.markdown("""
            <h2 style='text-align: center; color: #7979f8;'> Dietary Analysis</h2>
            <p class='subtitle' style='text-align: center;'>Dietary habits and their impact</p>
            <hr>
        """, unsafe_allow_html=True)

        diet_col = 'Dietary Habits' # Use cleaned name

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown(f"###  {diet_col} Distribution")
            if not df_filtered.empty and diet_col in df_filtered.columns:
                diet_counts = df_filtered[diet_col].value_counts().reset_index()
                diet_counts.columns = [diet_col, 'Count'] # Rename after reset_index
                if not diet_counts.empty:
                     fig = px.pie(diet_counts, values='Count', names=diet_col,
                                  title=f"Distribution of {diet_col} (Filtered)", hole=0.4,
                                  color_discrete_sequence=px.colors.sequential.Plasma)
                     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                     fig.update_traces(textposition='inside', textinfo='percent+label')
                     st.plotly_chart(fig, use_container_width=True)
                else: st.info("No data for Dietary Habits distribution after filtering.")
            else: st.info("Dietary Habits column missing or no data after filtering.")

        with col_d2:
            st.markdown(f"###  Depression by {diet_col}")
            if not df_filtered.empty and diet_col in df_filtered.columns and depression_col in df_filtered.columns:
                 diet_dep = df_filtered.groupby(diet_col, observed=True)[depression_col].mean().reset_index()
                 diet_dep['Depression Rate'] = diet_dep[depression_col] * 100
                 diet_dep = diet_dep.sort_values('Depression Rate', ascending=False)
                 if not diet_dep.empty:
                      fig = px.bar(diet_dep, x=diet_col, y='Depression Rate',
                                    title=f"Depression Rate by {diet_col} (%)", text_auto='.1f',
                                    color='Depression Rate', color_continuous_scale='Bluered')
                      fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white',
                                        yaxis=dict(title='Depression Rate (%)'))
                      st.plotly_chart(fig, use_container_width=True)
                 else: st.info(f"No data for Depression by {diet_col} after filtering.")
            else: st.info("Dietary Habits or Depression column missing or no data after filtering.")

        st.markdown(f"### {diet_col} and Other Factors")
        if not df_filtered.empty and diet_col in df_filtered.columns:
             # Compare diet against numeric variables
             potential_comp_cols_diet = ['Age', 'CGPA', 'Academic Pressure', 'Work Pressure', 'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
             available_numeric_diet = [col for col in potential_comp_cols_diet if col in df_filtered.columns and pd.api.types.is_numeric_dtype(df_filtered[col])]

             if available_numeric_diet:
                  selected_numeric_diet = st.selectbox(f"Select Numeric Factor to Compare with {diet_col}:", available_numeric_diet, key="diet_compare")
                  if selected_numeric_diet:
                       fig = px.box(df_filtered, x=diet_col, y=selected_numeric_diet,
                                    color=depression_col if depression_col in df_filtered.columns else None,
                                    color_discrete_map={0: 'lightgreen', 1: 'crimson'},
                                    title=f"{selected_numeric_diet} by {diet_col} (Filtered)", points="all")
                       fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                       st.plotly_chart(fig, use_container_width=True)
             else:
                  st.warning("No suitable numeric factors found to compare with Dietary Habits in the filtered data.")
        else: st.info("Dietary Habits column missing or no data after filtering.")


    # ----------------- ADVANCED ANALYTICS TAB -----------------
    with tabs[4]:
        st.markdown("""
            <h2 style='text-align: center; color: #7979f8;'> Advanced Analytics</h2>
            <p class='subtitle' style='text-align: center;'>Correlations and multivariate views (Filtered Data)</p>
            <hr>
        """, unsafe_allow_html=True)

        st.markdown("###  Correlation Matrix (Numeric Variables)")
        if not df_filtered.empty:
             # Select numeric columns available in the filtered data
             numeric_cols_filtered = df_filtered.select_dtypes(include=np.number).columns.tolist()
             # Optionally exclude the target variable if present
             if depression_col in numeric_cols_filtered:
                  numeric_cols_filtered.remove(depression_col)

             if len(numeric_cols_filtered) > 1:
                  corr_matrix = df_filtered[numeric_cols_filtered].corr()
                  fig_sns, ax = plt.subplots(figsize=(10, 8)) # Adjust size
                  sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='magma', ax=ax, linewidths=.5)
                  ax.set_title('Correlation Matrix (Filtered Numeric Variables)', fontsize=14, color='white')
                  plt.xticks(rotation=45, ha='right', color='white')
                  plt.yticks(rotation=0, color='white')
                  fig_sns.set_facecolor('#0e1117')
                  ax.set_facecolor('#0e1117')
                  plt.tight_layout()
                  st.pyplot(fig_sns)
             else:
                  st.warning(f"Not enough numeric columns found for correlation matrix in filtered data. Found: {numeric_cols_filtered}")
        else: st.info("No data available based on current filters.")


        st.markdown("### Multivariate Analysis (3D Scatter)")
        if not df_filtered.empty:
             # Use available numeric columns from filtered data
             scatter_cols_3d = numeric_cols_filtered # Use the list from correlation
             if len(scatter_cols_3d) >= 3:
                  col1, col2, col3 = st.columns(3)
                  with col1: x_var = st.selectbox("X-axis:", scatter_cols_3d, index=0, key="3d_x_adv")
                  with col2: y_var = st.selectbox("Y-axis:", scatter_cols_3d, index=min(1, len(scatter_cols_3d)-1), key="3d_y_adv")
                  with col3: z_var = st.selectbox("Z-axis:", scatter_cols_3d, index=min(2, len(scatter_cols_3d)-1), key="3d_z_adv")

                  color_var_3d = depression_col if depression_col in df_filtered.columns else None

                  if x_var and y_var and z_var and len(set([x_var, y_var, z_var])) == 3:
                       if color_var_3d:
                            fig = px.scatter_3d(df_filtered, x=x_var, y=y_var, z=z_var, color=color_var_3d,
                                                 color_discrete_map={0: 'lightgreen', 1: 'crimson'},
                                                 opacity=0.7, title=f"3D Scatter: {x_var} vs {y_var} vs {z_var}",
                                                 hover_data=[col for col in ['Profession', 'Degree', 'City'] if col in df_filtered.columns]) # Add hover info
                            fig.update_layout(scene=dict(xaxis_title=x_var, yaxis_title=y_var, zaxis_title=z_var,
                                                         bgcolor='#0e1117', # Match background
                                                         xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=True, zerolinecolor="gray", color='white'),
                                                         yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=True, zerolinecolor="gray", color='white'),
                                                         zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=True, zerolinecolor="gray", color='white')),
                                              margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor='#0e1117')
                            st.plotly_chart(fig, use_container_width=True)
                       else: st.warning("Cannot color by 'Depression' as it's missing.")
                  else: st.warning("Please select three different variables for the axes.")
             else:
                  st.warning(f"Need at least 3 numeric columns for 3D scatter plot. Found: {scatter_cols_3d}")
        else: st.info("No data available based on current filters.")


    # --- Footer / Summary ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #7979f8;'>End of Report</h3>", unsafe_allow_html=True)
    # Your summary cards remain the same...
    
    # --- Sidebar Footer ---
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard reflects data based on selected filters, except for Regression/Feature Importance which uses the full dataset.")


else:
    # This block runs if df is None or empty after load_data()
    st.error("Dashboard cannot be displayed: Data loading failed or the file is empty.")
