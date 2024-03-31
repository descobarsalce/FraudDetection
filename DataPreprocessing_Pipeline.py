# IMPORTS

# General libraries
import warnings
import requests
import zipfile
import io
import json
import numpy as np
import pandas as pd
from itertools import product

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit-learn utilities
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_score, f1_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

# GLOBAL CONFIGURATION
# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Set plot style
sns.set_style("whitegrid")

class DataPipeline:
    """Class to manage and preprocess data for machine learning tasks."""

    def __init__(self, time_var, target):
        """Initialize the DataPipeline with empty preprocessing containers."""
        self.time_var = time_var
        self.target = target
        self.encoders = None
        
    ############
    #  DATA LOADING?EXTRACTION 

    @staticmethod
    def open_csv_data(filename: str) -> pd.DataFrame:
        """Load a CSV file into a pandas DataFrame."""
        try:
            return pd.read_csv(filename)
        except (FileNotFoundError, IOError) as e:
            raise NameError('Error: File Not Found.') from e

    @staticmethod
    def read_txt(txt_file: str) -> pd.DataFrame:
        """Read a txt file and parse each line as JSON."""
        with io.open(txt_file) as file:
            data = [json.loads(line) for line in file.read().strip().split('\n')]
        return pd.DataFrame(data)

    def download_from_url(self, zip_url: str) -> pd.DataFrame:
        """Download and extract a JSON file from a given ZIP URL."""
        zip_bytes = self._download_zip_from_github(zip_url)
        if zip_bytes:
            return self._extract_json_from_zip(zip_bytes)
        else:
            print('Invalid download. Provide a valid URL.')
            return None

    @staticmethod
    def _download_zip_from_github(zip_url: str) -> bytes:
        """Helper method to download ZIP file content from a given URL."""
        try:
            return requests.get(zip_url).content
        except requests.RequestException as e:
            print(f"Error downloading from {zip_url}. Error: {e}")
            return None

    @staticmethod
    def _extract_json_from_zip(zip_bytes: bytes) -> pd.DataFrame:
        """Helper method to extract JSON content from ZIP bytes."""
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
            for name in zip_ref.namelist():
                with zip_ref.open(name) as txt_file:
                    content = txt_file.read().decode('utf-8')
                    data = [json.loads(line) for line in content.strip().split('\n')]
                    return pd.DataFrame(data)

    ############
    # DATA DESCRIPTION METHODS

    @staticmethod
    def describe_data(data: pd.DataFrame):
        """Describe the dataset with various statistics."""
        separator = "\n" + "-" * 40 + "\n"
        print(separator)
        print(f"Number of observations: {len(data)}")
        print(separator)
        print(pd.concat([data.sample(10), data.head(10)]))
        print(separator)

        # Define data types and their corresponding additional statistics
        data_types = {
            'object_bool': {
                'types': ['object', bool],
                'additional_stats': {
                    'null_or_empty': lambda x: (x.isnull() | (x == "")).sum()
                },
                'transpose': True  # Indicate that this description should be transposed
            },
            'numeric': {
                'types': [np.number],
                'additional_stats': {
                    'nunique': lambda x: x.nunique(),
                    'null_count': lambda x: x.isnull().sum()
                },
                'transpose': False  # No transposition for numeric columns
            }
        }

        # Iterate over the data types and compute the statistics
        for key, value in data_types.items():
            desc = data.describe(include=value['types'])
            additional_stats = {stat_name: stat_func(data.select_dtypes(include=value['types'])) for stat_name, stat_func in value['additional_stats'].items()}
            desc = pd.concat([desc, pd.DataFrame(additional_stats).T], axis=0)
            
            # Transpose if specified
            if value['transpose']:
                desc = desc.T
            
            # Print the results
            print(f"Description for {key} columns:")
            print(desc)
            print(separator)

    @staticmethod
    def histogram(data: pd.DataFrame, var: str, var_desc: str = None, bins: int = 30, threshold: float = None, display_totals: bool = False):
        """Plot a histogram for the specified variable in the dataset."""
        if not var_desc:
            var_desc = var

        plt.figure(figsize=(10, 6))
        limit = data[var].max() if not threshold else threshold
        data_to_plot = [limit if x > limit else x for x in data[var]]

        # Choose colors
        custom_palette = ["#004879", "#D22E1E"]

        if display_totals:
            # Compute the weighted histogram
            hist, bin_edges = np.histogram(data_to_plot, bins=bins, weights=data[var])
            plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align="edge", color=custom_palette[0], edgecolor='black', linewidth=1)
            plt.title(f'Total amount by {var_desc} ({var})', fontsize=20, fontweight='bold')
            plt.ylabel(f'Total sum of {var_desc}', fontsize=16)
        else:
            sns.histplot(data_to_plot, bins=bins, kde=True, color=custom_palette[0], edgecolor='black', linewidth=1)
            plt.title(f'Histogram of {var_desc} ({var})', fontsize=20, fontweight='bold')
            plt.ylabel('Frequency', fontsize=16)

        plt.xlabel('Value', fontsize=16)

        # Highlight the overflow bin
        if threshold:
            colors = sns.cubehelix_palette(8, start=2, rot=0, dark=0.2, light=0.8, reverse=True)
            plt.axvspan(limit - (threshold/bins), threshold, color=colors[5], alpha=0.2, label='Overflow Bin')

        plt.legend()
        plt.show()
        
    ############
    # DATA MANIPULATION AND FEATURE GENERATION:

    def sort_id_time(self, data: pd.DataFrame, id_var: str) -> pd.DataFrame:
        """Sorts the DataFrame by ID and time variables."""
        sorting_key = id_var + [self.time_var] # time must be the last one for 
        data.sort_values(by=sorting_key, inplace=True)
        data.reset_index(inplace=True, drop=True)
        return data
    
    @staticmethod
    def find_reversals(data: pd.DataFrame, var_amount: str, reversal_key: str) -> tuple:
        """Finds and categorizes reversed transactions in the data."""
        reversals = data[data.transactionType=='REVERSAL'][reversal_key]
        non_reversals = data[(data.transactionType!='REVERSAL') & (data.transactionType!='ADDRESS_VERIFICATION')]
        # Now join to find matches:
        unreversed_transactions = pd.merge(non_reversals, reversals, how='outer', on=reversal_key, indicator='merge_reversal')
        # merge dictionar= Both: reversed, left: unmatched/unreversed, right=unmatched_reversal
        reversed_transactions = unreversed_transactions[unreversed_transactions.merge_reversal=='both']
        unmatched_reversal = unreversed_transactions[unreversed_transactions.merge_reversal=='right_only']
        unreversed_transactions = unreversed_transactions[unreversed_transactions.merge_reversal=='left_only']
        return reversed_transactions, unmatched_reversal, unreversed_transactions
    
    @staticmethod
    def find_repeated(data: pd.DataFrame, key: str, time_threshold_range: list) -> tuple:
        """Finds repeated events in the data within a given time range."""
        data['time_diff'] = data.groupby(key)['transactionDateTime'].diff().dt.total_seconds()
        data['same_amount_next'] = data.groupby(key)['transactionAmount'].shift() == data['transactionAmount']
        results=[]
        results_count=[]
        threshold_values = list(time_threshold_range)
        for range_seconds in threshold_values:
            data['repeated'] = (data['same_amount_next']==1) & (data['time_diff'] < range_seconds)
            total_repeated_amount = data.loc[data['repeated'], 'transactionAmount'].sum()
            total_repeated_count = data.loc[data['repeated'], 'transactionAmount'].count()
            results.append(total_repeated_amount)
            results_count.append(total_repeated_count)
        return results, results_count
    
    @staticmethod
    def plot_results(results: list, title: str, xlabel: str, ylabel: str, xticks: list) -> None:
        """Plots results such as model performance metrics."""        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(xticks, results, '-o')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(xticks, [str(val) + "s" for val in xticks])
        plt.grid(True)
        plt.show()
        
    def add_delta_seconds(self, data: pd.DataFrame, var_id: list) -> pd.DataFrame:
        """Adjusts time data to handle simultaneous events."""
        # Adding a new variable to differentiate observations while sorting, allowing for repeated times and adding a flag for them.
        data['sequence'] = data.groupby(var_id + [self.time_var]).cumcount()
        data['simultaneous_trans'] = data.groupby(var_id + [self.time_var])[self.time_var].transform('size')
        # added a microsecond delay to treat the observations as time series with unique values but keep 
        data[self.time_var] = data[self.time_var] + pd.to_timedelta(data['sequence'], unit='ms')
        data = self.sort_id_time(data, var_id) # Verify order with new time delta:
        return data
        
    @staticmethod
    def rolling_windows_ROWS(data: pd.DataFrame, id_var: list, var_name: str, row_window: int = 3, suffix: str = '') -> (pd.DataFrame, set):
        """Computes rolling window statistics for given row windows."""
        
        old_columns = data.columns
        # Moving average (last X observations)
        data['MA_'+var_name +suffix+f'{row_window}'] = data.groupby(id_var, sort=False)[var_name].rolling(row_window, min_periods=0).sum().values
        data['MAn_'+var_name+suffix+f'{row_window}'] = data.groupby(id_var, sort=False)[var_name].rolling(row_window, min_periods=0).count().values
        data['MA_'+var_name+suffix+f'{row_window}'] = data['MA_'+var_name+suffix+f'{row_window}']-data[var_name]
        new_columns_names = set(data.columns)-set(old_columns)
        return data, new_columns_names
    
    def rolling_windows_TIME(self, data: pd.DataFrame, id_var: list, var_name: str, time_window: str = '2D', suffix: str = '') -> (pd.DataFrame, set):
        """Computes rolling window statistics within a given time window."""
        old_columns = data.columns
        # Total within range of time
        data['RW_'+var_name+suffix+f'{time_window}'] = data.groupby(id_var, group_keys=False, sort=False)\
                                     .apply(lambda group: group.set_index(self.time_var)[var_name]\
                                     .rolling(window=time_window, min_periods=0).sum())\
                                     .reset_index(level=0, drop=True).values
        data['RWn_'+var_name+suffix+f'{time_window}'] = data.groupby(id_var, group_keys=False, sort=False)\
                                     .apply(lambda group: group.set_index(self.time_var)[var_name]\
                                     .rolling(window=time_window, min_periods=0).count())\
                                     .reset_index(level=0, drop=True).values
        data['RW_'+var_name+f'{time_window}'] = data['RW_'+var_name+suffix+f'{time_window}']-data[var_name]
        new_columns_names = set(data.columns)-set(old_columns)
        return data, new_columns_names
    
    def generate_rolling_windows(self, data: pd.DataFrame, outcome_vars: list, row_ranges: list, time_ranges: list, suffixes: dict) -> pd.DataFrame:
        """Generates rolling window statistics for given data."""
        # First we generate rolling window using customer level
        all_new_columns = {}
        for key, value in suffixes.items():
            data = self.sort_id_time(data=data, id_var=value)
            for outcome in outcome_vars:
                for row_range in row_ranges:
                    data, new_columns_names_temp = self.rolling_windows_ROWS(data, value, outcome, row_window=row_range, suffix=key)
                    all_new_columns[key+'{}'.format(row_range)] = new_columns_names_temp                    
                for time_range in time_ranges:
                    data, new_columns_names_temp = self.rolling_windows_TIME(data, value, outcome, time_window=time_range, suffix=key)
                    all_new_columns[key+'{}'.format(time_range)] = new_columns_names_temp

        # Now create indirect link of the fraud of client even if from different merchant, and the frauds of anywhere of clients of given merchant
        for key, value in suffixes.items():        
            data = self.sort_id_time(data=data, id_var=value)
            for row_range in row_ranges:
                for outcome in all_new_columns[key+'{}'.format(time_range)]:
                    data, new_columns_names_temp = self.rolling_windows_ROWS(data, value, outcome, row_window=row_range, suffix=key)
            for  time_range in time_ranges:
                for outcome in all_new_columns[key+'{}'.format(time_range)]:
                    data, new_columns_names_temp = self.rolling_windows_TIME(data, value, outcome, time_window=time_range, suffix=key)   

        # Now using both keys to check if some client is a recurrent somewhere
        full_comb_key = []
        for key, value in suffixes.items():
            full_comb_key+=value
        data = self.sort_id_time(data=data, id_var=full_comb_key)
        key = 'd2'
        for row_range in row_ranges:
            for outcome in outcome_vars:
                data, new_columns_names_temp = self.rolling_windows_ROWS(data, full_comb_key, outcome, row_window=row_range, suffix=key)
        for time_range in time_ranges:
            for outcome in outcome_vars:
                data, new_columns_names_temp = self.rolling_windows_TIME(data, full_comb_key, outcome, time_window=time_range, suffix=key)     
        return data
    
    @staticmethod
    # Day/month/week trends
    def add_temporal_features(data: pd.DataFrame, time_var: str, prefix: str = '') -> pd.DataFrame:
        """Adds temporal features like year, month, day, and weekday."""
        data[prefix+'year'] = data[time_var].dt.year
        data[prefix+'month'] = data[time_var].dt.month
        data[prefix+'day'] = data[time_var].dt.day
        data[prefix+'weekday'] = data[time_var].dt.weekday
        return data
    
    # SPLIT AND TRAIN METHODS:
    def split_sample(self, df: pd.DataFrame, individual_id: str, n_splits: int, customer_key: str='accountNumber') -> list:
        """Splits data based on TimeSeriesSplit for unique individual IDs."""
        # I will also use an oversample model to capture more fraudulent observations:
        avg_values = df.groupby(customer_key)[self.target].mean()
        # Use the average values to determine oversampling weights
        weights = avg_values.loc[df[customer_key]].values
        # Sample from the DataFrame using the weights
        df = df.sample(n=len(df), replace=True, weights=weights).reset_index(drop=True)

        # Group by user_id and get unique user_ids
        unique_user_ids = df[individual_id].unique()
        # TimeSeriesSplit on unique user_ids
        tscv = TimeSeriesSplit(n_splits=n_splits)
        # Store train and test indices for each split
        train_test_splits = []
        for train_index, test_index in tscv.split(unique_user_ids):
            train_user_ids = unique_user_ids[train_index]
            test_user_ids = unique_user_ids[test_index]
            train_indices = df[df[individual_id].isin(train_user_ids)].index
            test_indices = df[df[individual_id].isin(test_user_ids)].index
            train_test_splits.append((train_indices, test_indices))
        return train_test_splits 
    
    def generate_sample(self, df: pd.DataFrame, cross_sample: tuple, features: list) -> tuple:
        """Generates training and testing samples from given indices."""
        train_index=cross_sample[0]
        test_index=cross_sample[1]
        train, test = df.iloc[train_index], df.iloc[test_index]
        X_train, y_train = train[features], train[self.target] # .fillna(mean) .dropna()
        X_test, y_test = test[features], test[self.target]
        return X_train, y_train, X_test, y_test
    
    def gen_one_hot_encoders(self, df: pd.DataFrame, columns: list, train: bool = True, handle_unknown: str = 'ignore', sparse: bool = False) -> pd.DataFrame:
        """Generates one-hot encoded data."""
        if train:
            encoder = OneHotEncoder(handle_unknown=handle_unknown, sparse=sparse)
            encoded_data = encoder.fit_transform(df[columns])
            self.encoders = encoder
        encoder = self.encoders
        encoded_data = encoder.transform(df[columns])
        encoded_data = pd.DataFrame(encoded_data)
        df.columns
        data = pd.concat([df.drop(columns, axis=1), encoded_data], axis=1)
        return data

    @staticmethod
    def evaluate_model(est_model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float=0.5) -> tuple:
        """Evaluates a model's performance using precision and F1 score."""
        y_pred = est_model.predict(X_test)
        predicted_labels = np.where(y_pred >= threshold, 1, 0)
        # Evaluate model on test data:
        prec = precision_score(y_test, pd.DataFrame(predicted_labels), zero_division=0)
        f1_sco = f1_score(y_test, pd.DataFrame(predicted_labels))
        return prec, f1_sco
    
    def fit_all_models(self, df: pd.DataFrame, train_test_splits: pd.Series, models: dict(), param_grid: dict(), threshold : float = 0.05):
        
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        # Define data pipeline to make preprocessing homogeneous between train and test set:
        data_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')), # impute median to NAN values to avoid dropping data that may be missing non-randomly
                ('robust', RobustScaler()),  # scale the features but asjust using IQR to make it robsut to outliers in data        
                ('poly_features', PolynomialFeatures(degree=2)),
            ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', data_pipe, numeric_features),
            ])# Create full pipeline for OLS model
        # Placeholder for results                     
        pipelines = {}
        best_params_prec = {}
        best_params_f1 = {}
        best_prec = {}
        best_f1 = {}
        
        i = 1 # This is to modify the name for each item
        for model_name, model in models.items():   
        
            print('---------------------\n')
            print('Model: {}'.format(model_name))
            i+=1 # This adds some value to each of the columns names of the grid:
            best_prec_temp = 0 
            best_f1_temp = 0
            
            # Get the parameter grid for the current model:
            curr_param_grid = param_grid.get(model_name, {})
            # Generate all combinations of hyperparameters
            keys, values = zip(*curr_param_grid.items())
            param_combinations = [dict(zip(keys, v)) for v in product(*values)]
           
            print('Total parameters combinations model: {}'.format(len(param_combinations)))
            
            # Iterate through all combinations of hyperparameters
            for param_set in param_combinations:    
                print(param_set)
                # Create a pipeline for each model
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    (model_name, model),
                ])
                
                # Set the parameters and fit the model
                pipeline.set_params(**param_set)
                samples = 1
                
                results_prec = []
                results_f1 = []
                for subsample in train_test_splits:
                    print('Subsample: {}'.format(samples))
                    samples+=1
                    X_train, y_train, X_test, y_test = self.generate_sample(df, subsample, numeric_features)
        
                    # Fit the pipeline to the training data
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate the model's performance on test data
                    prec, f1_sco = self.evaluate_model(pipeline, X_test, y_test, threshold)
                    
                    # Store the fitted pipeline values:
                    pipelines[model_name] = pipeline
                    results_prec.append(prec)
                    results_f1.append(f1_sco)
                # Now we average subsamples to get results.
                prec = sum(results_prec)/len(results_prec)
                f1_sco = sum(results_f1)/len(results_f1)
                print(prec)
                print(f1_sco)
                # Update best score and parameters if needed
                if prec > best_prec_temp:
                    best_prec_temp = prec
                if f1_sco > best_f1_temp:
                    best_f1_temp = f1_sco
                
        best_prec[model_name] = best_prec_temp
        best_params_prec[model_name] = param_set
           
        best_f1[model_name] = best_f1_temp
        best_params_f1[model_name] = param_set
        
        return best_prec, best_params_prec, best_f1, best_params_f1
         
    
