import os
import joblib
from sklearn.preprocessing import MinMaxScaler,RobustScaler


class Scaler_wrapper:
    def __init__(self, scaler) -> None:
        self.scaler = scaler
    
    def fit(self, df_train, feature_map):
        self.scaler.fit(df_train[feature_map])
    
    def transform(self, df, feature_map):
        # Use .loc for assignment to modify df in-place robustly
        transformed_values = self.scaler.transform(df[feature_map])
        df.loc[:, feature_map] = transformed_values
        return df

    def inverse(self, df, feature_map):
        # Use .loc for assignment to modify df in-place robustly
        inverted_values = self.scaler.inverse_transform(df[feature_map])
        df.loc[:, feature_map] = inverted_values
        return df

def scaler_wrapper(df, feature_map, flag, scaler_type, cache_dir='./cache/', label=''):
    """
    Applies scaling to the dataframe based on the specified strategy.
    - If flag is 'train':
        - If scaler cache file exists, loads it.
        - Else, creates, fits, and saves a new scaler.
    - If flag is 'val' or 'test':
        - If scaler cache file exists, loads it.
        - Else, raises FileNotFoundError.
    Transforms the dataframe using the obtained scaler.

    Args:
        df (pd.DataFrame): Input dataframe.
        feature_map (list): List of columns to scale.
        flag (str): 'train', 'val', or 'test'.
        scaler_type (str): 'minmax', 'robust', or 'None'.
        cache_dir (str): Directory to store/load cached scalers.

    Returns:
        tuple: (transformed_df, scaler_wrapper_instance or None)
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Cache directory created: {cache_dir}")
     
    scaler_filename = f"{scaler_type}_{label}_scaler.joblib"
    scaler_path = os.path.join(cache_dir, scaler_filename)

    if scaler_type == 'None':
        return df, None

    current_scaler_instance = None

    if os.path.exists(scaler_path):
        print(f"Scaler file found at {scaler_path}. Loading scaler.")
        current_scaler_instance = joblib.load(scaler_path)
    else:
        # Scaler file does not exist
        if  'train' in flag:
            print(f"Scaler file not found at {scaler_path}. Creating and fitting a new scaler (train mode).")
            if scaler_type == 'minmax':
                scaler_to_fit = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler_to_fit = RobustScaler()
            else:
                raise Exception(f"Scaler type '{scaler_type}' not supported")

            # Fit the scaler directly
            scaler_to_fit.fit(df[feature_map])
            current_scaler_instance = scaler_to_fit # The fitted scaler itself

            joblib.dump(current_scaler_instance, scaler_path)
            print(f"Scaler for '{scaler_type}' saved to {scaler_path}")
        else:  # flag is 'val' or 'test' and file does not exist
            raise FileNotFoundError(
                f"Scaler file not found at {scaler_path} for flag='{flag}'. "
                f"Ensure the scaler was trained and cached using flag='train' first for scaler_type='{scaler_type}'."
            )

    # At this point, current_scaler_instance is either loaded or newly fitted and saved.
    # Create a Scaler_wrapper for consistent interface (transform method and return type)
    final_scaler_wrapper = Scaler_wrapper(current_scaler_instance)

    # Modify 'df' in-place to save memory.
    # The Scaler_wrapper's transform method (updated to use .loc) will alter 'df'.
    # The caller should be aware that the input 'df' is modified.
    # Using .loc in the transform method helps avoid SettingWithCopyWarning.
    final_scaler_wrapper.transform(df, feature_map)
    # df_transformed = df # df is now the transformed (modified) DataFrame

    return df, final_scaler_wrapper