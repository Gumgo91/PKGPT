"""
Data Loader for Pharmacokinetic datasets
Analyzes dataset structure and extracts column information
"""

import pandas as pd
from typing import Dict, List, Tuple
import os


class PKDataLoader:
    """Load and analyze pharmacokinetic datasets"""

    def __init__(self, file_path: str):
        """
        Initialize data loader

        Args:
            file_path: Path to CSV dataset file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        self.file_path = file_path
        self.df = None
        self.columns = None
        self.metadata = {}

        self._load_data()
        self._analyze_structure()

    def _load_data(self):
        """Load CSV data"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"[OK] Loaded dataset: {self.file_path}")
            print(f"  Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        except Exception as e:
            raise Exception(f"Failed to load dataset: {e}")

    def _analyze_structure(self):
        """Analyze dataset structure and extract metadata"""
        self.columns = list(self.df.columns)

        # Analyze column types and statistics
        self.metadata = {
            'file_path': self.file_path,
            'n_rows': len(self.df),
            'n_cols': len(self.columns),
            'columns': self.columns,
            'column_info': {}
        }

        # Analyze each column
        for col in self.columns:
            col_info = {
                'dtype': str(self.df[col].dtype),
                'n_unique': self.df[col].nunique(),
                'n_missing': self.df[col].isna().sum(),
                'non_missing_count': self.df[col].notna().sum()
            }

            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                valid_data = self.df[col].dropna()
                if len(valid_data) > 0:
                    col_info.update({
                        'min': float(valid_data.min()),
                        'max': float(valid_data.max()),
                        'mean': float(valid_data.mean()),
                        'median': float(valid_data.median())
                    })

            self.metadata['column_info'][col] = col_info

        # Detect standard NONMEM columns
        self._detect_nonmem_columns()

        # Detect subject-level covariates
        self._detect_covariates()

    def _detect_nonmem_columns(self):
        """Detect standard NONMEM column types"""
        standard_cols = {
            'ID': ['ID', 'SUBJ', 'SUBJECT'],
            'TIME': ['TIME', 'TIME_POINT'],
            'AMT': ['AMT', 'DOSE', 'AMOUNT'],
            'DV': ['DV', 'CONC', 'CONCENTRATION'],
            'EVID': ['EVID', 'EVENT'],
            'CMT': ['CMT', 'COMP', 'COMPARTMENT'],
            'MDV': ['MDV', 'MISSING'],
            'RATE': ['RATE', 'INFUSION_RATE'],
            'DVID': ['DVID', 'DV_ID']
        }

        detected = {}
        for std_name, possible_names in standard_cols.items():
            for col in self.columns:
                if col.upper() in possible_names:
                    detected[std_name] = col
                    break

        self.metadata['nonmem_columns'] = detected

    def _detect_covariates(self):
        """Detect potential covariates (subject-level variables)"""
        if 'ID' not in self.metadata.get('nonmem_columns', {}):
            self.metadata['covariates'] = []
            return

        id_col = self.metadata['nonmem_columns']['ID']
        nonmem_core = set(self.metadata['nonmem_columns'].values())

        covariates = []
        for col in self.columns:
            if col in nonmem_core:
                continue

            # Check if column has constant value within each subject
            try:
                grouped = self.df.groupby(id_col)[col].nunique()
                if (grouped == 1).all():
                    covariates.append(col)
            except:
                pass

        self.metadata['covariates'] = covariates

    def get_column_summary(self) -> str:
        """
        Get human-readable summary of dataset columns

        Returns:
            Formatted string describing dataset structure
        """
        lines = []
        lines.append(f"Dataset: {os.path.basename(self.file_path)}")
        lines.append(f"Total rows: {self.metadata['n_rows']}")
        lines.append(f"Total columns: {self.metadata['n_cols']}")
        lines.append("")

        # NONMEM standard columns
        if self.metadata.get('nonmem_columns'):
            lines.append("Detected NONMEM columns:")
            for std_name, col_name in self.metadata['nonmem_columns'].items():
                info = self.metadata['column_info'][col_name]
                lines.append(f"  - {std_name} ({col_name}): {info['non_missing_count']} observations")
            lines.append("")

        # Covariates
        if self.metadata.get('covariates'):
            lines.append("Detected covariates (subject-level):")
            for cov in self.metadata['covariates']:
                info = self.metadata['column_info'][cov]
                lines.append(f"  - {cov}: {info['n_unique']} unique values")
            lines.append("")

        # Other columns
        other_cols = [
            col for col in self.columns
            if col not in self.metadata.get('nonmem_columns', {}).values()
            and col not in self.metadata.get('covariates', [])
        ]
        if other_cols:
            lines.append("Other columns:")
            for col in other_cols:
                info = self.metadata['column_info'][col]
                lines.append(f"  - {col}: {info['dtype']}, {info['n_unique']} unique values")

        return "\n".join(lines)

    def get_data_summary(self) -> str:
        """
        Get summary statistics for the dataset

        Returns:
            Formatted string with summary statistics
        """
        lines = []

        # Subject count
        if 'ID' in self.metadata.get('nonmem_columns', {}):
            id_col = self.metadata['nonmem_columns']['ID']
            n_subjects = self.df[id_col].nunique()
            lines.append(f"Number of subjects: {n_subjects}")

        # Observation count
        if 'MDV' in self.metadata.get('nonmem_columns', {}):
            mdv_col = self.metadata['nonmem_columns']['MDV']
            n_obs = (self.df[mdv_col] == 0).sum()
            lines.append(f"Number of observations: {n_obs}")

        # Dose records
        if 'EVID' in self.metadata.get('nonmem_columns', {}):
            evid_col = self.metadata['nonmem_columns']['EVID']
            n_doses = (self.df[evid_col] == 1).sum()
            lines.append(f"Number of dose records: {n_doses}")

        # Time range
        if 'TIME' in self.metadata.get('nonmem_columns', {}):
            time_col = self.metadata['nonmem_columns']['TIME']
            time_range = self.df[time_col].max() - self.df[time_col].min()
            lines.append(f"Time range: 0 to {self.df[time_col].max()} ({time_range} units)")

        return "\n".join(lines)

    def get_metadata(self) -> Dict:
        """Get complete metadata dictionary"""
        return self.metadata

    def get_dataframe(self) -> pd.DataFrame:
        """Get the loaded DataFrame"""
        return self.df
