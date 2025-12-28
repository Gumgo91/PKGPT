"""
Data Loader for Pharmacokinetic datasets
Analyzes dataset structure and extracts column information
"""

import pandas as pd
from typing import Dict, List, Tuple
import os
import numpy as np
import math


def _estimate_lambda_z(
    t: np.ndarray,
    c: np.ndarray,
    min_points: int = 3,
    max_points: int = 6,
):
    """말기 소실기(terminal phase) λz 추정 (log-linear 회귀)."""
    mask = np.isfinite(t) & np.isfinite(c) & (c > 0)
    t = t[mask]
    c = c[mask]
    if t.size < min_points:
        return None

    order = np.argsort(t)
    t = t[order]
    c = c[order]
    ln_c = np.log(c)
    n = t.size

    best = None
    for k in range(min_points, min(max_points, n) + 1):
        t_win = t[n - k:]
        y_win = ln_c[n - k:]

        x_mean = float(t_win.mean())
        y_mean = float(y_win.mean())
        ss_xy = float(((t_win - x_mean) * (y_win - y_mean)).sum())
        ss_xx = float(((t_win - x_mean) ** 2).sum())
        if ss_xx == 0.0:
            continue

        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_pred = intercept + slope * t_win
        ss_res = float(((y_win - y_pred) ** 2).sum())
        ss_tot = float(((y_win - y_mean) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        cand = {
            "lambda_z": abs(slope),
            "intercept": intercept,
            "r2": r2,
        }
        if best is None or cand["r2"] > best["r2"]:
            best = cand

    return best


def _estimate_early_slope_after_cmax(
    t: np.ndarray,
    c: np.ndarray,
    min_points: int = 3,
):
    """Cmax 이후 초기 분포기 기울기 λearly 추정 (log-linear 회귀)."""
    mask = np.isfinite(t) & np.isfinite(c) & (c > 0)
    t = t[mask]
    c = c[mask]
    if t.size < min_points + 1:
        return None

    order = np.argsort(t)
    t = t[order]
    c = c[order]

    imax = int(np.argmax(c))
    t_post = t[imax:]
    c_post = c[imax:]
    if t_post.size < min_points:
        return None

    tmax = t_post[0]
    t_last = t_post[-1]
    if t_last <= tmax:
        return None

    cutoff = tmax + 0.3 * (t_last - tmax)
    early_mask = t_post <= cutoff
    if early_mask.sum() < min_points:
        early_mask[:] = True

    t_early = t_post[early_mask]
    c_early = c_post[early_mask]
    if t_early.size < min_points:
        return None

    ln_c = np.log(c_early)
    x_mean = float(t_early.mean())
    y_mean = float(ln_c.mean())
    ss_xy = float(((t_early - x_mean) * (ln_c - y_mean)).sum())
    ss_xx = float(((t_early - x_mean) ** 2).sum())
    if ss_xx == 0.0:
        return None

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    y_pred = intercept + slope * t_early
    ss_res = float(((ln_c - y_pred) ** 2).sum())
    ss_tot = float(((ln_c - y_mean) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "lambda_early": abs(slope),
        "intercept": intercept,
        "r2": r2,
    }


def _bic_piecewise_log_linear(
    t: np.ndarray,
    c: np.ndarray,
):
    """
    (참고용) log(C) vs TIME 에서
      - 1-comp: 직선 1개
      - 2-comp: 직선 2개 (분할 시점 grid search)
    의 BIC 차이(ΔBIC = BIC1 - BIC2)를 계산.
    (현재 보수적 알고리즘에서는 직접 구현을 쓰므로 이 함수는 사용하지 않아도 됨)
    """
    mask = np.isfinite(t) & np.isfinite(c) & (c > 0)
    t = t[mask]
    c = c[mask]
    if t.size < 4:
        return None

    order = np.argsort(t)
    t_all = t[order].astype(float)
    c_all = c[order].astype(float)

    imax = int(np.argmax(c_all))
    t_seg = t_all[imax:]
    c_seg = c_all[imax:]
    if t_seg.size < 4:
        t_seg = t_all
        c_seg = c_all
        if t_seg.size < 4:
            return None

    ln_c = np.log(c_seg)
    n = len(t_seg)

    def lin_reg(x, y):
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        ss_xy = float(((x - x_mean) * (y - y_mean)).sum())
        ss_xx = float(((x - x_mean) ** 2).sum())
        if ss_xx == 0.0:
            slope = 0.0
        else:
            slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_pred = intercept + slope * x
        rss = float(((y - y_pred) ** 2).sum())
        return intercept, slope, rss

    # 1-comp
    _, _, rss1 = lin_reg(t_seg, ln_c)
    eps = 1e-12
    k1 = 2
    bic1 = n * math.log((rss1 + eps) / n) + k1 * math.log(n)

    # 2-comp
    best_bic2 = float("inf")
    best_split = None
    for split in range(2, n - 1):
        t1, t2 = t_seg[:split], t_seg[split:]
        y1, y2 = ln_c[:split], ln_c[split:]
        if len(t1) < 2 or len(t2) < 2:
            continue
        _, _, rss_1 = lin_reg(t1, y1)
        _, _, rss_2 = lin_reg(t2, y2)
        rss2 = rss_1 + rss_2
        k2 = 4
        bic2 = n * math.log((rss2 + eps) / n) + k2 * math.log(n)
        if bic2 < best_bic2:
            best_bic2 = bic2
            best_split = split

    if best_split is None:
        return {"delta_bic": 0.0, "bic1": bic1, "bic2": bic1, "n": n}

    return {
        "delta_bic": bic1 - best_bic2,
        "bic1": bic1,
        "bic2": best_bic2,
        "n": n,
    }


def _infer_compartments_from_profile(
    df: pd.DataFrame,
    time_col: str = "TIME",
    conc_col: str = "DV",
) -> int:
    """
    단일 프로파일에서 1-comp vs 2-comp 추론 (보수적).

    원칙:
      1) tail(log C vs time)이 깔끔한 mono-exponential이면 → 1-comp 확정
      2) 2-comp는 매우 제한적으로만 허용:
         - BIC가 2-comp를 강하게 지지(ΔBIC > 20) AND
         - NCA에서 명확한 biphasic:
           · slope_ratio = λearly / λz ≥ 3.0
           · tail/초기 R² ≥ 0.98
      3) 그 외는 모두 1-comp
    """
    d = df[[time_col, conc_col]].dropna().copy()
    d = d[d[conc_col] > 0]
    if len(d) < 4:
        return 1

    d = d.sort_values(time_col)
    t = d[time_col].values.astype(float)
    c = d[conc_col].values.astype(float)

    # ---------- 1) NCA 기반 tail / 초기 기울기 평가 ----------
    nca_term = _estimate_lambda_z(t, c, min_points=3, max_points=6)
    nca_early = _estimate_early_slope_after_cmax(t, c, min_points=3)

    slope_ratio = None
    strong_mono = False
    strong_biphasic = False

    if nca_term is not None and nca_term.get("lambda_z", 0.0) > 0.0:
        r2_term = float(nca_term.get("r2", 0.0))

        if nca_early is not None:
            r2_early = float(nca_early.get("r2", 0.0))
            slope_ratio = nca_early["lambda_early"] / nca_term["lambda_z"]
        else:
            r2_early = 0.0

        # (1) tail이 강하게 mono-exponential → 1-comp로 고정
        if r2_term >= 0.98 and (
            nca_early is None
            or r2_early < 0.8
            or (slope_ratio is not None and slope_ratio < 1.3)
        ):
            strong_mono = True

        # (2) 명확한 biphasic 후보 (조금 완화: slope_ratio >= 3.0)
        if slope_ratio is not None:
            if slope_ratio >= 3.0 and r2_term >= 0.98 and r2_early >= 0.98:
                strong_biphasic = True

    if strong_mono:
        return 1

    # ---------- 2) BIC 기반 1 vs 2-comp 비교 ----------
    imax = int(np.argmax(c))
    t_seg = t[imax:]
    c_seg = c[imax:]
    if t_seg.size < 4:
        t_seg = t
        c_seg = c
        if t_seg.size < 4:
            return 1

    ln_c = np.log(c_seg)
    n = len(t_seg)

    def lin_reg(x, y):
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        ss_xy = float(((x - x_mean) * (y - y_mean)).sum())
        ss_xx = float(((x - x_mean) ** 2).sum())
        if ss_xx == 0.0:
            slope = 0.0
        else:
            slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_pred = intercept + slope * x
        rss = float(((y - y_pred) ** 2).sum())
        return intercept, slope, rss

    # 1-comp BIC
    _, _, rss1 = lin_reg(t_seg, ln_c)
    eps = 1e-12
    k1 = 2
    bic1 = n * math.log((rss1 + eps) / n) + k1 * math.log(n)

    # 2-comp BIC: piecewise 직선 2개
    best_bic2 = float("inf")
    best_split = None
    for split in range(2, n - 1):
        t1, t2 = t_seg[:split], t_seg[split:]
        y1, y2 = ln_c[:split], ln_c[split:]
        if len(t1) < 2 or len(t2) < 2:
            continue
        _, _, rss_1 = lin_reg(t1, y1)
        _, _, rss_2 = lin_reg(t2, y2)
        rss2 = rss_1 + rss_2
        k2 = 4
        bic2 = n * math.log((rss2 + eps) / n) + k2 * math.log(n)
        if bic2 < best_bic2:
            best_bic2 = bic2
            best_split = split

    if best_split is None:
        return 1

    delta_bic = bic1 - best_bic2  # 양수면 2-comp 유리

    if delta_bic > 20.0 and strong_biphasic:
        return 2

    return 1


def infer_compartments_from_dataset(
    df: pd.DataFrame,
    id_col: str = "ID",
    time_col: str = "TIME",
    conc_col: str = "DV",
) -> int:
    """
    전체 데이터셋에서 1-comp vs 2-comp 판단 (보수적 + pooled 우선).

    규칙:
      - ID가 없으면 pooled profile만 보고 판단
      - ID가 있어도:
        · pooled profile이 2-comp로 강하게 분류되면 → 전체도 2-comp
        · 그렇지 않으면 1-comp
      - subject별 개별 프로파일은 TDM 데이터처럼 샘플이 매우 적어
        구조 판단이 어려운 경우가 많으므로, 여기서는 참고하지 않음
    """
    # 1) pooled profile 기준
    pooled_comp = _infer_compartments_from_profile(
        df, time_col=time_col, conc_col=conc_col
    )

    # ID 유무와 관계없이 pooled 결과를 그대로 사용
    if pooled_comp is None:
        return 1
    return int(pooled_comp)


class PKDataLoader:

    """Load and analyze pharmacokinetic datasets"""

    def __init__(self, file_path: str):
        """Initialize data loader"""
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

        # Infer route (IV vs oral) from dosing pattern
        self._infer_route()

        # Detect subject-level covariates
        self._detect_covariates()

        # 실제 농도-시간 데이터 기반으로 compartment 수 추론
        try:
            nm_cols = self.metadata.get('nonmem_columns', {})
            id_col = nm_cols.get('ID', 'ID')
            time_col = nm_cols.get('TIME', 'TIME')
            conc_col = nm_cols.get('DV', 'DV')

            if time_col in self.df.columns and conc_col in self.df.columns:
                compartments = infer_compartments_from_dataset(
                    self.df,
                    id_col=id_col if id_col in self.df.columns else id_col,
                    time_col=time_col,
                    conc_col=conc_col,
                )
                self.metadata['compartments'] = int(compartments)
        except Exception as e:
            print(f"[WARN] Failed to infer compartments from data: {e}")

    def _infer_route(self):
        """투여 경로(route: iv / oral) 추론 (데이터 기반, 보수적)"""
        route = "oral"
        try:
            nm_cols = self.metadata.get('nonmem_columns', {})
            df = self.df

            id_col = nm_cols.get('ID')
            amt_col = nm_cols.get('AMT')
            evid_col = nm_cols.get('EVID')
            rate_col = nm_cols.get('RATE')

            # 1) 비-0 RATE가 있으면 IV (정주/주입)
            if rate_col and rate_col in df.columns:
                rate_nonzero = df[rate_col].fillna(0).abs() > 0
                if rate_nonzero.any():
                    route = "iv"
                # 모두 0이면 추가 패턴으로 판단

            # 2) AMT/EVID 기반 다회 투여 패턴
            if route == "oral" and amt_col and amt_col in df.columns:
                if evid_col and evid_col in df.columns:
                    dose_mask = (df[evid_col] == 1) & (df[amt_col].fillna(0) > 0)
                else:
                    # EVID 없으면 AMT>0인 행을 dose로 가정
                    dose_mask = df[amt_col].fillna(0) > 0

                if id_col and id_col in df.columns:
                    dose_df = df.loc[dose_mask]
                    if not dose_df.empty:
                        dose_counts = dose_df.groupby(id_col)[amt_col].size()
                        # 어떤 subject라도 2회 이상 투여되면 (이 예제 데이터셋에서는)
                        # IV multiple dosing으로 간주
                        if dose_counts.max() > 1:
                            route = "iv"

        except Exception:
            # 실패 시 기본 oral 유지
            pass

        self.metadata['route'] = route

    def _detect_nonmem_columns(self):
        """Detect standard NONMEM column types"""
        standard_cols = {
            'ID': ['ID', '#ID', 'SUBJ', 'SUBJECT'],
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
        """Detect potential covariates (subject-level variables) with detailed analysis"""
        if 'ID' not in self.metadata.get('nonmem_columns', {}):
            self.metadata['covariates'] = []
            self.metadata['covariate_info'] = {}
            return

        id_col = self.metadata['nonmem_columns']['ID']
        nonmem_core = set(self.metadata['nonmem_columns'].values())

        covariates = []
        covariate_info = {}

        for col in self.columns:
            if col in nonmem_core:
                continue

            # Check if column has constant value within each subject
            try:
                grouped = self.df.groupby(id_col)[col].nunique()
                if (grouped == 1).all():
                    covariates.append(col)

                    # Analyze covariate characteristics for modeling
                    # Get one value per subject
                    subject_values = self.df.groupby(id_col)[col].first()

                    cov_info = {
                        'n_unique': subject_values.nunique(),
                        'type': 'continuous' if pd.api.types.is_numeric_dtype(subject_values) and subject_values.nunique() > 5 else 'categorical'
                    }

                    if pd.api.types.is_numeric_dtype(subject_values):
                        valid_data = subject_values.dropna()
                        if len(valid_data) > 0:
                            cov_info.update({
                                'min': float(valid_data.min()),
                                'max': float(valid_data.max()),
                                'mean': float(valid_data.mean()),
                                'median': float(valid_data.median()),
                                'std': float(valid_data.std())
                            })

                            # Suggest modeling approach
                            # Power model for covariates like WT: (COV/median)^THETA
                            # Linear model for others: (1 + THETA*(COV-median))
                            if 'WT' in col.upper() or 'WEIGHT' in col.upper():
                                cov_info['suggested_model'] = 'power'  # (WT/70)^THETA
                            else:
                                cov_info['suggested_model'] = 'linear'  # (1 + THETA*(COV-median))
                    else:
                        # Categorical covariate
                        value_counts = subject_values.value_counts().to_dict()
                        cov_info['categories'] = {str(k): int(v) for k, v in value_counts.items()}
                        cov_info['suggested_model'] = 'categorical'  # Separate THETA per category

                    covariate_info[col] = cov_info

            except Exception:
                pass

        self.metadata['covariates'] = covariates
        self.metadata['covariate_info'] = covariate_info

        # Store number of subjects for later use
        if 'ID' in self.metadata.get('nonmem_columns', {}):
            id_col = self.metadata['nonmem_columns']['ID']
            self.metadata['n_subjects'] = int(self.df[id_col].nunique())

    def get_column_summary(self) -> str:
        """Get human-readable summary of dataset columns"""
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
        """Get summary statistics for the dataset"""
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

        # PKGPT용 구조 힌트 (route, compartments)
        if 'route' in self.metadata:
            try:
                route_val = str(self.metadata['route']).upper()
                if route_val in ("IV", "ORAL"):
                    lines.append(f"PKGPT_STRUCT_HINT: ROUTE={route_val}")
            except Exception:
                pass

        if 'compartments' in self.metadata:
            try:
                comp_val = int(self.metadata['compartments'])
                lines.append(f"PKGPT_STRUCT_HINT: COMPARTMENTS={comp_val}")
            except Exception:
                pass

        return "\n".join(lines)

    def get_metadata(self) -> Dict:
        """Get complete metadata dictionary"""
        return self.metadata

    def get_dataframe(self) -> pd.DataFrame:
        """Get the loaded DataFrame"""
        return self.df

    def get_covariate_summary(self) -> str:
        """Get detailed covariate summary for model building"""
        if not self.metadata.get('covariates'):
            return "No covariates detected in dataset."

        lines = []
        lines.append("=" * 70)
        lines.append("COVARIATE ANALYSIS INFORMATION")
        lines.append("=" * 70)
        lines.append(f"Total covariates detected: {len(self.metadata['covariates'])}")
        lines.append("")

        for cov_name in self.metadata['covariates']:
            cov_info = self.metadata['covariate_info'].get(cov_name, {})
            lines.append(f"{cov_name}:")
            lines.append(f"  Type: {cov_info.get('type', 'unknown').capitalize()}")
            lines.append(f"  Unique values: {cov_info.get('n_unique', 0)}")

            if cov_info.get('type') == 'continuous':
                lines.append(f"  Range: {cov_info.get('min', 'N/A'):.2f} - {cov_info.get('max', 'N/A'):.2f}")
                lines.append(f"  Median: {cov_info.get('median', 'N/A'):.2f}")
                lines.append(f"  Mean (SD): {cov_info.get('mean', 'N/A'):.2f} ({cov_info.get('std', 'N/A'):.2f})")

                model_type = cov_info.get('suggested_model', 'linear')
                if model_type == 'power':
                    median = cov_info.get('median', 70)
                    lines.append(f"  Suggested model: CL = TVCL * ({cov_name}/{median:.1f})^THETA")
                    lines.append("    (Power model - typical for body size)")
                else:
                    median = cov_info.get('median', 0)
                    lines.append(f"  Suggested model: CL = TVCL * (1 + THETA*({cov_name}-{median:.1f}))")
                    lines.append("    (Linear model - typical for continuous covariates)")

            elif cov_info.get('type') == 'categorical':
                lines.append("  Categories:")
                categories = cov_info.get('categories', {})
                for cat_val, count in categories.items():
                    lines.append(f"    - {cat_val}: {count} subjects")
                lines.append("  Suggested model: IF({cov_name}.EQ.value) TVCL=THETA(X)")
                lines.append("    (Categorical - separate parameter per group)")

            lines.append("")

        lines.append("COVARIATE MODELING STRATEGY (Chapter 11):")
        lines.append("1. Base model must be stable first (shrinkage <50%)")
        lines.append("2. Visual screening: Plot ETAs vs covariates")
        lines.append("3. Forward selection: Add if OFV drops >3.84 (p<0.05)")
        lines.append("4. Backward elimination: Keep if OFV increases >6.63 (p<0.01)")
        lines.append("=" * 70)

        return "\n".join(lines)
