#!/usr/bin/env python3
"""
Synthetic dataset generator for "Minimal Feature Subset" testing
----------------------------------------------------------------
- Creates a sizeable dataset with the exact schemas and value sets you specified
- Enforces per-row constraints: ftds <= ils_count <= accounts
- Computes: l2ftd = ftds / accounts, ils_l2ftd = ftds / ils_count
- Plants several "winner" segments to validate that your search finds them

Outputs:
  - mfs_synth.parquet (preferred) or mfs_synth.csv (fallback if no pyarrow/fastparquet)
  - mfs_planted_segments_summary.csv

You can adjust N_BASE and N_PER_SEG for size.
"""

import argparse
import sys
import numpy as np
import pandas as pd

# -----------------------------
# Utilities
# -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def logit(p):
    return np.log(p / (1 - p))

def try_to_parquet(df: pd.DataFrame, path: str) -> bool:
    """Try saving to Parquet; return True if succeeded, else False."""
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception as e:
        print(f"[info] Parquet save failed ({e.__class__.__name__}: {e}). Will write CSV instead.", file=sys.stderr)
        return False

# -----------------------------
# Main generator
# -----------------------------
def main(N_BASE: int, N_PER_SEG: int, seed: int):
    rng = np.random.default_rng(seed)

    # 1) Define categorical domains (exactly as requested)
    master_country_code_vals = np.array([
     'AD','AE','AF','AG','AI','AL','AM','AO','AR','AS','AT','AU','AW','AX','AZ','BA','BB','BD','BE','BF','BG','BH','BJ','BM','BN','BO','BR','BS','BT','BW','BY','BZ','CA','CD','CF','CG','CH','CI','CL','CM','CN','CO','CR','CS','CU','CV','CY','CZ','DE','DJ','DK','DM','DO','DZ','EC','EE','EG','EH','ER','ES','ET','FI','FK','FO','FR','GA','GB','GD','GE','GF','GG','GH','GI','GM','GN','GP','GQ','GR','GT','GU','GY','HK','HN','HR','HT','HU','ID','IE','IL','IN','IO','IQ','IR','IS','IT','JM','JO','JP','KE','KG','KH','KI','KM','KN','KP','KR','KW','KY','KZ','LA','LB','LC','LI','LK','LR','LS','LT','LU','LV','LY','MA','MD','MG','MK','ML','MM','MN','MO','MP','MQ','MR','MT','MU','MV','MW','MX','MY','MZ','NA','NE','NF','NG','NI','NL','NO','NP','NR','NZ','OM','PA','PE','PF','PG','PH','PK','PL','PR','PS','PT','PW','PY','QA','RO','RS','RU','RW','SA','SC','SD','SE','SG','SH','SI','SJ','SK','SL','SM','SN','SO','SR','ST','SV','SY','TC','TD','TG','TH','TJ','TK','TL','TM','TN','TO','TR','TT','TV','TW','TZ','UA','UG','UM','US','UY','UZ','VA','VC','VE','VI','VN','VU','WF','WS','YE','YT','ZA','ZM','ZW'
    ])

    initial_platform_vals = np.array(['fx','vp','op','ss','vr'])
    account_license_vals  = np.array(['fx','vp','op','ss'])
    account_status_vals   = np.array(['demo registered','lead','real registered','real verified'])
    initial_lead_status_vals = np.array(['gecersiz','potansiyel yok',None,'belirlenemedi','cevapsiz','potansiyel var','yuksek potansiyel (50k uzeri)','50k potensiyel'], dtype=object)
    lead_status_vals = np.array(['gecersiz','cevapsiz',None,'belirlenemedi','potansiyel var','potansiyel yok','yuksek potansiyel (50k uzeri)'], dtype=object)

    age_group_vals = np.array(['','18_24_age','45_54_age','35_44_age','under_18_age','55_64_age','25_34_age','75_plus_age','65_74_age'])
    annual_income_vals = np.array(['','15000_50000_annual','less_than_15000_annual','100000_250000_annual','more_than_250000_annual'])
    savings_vals = np.array(['','25000_50000_savings','less_than_5000_savings','100000_250000_savings','250000_500000_savings','5000_25000_savings','50000_100000_savings'])
    knowledge_vals = np.array(['','none','role_in_financial_services','financial_qualification','previous_trading_experience'])
    os_name_vals = np.array(['','Android','Linux','Windows','Mac OS X','iOS','Chrome OS','Tizen','CentOS','Ubuntu','Other'])
    device_type_vals = np.array(['Other','Mobile','Desktop','Tablet'])

    sms_demo_vals = np.array([np.nan, 0.0, 1.0], dtype=float)  # for sms_verification and demo_trade_flag

    # Slightly favor boosted countries so segments have support
    country_weights = np.ones_like(master_country_code_vals, dtype=float)
    for c in ['GB','DE','AE','US','SG','AU','FR','IT']:
        country_weights[master_country_code_vals == c] = 4.0
    country_weights = country_weights / country_weights.sum()

    # 2) Planted "winner" segments
    planted_segments = [
        {"name": "GB_fx_realverified", "conds": {"master_country_code":"GB","initial_platform":"fx","account_status":"real verified"}, "ils_boost": 0.8, "conv_boost": 1.5},
        {"name": "DE_vp_mobile", "conds": {"master_country_code":"DE","initial_platform":"vp","device_type":"Mobile"}, "ils_boost": 0.5, "conv_boost": 1.1},
        {"name": "AE_fx_finqual", "conds": {"master_country_code":"AE","account_license":"fx","knowledge_of_trading":"financial_qualification"}, "ils_boost": 0.6, "conv_boost": 1.8},
        {"name": "US_fx_ultra_leadstatus", "conds": {"master_country_code":"US","initial_platform":"fx","lead_status":"yuksek potansiyel (50k uzeri)"}, "ils_boost": 0.7, "conv_boost": 1.6},
        {"name": "SG_hiwealth_roleFS", "conds": {"master_country_code":"SG","savings":"250000_500000_savings","knowledge_of_trading":"role_in_financial_services","annual_income":"more_than_250000_annual"}, "ils_boost": 0.7, "conv_boost": 1.7},
        {"name": "AU_ios_35_44", "conds": {"master_country_code":"AU","os_name":"iOS","age_group":"35_44_age"}, "ils_boost": 0.3, "conv_boost": 0.9},
        {"name": "FR_op_realverified", "conds": {"master_country_code":"FR","initial_platform":"op","account_status":"real verified"}, "ils_boost": 0.4, "conv_boost": 1.0},
        {"name": "IT_desktop_verified_demo", "conds": {"master_country_code":"IT","device_type":"Desktop","account_status":"real verified","demo_trade_flag":1.0}, "ils_boost": 0.2, "conv_boost": 0.8},
    ]

    def sample_from(vals, size, p=None):
        return rng.choice(vals, size=size, p=p)

    def sample_float_nan(size, probs=(0.2, 0.4, 0.4)):
        return rng.choice(sms_demo_vals, size=size, p=probs)

    # 3) Base random frame
    base = pd.DataFrame({
        "master_country_code": sample_from(master_country_code_vals, N_BASE, p=country_weights),
        "initial_platform":    sample_from(initial_platform_vals, N_BASE),
        "account_license":     sample_from(account_license_vals,  N_BASE),
        "account_status":      sample_from(account_status_vals,   N_BASE),
        "initial_lead_status": sample_from(initial_lead_status_vals, N_BASE),
        "lead_status":         sample_from(lead_status_vals,      N_BASE),
        "age_group":           sample_from(age_group_vals,        N_BASE),
        "annual_income":       sample_from(annual_income_vals,    N_BASE),
        "savings":             sample_from(savings_vals,          N_BASE),
        "knowledge_of_trading":sample_from(knowledge_vals,        N_BASE),
        "os_name":             sample_from(os_name_vals,          N_BASE),
        "device_type":         sample_from(device_type_vals,      N_BASE),
        "sms_verification":    sample_float_nan(N_BASE, probs=(0.25,0.35,0.40)),
        "demo_trade_flag":     sample_float_nan(N_BASE, probs=(0.30,0.35,0.35)),
        "self_reg_real":       rng.choice([0,1], size=N_BASE, p=[0.7,0.3]),
        "dummy_db_flag":       rng.choice([0,1], size=N_BASE, p=[0.9,0.1]),
        "dummy":               rng.choice([0,1], size=N_BASE, p=[0.5,0.5]),
    })

    # Segment-specific frames
    seg_frames = []
    for seg in planted_segments:
        df = pd.DataFrame({
            "master_country_code": sample_from(master_country_code_vals, N_PER_SEG, p=country_weights),
            "initial_platform":    sample_from(initial_platform_vals,    N_PER_SEG),
            "account_license":     sample_from(account_license_vals,     N_PER_SEG),
            "account_status":      sample_from(account_status_vals,      N_PER_SEG),
            "initial_lead_status": sample_from(initial_lead_status_vals, N_PER_SEG),
            "lead_status":         sample_from(lead_status_vals,         N_PER_SEG),
            "age_group":           sample_from(age_group_vals,           N_PER_SEG),
            "annual_income":       sample_from(annual_income_vals,       N_PER_SEG),
            "savings":             sample_from(savings_vals,             N_PER_SEG),
            "knowledge_of_trading":sample_from(knowledge_vals,           N_PER_SEG),
            "os_name":             sample_from(os_name_vals,             N_PER_SEG),
            "device_type":         sample_from(device_type_vals,         N_PER_SEG),
            "sms_verification":    sample_float_nan(N_PER_SEG, probs=(0.15,0.35,0.50)),
            "demo_trade_flag":     sample_float_nan(N_PER_SEG, probs=(0.15,0.30,0.55)),
            "self_reg_real":       rng.choice([0,1], size=N_PER_SEG, p=[0.4,0.6]),
            "dummy_db_flag":       rng.choice([0,1], size=N_PER_SEG, p=[0.95,0.05]),
            "dummy":               rng.choice([0,1], size=N_PER_SEG, p=[0.5,0.5]),
        })
        for k, v in seg["conds"].items():
            df[k] = v
        df["__seg_name__"] = seg["name"]
        seg_frames.append(df)

    df = pd.concat([base] + seg_frames, ignore_index=True)

    # 4) Generate metrics with boosted likelihoods
    ils_base_logit  = logit(0.35)   # base probability that an account creates ILS
    conv_base_logit = logit(0.07)   # base probability that an ILS converts to FTD

    ils_lin  = ils_base_logit  + rng.normal(0, 0.15, size=len(df))
    conv_lin = conv_base_logit + rng.normal(0, 0.20, size=len(df))

    for seg in planted_segments:
        mask = np.ones(len(df), dtype=bool)
        for k, v in seg["conds"].items():
            if isinstance(v, float) and np.isnan(v):
                mask &= df[k].isna().to_numpy()
            else:
                mask &= (df[k].to_numpy() == v)
        ils_lin[mask]  += seg["ils_boost"]
        conv_lin[mask] += seg["conv_boost"]

    # Optional soft correlations
    conv_lin += np.where(df["knowledge_of_trading"].isin(
        ["financial_qualification","previous_trading_experience","role_in_financial_services"]
    ).to_numpy(), 0.15, 0.0)

    is_verified = (df["account_status"].to_numpy() == "real verified")
    ils_lin  += np.where(is_verified, 0.10, 0.0)
    conv_lin += np.where(is_verified, 0.20, 0.0)

    p_ils  = sigmoid(ils_lin)
    p_conv = sigmoid(conv_lin)

    # Accounts 1..8, biased lower
    accounts = rng.integers(1, 9, size=len(df))
    ils_count = 1 + rng.binomial(np.maximum(accounts-1, 0), p_ils)
    ftds      = rng.binomial(ils_count, p_conv)

    # Sanity: constraints
    assert np.all(ftds <= ils_count), "Constraint violated: ftds <= ils_count"
    assert np.all(ils_count <= accounts), "Constraint violated: ils_count <= accounts"
    assert np.all(accounts >= 1), "Accounts must be >= 1"

    # Ratios
    df["accounts"]  = accounts.astype("int64")
    df["ils_count"] = ils_count.astype("int64")
    df["ftds"]      = ftds.astype("int64")
    df["l2ftd"]     = df["ftds"] / df["accounts"]
    df["ils_l2ftd"] = df["ftds"] / df["ils_count"]

    # 5) Persist
    out_parquet = "mfs_synth.parquet"
    out_csv     = "mfs_synth.csv"
    saved_parquet = try_to_parquet(df.drop(columns=["__seg_name__"], errors="ignore"), out_parquet)
    if not saved_parquet:
        df.drop(columns=["__seg_name__"], errors="ignore").to_csv(out_csv, index=False)
        print(f"[ok] Wrote CSV: {out_csv}")
    else:
        print(f"[ok] Wrote Parquet: {out_parquet}")

    # 6) Summarise planted winners and persist
    summary_rows = []
    for seg in planted_segments:
        mask = pd.Series(True, index=df.index)
        for k, v in seg["conds"].items():
            if isinstance(v, float) and np.isnan(v):
                mask &= df[k].isna()
            else:
                mask &= (df[k] == v)
        sub = df[mask]
        acc_sum = int(sub["accounts"].sum())
        ils_sum = int(sub["ils_count"].sum())
        ftd_sum = int(sub["ftds"].sum())
        l2      = ftd_sum / acc_sum if acc_sum > 0 else 0.0
        ils_l2  = ftd_sum / ils_sum if ils_sum > 0 else 0.0
        summary_rows.append({
            "segment_name": seg["name"],
            "num_rows": len(sub),
            "accounts_sum": acc_sum,
            "ils_count_sum": ils_sum,
            "ftds_sum": ftd_sum,
            "l2ftd": l2,
            "ils_l2ftd": ils_l2,
            "conditions": seg["conds"],
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["ils_l2ftd","ftds_sum"], ascending=[False, False]
    ).reset_index(drop=True)

    summary_df.to_csv("mfs_planted_segments_summary.csv", index=False)
    print("[ok] Wrote planted winners summary: mfs_planted_segments_summary.csv\n")

    # Print a concise overview to stdout
    print("Top planted segments by ils_l2ftd:")
    print(summary_df[["segment_name","num_rows","accounts_sum","ils_count_sum","ftds_sum","l2ftd","ils_l2ftd"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for Minimal Feature Subset testing.")
    parser.add_argument("--base", type=int, default=200_000, help="Number of background rows (default: 200000)")
    parser.add_argument("--per-seg", type=int, default=8_000, help="Rows per planted segment (default: 8000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()
    main(args.base, args.per_seg, args.seed)
