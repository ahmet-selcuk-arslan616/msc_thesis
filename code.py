import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from matplotlib.ticker import PercentFormatter, MaxNLocator
from matplotlib import cm, colors as mcolors
from itertools import combinations
from country_converter import CountryConverter
from itertools import combinations
from mesa import Agent, Model
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
os.environ['PYTHONHASHSEED'] = str(616)
random.seed(616)
np.random.seed(616)


def create_IVs_with_weights(
    cepii_data = None,
    cow_alliances_data = None,
    cow_conflicts_data = None,
    cow_capabilities_data = None,
    cow_to_iso_data = None,
    vdem_FOE_data = None,
    ucdp_BRDs_data = None,
    sipri_ME_data = None
):
    ### strategic environment index
    ## importing relevant raw datasets

    # CEPII GeoDist:
    distance_df = cepii_data.copy()

    # Correlates of War (COW) Formal Alliances:
    cow_alliances_df = cow_alliances_data.copy()
    cow_alliances_df = cow_alliances_df[cow_alliances_df["dyad_st_year"] > 1994].reset_index(drop=True)

    # Correlates of War (COW) Militarized Interstate Disputes (MID):
    cow_conflicts_df = cow_conflicts_data.copy()
    cow_conflicts_df = cow_conflicts_df[cow_conflicts_df["styear"] > 1994].reset_index(drop=True)

    # Correlates of War (COW) National Material Capabilities (NMC):
    cow_capabilities_df = cow_capabilities_data.copy()
    cow_capabilities_df = cow_capabilities_df[cow_capabilities_df["year"] > 1994].reset_index(drop=True)

    # COW to ISO3 mapping
    cow_to_iso = cow_to_iso_data.copy()
    cow_to_iso = cow_to_iso.dropna(subset=["cow_id", "iso_id"])
    cow_to_iso["cow_id"] = cow_to_iso["cow_id"].astype(int)
    cow_to_iso["iso_id"] = cow_to_iso["iso_id"].astype(int)
    cow_to_iso["iso3"] = cow_to_iso["iso3"].astype(str)
    COW_TO_ISO3 = {cow_id: iso3 for cow_id, iso3 in zip(cow_to_iso["cow_id"], cow_to_iso["iso3"])}

    # Alliances formatting
    cow_alliances_df["state_name1_iso"] = cow_alliances_df["ccode1"].map(COW_TO_ISO3)
    cow_alliances_df["state_name2_iso"] = cow_alliances_df["ccode2"].map(COW_TO_ISO3)
    cow_alliances_df["state_name1_iso"] = cow_alliances_df["state_name1_iso"].replace("not found", "YUG")
    cow_alliances_df["state_name2_iso"] = cow_alliances_df["state_name2_iso"].replace("not found", "YUG")

    for col in ["defense", "neutrality", "nonaggression", "entente"]:
        if col in cow_alliances_df.columns:
            cow_alliances_df[col] = cow_alliances_df[col].fillna(0).astype(int)
        else:
            cow_alliances_df[col] = 0

    weights = [4, 3, 2, 1]  
    cow_alliances_df["alliance_strength"] = cow_alliances_df[["defense","neutrality","nonaggression","entente"]].mul(weights, axis=1).max(axis=1)
    cow_alliances_df["alliance_strength"] = cow_alliances_df["alliance_strength"].fillna(0).astype(int)

    # Capabilities formatting
    cow_capabilities_df["stateabb"] = cow_capabilities_df["ccode"].map(COW_TO_ISO3)
    cleaned_cow_capabilities_df = cow_capabilities_df[["stateabb", "year", "cinc"]].copy()

    # Conflicts formatting
    cow_conflicts_df.loc[cow_conflicts_df['hostlev'] < 0, 'hostlev'] = 0
    cow_conflicts_df["stabb"] = cow_conflicts_df["ccode"].map(COW_TO_ISO3)
    cow_conflicts_df = cow_conflicts_df.dropna(subset=["stabb"]).reset_index(drop=True)
    cow_conflicts_df.columns = cow_conflicts_df.columns.str.lower()

    col_disp = "dispnum"
    col_sty  = "styear"
    col_endy = "endyear"
    col_host = "hostlev"
    col_iso3 = "stabb"

    if "sidea" in cow_conflicts_df.columns:
        side_series = np.where(cow_conflicts_df["sidea"].astype(int) == 1, "A", "B")
    elif "side" in cow_conflicts_df.columns:
        side_series = cow_conflicts_df["side"].map({1: "A", 2: "B"}).fillna("B")

    cow_conflicts_df = cow_conflicts_df[[col_disp, col_sty, col_endy, col_host, col_iso3]].copy()
    cow_conflicts_df["side"] = side_series

    cow_conflicts_df["styear"]  = cow_conflicts_df[col_sty].astype(int)
    cow_conflicts_df["endyear"] = cow_conflicts_df[col_endy].astype(int)
    cow_conflicts_df["years"]   = cow_conflicts_df.apply(lambda r: list(range(r["styear"], r["endyear"] + 1)), axis=1)
    cow_conflicts_df = cow_conflicts_df.explode("years", ignore_index=True).rename(columns={"years": "year"})

    max_host_by_disp_year = (
        cow_conflicts_df.groupby([col_disp, "year"], as_index=False)[col_host].max()
        .rename(columns={col_host: "disp_year_host_max"})
    )

    participants = (
        cow_conflicts_df.groupby([col_disp, "year", "side"])[col_iso3]
        .unique()
        .reset_index()
        .pivot(index=[col_disp, "year"], columns="side", values=col_iso3)
        .reset_index()
    )
    participants = participants.merge(max_host_by_disp_year, on=[col_disp, "year"], how="left")

    def _to_list(x):
        if x is None: return []
        if isinstance(x, float) and pd.isna(x): return []
        if isinstance(x, (list, tuple)): return list(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return [x]

    def make_pairs(row):
        a_list = _to_list(row.get("A"))
        b_list = _to_list(row.get("B"))
        if len(a_list) == 0 or len(b_list) == 0: return []
        pairs = []
        yr = int(row["year"])
        host = float(row["disp_year_host_max"]) if pd.notna(row["disp_year_host_max"]) else np.nan
        for a in a_list:
            for b in b_list:
                iso1, iso2 = sorted([a, b])
                pairs.append((iso1, iso2, yr, host))
        return pairs

    pair_rows = []
    for _, r in participants.iterrows():
        pair_rows.extend(make_pairs(r))

    dyad_year_from_disputes = pd.DataFrame(pair_rows, columns=["iso3_1", "iso3_2", "year", "host_from_disp"])

    dyad_year_host = (
        dyad_year_from_disputes
        .groupby(["iso3_1", "iso3_2", "year"], as_index=False)["host_from_disp"]
        .max()
        .rename(columns={"host_from_disp": "max_hostility"})
    )

    all_countries = sorted(cow_conflicts_df[col_iso3].dropna().unique().tolist())
    min_year = int(cow_conflicts_df["year"].min())
    max_year = 2012
    all_dyads = list(combinations(all_countries, 2))
    universe = pd.DataFrame(
        [(a, b, y) for (a, b) in all_dyads for y in range(min_year, max_year + 1)],
        columns=["iso3_1", "iso3_2", "year"]
    ).assign(max_hostility=0)

    base_cow_conflicts_df = universe.merge(dyad_year_host, on=["iso3_1", "iso3_2", "year"], how="left", suffixes=("", "_obs"))
    base_cow_conflicts_df["max_hostility"] = base_cow_conflicts_df["max_hostility_obs"].fillna(base_cow_conflicts_df["max_hostility"])
    base_cow_conflicts_df = base_cow_conflicts_df.drop(columns=["max_hostility_obs"]).sort_values(["year", "iso3_1", "iso3_2"]).reset_index(drop=True)
    base_cow_conflicts_df = base_cow_conflicts_df.rename(columns={"iso3_1": "iso3_i", "iso3_2": "iso3_j"})
    base_cow_conflicts_df["max_hostility"] = base_cow_conflicts_df["max_hostility"].astype(int)

    base_cow_conflicts_df = pd.concat([
        base_cow_conflicts_df,
        base_cow_conflicts_df.rename(columns={'iso3_i': 'iso3_j', 'iso3_j': 'iso3_i'})
    ], ignore_index=True).drop_duplicates()

    # Capabilities merge
    pre_conflict_and_capabilities_df = base_cow_conflicts_df.merge(cleaned_cow_capabilities_df, how="left", left_on=["iso3_i", "year"], right_on=["stateabb", "year"])
    pre_conflict_and_capabilities_df = pre_conflict_and_capabilities_df.rename(columns={"cinc": "cinc_i"}).drop(columns=["stateabb"])

    conflict_and_capabilities_df = pre_conflict_and_capabilities_df.merge(cleaned_cow_capabilities_df, how="left", left_on=["iso3_j", "year"], right_on=["stateabb", "year"])
    conflict_and_capabilities_df = conflict_and_capabilities_df.rename(columns={"cinc": "cinc_j"}).drop(columns=["stateabb"])

    # Distances merge
    conflict_and_capabilities_df["k_min"] = np.minimum(conflict_and_capabilities_df["iso3_i"].values, conflict_and_capabilities_df["iso3_j"].values)
    conflict_and_capabilities_df["k_max"] = np.maximum(conflict_and_capabilities_df["iso3_i"].values, conflict_and_capabilities_df["iso3_j"].values)

    distance_df_key = distance_df.copy()
    distance_df_key["k_min"] = np.minimum(distance_df_key["iso_o"].values, distance_df_key["iso_d"].values)
    distance_df_key["k_max"] = np.maximum(distance_df_key["iso_o"].values, distance_df_key["iso_d"].values)
    distance_df_key = distance_df_key[["k_min","k_max","distcap"]].drop_duplicates()

    conflict_and_capabilities_and_distances_df = conflict_and_capabilities_df.merge(distance_df_key, on=["k_min","k_max"], how="left").drop(columns=["k_min","k_max"])

    # Master Dataset prep
    pairs = np.sort(conflict_and_capabilities_and_distances_df[['iso3_i','iso3_j']].to_numpy(), axis=1)
    conflict_and_capabilities_and_distances_df['pair_first'] = pairs[:, 0]
    conflict_and_capabilities_and_distances_df['pair_second'] = pairs[:, 1]

    pairs2 = np.sort(cow_alliances_df[['state_name1_iso','state_name2_iso']].to_numpy(), axis=1)
    cow_alliances_df['pair_first'] = pairs2[:, 0]
    cow_alliances_df['pair_second'] = pairs2[:, 1]

    conflict_and_capabilities_and_distances_df['year'] = pd.to_numeric(conflict_and_capabilities_and_distances_df['year'], errors='coerce').astype('Int64')
    cow_alliances_df['year'] = pd.to_numeric(cow_alliances_df['year'], errors='coerce').astype('int64')

    alli_yearly = cow_alliances_df.groupby(['pair_first','pair_second','year'], as_index=False)['alliance_strength'].max()

    master_dataset = conflict_and_capabilities_and_distances_df.merge(alli_yearly[['pair_first','pair_second','year','alliance_strength']], how='left', on=['pair_first','pair_second','year'], validate='m:1')
    master_dataset = master_dataset.drop(columns=['pair_first','pair_second']).dropna(subset=["cinc_i", "cinc_j", "distcap"]).assign(alliance_strength=lambda df: df["alliance_strength"].fillna(0))
    master_dataset["alliance_strength"] = master_dataset["alliance_strength"].astype(int)
    master_dataset = master_dataset[master_dataset['year'] <= 2012].reset_index(drop=True)
    pairs = np.sort(master_dataset[['iso3_i','iso3_j']].to_numpy(), axis=1)
    master_dataset['first'] = pairs[:, 0]
    master_dataset['second'] = pairs[:, 1]
    full_years = set(range(1995, 2013))
    mask = master_dataset.groupby(['first','second'])['year'].transform(lambda s: set(s) == full_years)
    master_dataset = master_dataset[mask].drop(columns=['first','second'])
    def modify_weights_for_w_ij(master_data = master_dataset):
        df = master_data.copy()
        df["proximity_term"] = np.exp(-(df["distcap"] / 1000.0))
        df["norm_hostility"] = df["max_hostility"] / 5.0
        df["norm_alliance"] = df["alliance_strength"] / 4.0
        df["relationship_term"] = (1.0 + df["norm_hostility"]) / (1.0 + df["norm_alliance"])
        df["w_ij"] = df["proximity_term"] * df["relationship_term"]
        df["si_component"] = df["w_ij"] * df["cinc_j"]

        return df

    w_ij_df = modify_weights_for_w_ij()

    SEI_df = w_ij_df.groupby(["iso3_i", "year"], as_index=False)["si_component"].sum().rename(columns={"si_component": "Strategic_Environment_Index"})

    ### CLARITY INDEX
    master_dataset_clarity = master_dataset.copy()
    master_dataset_clarity["O_obviousness"] = (master_dataset_clarity["cinc_i"] - master_dataset_clarity["cinc_j"]).abs() / (master_dataset_clarity["cinc_i"] + master_dataset_clarity["cinc_j"])

    # VDem FOE
    vdem_FOE_df = vdem_FOE_data.copy()[["Code", "Year", "Freedom of expression and alternative sources of information index (central estimate)"]]
    vdem_FOE_df = vdem_FOE_df[vdem_FOE_df["Year"] > 1994].reset_index(drop=True).rename(columns={"Freedom of expression and alternative sources of information index (central estimate)": "V_visibility"})

    # UCDP
    ucdp_BRDs_df = ucdp_BRDs_data.copy()[["region", "year", "bd_best"]]
    ucdp_BRDs_df = ucdp_BRDs_df.groupby(['region', 'year'], as_index=False)['bd_best'].sum().rename(columns={'bd_best': 'total_regional_deaths'}).sort_values(['region', 'year'])
    ucdp_BRDs_df["region_list"] = ucdp_BRDs_df["region"].astype(str).str.split(",").apply(lambda xs: [int(x.strip()) for x in xs if x.strip().isdigit()])
    ucdp_BRDs_df["deaths_share"] = ucdp_BRDs_df["total_regional_deaths"] / ucdp_BRDs_df["region_list"].str.len()
    ucdp_BRDs_df = ucdp_BRDs_df.explode("region_list", ignore_index=True)
    ucdp_BRDs_df["region"] = ucdp_BRDs_df["region_list"].astype(int)
    ucdp_BRDs_df = ucdp_BRDs_df.groupby(["region", "year"], as_index=False).agg(total_deaths=("deaths_share", "sum")).sort_values(["region", "year"])

    ucdp_BRDs_df["prev3_mean"] = ucdp_BRDs_df.groupby("region")["total_deaths"].transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    previous = ucdp_BRDs_df["prev3_mean"]
    current  = ucdp_BRDs_df["total_deaths"]
    ucdp_BRDs_df["M_shock"] = np.where(
        previous.isna(), 0,
        np.where(previous == 0, (current > 0).astype(int), (current > 2 * previous).astype(int))
    ).astype(int)
    ucdp_BRDs_df = ucdp_BRDs_df[ucdp_BRDs_df["year"] > 1994].reset_index(drop=True)

    # SIPRI
    sipri_ME_df = sipri_ME_data.copy().iloc[1:, :]
    region_codes = {"Europe": 1, "Middle East": 2, "Asia & Oceania": 3, "Africa": 4, "Americas": 5}
    region_headers = set(region_codes.keys())
    sipri_ME_df["region"] = sipri_ME_df["Country"].where(sipri_ME_df["Country"].isin(region_headers)).ffill()
    sipri_ME_df["region_code"] = sipri_ME_df["region"].map(region_codes)
    regions_to_drop = ["Africa", "North Africa", "sub-Saharan Africa", "Americas", "Central America and the Caribbean", "North America", "South America", "Asia & Oceania", "Oceania", "South Asia", "East Asia", "South East Asia", "Central Asia", "Europe", "Central Europe", "Eastern Europe", "Western Europe", "Middle East", "European Union"]
    sipri_ME_df = sipri_ME_df.dropna(subset=["Country"])
    sipri_ME_df = sipri_ME_df[~sipri_ME_df["Country"].isin(regions_to_drop)].replace(["xxx", ". .", "..."], pd.NA).convert_dtypes().reset_index(drop = True)
    sipri_ME_df = pd.melt(sipri_ME_df, id_vars = ["Country", "region", "region_code"], var_name = "Year", value_name = "H_time-horizon")
    sipri_ME_df["Year"] = sipri_ME_df["Year"].astype(int)
    sipri_ME_df["iso3"] = CountryConverter().convert(names=sipri_ME_df["Country"], to='ISO3', not_found=None)
    sipri_ME_df = sipri_ME_df.dropna(subset=["iso3"]).reset_index(drop=True)
    mask = sipri_ME_df["Country"].astype(str).str.strip().str.casefold().eq("yugoslavia")
    sipri_ME_df.loc[mask, "iso3"] = "YUG"
    sipri_ME_df["iso3"] = sipri_ME_df["iso3"].replace({"not found": pd.NA, "None": pd.NA, "": pd.NA, None: pd.NA})
    sipri_ME_df = sipri_ME_df[~sipri_ME_df["iso3"].isin(["German Democratic Republic", "USSR"])].reset_index(drop=True)

    # ------------------------------------------
    clarity_index_df = master_dataset_clarity[["iso3_i", "iso3_j", "year"]].copy()

    # V_visibility
    clarity_index_df = clarity_index_df.merge(vdem_FOE_df, left_on=['iso3_j', 'year'], right_on=['Code', 'Year'], how='left').drop(columns=['Code', 'Year']).rename(columns={'V_visibility': 'V_visibility_j'})

    # H_time-horizon
    lk = sipri_ME_df.rename(columns={"Year": "year", "H_time-horizon": "H_time_horizon"})[["iso3", "year", "region", "region_code", "H_time_horizon"]].drop_duplicates(subset=["iso3", "year"])
    lk_i = lk[["iso3", "year", "region", "region_code"]].rename(columns={"iso3": "iso3_i", "region": "region_i", "region_code": "region_code_i"})
    clarity_index_df = clarity_index_df.merge(lk_i, on=["iso3_i", "year"], how="left")
    lk_j = lk.rename(columns={"iso3": "iso3_j", "region": "region_j", "region_code": "region_code_j", "H_time_horizon": "H_time_horizon_j"})[["iso3_j", "year", "region_j", "region_code_j", "H_time_horizon_j"]]
    clarity_index_df = clarity_index_df.merge(lk_j, on=["iso3_j", "year"], how="left")

    # O_obviousness
    clarity_index_df = clarity_index_df.merge(master_dataset_clarity[["iso3_i", "iso3_j", "year", "O_obviousness"]], on=["iso3_i", "iso3_j", "year"], how="left")

    # M_shock
    ucdp_BRDs_df["region"] = ucdp_BRDs_df["region"].astype(int)
    clarity_index_df = clarity_index_df.merge(ucdp_BRDs_df[["region", "year", "M_shock"]], left_on=["region_code_j", "year"], right_on=["region", "year"], how="left").drop(columns="region")
    clarity_index_df = clarity_index_df.rename(columns={"M_shock": "M_shock_j"})
    clarity_index_df["M_shock_j"] = clarity_index_df["M_shock_j"].fillna(0).astype(int)

    clarity_index_df = clarity_index_df.merge(w_ij_df[["iso3_i", "iso3_j", "year", "w_ij"]], on=["iso3_i", "iso3_j", "year"], how="left")
    df = clarity_index_df.copy()

    for col in ["V_visibility_j", "H_time_horizon_j", "O_obviousness"]:
        by_country_mean = df.groupby("iso3_j")[col].transform("mean")
        global_mean = df[col].mean()
        df[col + "_filled"] = df[col].fillna(by_country_mean).fillna(global_mean).clip(0, 1)

    core = (
        df["V_visibility_j_filled"] *
        df["H_time_horizon_j_filled"] *
        df["O_obviousness_filled"]
    ) ** (1.0 / 3.0)

    mat = 1.0 + df["M_shock_j"].fillna(0)
    df["C_ij"] = np.minimum(1.0, core * mat)

    by_country_mean_w = df.groupby("iso3_i")["w_ij"].transform("mean")
    global_mean_w = df["w_ij"].mean()
    df["w_ij_filled"] = df["w_ij"].fillna(by_country_mean_w).fillna(global_mean_w)

    df["weighted_clarity"] = df["C_ij"] * df["w_ij_filled"]

    agg_df = df.groupby(["iso3_i", "year"], as_index=False).agg(
        sum_weighted_clarity=("weighted_clarity", "sum"),
        sum_weights=("w_ij_filled", "sum")
    )

    agg_df["C_i_agg"] = np.where(
        agg_df["sum_weights"] > 0,
        agg_df["sum_weighted_clarity"] / agg_df["sum_weights"],
        0.0
    )

    df = df.merge(agg_df[["iso3_i", "year", "C_i_agg"]], on=["iso3_i", "year"], how="left")

    df["Clarity_Index"] = (df["C_ij"] + df["C_i_agg"]) / 2.0
    final_df = pd.merge(
        SEI_df,
        agg_df[["iso3_i", "year", "C_i_agg"]].rename(columns={"C_i_agg": "C_i_sys"}), # We keep the C_i_sys name purely for downstream code compatibility
        on=["iso3_i", "year"],
        how="inner"
    )
    
    return final_df, w_ij_df, cow_alliances_df

def create_IVVs_with_weights(
    IV_data = None,
    plad_data = None,
    CSP_data = None,
    vdem_data = None,
    wvs_data = None,
    RPC_data = None,
    PolityV_data = None,
    conflict_history_window_year = 40 # strategic culture, number of past years to take into consideration when creating conflict history
):

    final_df = IV_data.copy()
    plad_leaders_df = plad_data.copy()
    political_violence_df = CSP_data.copy()
    vdem = vdem_data.copy()
    wvs = wvs_data.copy()
    rpc_df = RPC_data.copy()
    polity_EC_df = PolityV_data.copy()

    #### IVVs
    ### leader images

    plad_leaders_df["startyear"] = pd.to_numeric(plad_leaders_df["startyear"], errors="coerce").astype("int64")
    plad_leaders_df["endyear"] = pd.to_numeric(plad_leaders_df["endyear"],   errors="coerce").astype("int64")
    plad_leaders_df = plad_leaders_df.sort_values(["idacr", "startyear", "endyear"], kind="mergesort").reset_index(drop=True)
    next_start = plad_leaders_df.groupby("idacr")["startyear"].shift(-1)
    plad_leaders_df["endyear_trim"] = np.where(
        next_start.notna() & (plad_leaders_df["endyear"] >= next_start),
        next_start - 1,
        plad_leaders_df["endyear"]
    ).astype("int64")
    df_trimmed = plad_leaders_df[plad_leaders_df["endyear_trim"] >= plad_leaders_df["startyear"]].copy()
    cols_to_keep = [c for c in df_trimmed.columns if c not in ["endyear_trim"]]
    records = []
    for abc, row in df_trimmed.iterrows():
        for y in range(int(row["startyear"]), int(row["endyear_trim"]) + 1):
            r = {c: row[c] for c in cols_to_keep}
            r["year"] = y
            records.append(r)

    plad_leaders_df = pd.DataFrame.from_records(records)

    plad_leaders_df["experience"] = plad_leaders_df.groupby(["idacr", "leader"])["year"].transform(lambda x: x - x.min())
    plad_leaders_df["risk_tol"] = np.where(plad_leaders_df["entry"].isin(["Irregular", "Foreign Imposition"]), 1, 0)

    plad_leaders_df["risk_normalized"] = (plad_leaders_df["risk_tol"] - plad_leaders_df["risk_tol"].min()) / (plad_leaders_df["risk_tol"].max() - plad_leaders_df["risk_tol"].min())
    plad_leaders_df["experience_normalized"] = (plad_leaders_df["experience"] - plad_leaders_df["experience"].min()) / (plad_leaders_df["experience"].max() - plad_leaders_df["experience"].min())
    plad_leaders_df["Leader_Profile_Index"] = (plad_leaders_df["risk_normalized"] + (1.0 - plad_leaders_df["experience_normalized"])) / 2.0

    plad_leaders_df["iso3"] = CountryConverter().convert(names=plad_leaders_df["country"], to='ISO3', not_found=None)
    plad_leaders_df = plad_leaders_df.dropna(subset=["iso3"]).reset_index(drop=True)
    mask = plad_leaders_df["country"].astype(str).str.strip().str.casefold().eq("yugoslavia")
    plad_leaders_df.loc[mask, "iso3"] = "YUG"
    plad_leaders_df["iso3"] = plad_leaders_df["iso3"].replace({"not found": pd.NA, "None": pd.NA, "": pd.NA, None: pd.NA})
    plad_leaders_df = plad_leaders_df[~plad_leaders_df["iso3"].isin(["German Democratic Republic", "USSR"])].reset_index(drop=True)
    plad_leaders_df = plad_leaders_df[(plad_leaders_df["year"] > 1994) & (plad_leaders_df["year"] < 2013)].reset_index(drop=True)
    plad_leaders_df = plad_leaders_df[["iso3", "year", "Leader_Profile_Index"]]

    # ------------------------------------------

    ### strategic culture
    political_violence_df = political_violence_df.rename(columns={"scode": "iso3"})

    vdem_MD_df = vdem.copy()[["country_name", "country_text_id", "year", "v2x_ex_military"]]
    vdem_MD_df = vdem_MD_df[(vdem_MD_df["year"] > 1994) & (vdem_MD_df["year"] < 2013)].reset_index(drop=True)
    vdem_MD_df = vdem_MD_df.rename(columns={"country_text_id": "iso3", "v2x_ex_military": "Military_dimension_index"}).drop(columns=["country_name"])

    wvs = wvs[["S002VS", "COUNTRY_ALPHA", "S020", "G006", "E012", "E037", "E114", "E116", "E125", "E128"]]
    wvs = wvs.rename(columns={
        "S002VS":"Chronology_of_EVS_WVS_waves","COUNTRY_ALPHA":"iso3","S020":"year",
        "G006":"How_proud_of_nationality","E012":"Willingness_to_fight_for_country",
        "E037":"Government_responsibility","E114":"Political_system_Having_a_strong_leader",
        "E116":"Having_the_army_rule","E125":"Satisfaction_with_the_people_in_national_office",
        "E128":"Country_is_run_by_big_interest_vs_for_all_people's_benefit"
    })
    wvs = wvs[(wvs["year"] >= 1995) & (wvs["year"] <= 2012)].reset_index(drop=True)

    cols = [c for c in wvs.columns if c not in ["Chronology_of_EVS_WVS_waves","iso3","year"]]
    wvs[cols] = wvs[cols].where(wvs[cols] >= 0, np.nan)

    wvs["Pride_norm"]        = (4 - wvs["How_proud_of_nationality"]) / 3
    wvs["Fight_norm"]        = wvs["Willingness_to_fight_for_country"]
    wvs["ArmyRule_norm"]     = (4 - wvs["Having_the_army_rule"]) / 3
    wvs["StrongLeader_norm"] = (4 - wvs["Political_system_Having_a_strong_leader"]) / 3
    wvs["GovtResp_norm"]     = (wvs["Government_responsibility"] - 1) / 9
    wvs["Satisfaction_norm"] = (4 - wvs["Satisfaction_with_the_people_in_national_office"]) / 3
    wvs["RunForAll_norm"]    = (wvs["Country_is_run_by_big_interest_vs_for_all_people's_benefit"] - 1)

    norm_cols = ["Pride_norm","Fight_norm","ArmyRule_norm","StrongLeader_norm",
                "GovtResp_norm","Satisfaction_norm","RunForAll_norm"]

    wvs = wvs.groupby(["iso3","year"], as_index=False)[norm_cols].mean().sort_values(["iso3","year"]).reset_index(drop=True)

    wave_years = {3:(1994,1998), 4:(1999,2004), 5:(2005,2009), 6:(2010,2012), 7:(2017,2022)}

    def year_to_wave(y):
        for w,(a,b) in wave_years.items():
            if a <= y <= b: return w
        return np.nan

    wvs["wave"] = wvs["year"].map(year_to_wave).astype("Int64")

    YEARS = range(1995, 2013)
    years_min, years_max = 1995, 2012

    out = []
    for (iso, w), g in wvs.groupby(["iso3","wave"]):
        if pd.isna(w):
            out.append(g.set_index("year").sort_index())
            continue
        start, end = wave_years[int(w)]
        start, end = max(start, years_min), min(end, years_max)
        if start > end: continue
        idx = pd.Index(range(start, end+1), name="year")
        g2 = (g.set_index("year").reindex(idx).assign(iso3=iso, wave=w))
        g2[norm_cols] = g2[norm_cols].interpolate("linear", limit_direction="both").ffill().bfill().clip(0,1)
        out.append(g2)

    wvs_f = pd.concat(out).reset_index().loc[:, ["iso3","year",*norm_cols,"wave"]].sort_values(["iso3","year"]).reset_index(drop=True)
    all_idx = pd.MultiIndex.from_product([wvs_f["iso3"].unique(), YEARS], names=["iso3","year"])
    wvs = wvs_f.set_index(["iso3","year"]).reindex(all_idx).reset_index()

    y = wvs["year"].to_numpy()
    wvs["wave"] = np.select(
        [(y>=1994)&(y<=1998),(y>=1999)&(y<=2004),(y>=2005)&(y<=2009),(y>=2010)&(y<=2012),(y>=2017)&(y<=2022)],
        [3,4,5,6,7], default=np.nan
    ).astype("int64")

    wvs[norm_cols] = wvs.groupby("iso3", group_keys=False)[norm_cols].apply(lambda g: g.interpolate("linear", limit_direction="both").ffill().bfill().clip(0,1))
    wvs = wvs.sort_values(["iso3","year"]).reset_index(drop=True)
    wvs[norm_cols].dropna()

    wvs_str_culture_df = wvs.copy()[["iso3", "year", "GovtResp_norm", "StrongLeader_norm", "Pride_norm", "Fight_norm", "ArmyRule_norm"]]
    wvs_str_culture_df["SocietalAttitudes"] = wvs_str_culture_df.iloc[:,2:].mean(axis=1)
    wvs_str_culture_df = wvs_str_culture_df[["iso3", "year", "SocietalAttitudes"]]

    political_violence_df = political_violence_df.sort_values(['iso3','year'])
    def linear_decay_weighted_sum(window_vals):
        w = np.arange(1, len(window_vals)+1)
        return np.dot(window_vals, w)
        
    political_violence_df[f'ConflictHistory_{conflict_history_window_year}y'] = political_violence_df \
        .groupby('iso3')['actotal'] \
        .apply(lambda s: s.astype(float).shift(1) \
            .rolling(conflict_history_window_year, min_periods=conflict_history_window_year)
            .apply(linear_decay_weighted_sum, raw=True)) \
        .reset_index(level=0, drop=True)

    political_violence_df[f'ConflictHistory_{conflict_history_window_year}y'] = political_violence_df \
        .groupby('iso3')[f'ConflictHistory_{conflict_history_window_year}y'] \
        .transform(lambda s: (s - s.min()) / (s.max() - s.min()))
    political_violence_df = political_violence_df.fillna(0)[["iso3", "year", f'ConflictHistory_{conflict_history_window_year}y']]

    merged_str = pd.merge(wvs_str_culture_df, vdem_MD_df, on=["iso3", "year"], how="left")
    strategic_culture_df = pd.merge(merged_str, political_violence_df, on=["iso3", "year"], how="left")

    strategic_culture_df["Strategic_Culture_Index"] = (
        strategic_culture_df["SocietalAttitudes"] +
        strategic_culture_df["Military_dimension_index"] +
        strategic_culture_df[f"ConflictHistory_{conflict_history_window_year}y"]
    ) / 3.0

    strategic_culture_df = strategic_culture_df[["iso3", "year", "Strategic_Culture_Index"]]

    # ------------------------------------------
    ### state-society relations
    vdem_PP_df = vdem.copy()[["country_name", "country_text_id", "year", "v2cacamps"]]
    vdem_PP_df = vdem_PP_df[(vdem_PP_df["year"] > 1994) & (vdem_PP_df["year"] < 2013)].reset_index(drop=True).rename(columns={"country_text_id": "iso3"})
    vdem_PP_df = vdem_PP_df.assign(political_polarization = lambda x: (x["v2cacamps"] - x["v2cacamps"].min()) / (x["v2cacamps"].max() - x["v2cacamps"].min())).drop(columns=["v2cacamps", "country_name"])

    rpc_df = rpc_df[["ISO3", "year", "rpe_gdp"]].rename(columns={"ISO3": "iso3", "rpe_gdp": "relative_political_extraction_gdp"})
    rpc_df = rpc_df.assign(relative_political_extraction_gdp = lambda s: (s["relative_political_extraction_gdp"] - s["relative_political_extraction_gdp"].min()) / (s["relative_political_extraction_gdp"].max() - s["relative_political_extraction_gdp"].min()))

    state_society_df = pd.merge(left=vdem_PP_df, right=rpc_df, how="left", on=["iso3", "year"])
    state_society_df = state_society_df.assign(SCM_Index = lambda s: s["relative_political_extraction_gdp"] * (1.0 - s["political_polarization"]))
    state_society_df = state_society_df.drop(columns=["political_polarization", "relative_political_extraction_gdp"])

    # ------------------------------------------
    ### domestic institutions
    polity_EC_df = polity_EC_df[["country", "scode", "year", "xconst"]].rename(columns={"scode": "iso3"})
    polity_EC_df = polity_EC_df[(polity_EC_df["year"] > 1994) & (polity_EC_df["year"] < 2013)].reset_index(drop=True)
    special_codes = [-66, -77, -88]
    polity_EC_df["xconst"] = polity_EC_df["xconst"].mask(polity_EC_df["xconst"].isin(special_codes), 7)
    polity_EC_df = polity_EC_df.assign(executive_constraints = lambda x: (x["xconst"] - x["xconst"].min()) / (x["xconst"].max() - x["xconst"].min())).drop(columns="xconst")
    vdem_domestic_inst_df = vdem.copy()[["country_name", "country_text_id", "year", "v2xlg_legcon", "v2x_jucon"]]
    vdem_domestic_inst_df = vdem_domestic_inst_df[(vdem_domestic_inst_df["year"] > 1994) & (vdem_domestic_inst_df["year"] < 2013)].reset_index(drop=True)
    vdem_domestic_inst_df = vdem_domestic_inst_df.rename(columns={"country_text_id": "iso3", "v2xlg_legcon": "legislature_constraints", "v2x_jucon": "judicial_constraints"})
    domestic_institutions_df = pd.merge(left=vdem_domestic_inst_df, right=polity_EC_df, how="left", on=["iso3", "year"]).drop(columns=["country", "country_name"])
    domestic_institutions_df = domestic_institutions_df.assign(DIVG_Index = lambda x: (x["legislature_constraints"] + x["judicial_constraints"] + x["executive_constraints"]) / 3.0).drop(columns=["legislature_constraints", "judicial_constraints", "executive_constraints"])
    temp_master_df = final_df.merge(plad_leaders_df, left_on=['iso3_i', 'year'], right_on=['iso3', 'year'], how='left')
    temp_master_df['Clarity_modified_i_temp'] = temp_master_df['C_i_sys'] + (1.0 - temp_master_df['C_i_sys']) * temp_master_df['Leader_Profile_Index']
    temp_master_df = temp_master_df[['iso3_i', 'year', 'Strategic_Environment_Index', 'C_i_sys', 'Leader_Profile_Index', 'Clarity_modified_i_temp']]
    temp_master_df = temp_master_df.merge(strategic_culture_df, left_on=['iso3_i', 'year'], right_on=['iso3', 'year'], how='left')
    temp_master_df['Clarity_Index_Modified'] = temp_master_df['Clarity_modified_i_temp'] + (1.0 - temp_master_df['Clarity_modified_i_temp']) * temp_master_df['Strategic_Culture_Index']
    temp_master_df['Strategic_Environment_i_temp'] = temp_master_df['Strategic_Environment_Index'] / (1.0 + temp_master_df['Strategic_Culture_Index'])
    temp_master_df = temp_master_df.merge(state_society_df, left_on=['iso3_i', 'year'], right_on=['iso3', 'year'], how='left')
    temp_master_df['Strategic_Environment_prime'] = temp_master_df['Strategic_Environment_i_temp'] * (2.0 - temp_master_df['SCM_Index'])
    temp_master_df = temp_master_df.drop(columns="iso3_x")
    temp_master_df = temp_master_df.merge(domestic_institutions_df, left_on=['iso3_i', 'year'], right_on=['iso3', 'year'], how='left')
    temp_master_df['Strategic_Environment_Index_Modified'] = temp_master_df['Strategic_Environment_prime'] * (1.0 + temp_master_df['DIVG_Index'])
    master_dataset_final_df = temp_master_df[["iso3_i", "year", "C_i_sys", "Strategic_Environment_Index", "Clarity_Index_Modified", "Strategic_Environment_Index_Modified"]]
    master_dataset_final_df = master_dataset_final_df.dropna().reset_index(drop=True)

    valid_countries = master_dataset_final_df.groupby('iso3_i')['year'].apply(lambda yrs: set(yrs) == set(range(1995, 2013)))
    valid_countries = valid_countries[valid_countries].index
    master_dataset_final_df = master_dataset_final_df[master_dataset_final_df['iso3_i'].isin(valid_countries)].reset_index(drop=True)
    master_dataset_final_df = master_dataset_final_df.rename(columns={"iso3_i": "iso3"})
    
    return master_dataset_final_df

def create_NCR_dataset(
    cepii_data = pd.read_excel("data/CEPII GeoDist/dist_cepii.xls"),
    cow_alliances_data = pd.read_csv("data/Correlates of War (COW) Formal Alliances/alliance_v4.1_by_dyad_yearly.csv", encoding = "latin-1"),
    cow_conflicts_data = pd.read_csv("data/Correlates of War (COW) Militarized Interstate Disputes (MID)/MIDIP 5.0.csv", encoding = "latin-1"),
    cow_capabilities_data = pd.read_csv("data/Correlates of War (COW) National Material Capabilities (NMC)/NMC-60-wsupplementary.csv", encoding = "latin-1"),
    cow_to_iso_data = pd.read_csv("data/cow2iso.csv", encoding = "latin-1"),
    vdem_FOE_data = pd.read_csv("data/VDem - Our World in Data - FOE/key-features-of-electoral-democracy.csv"),
    ucdp_BRDs_data = pd.read_csv("data/UCDP Battle-related Deaths/BattleDeaths_v25_1_conf.csv"),
    sipri_ME_data = pd.read_excel("data/SIPRI-Milex-data-1992-2024.xlsx", sheet_name="Share of GDP"),
    plad_data = pd.read_excel("data//IVVs/Leader Images/The Political Leaders Affiliation Database (PLAD)/PLAD_April_2024.xls"),
    CSP_data = pd.read_excel("data/IVVs/Strategic Culture/Center for Systemic Peace (CSP) - Major Episodes of Political Violence (MEPV), 1946-2018/MEPVv2018.xls"),
    vdem_data = pd.read_csv("data/VDem - vanilla/V-Dem-CY-Full+Others-v15.csv"),
    wvs_data = pd.read_csv("data/World Value Survey/WVS_Time_Series_1981-2022_csv_v5_0.csv"),
    RPC_data = pd.read_excel("data/IVVs/State Society/arpc_2020.xlsx"),
    PolityV_data = pd.read_excel("data/IVVs/Domestic Institutions/Polity V/p5v2018.xls"),
    conflict_history_window_year = 40
):

    final_df, dyadic_df, alliances_df = create_IVs_with_weights(
        cepii_data = cepii_data,
        cow_alliances_data = cow_alliances_data,
        cow_conflicts_data = cow_conflicts_data,
        cow_capabilities_data = cow_capabilities_data,
        cow_to_iso_data = cow_to_iso_data,
        vdem_FOE_data = vdem_FOE_data,
        ucdp_BRDs_data = ucdp_BRDs_data,
        sipri_ME_data = sipri_ME_data
    )
    
    master_dataset = create_IVVs_with_weights(
        IV_data = final_df,
        plad_data = plad_data,
        CSP_data = CSP_data,
        vdem_data = vdem_data,
        wvs_data = wvs_data,
        RPC_data = RPC_data,
        PolityV_data = PolityV_data,
        conflict_history_window_year = conflict_history_window_year
    )
    
    return master_dataset, dyadic_df, alliances_df

master_df, dyadic_df, alliances_df = create_NCR_dataset()
valid_iso3 = master_df["iso3"].unique()
dyadic_df = dyadic_df[
    dyadic_df["iso3_i"].isin(valid_iso3) &
    dyadic_df["iso3_j"].isin(valid_iso3)
].reset_index(drop=True)

alliances_df = alliances_df[
    alliances_df["state_name1_iso"].isin(valid_iso3) &
    alliances_df["state_name2_iso"].isin(valid_iso3)
].reset_index(drop=True)
alliances_df = alliances_df[alliances_df['year'] != 0]

# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------
# ------------------------------------------


def edge_key(u, v):
    return tuple(sorted((str(u), str(v))))


def filter_graph_by_strength(G, min_strength=1):
    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        strength = data.get("strength", min_strength)

        try:
            strength = float(strength)
        except Exception:
            strength = min_strength

        if strength >= min_strength:
            H.add_edge(u, v, **data)

    return H


def build_cow_alliance_graph(year, countries, alliances_df, min_strength=1):
    G = nx.Graph()
    G.add_nodes_from(countries)

    ali = alliances_df.copy()
    ali["year"] = pd.to_numeric(ali["year"], errors="coerce")
    ali = ali.dropna(subset=["year"])
    ali["year"] = ali["year"].astype(int)

    ali = ali[ali["year"] >= 1900]

    ali_y = ali[
        (ali["year"] == int(year)) &
        (ali["alliance_strength"] >= min_strength)
    ]

    for _, row in ali_y.iterrows():
        u = str(row["state_name1_iso"])
        v = str(row["state_name2_iso"])

        if u in countries and v in countries and u != v:
            G.add_edge(
                u,
                v,
                strength=float(row["alliance_strength"]),
                source="COW"
            )

    return G


def graph_edge_set(G, min_strength=1):
    H = filter_graph_by_strength(G, min_strength=min_strength)
    return {edge_key(u, v) for u, v in H.edges()}


def safe_modularity(G, min_strength=1):
    H = filter_graph_by_strength(G, min_strength=min_strength)

    if H.number_of_edges() == 0:
        return 0.0

    communities = list(nx_comm.greedy_modularity_communities(H))
    return nx_comm.modularity(H, communities)


def community_labels(G, countries, min_strength=1):
    H = filter_graph_by_strength(G, min_strength=min_strength)

    if H.number_of_edges() == 0:
        return list(range(len(countries)))

    communities = list(nx_comm.greedy_modularity_communities(H))

    labels = {}
    for k, comm in enumerate(communities):
        for node in comm:
            labels[node] = k

    return [labels.get(c, -1) for c in countries]


def compare_networks(G_pred, G_obs, countries, min_strength=1):
    G_pred_f = filter_graph_by_strength(G_pred, min_strength=min_strength)
    G_obs_f = filter_graph_by_strength(G_obs, min_strength=min_strength)

    all_possible_edges = {edge_key(a, b) for a, b in combinations(countries, 2)}

    pred_edges = graph_edge_set(G_pred_f, min_strength=min_strength)
    obs_edges = graph_edge_set(G_obs_f, min_strength=min_strength)

    tp = len(pred_edges & obs_edges)
    fp = len(pred_edges - obs_edges)
    fn = len(obs_edges - pred_edges)
    tn = len(all_possible_edges - (pred_edges | obs_edges))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    f1 = (
        2 * precision * recall / (precision + recall)
        if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0
        else 0.0
    )

    jaccard = tp / len(pred_edges | obs_edges) if len(pred_edges | obs_edges) > 0 else 1.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    balanced_accuracy = np.nanmean([recall, specificity])

    pred_density = nx.density(G_pred_f)
    obs_density = nx.density(G_obs_f)

    pred_modularity = safe_modularity(G_pred_f, min_strength=min_strength)
    obs_modularity = safe_modularity(G_obs_f, min_strength=min_strength)

    pred_labels = community_labels(G_pred_f, countries, min_strength=min_strength)
    obs_labels = community_labels(G_obs_f, countries, min_strength=min_strength)

    return {
        "pred_edges": len(pred_edges),
        "obs_edges": len(obs_edges),

        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
        "edge_tn": tn,

        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_jaccard": jaccard,
        "balanced_accuracy": balanced_accuracy,

        "pred_density": pred_density,
        "obs_density": obs_density,
        "density_diff": pred_density - obs_density,

        "pred_modularity": pred_modularity,
        "obs_modularity": obs_modularity,
        "modularity_diff": pred_modularity - obs_modularity,
        "abs_modularity_error": abs(pred_modularity - obs_modularity),

        "community_nmi": normalized_mutual_info_score(obs_labels, pred_labels),
        "community_ari": adjusted_rand_score(obs_labels, pred_labels),
    }

class CountryAgent(Agent):
    def __init__(self, model, iso3, initial_cinc):
        super().__init__(model)
        self.iso3 = str(iso3)
        self.policy = "Status_Quo"
        self.s_i = 0.0
        self.clarity_i = 0.0
        self.cinc = float(initial_cinc)
        self.s_i_norm = 0.0
        self.clarity_i_norm = 0.0
        self.cinc_norm = 0.0
        self.primary_threat_iso3 = None
        self.primary_threat_pressure = 0.0
        self.threat_capacity = 0.0
        self.own_capacity_effective = 0.0
        self.ally_capacity = 0.0
        self.coalition_capacity = 0.0
        self.relative_resistance = 0.0
        self.ally_support = 0.0
        self.threat_score = 0.0
        self.clarity_score = 0.0
        self.capacity_score = 0.0
        self.resistance_score = 0.0
        self.vulnerability_score = 0.0
        self.action_pressure = 0.0
        self.u_balance = 0.0
        self.u_bandwagon = 0.0
        self.decision_gap = 0.0
        self.effective_decision_margin = 0.0
        self.formalization_allowed = 0.0
        self.best_resistance_gain = 0.0
        self.proposed_edge_count = 0
        self.removed_edge_count = 0
        self.pending_additions = []
        self.pending_removals = []
        self.logit_threat = 0.0
        self.logit_clarity = 0.0
        self.logit_capacity = 0.0
        self.logit_total = 0.0

    def observe_and_decide(self):
        self.pending_additions.clear()
        self.pending_removals.clear()

        self.formalization_allowed = 0.0
        self.best_resistance_gain = 0.0
        self.proposed_edge_count = 0
        self.removed_edge_count = 0

        agent_scores = self.model.agent_data.loc[(self.iso3, self.model.current_year)]

        if isinstance(agent_scores, pd.DataFrame):
            agent_scores = agent_scores.iloc[0]

        self.s_i = float(agent_scores["Strategic_Environment_Index_Modified"])
        self.clarity_i = float(agent_scores["Clarity_Index_Modified"])

        self.s_i_norm = float(agent_scores["Strategic_Environment_Index_Modified_norm"])
        self.clarity_i_norm = float(agent_scores["Clarity_Index_Modified_norm"])

        dy = self.model.dyadic_data.loc[(self.iso3, self.model.current_year)]

        if isinstance(dy, pd.Series):
            dy = dy.to_frame().T

        dy = dy.copy()

        if "dyadic_pressure" not in dy.columns:
            dy["dyadic_pressure"] = dy["w_ij"] * dy["cinc_j"]

        top_row = dy.sort_values("dyadic_pressure", ascending=False).iloc[0]

        self.primary_threat_iso3 = str(top_row["iso3_j"])
        self.primary_threat_pressure = float(top_row["dyadic_pressure"])

        self.cinc = float(top_row["cinc_i"])
        self.cinc_norm = float(top_row["cinc_i_norm"])
        self.threat_capacity = float(top_row["cinc_j"])

        actual_threat_score = np.clip(self.s_i_norm, 0.0, 1.0)
        actual_clarity_score = np.clip(self.clarity_i_norm, 0.0, 1.0)
        actual_capacity_score = np.clip(self.cinc_norm, 0.0, 1.0)

        if self.model.neutralize == "strategic_env":
            self.threat_score = self.model.neutral_threat_score
        else:
            self.threat_score = actual_threat_score

        if self.model.neutralize == "clarity":
            self.clarity_score = self.model.neutral_clarity_score
        else:
            self.clarity_score = actual_clarity_score

        if self.model.neutralize == "capacity":
            self.capacity_score = self.model.neutral_capacity_score
        else:
            self.capacity_score = actual_capacity_score

        resistance_info = self.model.compute_relative_resistance(
            ego_iso3=self.iso3,
            threat_iso3=self.primary_threat_iso3,
            own_cinc=self.cinc,
            threat_cinc=self.threat_capacity,
            neutralize_capacity=(self.model.neutralize == "capacity")
        )

        self.own_capacity_effective = resistance_info["own_capacity_effective"]
        self.ally_capacity = resistance_info["ally_capacity"]
        self.coalition_capacity = resistance_info["coalition_capacity"]
        self.threat_capacity = resistance_info["threat_capacity"]
        self.relative_resistance = resistance_info["relative_resistance"]
        self.ally_support = resistance_info["ally_support"]

        self.resistance_score = self.relative_resistance
        self.vulnerability_score = 1.0 - self.resistance_score

        self.action_pressure = self.threat_score * (
            (1.0 - self.model.clarity_action_weight)
            + self.model.clarity_action_weight * self.clarity_score
        )

        self.u_balance = self.action_pressure * self.resistance_score
        self.u_bandwagon = self.action_pressure * self.vulnerability_score

        self.decision_gap = self.u_balance - self.u_bandwagon

        self.effective_decision_margin = (
            self.model.decision_margin
            + self.model.uncertainty_margin * (1.0 - self.clarity_score)
        )

        if self.action_pressure < self.model.threat_activation_threshold:
            self.policy = "Status_Quo"

        elif self.decision_gap > self.effective_decision_margin:
            self.policy = "Balance"

        elif self.decision_gap < -self.effective_decision_margin:
            self.policy = "Bandwagon"

        else:
            self.policy = "Status_Quo"

        if (
            self.policy not in ["No_Data", "Status_Quo"]
            and self.action_pressure >= self.model.formalization_threshold
        ):
            self.formalization_allowed = 1.0
        else:
            self.formalization_allowed = 0.0

        self.logit_threat = self.threat_score
        self.logit_clarity = self.clarity_score
        self.logit_capacity = self.capacity_score
        self.logit_total = self.decision_gap

    def stage_policy_changes(self):
        self.pending_additions.clear()
        self.pending_removals.clear()
        self.proposed_edge_count = 0
        self.removed_edge_count = 0
        self.best_resistance_gain = 0.0

        if (
            self.policy in ["No_Data", "Status_Quo"]
            or not self.primary_threat_iso3
            or self.formalization_allowed == 0.0
        ):
            return

        current_allies = list(self.model.G.neighbors(self.iso3))
        current_allies_set = set(current_allies)
        if self.policy == "Balance" and self.primary_threat_iso3 in current_allies_set:
            self.pending_removals.append(self.primary_threat_iso3)

        if self.policy == "Bandwagon":
            if not self.model.G.has_edge(self.iso3, self.primary_threat_iso3):
                proposal_score = float(self.u_bandwagon)

                self.pending_additions.append(
                    (
                        self.primary_threat_iso3,
                        "bandwagon_alignment",
                        self.model.bandwagon_alignment_strength,
                        proposal_score
                    )
                )

        elif self.policy == "Balance":
            threat_allies = set(self.model.G.neighbors(self.primary_threat_iso3))

            best_ally = None
            best_score = -float("inf")
            best_gain = 0.0

            current_resistance = self.relative_resistance

            for cand in self.model.agents:
                cand_iso3 = cand.iso3

                if cand_iso3 in (self.iso3, self.primary_threat_iso3):
                    continue

                if cand_iso3 in current_allies_set:
                    continue

                if cand_iso3 in threat_allies:
                    continue

                candidate_resistance = self.model.compute_expected_resistance_with_candidate(
                    ego_iso3=self.iso3,
                    threat_iso3=self.primary_threat_iso3,
                    candidate_iso3=cand_iso3,
                    neutralize_capacity=(self.model.neutralize == "capacity")
                )

                resistance_gain = max(0.0, candidate_resistance - current_resistance)

                shared_threat_bonus = (
                    self.model.shared_threat_bonus
                    if cand.primary_threat_iso3 == self.primary_threat_iso3
                    else 0.0
                )

                ally_score = resistance_gain + shared_threat_bonus

                if ally_score > best_score:
                    best_score = ally_score
                    best_gain = resistance_gain
                    best_ally = cand_iso3

            self.best_resistance_gain = float(best_gain)

            if best_ally and not self.model.G.has_edge(self.iso3, best_ally):
                proposal_score = float(self.u_balance + self.best_resistance_gain)

                self.pending_additions.append(
                    (
                        best_ally,
                        "balancing_alignment",
                        self.model.balancing_alignment_strength,
                        proposal_score
                    )
                )

        self.proposed_edge_count = len(self.pending_additions)
        self.removed_edge_count = len(self.pending_removals)


class TypeIIINCRModel(Model):

    def __init__(
        self,
        agent_data: pd.DataFrame,
        dyadic_data: pd.DataFrame,
        alliance_data: pd.DataFrame,
        start_year=1995,
        neutralize=None,
        removed_variable="None",
        threat_activation_threshold=0.08,
        decision_margin=0.02,
        uncertainty_margin=0.08,
        clarity_action_weight=0.50,
        formalization_threshold=0.12,
        bandwagon_alignment_strength=2,
        balancing_alignment_strength=4,
        shared_threat_bonus=0.10,
        formalization_lookback=5,

        min_strength=1,
        seed=None
    ):
        super().__init__(seed=seed)

        self.neutralize = neutralize
        self.removed_variable = removed_variable

        self.threat_activation_threshold = threat_activation_threshold
        self.decision_margin = decision_margin
        self.uncertainty_margin = uncertainty_margin
        self.clarity_action_weight = clarity_action_weight
        self.formalization_threshold = formalization_threshold

        self.bandwagon_alignment_strength = bandwagon_alignment_strength
        self.balancing_alignment_strength = balancing_alignment_strength
        self.shared_threat_bonus = shared_threat_bonus

        self.formalization_lookback = formalization_lookback
        self.min_strength = min_strength

        self.agent_data_raw = agent_data.copy()
        self.dyadic_data_raw = dyadic_data.copy()
        self.alliance_data = alliance_data.copy()

        self.agent_data, self.dyadic_data = self.preprocess_data(
            self.agent_data_raw,
            self.dyadic_data_raw
        )

        self.neutral_threat_score = float(
            self.agent_data["Strategic_Environment_Index_Modified_norm"].mean()
        )
        self.neutral_clarity_score = float(
            self.agent_data["Clarity_Index_Modified_norm"].mean()
        )
        capacity_country_year = (
            self.dyadic_data[["iso3_i", "year", "cinc_i", "cinc_i_norm"]]
            .drop_duplicates(subset=["iso3_i", "year"])
        )

        self.neutral_capacity_score = float(capacity_country_year["cinc_i_norm"].mean())
        self.neutral_capacity_raw = float(capacity_country_year["cinc_i"].mean())

        self.agent_data = self.agent_data.set_index(["iso3", "year"]).sort_index()
        self.dyadic_data = self.dyadic_data.set_index(["iso3_i", "year"]).sort_index()

        self.all_ccodes = sorted(
            self.agent_data.index.get_level_values("iso3").unique().astype(str).tolist()
        )

        self.all_years = sorted(
            self.agent_data.index.get_level_values("year").unique().astype(int).tolist()
        )

        if start_year not in self.all_years:
            self.current_year_index = 0
        else:
            self.current_year_index = self.all_years.index(start_year)

        self.current_year = self.all_years[self.current_year_index]
        self.running = True

        self.validation_target_year = None

        self.capacity_raw_map = {}
        self.capacity_norm_map = {}

        self.observed_change_history_df = self.prepare_observed_change_history()
        self.default_formal_change_budget = self.estimate_default_formal_change_budget()

        self.G = nx.Graph()
        self.grid = NetworkGrid(self.G)

        for ccode in self.all_ccodes:
            self.G.add_node(ccode)
            self.G.nodes[ccode]["agent"] = []

        self.agent_by_iso = {}

        for ccode in self.all_ccodes:
            try:
                dy0 = self.dyadic_data.loc[(ccode, self.current_year)]

                if isinstance(dy0, pd.Series):
                    dy0 = dy0.to_frame().T

                initial_cinc = float(dy0.iloc[0]["cinc_i"])

            except KeyError:
                initial_cinc = 0.0

            ag = CountryAgent(self, iso3=ccode, initial_cinc=initial_cinc)
            self.grid.place_agent(ag, ccode)
            self.agent_by_iso[ccode] = ag

        self.reset_validation_metrics()
        self.reset_formalization_metrics()
        self.systemic_conflict_level = 0.0
        self.datacollector = DataCollector(
            agent_reporters={
                "iso3": lambda a: a.iso3,
                "year": lambda a: a.model.current_year,
                "ablation": lambda a: a.model.ablation_name,
                "removed_variable": lambda a: a.model.removed_variable,
                "policy": lambda a: a.policy,
                "primary_threat": lambda a: a.primary_threat_iso3,

                "Strategic_Environment_Index_Modified_norm": lambda a: a.s_i_norm,
                "Clarity_Index_Modified_norm": lambda a: a.clarity_i_norm,
                "Capacity_norm": lambda a: a.cinc_norm,

                "threat_score": lambda a: a.threat_score,
                "clarity_score": lambda a: a.clarity_score,
                "capacity_score": lambda a: a.capacity_score,

                "threat_capacity": lambda a: a.threat_capacity,
                "own_capacity_effective": lambda a: a.own_capacity_effective,
                "ally_capacity": lambda a: a.ally_capacity,
                "coalition_capacity": lambda a: a.coalition_capacity,
                "relative_resistance": lambda a: a.relative_resistance,
                "ally_support": lambda a: a.ally_support,
                "vulnerability_score": lambda a: a.vulnerability_score,

                "action_pressure": lambda a: a.action_pressure,
                "u_balance": lambda a: a.u_balance,
                "u_bandwagon": lambda a: a.u_bandwagon,
                "decision_gap": lambda a: a.decision_gap,
                "effective_decision_margin": lambda a: a.effective_decision_margin,

                "formalization_allowed": lambda a: a.formalization_allowed,
                "best_resistance_gain": lambda a: a.best_resistance_gain,
                "proposed_edge_count": lambda a: a.proposed_edge_count,
                "removed_edge_count": lambda a: a.removed_edge_count,

                "logit_threat": lambda a: a.logit_threat,
                "logit_clarity": lambda a: a.logit_clarity,
                "logit_capacity": lambda a: a.logit_capacity,
                "logit_total": lambda a: a.logit_total,
            },

            model_reporters={
                "Year": "current_year",
                "Target_Year": "validation_target_year",
                "ablation": lambda m: m.ablation_name,
                "removed_variable": lambda m: m.removed_variable,
                "neutralize": lambda m: m.neutralize,

                "Systemic_Conflict_Level": "systemic_conflict_level",

                "formal_change_budget": "formal_change_budget",
                "candidate_additions_count": "candidate_additions_count",
                "candidate_removals_count": "candidate_removals_count",
                "committed_additions_count": "committed_additions_count",
                "committed_removals_count": "committed_removals_count",
                "committed_changes_count": "committed_changes_count",

                "Edge_Precision": "edge_precision",
                "Edge_Recall": "edge_recall",
                "Edge_F1": "edge_f1",
                "Edge_Jaccard": "edge_jaccard",
                "Balanced_Accuracy": "balanced_accuracy",

                "Predicted_Edges": "pred_edges",
                "Observed_COW_Edges": "obs_edges",

                "Predicted_Density": "pred_density",
                "Observed_COW_Density": "obs_density",
                "Density_Difference": "density_diff",

                "Predicted_Modularity": "pred_modularity",
                "Observed_COW_Modularity": "obs_modularity",
                "Modularity_Difference": "modularity_diff",
                "Abs_Modularity_Error": "abs_modularity_error",

                "Community_NMI": "community_nmi",
                "Community_ARI": "community_ari",
            }
        )

        self.ablation_name = "baseline"

    def preprocess_data(self, agent_df, dyadic_df):
        agent_df = agent_df.copy()
        dyadic_df = dyadic_df.copy()

        agent_df["year"] = pd.to_numeric(agent_df["year"], errors="coerce").astype(int)
        dyadic_df["year"] = pd.to_numeric(dyadic_df["year"], errors="coerce").astype(int)

        agent_df["iso3"] = agent_df["iso3"].astype(str)
        dyadic_df["iso3_i"] = dyadic_df["iso3_i"].astype(str)
        dyadic_df["iso3_j"] = dyadic_df["iso3_j"].astype(str)

        for col in ["Strategic_Environment_Index_Modified", "Clarity_Index_Modified"]:
            agent_df[col] = pd.to_numeric(agent_df[col], errors="coerce")

            min_val = agent_df[col].min()
            max_val = agent_df[col].max()
            denom = max_val - min_val

            if pd.isna(denom) or denom == 0:
                agent_df[f"{col}_norm"] = 0.0
            else:
                agent_df[f"{col}_norm"] = (agent_df[col] - min_val) / denom

        dyadic_df["cinc_i"] = pd.to_numeric(dyadic_df["cinc_i"], errors="coerce")
        dyadic_df["cinc_j"] = pd.to_numeric(dyadic_df["cinc_j"], errors="coerce")
        dyadic_df["w_ij"] = pd.to_numeric(dyadic_df["w_ij"], errors="coerce")

        cinc_min = dyadic_df["cinc_i"].min()
        cinc_max = dyadic_df["cinc_i"].max()
        cinc_denom = cinc_max - cinc_min

        if pd.isna(cinc_denom) or cinc_denom == 0:
            dyadic_df["cinc_i_norm"] = 0.0
        else:
            dyadic_df["cinc_i_norm"] = (dyadic_df["cinc_i"] - cinc_min) / cinc_denom

        dyadic_df["dyadic_pressure"] = dyadic_df["w_ij"] * dyadic_df["cinc_j"]

        return agent_df, dyadic_df

    def prepare_observed_change_history(self):
        records = []

        years = sorted(self.all_years)

        for idx in range(len(years) - 1):
            year_t = years[idx]
            year_next = years[idx + 1]

            G_t = build_cow_alliance_graph(
                year=year_t,
                countries=self.all_ccodes,
                alliances_df=self.alliance_data,
                min_strength=self.min_strength
            )

            G_next = build_cow_alliance_graph(
                year=year_next,
                countries=self.all_ccodes,
                alliances_df=self.alliance_data,
                min_strength=self.min_strength
            )

            e_t = graph_edge_set(G_t, min_strength=self.min_strength)
            e_next = graph_edge_set(G_next, min_strength=self.min_strength)

            additions = len(e_next - e_t)
            removals = len(e_t - e_next)
            changes = additions + removals

            records.append({
                "Year": year_t,
                "Target_Year": year_next,
                "observed_additions": additions,
                "observed_removals": removals,
                "observed_changes": changes
            })

        return pd.DataFrame(records)

    def estimate_default_formal_change_budget(self):
        return 0

    def get_formalization_budget(self, year):
        if self.observed_change_history_df.empty:
            return 0

        past = self.observed_change_history_df[
            self.observed_change_history_df["Year"] < int(year)
        ].copy()

        if past.empty:
            return self.default_formal_change_budget

        past = past.sort_values("Year").tail(self.formalization_lookback)

        budget = int(round(past["observed_changes"].mean()))

        return max(0, budget)

    def update_yearly_capacity_maps(self):
        year_data = self.dyadic_data.loc[(slice(None), self.current_year), :].reset_index()

        cap_raw = year_data.groupby("iso3_i")["cinc_i"].first()
        cap_norm = year_data.groupby("iso3_i")["cinc_i_norm"].first()

        self.capacity_raw_map = cap_raw.to_dict()
        self.capacity_norm_map = cap_norm.to_dict()


    def get_effective_capacity_raw(self, iso3, fallback=0.0, neutralize_capacity=False):
        if neutralize_capacity:
            return float(self.neutral_capacity_raw)

        return float(self.capacity_raw_map.get(iso3, fallback))

    def get_current_ally_capacity(self, ego_iso3, threat_iso3, neutralize_capacity=False):
        if ego_iso3 not in self.G.nodes:
            return 0.0

        allies = list(self.G.neighbors(ego_iso3))

        valid_allies = [
            ally for ally in allies
            if ally != threat_iso3 and ally in self.all_ccodes
        ]

        return sum(
            self.get_effective_capacity_raw(
                ally,
                fallback=0.0,
                neutralize_capacity=neutralize_capacity
            )
            for ally in valid_allies
        )

    def compute_relative_resistance(
        self,
        ego_iso3,
        threat_iso3,
        own_cinc,
        threat_cinc,
        neutralize_capacity=False
    ):
        own_capacity_effective = self.get_effective_capacity_raw(
            ego_iso3,
            fallback=own_cinc,
            neutralize_capacity=neutralize_capacity
        )

        ally_capacity = self.get_current_ally_capacity(
            ego_iso3=ego_iso3,
            threat_iso3=threat_iso3,
            neutralize_capacity=neutralize_capacity
        )

        threat_capacity = self.get_effective_capacity_raw(
            threat_iso3,
            fallback=threat_cinc,
            neutralize_capacity=neutralize_capacity
        )

        coalition_capacity = own_capacity_effective + ally_capacity
        denom = coalition_capacity + threat_capacity

        if denom > 0:
            relative_resistance = coalition_capacity / denom
        else:
            relative_resistance = 0.0

        if ally_capacity + threat_capacity > 0:
            ally_support = ally_capacity / (ally_capacity + threat_capacity)
        else:
            ally_support = 0.0

        return {
            "own_capacity_effective": float(own_capacity_effective),
            "ally_capacity": float(ally_capacity),
            "coalition_capacity": float(coalition_capacity),
            "threat_capacity": float(threat_capacity),
            "relative_resistance": float(np.clip(relative_resistance, 0.0, 1.0)),
            "ally_support": float(np.clip(ally_support, 0.0, 1.0)),
        }

    def compute_expected_resistance_with_candidate(
        self,
        ego_iso3,
        threat_iso3,
        candidate_iso3,
        neutralize_capacity=False
    ):
        own_capacity = self.get_effective_capacity_raw(
            ego_iso3,
            fallback=0.0,
            neutralize_capacity=neutralize_capacity
        )

        current_ally_capacity = self.get_current_ally_capacity(
            ego_iso3=ego_iso3,
            threat_iso3=threat_iso3,
            neutralize_capacity=neutralize_capacity
        )

        candidate_capacity = self.get_effective_capacity_raw(
            candidate_iso3,
            fallback=0.0,
            neutralize_capacity=neutralize_capacity
        )

        threat_capacity = self.get_effective_capacity_raw(
            threat_iso3,
            fallback=0.0,
            neutralize_capacity=neutralize_capacity
        )

        new_coalition_capacity = own_capacity + current_ally_capacity + candidate_capacity
        denom = new_coalition_capacity + threat_capacity

        if denom <= 0:
            return 0.0

        return float(np.clip(new_coalition_capacity / denom, 0.0, 1.0))

    def reset_validation_metrics(self):
        self.edge_precision = np.nan
        self.edge_recall = np.nan
        self.edge_f1 = np.nan
        self.edge_jaccard = np.nan
        self.balanced_accuracy = np.nan

        self.pred_edges = np.nan
        self.obs_edges = np.nan

        self.pred_density = np.nan
        self.obs_density = np.nan
        self.density_diff = np.nan

        self.pred_modularity = np.nan
        self.obs_modularity = np.nan
        self.modularity_diff = np.nan
        self.abs_modularity_error = np.nan

        self.community_nmi = np.nan
        self.community_ari = np.nan

    def reset_formalization_metrics(self):
        self.formal_change_budget = 0
        self.candidate_additions_count = 0
        self.candidate_removals_count = 0
        self.committed_additions_count = 0
        self.committed_removals_count = 0
        self.committed_changes_count = 0

    def set_network_to_observed_year(self, year):
        G_obs = build_cow_alliance_graph(
            year=year,
            countries=self.all_ccodes,
            alliances_df=self.alliance_data,
            min_strength=self.min_strength
        )

        self.G.clear_edges()

        for u, v, data in G_obs.edges(data=True):
            self.G.add_edge(u, v, **data)

    def validate_against_cow(self, target_year):
        self.reset_validation_metrics()

        if target_year is None:
            return

        available_years = set(
            pd.to_numeric(self.alliance_data["year"], errors="coerce")
            .dropna()
            .astype(int)
            .tolist()
        )

        if int(target_year) not in available_years:
            return

        G_obs = build_cow_alliance_graph(
            year=target_year,
            countries=self.all_ccodes,
            alliances_df=self.alliance_data,
            min_strength=self.min_strength
        )

        metrics = compare_networks(
            G_pred=self.G,
            G_obs=G_obs,
            countries=self.all_ccodes,
            min_strength=self.min_strength
        )

        self.edge_precision = metrics["edge_precision"]
        self.edge_recall = metrics["edge_recall"]
        self.edge_f1 = metrics["edge_f1"]
        self.edge_jaccard = metrics["edge_jaccard"]
        self.balanced_accuracy = metrics["balanced_accuracy"]

        self.pred_edges = metrics["pred_edges"]
        self.obs_edges = metrics["obs_edges"]

        self.pred_density = metrics["pred_density"]
        self.obs_density = metrics["obs_density"]
        self.density_diff = metrics["density_diff"]

        self.pred_modularity = metrics["pred_modularity"]
        self.obs_modularity = metrics["obs_modularity"]
        self.modularity_diff = metrics["modularity_diff"]
        self.abs_modularity_error = metrics["abs_modularity_error"]

        self.community_nmi = metrics["community_nmi"]
        self.community_ari = metrics["community_ari"]

    def accepts_alliance(self, receiver_iso3, proposer_iso3):
        receiver_agent = self.agent_by_iso.get(receiver_iso3)

        if receiver_agent is None:
            return False

        if receiver_agent.policy == "No_Data":
            return False

        if (
            receiver_agent.policy == "Balance"
            and receiver_agent.primary_threat_iso3 == proposer_iso3
        ):
            return False

        return True

    def commit_all_policy_changes(self):
        self.reset_formalization_metrics()

        self.formal_change_budget = self.get_formalization_budget(self.current_year)

        candidate_additions = []
        candidate_removals = []

        for agent in self.agents:
            for target, type_, strength, score in agent.pending_additions:
                u, v = edge_key(agent.iso3, target)

                if self.G.has_edge(u, v):
                    continue

                if not self.accepts_alliance(receiver_iso3=target, proposer_iso3=agent.iso3):
                    continue

                candidate_additions.append({
                    "kind": "addition",
                    "u": u,
                    "v": v,
                    "type": type_,
                    "strength": strength,
                    "score": float(score),
                    "proposer": agent.iso3
                })

            for target in agent.pending_removals:
                u, v = edge_key(agent.iso3, target)

                if self.G.has_edge(u, v):
                    candidate_removals.append({
                        "kind": "removal",
                        "u": u,
                        "v": v,
                        "score": float(agent.action_pressure),
                        "proposer": agent.iso3
                    })

        self.candidate_additions_count = len(candidate_additions)
        self.candidate_removals_count = len(candidate_removals)

        all_candidates = candidate_additions + candidate_removals
        all_candidates = sorted(all_candidates, key=lambda x: x["score"], reverse=True)

        accepted_candidates = all_candidates[:self.formal_change_budget]

        for cand in accepted_candidates:
            u = cand["u"]
            v = cand["v"]

            if cand["kind"] == "addition":
                if not self.G.has_edge(u, v):
                    self.G.add_edge(
                        u,
                        v,
                        type=cand["type"],
                        strength=float(cand["strength"]),
                        source="ABM"
                    )
                    self.committed_additions_count += 1

            elif cand["kind"] == "removal":
                if self.G.has_edge(u, v):
                    self.G.remove_edge(u, v)
                    self.committed_removals_count += 1

        self.committed_changes_count = (
            self.committed_additions_count + self.committed_removals_count
        )

    def step(self):
        self.reset_validation_metrics()
        self.reset_formalization_metrics()
        self.set_network_to_observed_year(self.current_year)
        self.update_yearly_capacity_maps()

        if self.current_year_index + 1 < len(self.all_years):
            target_year = self.all_years[self.current_year_index + 1]
        else:
            target_year = None

        self.validation_target_year = target_year

        self.agents.do("observe_and_decide")
        self.agents.do("stage_policy_changes")
        self.commit_all_policy_changes()

        try:
            year_data = self.dyadic_data.loc[(slice(None), self.current_year), :]
            self.systemic_conflict_level = year_data["max_hostility"].sum() / 2.0
        except KeyError:
            self.systemic_conflict_level = 0.0

        self.validate_against_cow(target_year)
        self.datacollector.collect(self)
        self.current_year_index += 1

        if self.current_year_index < len(self.all_years):
            self.current_year = self.all_years[self.current_year_index]
        else:
            self.running = False

    def run_all_years(self):
        while self.running:
            self.step()


def persistence_baseline(master_df, alliances_df, min_strength=1):
    countries = sorted(master_df["iso3"].astype(str).unique().tolist())

    years = sorted(
        pd.to_numeric(master_df["year"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    available_years = set(
        pd.to_numeric(alliances_df["year"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )

    records = []

    for i in range(len(years) - 1):
        year_t = years[i]
        year_next = years[i + 1]

        if year_next not in available_years:
            continue

        G_pred = build_cow_alliance_graph(
            year=year_t,
            countries=countries,
            alliances_df=alliances_df,
            min_strength=min_strength
        )

        G_obs = build_cow_alliance_graph(
            year=year_next,
            countries=countries,
            alliances_df=alliances_df,
            min_strength=min_strength
        )

        metrics = compare_networks(
            G_pred=G_pred,
            G_obs=G_obs,
            countries=countries,
            min_strength=min_strength
        )

        metrics["Year"] = year_t
        metrics["Target_Year"] = year_next
        metrics["model"] = "persistence_anchor"
        metrics["min_strength"] = min_strength

        records.append(metrics)

    return pd.DataFrame(records)

ABLATION_EXPERIMENTS = [
    {
        "name": "baseline",
        "removed_variable": "None",
        "neutralize": None,
    },
    {
        "name": "neutralized_clarity",
        "removed_variable": "Clarity",
        "neutralize": "clarity",
    },
    {
        "name": "neutralized_strategic_env",
        "removed_variable": "Strategic Environment",
        "neutralize": "strategic_env",
    },
    {
        "name": "neutralized_capacity",
        "removed_variable": "Capacity",
        "neutralize": "capacity",
    },
]


def run_abm_experiments(
    master_df,
    dyadic_df,
    alliances_df,
    start_year=1995,
    min_strength=1,

    threat_activation_threshold=0.08,
    decision_margin=0.02,
    uncertainty_margin=0.08,
    clarity_action_weight=0.50,
    formalization_threshold=0.12,

    bandwagon_alignment_strength=2,
    balancing_alignment_strength=4,
    shared_threat_bonus=0.10,

    formalization_lookback=5,
    seed=616
):
    all_model_results_list = []
    all_agent_results_list = []

    for exp in ABLATION_EXPERIMENTS:

        model = TypeIIINCRModel(
            agent_data=master_df,
            dyadic_data=dyadic_df,
            alliance_data=alliances_df,
            start_year=start_year,

            neutralize=exp["neutralize"],
            removed_variable=exp["removed_variable"],

            threat_activation_threshold=threat_activation_threshold,
            decision_margin=decision_margin,
            uncertainty_margin=uncertainty_margin,
            clarity_action_weight=clarity_action_weight,
            formalization_threshold=formalization_threshold,

            bandwagon_alignment_strength=bandwagon_alignment_strength,
            balancing_alignment_strength=balancing_alignment_strength,
            shared_threat_bonus=shared_threat_bonus,

            formalization_lookback=formalization_lookback,
            min_strength=min_strength,
            seed=seed
        )

        model.ablation_name = exp["name"]

        model.run_all_years()

        model_data = model.datacollector.get_model_vars_dataframe().reset_index()
        model_data["ablation"] = exp["name"]
        model_data["removed_variable"] = exp["removed_variable"]
        model_data["neutralize"] = exp["neutralize"]
        model_data["min_strength"] = min_strength

        agent_data = model.datacollector.get_agent_vars_dataframe().reset_index()
        agent_data["ablation"] = exp["name"]
        agent_data["removed_variable"] = exp["removed_variable"]
        agent_data["neutralize"] = exp["neutralize"]
        agent_data["min_strength"] = min_strength

        all_model_results_list.append(model_data)
        all_agent_results_list.append(agent_data)

    abm_validation_df = pd.concat(all_model_results_list, ignore_index=True)
    policy_debug_df = pd.concat(all_agent_results_list, ignore_index=True)

    return abm_validation_df, policy_debug_df


def policy_distribution_table(policy_debug_df):
    out = (
        policy_debug_df
        .groupby(["ablation", "removed_variable", "policy"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        out
        .groupby(["ablation", "removed_variable"])["count"]
        .sum()
        .reset_index(name="total")
    )

    out = out.merge(totals, on=["ablation", "removed_variable"], how="left")
    out["share"] = out["count"] / out["total"]

    return out.sort_values(["ablation", "policy"]).reset_index(drop=True)


def mechanism_summary_table(policy_debug_df):
    cols = [
        "threat_score",
        "clarity_score",
        "capacity_score",
        "relative_resistance",
        "vulnerability_score",
        "action_pressure",
        "u_balance",
        "u_bandwagon",
        "decision_gap",
        "formalization_allowed",
        "best_resistance_gain",
        "proposed_edge_count",
    ]

    out = (
        policy_debug_df
        .groupby(["ablation", "removed_variable"], as_index=False)[cols]
        .mean()
    )

    return out


def formalization_summary_table(abm_validation_df):
    cols = [
        "formal_change_budget",
        "candidate_additions_count",
        "candidate_removals_count",
        "committed_additions_count",
        "committed_removals_count",
        "committed_changes_count",
    ]

    out = (
        abm_validation_df
        .dropna(subset=["Target_Year"])
        .groupby(["ablation", "removed_variable"], as_index=False)[cols]
        .mean()
    )

    return out


def historical_anchor_plausibility_table(abm_validation_df, baseline_df):
    baseline_abm = (
        abm_validation_df
        .dropna(subset=["Target_Year"])
        .query("ablation == 'baseline'")
        [[
            "Predicted_Edges",
            "Observed_COW_Edges",
            "Predicted_Density",
            "Observed_COW_Density",
            "Abs_Modularity_Error",
            "Community_NMI",
            "Community_ARI",
        ]]
        .mean()
        .to_frame()
        .T
    )

    baseline_abm.insert(0, "Model", "Baseline_ABM")

    persistence = (
        baseline_df
        [[
            "pred_edges",
            "obs_edges",
            "pred_density",
            "obs_density",
            "abs_modularity_error",
            "community_nmi",
            "community_ari",
        ]]
        .mean()
        .to_frame()
        .T
        .rename(columns={
            "pred_edges": "Predicted_Edges",
            "obs_edges": "Observed_COW_Edges",
            "pred_density": "Predicted_Density",
            "obs_density": "Observed_COW_Density",
            "abs_modularity_error": "Abs_Modularity_Error",
            "community_nmi": "Community_NMI",
            "community_ari": "Community_ARI",
        })
    )

    persistence.insert(0, "Model", "Persistence_Anchor")

    return pd.concat([baseline_abm, persistence], ignore_index=True)


def calculate_policy_deviation(policy_dist_df, ablation_name):
    base = (
        policy_dist_df
        .query("ablation == 'baseline'")
        .set_index("policy")["share"]
    )

    alt = (
        policy_dist_df
        .query("ablation == @ablation_name")
        .set_index("policy")["share"]
    )

    policies = sorted(set(base.index) | set(alt.index))

    base = base.reindex(policies).fillna(0.0)
    alt = alt.reindex(policies).fillna(0.0)

    return float(0.5 * np.abs(base - alt).sum())


def calculate_mechanism_deviation(mechanism_df, ablation_name):
    mechanism_cols = [
        "threat_score",
        "clarity_score",
        "capacity_score",
        "relative_resistance",
        "vulnerability_score",
        "action_pressure",
        "u_balance",
        "u_bandwagon",
        "formalization_allowed",
        "best_resistance_gain",
    ]

    base = (
        mechanism_df
        .query("ablation == 'baseline'")
        [mechanism_cols]
        .iloc[0]
        .astype(float)
    )

    alt = (
        mechanism_df
        .query("ablation == @ablation_name")
        [mechanism_cols]
        .iloc[0]
        .astype(float)
    )

    return float(np.abs(base - alt).mean())


def calculate_system_deviation(abm_validation_df, ablation_name):
    df = abm_validation_df.dropna(subset=["Target_Year"]).copy()

    countries_n = len(master_df["iso3"].astype(str).unique())
    possible_edges = countries_n * (countries_n - 1) / 2

    df["Predicted_Edges_Share"] = df["Predicted_Edges"] / possible_edges
    df["Committed_Changes_Share"] = df["committed_changes_count"] / possible_edges

    system_cols = [
        "Predicted_Density",
        "Predicted_Modularity",
        "Predicted_Edges_Share",
        "Committed_Changes_Share",
    ]

    base = (
        df
        .query("ablation == 'baseline'")
        .set_index("Year")[system_cols]
    )

    alt = (
        df
        .query("ablation == @ablation_name")
        .set_index("Year")[system_cols]
    )

    common_years = sorted(set(base.index) & set(alt.index))

    if len(common_years) == 0:
        return np.nan

    diff = (base.loc[common_years] - alt.loc[common_years]).abs()

    return float(diff.mean().mean())


def final_ablation_ranking_table(policy_dist_df, mechanism_df, abm_validation_df):
    records = []

    ablations = [
        a for a in abm_validation_df["ablation"].unique().tolist()
        if a != "baseline"
    ]

    removed_map = (
        abm_validation_df
        .drop_duplicates("ablation")
        .set_index("ablation")["removed_variable"]
        .to_dict()
    )

    for ablation in ablations:
        policy_dev = calculate_policy_deviation(policy_dist_df, ablation)
        mechanism_dev = calculate_mechanism_deviation(mechanism_df, ablation)
        system_dev = calculate_system_deviation(abm_validation_df, ablation)

        overall = np.nanmean([policy_dev, mechanism_dev, system_dev])

        records.append({
            "Removed_Variable": removed_map.get(ablation, ablation),
            "ablation": ablation,
            "Overall_Determinacy_Score": overall,
            "Policy_Deviation": policy_dev,
            "Mechanism_Deviation": mechanism_dev,
            "System_Deviation": system_dev,
        })

    out = pd.DataFrame(records)
    out = out.sort_values("Overall_Determinacy_Score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))

    return out

MAIN_MIN_STRENGTH = 1

abm_validation_df, policy_debug_df = run_abm_experiments(
    master_df=master_df,
    dyadic_df=dyadic_df,
    alliances_df=alliances_df,
    start_year=1995,
    min_strength=MAIN_MIN_STRENGTH,

    threat_activation_threshold=0.08,
    decision_margin=0.02,
    uncertainty_margin=0.08,
    clarity_action_weight=0.50,
    formalization_threshold=0.12,

    bandwagon_alignment_strength=2,
    balancing_alignment_strength=4,
    shared_threat_bonus=0.10,

    formalization_lookback=5,
    seed=616
)

baseline_df = persistence_baseline(
    master_df=master_df,
    alliances_df=alliances_df,
    min_strength=MAIN_MIN_STRENGTH
)

policy_dist = policy_distribution_table(policy_debug_df)
mechanism_summary = mechanism_summary_table(policy_debug_df)
formalization_summary = formalization_summary_table(abm_validation_df)
historical_anchor_plausibility = historical_anchor_plausibility_table(
    abm_validation_df=abm_validation_df,
    baseline_df=baseline_df
)
ablation_ranking = final_ablation_ranking_table(
    policy_dist_df=policy_dist,
    mechanism_df=mechanism_summary,
    abm_validation_df=abm_validation_df
)

neutralization_check = (
    policy_debug_df
    .groupby(["ablation", "removed_variable"])[
        ["threat_score", "clarity_score", "capacity_score"]
    ]
    .agg(["mean", "std"])
)

policy_dist
mechanism_summary
formalization_summary
historical_anchor_plausibility
ablation_ranking
neutralization_check

FIG_DIR = "abm_figures"
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
FIGSIZE_WIDE = (11.5, 6.2)
FIGSIZE_PANEL = (13.5, 5.2)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.7,
})

ABM_ORDER = [
    "baseline",
    "neutralized_capacity",
    "neutralized_strategic_env",
    "neutralized_clarity",
]

ABM_LABELS = {
    "baseline": "Baseline",
    "neutralized_capacity": "Neutralized Capacity",
    "neutralized_strategic_env": "Neutralized Strategic Environment",
    "neutralized_clarity": "Neutralized Clarity",
}

ABM_COLORS = {
    "baseline": "#222222",
    "neutralized_capacity": "#D55E00",
    "neutralized_strategic_env": "#0072B2",
    "neutralized_clarity": "#009E73",
}

ABM_LINESTYLES = {
    "baseline": "-",
    "neutralized_capacity": "--",
    "neutralized_strategic_env": "-.",
    "neutralized_clarity": ":",
}

ABM_MARKERS = {
    "baseline": "o",
    "neutralized_capacity": "s",
    "neutralized_strategic_env": "^",
    "neutralized_clarity": "D",
}

POLICY_ORDER = ["Status_Quo", "Balance", "Bandwagon"]
POLICY_LABELS = {
    "Status_Quo": "Status Quo",
    "Balance": "Balance",
    "Bandwagon": "Bandwagon",
}

VARIABLE_LABELS = {
    "action_pressure": "Action pressure",
    "relative_resistance": "Relative resistance",
    "vulnerability_score": "Vulnerability",
    "decision_gap": "Balance–bandwagon gap",
    "formalization_allowed": "Formalization gate",
    "Predicted_Edges": "Alliance edges",
    "Predicted_Density": "Density",
    "Predicted_Modularity": "Modularity",
    "Abs_Modularity_Error": "Absolute modularity error",
}


def save_fig(filename):
    path = os.path.join(FIG_DIR, f"{filename}.png")
    plt.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close()


def get_year_col(df):
    if "Year" in df.columns:
        return "Year"
    if "year" in df.columns:
        return "year"


def ordered_ablations(df):
    available = df["ablation"].dropna().unique().tolist()
    ordered = [a for a in ABM_ORDER if a in available]
    remaining = [a for a in available if a not in ordered]
    return ordered + remaining


def clean_label(ablation):
    return ABM_LABELS.get(ablation, str(ablation))


def style_line(ax, x, y, ablation, label=None, linewidth=2.1, markersize=5.2):
    ax.plot(
        x,
        y,
        color=ABM_COLORS.get(ablation, "#444444"),
        linestyle=ABM_LINESTYLES.get(ablation, "-"),
        marker=ABM_MARKERS.get(ablation, "o"),
        linewidth=linewidth,
        markersize=markersize,
        label=label if label is not None else clean_label(ablation),
    )

def finish_time_axis(ax, xlabel="Year"):
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="both", alpha=0.22)

def make_policy_time_series(policy_debug_df):
    year_col = get_year_col(policy_debug_df)

    counts = (
        policy_debug_df
        .groupby([year_col, "ablation", "removed_variable", "policy"])
        .size()
        .reset_index(name="count")
    )

    totals = (
        counts
        .groupby([year_col, "ablation", "removed_variable"])["count"]
        .sum()
        .reset_index(name="total")
    )

    counts = counts.merge(
        totals,
        on=[year_col, "ablation", "removed_variable"],
        how="left"
    )

    counts["share"] = counts["count"] / counts["total"]

    full_index = pd.MultiIndex.from_product(
        [
            sorted(counts[year_col].unique()),
            counts["ablation"].unique(),
            POLICY_ORDER,
        ],
        names=[year_col, "ablation", "policy"]
    )

    filled = (
        counts
        .set_index([year_col, "ablation", "policy"])
        .reindex(full_index)
        .reset_index()
    )

    filled["share"] = filled["share"].fillna(0.0)
    filled["count"] = filled["count"].fillna(0)
    filled = filled.rename(columns={year_col: "Year"})

    return filled


def plot_policy_composition_panel(policy_debug_df):
    policy_ts = make_policy_time_series(policy_debug_df)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=FIGSIZE_PANEL,
        sharex=True,
        sharey=False
    )

    for ax, policy in zip(axes, POLICY_ORDER):
        for ablation in ordered_ablations(policy_ts):
            sub = policy_ts[
                (policy_ts["ablation"] == ablation) &
                (policy_ts["policy"] == policy)
            ].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("Year")

            style_line(
                ax=ax,
                x=sub["Year"],
                y=sub["share"],
                ablation=ablation,
                label=clean_label(ablation)
            )

        ax.set_title(POLICY_LABELS.get(policy, policy))
        ax.set_ylabel("Share of agents")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        finish_time_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06)
    )

    fig.suptitle("Policy Orientation over Time", y=1.03)
    fig.tight_layout()
    save_fig("fig01_policy_orientation_panel")

def make_mechanism_time_series(policy_debug_df):
    year_col = get_year_col(policy_debug_df)

    mechanism_cols = [
        "threat_score",
        "clarity_score",
        "capacity_score",
        "relative_resistance",
        "vulnerability_score",
        "action_pressure",
        "u_balance",
        "u_bandwagon",
        "decision_gap",
        "formalization_allowed",
        "best_resistance_gain",
        "proposed_edge_count",
        "removed_edge_count",
    ]

    available_cols = [c for c in mechanism_cols if c in policy_debug_df.columns]

    mechanism_ts = (
        policy_debug_df
        .groupby([year_col, "ablation", "removed_variable"], as_index=False)[available_cols]
        .mean()
        .rename(columns={year_col: "Year"})
    )

    return mechanism_ts


def make_mechanism_deviation_from_baseline(policy_debug_df):
    mechanism_ts = make_mechanism_time_series(policy_debug_df)

    variables = [
        "action_pressure",
        "relative_resistance",
        "vulnerability_score",
        "decision_gap",
        "formalization_allowed",
    ]

    variables = [v for v in variables if v in mechanism_ts.columns]

    base = (
        mechanism_ts[mechanism_ts["ablation"] == "baseline"]
        .set_index("Year")[variables]
        .rename(columns={v: f"{v}_baseline" for v in variables})
    )

    records = []

    for ablation in ordered_ablations(mechanism_ts):
        if ablation == "baseline":
            continue

        sub = (
            mechanism_ts[mechanism_ts["ablation"] == ablation]
            .set_index("Year")
        )

        common_years = sorted(set(base.index) & set(sub.index))

        for year in common_years:
            row = {
                "Year": year,
                "ablation": ablation,
            }

            for v in variables:
                row[v] = abs(sub.loc[year, v] - base.loc[year, f"{v}_baseline"])

            records.append(row)

    return pd.DataFrame(records)


def plot_mechanism_deviation_panel(policy_debug_df):
    dev = make_mechanism_deviation_from_baseline(policy_debug_df)

    variables = [
        "action_pressure",
        "relative_resistance",
        "decision_gap",
        "formalization_allowed",
    ]

    variables = [v for v in variables if v in dev.columns]

    fig, axes = plt.subplots(
        1,
        len(variables),
        figsize=(14.5, 4.8),
        sharex=True
    )

    if len(variables) == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        for ablation in [a for a in ordered_ablations(dev) if a != "baseline"]:
            sub = dev[dev["ablation"] == ablation].copy()

            if sub.empty:
                continue

            sub = sub.sort_values("Year")

            style_line(
                ax=ax,
                x=sub["Year"],
                y=sub[var],
                ablation=ablation,
                label=clean_label(ablation),
                linewidth=2.0,
                markersize=4.8
            )

        ax.set_title(VARIABLE_LABELS.get(var, var))
        ax.set_ylabel("Absolute deviation")
        finish_time_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07)
    )

    fig.suptitle("Mechanism Deviation from Baseline over Time", y=1.04)
    fig.tight_layout()
    save_fig("fig02_mechanism_deviation_from_baseline_panel")

def plot_formalization_diagnostics(abm_validation_df):

    df = abm_validation_df.dropna(subset=["Target_Year"]).copy()
    df = df.sort_values(["ablation", "Year"])

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_PANEL)

    ax = axes[0]

    for ablation in ordered_ablations(df):
        sub = df[df["ablation"] == ablation].copy()

        if sub.empty:
            continue

        style_line(
            ax=ax,
            x=sub["Year"],
            y=sub["candidate_additions_count"],
            ablation=ablation,
            label=clean_label(ablation)
        )

    ax.set_title("Candidate Alliance Additions")
    ax.set_ylabel("Candidate additions")
    finish_time_axis(ax)

    ax = axes[1]
    base = df[df["ablation"] == "baseline"].copy()

    if not base.empty:
        ax.plot(
            base["Year"],
            base["formal_change_budget"],
            color="#222222",
            linestyle="-",
            marker="o",
            linewidth=2.1,
            markersize=5.2,
            label="Formalization budget"
        )

        ax.plot(
            base["Year"],
            base["committed_changes_count"],
            color="#D55E00",
            linestyle="--",
            marker="s",
            linewidth=2.1,
            markersize=5.2,
            label="Committed changes"
        )

    ax.set_title("Budget and Accepted Changes: Baseline")
    ax.set_ylabel("Count")
    finish_time_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.08, -0.08)
    )

    handles2, labels2 = axes[1].get_legend_handles_labels()
    axes[1].legend(handles2, labels2, frameon=False, loc="upper right")

    fig.suptitle("Formalization Process over Time", y=1.04)
    fig.tight_layout()
    save_fig("fig03_formalization_diagnostics_panel")

def plot_historical_anchor_panel(abm_validation_df):
    base = (
        abm_validation_df
        .dropna(subset=["Target_Year"])
        .query("ablation == 'baseline'")
        .copy()
        .sort_values("Year")
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    specs = [
        (
            "Predicted_Edges",
            "Observed_COW_Edges",
            "Alliance edges",
            "Number of edges"
        ),
        (
            "Predicted_Density",
            "Observed_COW_Density",
            "Alliance density",
            "Density"
        ),
        (
            "Predicted_Modularity",
            "Observed_COW_Modularity",
            "Alliance modularity",
            "Modularity"
        ),
    ]

    for ax, (pred_col, obs_col, title, ylabel) in zip(axes, specs):
        ax.plot(
            base["Year"],
            base[pred_col],
            color="#222222",
            linestyle="-",
            marker="o",
            linewidth=2.1,
            markersize=5.2,
            label="Baseline ABM"
        )

        ax.plot(
            base["Year"],
            base[obs_col],
            color="#D55E00",
            linestyle="--",
            marker="s",
            linewidth=2.1,
            markersize=5.2,
            label="Observed COW"
        )

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        finish_time_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07)
    )

    fig.suptitle("Historical Anchor Plausibility: Baseline ABM vs Observed COW", y=1.04)
    fig.tight_layout()
    save_fig("fig04_historical_anchor_plausibility_panel")

def plot_system_deviation_from_baseline(abm_validation_df):
    df = abm_validation_df.dropna(subset=["Target_Year"]).copy()

    system_cols = [
        "Predicted_Density",
        "Predicted_Modularity",
        "Predicted_Edges",
        "committed_changes_count",
    ]

    system_cols = [c for c in system_cols if c in df.columns]

    base = (
        df[df["ablation"] == "baseline"]
        .set_index("Year")[system_cols]
        .rename(columns={c: f"{c}_baseline" for c in system_cols})
    )

    records = []

    for ablation in ordered_ablations(df):
        if ablation == "baseline":
            continue

        sub = df[df["ablation"] == ablation].set_index("Year")

        common_years = sorted(set(base.index) & set(sub.index))

        for year in common_years:
            row = {
                "Year": year,
                "ablation": ablation,
            }

            for c in system_cols:
                row[c] = abs(sub.loc[year, c] - base.loc[year, f"{c}_baseline"])

            records.append(row)

    dev = pd.DataFrame(records)

    variables = [
        "Predicted_Density",
        "Predicted_Modularity",
        "Predicted_Edges",
        "committed_changes_count",
    ]

    variables = [v for v in variables if v in dev.columns]

    fig, axes = plt.subplots(1, len(variables), figsize=(14.5, 4.8))

    if len(variables) == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        for ablation in [a for a in ordered_ablations(dev) if a != "baseline"]:
            sub = dev[dev["ablation"] == ablation].copy()

            if sub.empty:
                continue

            style_line(
                ax=ax,
                x=sub["Year"],
                y=sub[var],
                ablation=ablation,
                label=clean_label(ablation),
                linewidth=2.0,
                markersize=4.8
            )

        ax.set_title(VARIABLE_LABELS.get(var, var))
        ax.set_ylabel("Absolute deviation")
        finish_time_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07)
    )

    fig.suptitle("System-Level Deviation from Baseline over Time", y=1.04)
    fig.tight_layout()
    save_fig("fig05_system_deviation_from_baseline_panel")

def variable_color(variable):
    mapping = {
        "Capacity": "#D55E00",
        "Strategic Environment": "#0072B2",
        "Clarity": "#009E73",
    }
    return mapping.get(variable, "#555555")


def plot_determinacy_ranking(ablation_ranking):
    df = ablation_ranking.copy()
    df = df.sort_values("Overall_Determinacy_Score", ascending=True)

    colors = [variable_color(v) for v in df["Removed_Variable"]]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.barh(
        df["Removed_Variable"],
        df["Overall_Determinacy_Score"],
        color=colors,
        alpha=0.9
    )

    for i, value in enumerate(df["Overall_Determinacy_Score"]):
        ax.text(
            value,
            i,
            f" {value:.3f}",
            va="center",
            fontsize=10
        )

    ax.set_title("Final Neutralized-Ablation Ranking")
    ax.set_xlabel("Overall determinacy score")
    ax.set_ylabel("Neutralized variable")
    ax.grid(True, axis="x", alpha=0.22)
    ax.grid(False, axis="y")

    save_fig("fig06_final_determinacy_ranking")


def plot_determinacy_components(ablation_ranking):
    df = ablation_ranking.copy()
    df = df.sort_values("Overall_Determinacy_Score", ascending=True)

    component_cols = [
        "Policy_Deviation",
        "Mechanism_Deviation",
        "System_Deviation",
    ]

    component_labels = {
        "Policy_Deviation": "Policy",
        "Mechanism_Deviation": "Mechanism",
        "System_Deviation": "System",
    }

    component_colors = {
        "Policy_Deviation": "#222222",
        "Mechanism_Deviation": "#0072B2",
        "System_Deviation": "#D55E00",
    }

    y = np.arange(len(df))
    height = 0.22

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for idx, col in enumerate(component_cols):
        if col not in df.columns:
            continue

        ax.barh(
            y + (idx - 1) * height,
            df[col],
            height=height,
            color=component_colors[col],
            alpha=0.9,
            label=component_labels[col]
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["Removed_Variable"])
    ax.set_title("Determinacy Components by Neutralized Variable")
    ax.set_xlabel("Deviation from baseline")
    ax.set_ylabel("Neutralized variable")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.22)
    ax.grid(False, axis="y")

    save_fig("fig07_determinacy_components")

def plot_neutralization_diagnostic(policy_debug_df):
    score_cols = ["threat_score", "clarity_score", "capacity_score"]
    score_cols = [c for c in score_cols if c in policy_debug_df.columns]

    check = (
        policy_debug_df
        .groupby(["ablation", "removed_variable"])[score_cols]
        .std()
        .reset_index()
    )
    order = [a for a in ABM_ORDER if a in check["ablation"].unique()]
    check["ablation_order"] = check["ablation"].apply(
        lambda x: order.index(x) if x in order else 999
    )
    check = check.sort_values("ablation_order").reset_index(drop=True)

    matrix = check[score_cols].values
    row_labels = [clean_label(a) for a in check["ablation"]]
    col_labels = [
        "Threat score",
        "Clarity score",
        "Capacity score"
    ][:len(score_cols)]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    im = ax.imshow(matrix, aspect="auto", cmap="Greys")

    ax.set_xticks(np.arange(len(score_cols)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if val > np.nanmax(matrix) * 0.55 else "black"
            )

    ax.set_title("Neutralization Diagnostic: Standard Deviation of Decision Inputs")
    ax.set_xlabel("Decision input")
    ax.set_ylabel("Model condition")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Standard deviation")

    ax.grid(False)
    fig.tight_layout()
    save_fig("fig08_neutralization_diagnostic_heatmap")

def plot_historical_stress_context_clean(abm_validation_df):

    base = (
        abm_validation_df
        .dropna(subset=["Target_Year"])
        .query("ablation == 'baseline'")
        .copy()
        .sort_values("Year")
    )

    def minmax(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.max() == s.min():
            return s * 0.0
        return (s - s.min()) / (s.max() - s.min())

    base["Systemic_Conflict_Level_norm"] = minmax(base["Systemic_Conflict_Level"])
    base["candidate_additions_count_norm"] = minmax(base["candidate_additions_count"])

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.plot(
        base["Year"],
        base["Systemic_Conflict_Level_norm"],
        color="#222222",
        linestyle="-",
        marker="o",
        linewidth=2.1,
        markersize=5.2,
        label="Systemic conflict level"
    )

    ax.plot(
        base["Year"],
        base["candidate_additions_count_norm"],
        color="#D55E00",
        linestyle="--",
        marker="s",
        linewidth=2.1,
        markersize=5.2,
        label="Candidate alliance additions"
    )

    ax.set_title("Historical Systemic Stress and Baseline Alliance Pressure")
    ax.set_xlabel("Year")
    ax.set_ylabel("Min–max normalized value")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="upper right")


def make_academic_abm_figures(
    abm_validation_df,
    policy_debug_df,
    ablation_ranking,
    include_appendix=True,
    include_historical_stress=False
):
    plot_policy_composition_panel(policy_debug_df)
    plot_mechanism_deviation_panel(policy_debug_df)
    plot_formalization_diagnostics(abm_validation_df)
    plot_historical_anchor_panel(abm_validation_df)
    plot_system_deviation_from_baseline(abm_validation_df)
    plot_determinacy_ranking(ablation_ranking)
    plot_determinacy_components(ablation_ranking)

    if include_appendix:
        plot_neutralization_diagnostic(policy_debug_df)

    if include_historical_stress:
        plot_historical_stress_context_clean(abm_validation_df)


make_academic_abm_figures(
    abm_validation_df=abm_validation_df,
    policy_debug_df=policy_debug_df,
    ablation_ranking=ablation_ranking,
    include_appendix=True,
    include_historical_stress=False
)