# app.py
# INRC-II Workforce Scheduler (Single ZIP upload) + GA sliders + outputs
# Run: streamlit run app.py

import io
import json
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page setup
# =========================
st.set_page_config(page_title="INRC-II Workforce Scheduler", layout="wide")


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# =========================
# Utilities: ZIP loading
# =========================
@st.cache_data(show_spinner=False)
def load_inrc2_zip(zip_bytes: bytes):
    """Return: scenario(dict), weekdatas(dict name->dict), histories(dict name->dict), err(str|None)"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            json_map = {}
            for n in names:
                if n.lower().endswith(".json"):
                    with zf.open(n) as f:
                        json_map[Path(n).name] = json.load(f)

        sc_files = sorted([k for k in json_map if k.startswith("Sc-")])
        wd_files = sorted([k for k in json_map if k.startswith("WD-")])
        h_files = sorted([k for k in json_map if k.startswith("H0-")])

        if not sc_files or not wd_files:
            return None, None, None, "ZIP لازم يحتوي على Sc-*.json و WD-*.json على الأقل."

        scenario = json_map[sc_files[0]]
        weekdatas = {k: json_map[k] for k in wd_files}
        histories = {k: json_map[k] for k in h_files}
        return scenario, weekdatas, histories, None

    except zipfile.BadZipFile:
        return None, None, None, "الملف ده مش ZIP صحيح."
    except Exception as e:
        return None, None, None, f"خطأ أثناء القراءة: {e}"


def default_history_from_scenario(scenario: dict) -> dict:
    nurses = scenario.get("nurses", [])
    nurse_hist = []
    for n in nurses:
        nid = n["id"] if isinstance(n, dict) and "id" in n else str(n)
        nurse_hist.append(
            {
                "nurse": nid,
                "numberOfAssignments": 0,
                "numberOfWorkingWeekends": 0,
                "lastAssignedShiftType": "None",
                "numberOfConsecutiveAssignments": 0,
                "numberOfConsecutiveWorkingDays": 0,
                "numberOfConsecutiveDaysOff": 0,
            }
        )
    return {"week": 0, "scenario": scenario.get("id", "unknown"), "nurseHistory": nurse_hist}


# =========================
# Parsing scenario/weekdata
# =========================
def build_maps(scenario: dict):
    nurses = scenario.get("nurses", [])
    nurse_ids = [n["id"] for n in nurses]
    nurse_contract = {n["id"]: n.get("contract", "Unknown") for n in nurses}
    nurse_skills = {n["id"]: set(n.get("skills", [])) for n in nurses}

    contracts = {c["id"]: c for c in scenario.get("contracts", [])}

    shift_types = [s["id"] if isinstance(s, dict) else s for s in scenario.get("shiftTypes", [])]
    skills = [s["id"] if isinstance(s, dict) else s for s in scenario.get("skills", [])]

    # forbidden successions
    forbidden = set()
    for r in scenario.get("forbiddenShiftTypeSuccessions", []):
        pre = r.get("precedingShiftType")
        for suc in r.get("succeedingShiftTypes", []):
            forbidden.add((pre, suc))

    return nurse_ids, nurse_contract, nurse_skills, contracts, shift_types, skills, forbidden


def parse_requirements(weekdata: dict) -> pd.DataFrame:
    """Return DF rows: day, shiftType, skill, min, opt"""
    rows = []
    reqs = weekdata.get("requirements", [])
    for r in reqs:
        shiftType = r.get("shiftType")
        skill = r.get("skill")
        for day in DAYS:
            key = f"requirementOn{day}"
            if key in r and isinstance(r[key], dict):
                mn = int(r[key].get("minimum", 0))
                op = int(r[key].get("optimal", 0))
            else:
                mn, op = 0, 0
            if mn != 0 or op != 0:
                rows.append({"day": day, "shiftType": shiftType, "skill": skill, "min": mn, "opt": op})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["day", "shiftType", "skill", "min", "opt"])
    return df


def parse_shift_off_requests(weekdata: dict) -> pd.DataFrame:
    """Return DF rows: nurse, day, shiftType"""
    rows = []
    for r in weekdata.get("shiftOffRequests", []):
        rows.append({"nurse": r.get("nurse"), "day": r.get("day"), "shiftType": r.get("shiftType", "Any")})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["nurse", "day", "shiftType"])
    return df


# =========================
# Schedule representation
# =========================
# We store two arrays:
#   shift_assign[n_idx, d_idx] in {0..S}  (0=OFF, else index+1 of shift_types)
#   skill_assign[n_idx, d_idx] in {-1..K-1} (skill index, -1 for OFF)


def schedule_to_long_df(nurse_ids, shift_types, skills, shift_assign, skill_assign) -> pd.DataFrame:
    rows = []
    for i, nid in enumerate(nurse_ids):
        for d, day in enumerate(DAYS):
            sidx = int(shift_assign[i, d])
            if sidx == 0:
                continue
            shiftType = shift_types[sidx - 1]
            kidx = int(skill_assign[i, d])
            skill = skills[kidx] if kidx >= 0 else None
            rows.append({"nurse": nid, "day": day, "shiftType": shiftType, "skill": skill})
    return pd.DataFrame(rows)


def schedule_to_pivot_df(nurse_ids, shift_types, skills, shift_assign, skill_assign) -> pd.DataFrame:
    mat = []
    for i, nid in enumerate(nurse_ids):
        row = {"nurse": nid}
        for d, day in enumerate(DAYS):
            sidx = int(shift_assign[i, d])
            if sidx == 0:
                row[day] = "OFF"
            else:
                shiftType = shift_types[sidx - 1]
                kidx = int(skill_assign[i, d])
                skill = skills[kidx] if kidx >= 0 else ""
                row[day] = f"{shiftType}:{skill}" if skill else f"{shiftType}"
        mat.append(row)
    df = pd.DataFrame(mat).set_index("nurse")
    return df


# =========================
# Coverage diagnostics
# =========================
def coverage_diag(req_df: pd.DataFrame, nurse_ids, shift_types, skills, shift_assign, skill_assign) -> pd.DataFrame:
    if req_df.empty:
        return pd.DataFrame(columns=["day", "shiftType", "skill", "cov", "min", "opt", "under_min", "missing_opt"])

    # counts per (day, shiftType, skill)
    cov = {(day, st, sk): 0 for day in DAYS for st in shift_types for sk in skills}

    for i, nid in enumerate(nurse_ids):
        for d, day in enumerate(DAYS):
            sidx = int(shift_assign[i, d])
            if sidx == 0:
                continue
            st = shift_types[sidx - 1]
            kidx = int(skill_assign[i, d])
            if kidx >= 0:
                sk = skills[kidx]
                cov[(day, st, sk)] += 1

    out = []
    for _, r in req_df.iterrows():
        day, st, sk, mn, op = r["day"], r["shiftType"], r["skill"], int(r["min"]), int(r["opt"])
        c = cov.get((day, st, sk), 0)
        under = max(0, mn - c)
        miss = max(0, op - c)
        out.append({"day": day, "shiftType": st, "skill": sk, "cov": c, "min": mn, "opt": op, "under_min": under, "missing_opt": miss})
    return pd.DataFrame(out).sort_values(["day", "shiftType", "skill"]).reset_index(drop=True)


# =========================
# Cost function (constraints)
# =========================
@dataclass
class Weights:
    under_min: int = 1_000_000
    missing_opt: int = 10
    pref_viol: int = 200
    illegal_succ: int = 300
    cons_rule: int = 30


def evaluate_schedule(
    req_df: pd.DataFrame,
    off_df: pd.DataFrame,
    nurse_ids: List[str],
    nurse_contract: Dict[str, str],
    nurse_skills: Dict[str, set],
    contracts: Dict[str, dict],
    shift_types: List[str],
    skills: List[str],
    forbidden_pairs: set,
    history: dict,
    shift_assign: np.ndarray,
    skill_assign: np.ndarray,
    w: Weights,
) -> Tuple[int, dict, pd.DataFrame]:
    # Coverage penalties
    diag = coverage_diag(req_df, nurse_ids, shift_types, skills, shift_assign, skill_assign)
    under_min = int(diag["under_min"].sum()) if not diag.empty else 0
    missing_opt = int(diag["missing_opt"].sum()) if not diag.empty else 0

    # Preference violations (shift off requests)
    pref_viol = 0
    if not off_df.empty:
        reqs = off_df.to_dict("records")
        nurse_index = {nid: i for i, nid in enumerate(nurse_ids)}
        shift_index = {st: s + 1 for s, st in enumerate(shift_types)}  # 1..S
        for r in reqs:
            nid = r["nurse"]
            day = r["day"]
            st_req = r.get("shiftType", "Any")
            if nid not in nurse_index or day not in DAYS:
                continue
            i = nurse_index[nid]
            d = DAYS.index(day)
            sidx = int(shift_assign[i, d])
            if st_req == "Any":
                if sidx != 0:
                    pref_viol += 1
            else:
                if sidx == shift_index.get(st_req, -999):
                    pref_viol += 1

    # Illegal successions (including previous week last shift)
    illegal_succ = 0
    last_shift = {}
    if history and isinstance(history, dict):
        for h in history.get("nurseHistory", []):
            last_shift[h.get("nurse")] = h.get("lastAssignedShiftType", "None")

    for i, nid in enumerate(nurse_ids):
        prev = last_shift.get(nid, "None")
        for d in range(len(DAYS)):
            cur_idx = int(shift_assign[i, d])
            cur = "OFF" if cur_idx == 0 else shift_types[cur_idx - 1]
            # If prev is a real shiftType and cur is a real shiftType, check forbidden
            if prev in shift_types and cur in shift_types:
                if (prev, cur) in forbidden_pairs:
                    illegal_succ += 1
            prev = cur

    # Contract consecutive / assignments / weekend completeness
    cons_shift = 0
    cons_work = 0
    cons_off = 0
    complete_weekend = 0

    # Pull initial streaks from history
    hist_map = {}
    if history and isinstance(history, dict):
        for h in history.get("nurseHistory", []):
            hist_map[h.get("nurse")] = h

    for i, nid in enumerate(nurse_ids):
        c_id = nurse_contract.get(nid, "Unknown")
        c = contracts.get(c_id, {})

        minA = int(c.get("minimumNumberOfAssignments", 0))
        maxA = int(c.get("maximumNumberOfAssignments", 10**9))
        minCW = int(c.get("minimumNumberOfConsecutiveWorkingDays", 0))
        maxCW = int(c.get("maximumNumberOfConsecutiveWorkingDays", 10**9))
        minCO = int(c.get("minimumNumberOfConsecutiveDaysOff", 0))
        maxCO = int(c.get("maximumNumberOfConsecutiveDaysOff", 10**9))
        completeW = int(c.get("completeWeekends", 0))
        maxWW = int(c.get("maximumNumberOfWorkingWeekends", 10**9))

        h = hist_map.get(nid, {})
        prev_work_streak = int(h.get("numberOfConsecutiveWorkingDays", 0))
        prev_off_streak = int(h.get("numberOfConsecutiveDaysOff", 0))
        prev_work_weekends = int(h.get("numberOfWorkingWeekends", 0))

        worked = (shift_assign[i, :] != 0).astype(int)
        totalA = int(worked.sum())

        # assignment min/max
        if totalA < minA:
            cons_shift += (minA - totalA)
        if totalA > maxA:
            cons_shift += (totalA - maxA)

        # working/off runs
        # incorporate previous streaks: if first day is work then run starts from prev_work_streak+1 else new
        work_runs = []
        off_runs = []
        cur = worked[0]
        run = 1
        for d in range(1, len(DAYS)):
            if worked[d] == cur:
                run += 1
            else:
                if cur == 1:
                    work_runs.append(run)
                else:
                    off_runs.append(run)
                cur = worked[d]
                run = 1
        if cur == 1:
            work_runs.append(run)
        else:
            off_runs.append(run)

        # adjust first run using history
        if worked[0] == 1 and work_runs:
            work_runs[0] += prev_work_streak
        if worked[0] == 0 and off_runs:
            off_runs[0] += prev_off_streak

        for rlen in work_runs:
            if rlen > maxCW:
                cons_work += (rlen - maxCW)
            if minCW > 0 and rlen < minCW:
                cons_work += (minCW - rlen)

        for rlen in off_runs:
            if rlen > maxCO:
                cons_off += (rlen - maxCO)
            if minCO > 0 and rlen < minCO:
                cons_off += (minCO - rlen)

        # weekends count + complete weekends
        sat = DAYS.index("Saturday")
        sun = DAYS.index("Sunday")
        works_weekend = int((worked[sat] == 1) or (worked[sun] == 1))
        working_weekends = prev_work_weekends + works_weekend
        if working_weekends > maxWW:
            cons_shift += (working_weekends - maxWW)

        if completeW == 1:
            # If works weekend, must work both Sat & Sun (no split)
            if (worked[sat] + worked[sun]) == 1:
                complete_weekend += 1

    cons_cost = w.cons_rule * (cons_shift + cons_work + cons_off + complete_weekend)

    cost = (
        w.under_min * under_min
        + w.missing_opt * missing_opt
        + w.pref_viol * pref_viol
        + w.illegal_succ * illegal_succ
        + cons_cost
    )

    breakdown = {
        "under_min": under_min,
        "missing_opt": missing_opt,
        "pref_viol": pref_viol,
        "illegal_succ": illegal_succ,
        "cons_shift": cons_shift,
        "cons_work": cons_work,
        "cons_off": cons_off,
        "complete_weekend": complete_weekend,
        "cons_cost": cons_cost,
        "cost": int(cost),
    }
    return int(cost), breakdown, diag


# =========================
# Greedy random initializer
# =========================
def greedy_build(
    req_df: pd.DataFrame,
    nurse_ids: List[str],
    nurse_contract: Dict[str, str],
    nurse_skills: Dict[str, set],
    contracts: Dict[str, dict],
    shift_types: List[str],
    skills: List[str],
    seed: int = 0,
    fill_opt_prob: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N = len(nurse_ids)
    S = len(shift_types)
    K = len(skills)

    shift_assign = np.zeros((N, 7), dtype=np.int16)  # 0=OFF, else 1..S
    skill_assign = np.full((N, 7), -1, dtype=np.int16)

    nurse_idx = {nid: i for i, nid in enumerate(nurse_ids)}
    skill_idx = {sk: k for k, sk in enumerate(skills)}
    shift_idx = {st: s for s, st in enumerate(shift_types)}  # 0..S-1

    # Track max assignments
    maxA = []
    for nid in nurse_ids:
        c = contracts.get(nurse_contract.get(nid, "Unknown"), {})
        maxA.append(int(c.get("maximumNumberOfAssignments", 9999)))
    maxA = np.array(maxA, dtype=int)
    usedA = np.zeros(N, dtype=int)

    # Helper: pick candidate nurses for (skill, day) not assigned yet that day and not over maxA
    def candidates(day_i: int, required_skill: str):
        cands = []
        for nid in nurse_ids:
            i = nurse_idx[nid]
            if shift_assign[i, day_i] != 0:
                continue
            if usedA[i] >= maxA[i]:
                continue
            if required_skill in nurse_skills[nid]:
                cands.append(i)
        rng.shuffle(cands)
        # prefer those with lower used assignments
        cands.sort(key=lambda x: usedA[x])
        return cands

    if req_df.empty:
        return shift_assign, skill_assign

    # 1) Satisfy minimum demand
    for day in DAYS:
        day_i = DAYS.index(day)
        day_reqs = req_df[req_df["day"] == day]
        for _, r in day_reqs.iterrows():
            st = r["shiftType"]
            sk = r["skill"]
            mn = int(r["min"])
            if mn <= 0:
                continue
            for _k in range(mn):
                cands = candidates(day_i, sk)
                if not cands:
                    break
                i = cands[0]
                shift_assign[i, day_i] = shift_idx[st] + 1
                skill_assign[i, day_i] = skill_idx[sk]
                usedA[i] += 1

    # 2) Try to satisfy some optimal (soft) demand
    for day in DAYS:
        day_i = DAYS.index(day)
        day_reqs = req_df[req_df["day"] == day]
        for _, r in day_reqs.iterrows():
            st = r["shiftType"]
            sk = r["skill"]
            op = int(r["opt"])
            mn = int(r["min"])
            extra = max(0, op - mn)
            for _k in range(extra):
                if rng.random() > fill_opt_prob:
                    continue
                cands = candidates(day_i, sk)
                if not cands:
                    break
                i = cands[0]
                shift_assign[i, day_i] = shift_idx[st] + 1
                skill_assign[i, day_i] = skill_idx[sk]
                usedA[i] += 1

    return shift_assign, skill_assign


# =========================
# Genetic Algorithm
# =========================
def tournament_select(costs: np.ndarray, k: int, rng) -> int:
    idxs = rng.integers(0, len(costs), size=k)
    best = idxs[0]
    for j in idxs[1:]:
        if costs[j] < costs[best]:
            best = j
    return best


def crossover(a_shift, a_skill, b_shift, b_skill, rng) -> Tuple[np.ndarray, np.ndarray]:
    # uniform crossover by cell
    mask = rng.random(a_shift.shape) < 0.5
    child_shift = a_shift.copy()
    child_skill = a_skill.copy()
    child_shift[mask] = b_shift[mask]
    child_skill[mask] = b_skill[mask]
    return child_shift, child_skill


def mutate(shift_assign, skill_assign, n_shift_types, n_skills, rate: float, rng):
    # random mutation: flip some cells to OFF or random shift; random skill if shift
    N, D = shift_assign.shape
    for i in range(N):
        for d in range(D):
            if rng.random() < rate:
                if rng.random() < 0.25:
                    shift_assign[i, d] = 0
                    skill_assign[i, d] = -1
                else:
                    s = rng.integers(1, n_shift_types + 1)
                    shift_assign[i, d] = s
                    skill_assign[i, d] = int(rng.integers(0, n_skills))
    return shift_assign, skill_assign


def run_ga(
    req_df,
    off_df,
    nurse_ids,
    nurse_contract,
    nurse_skills,
    contracts,
    shift_types,
    skills,
    forbidden_pairs,
    history,
    generations: int,
    pop_size: int,
    mutation_rate: float,
    elitism: int,
    weights: Weights,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    # init population (greedy + randomness)
    pop_shift = []
    pop_skill = []
    for p in range(pop_size):
        sa, ka = greedy_build(
            req_df, nurse_ids, nurse_contract, nurse_skills, contracts, shift_types, skills,
            seed=int(rng.integers(0, 1_000_000)), fill_opt_prob=float(rng.uniform(0.2, 0.55))
        )
        pop_shift.append(sa)
        pop_skill.append(ka)

    pop_shift = np.array(pop_shift, dtype=np.int16)
    pop_skill = np.array(pop_skill, dtype=np.int16)

    best = None
    best_cost = 10**18
    best_break = None
    best_diag = None

    prog = st.progress(0)
    status = st.empty()

    for g in range(1, generations + 1):
        costs = np.zeros(pop_size, dtype=np.int64)
        breaks = [None] * pop_size
        diags = [None] * pop_size

        for i in range(pop_size):
            c, br, dg = evaluate_schedule(
                req_df, off_df, nurse_ids, nurse_contract, nurse_skills, contracts,
                shift_types, skills, forbidden_pairs, history,
                pop_shift[i], pop_skill[i], weights
            )
            costs[i] = c
            breaks[i] = br
            diags[i] = dg

        # update best
        idx_best = int(np.argmin(costs))
        if costs[idx_best] < best_cost:
            best_cost = int(costs[idx_best])
            best = (pop_shift[idx_best].copy(), pop_skill[idx_best].copy())
            best_break = breaks[idx_best]
            best_diag = diags[idx_best]

        # UI update
        if g == 1 or g % max(1, generations // 10) == 0 or g == generations:
            status.write(f"Gen {g}/{generations} | Best fitness = {-best_cost} | cost={best_cost} | {best_break}")
        prog.progress(int(100 * g / generations))

        # evolve
        # elitism keep top
        elite_idx = np.argsort(costs)[:max(0, elitism)]
        new_shift = []
        new_skill = []
        for ei in elite_idx:
            new_shift.append(pop_shift[ei].copy())
            new_skill.append(pop_skill[ei].copy())

        # rest by selection + crossover + mutation
        while len(new_shift) < pop_size:
            pa = tournament_select(costs, k=3, rng=rng)
            pb = tournament_select(costs, k=3, rng=rng)
            child_s, child_k = crossover(pop_shift[pa], pop_skill[pa], pop_shift[pb], pop_skill[pb], rng)
            child_s, child_k = mutate(child_s, child_k, len(shift_types), len(skills), mutation_rate, rng)
            new_shift.append(child_s)
            new_skill.append(child_k)

        pop_shift = np.array(new_shift, dtype=np.int16)
        pop_skill = np.array(new_skill, dtype=np.int16)

    prog.empty()
    return best_cost, best_break, best_diag, best


# =========================
# UI
# =========================
st.markdown("# 🧬 INRC-II Workforce Scheduler")
st.caption("Upload INRC-II ZIP ➜ tune GA params/weights ➜ run optimizer ➜ view schedule + diagnostics + download outputs.")


# Sidebar
st.sidebar.title("⚙️ Settings")

zip_file = st.sidebar.file_uploader("Upload dataset ZIP (e.g., n005w4.zip)", type=["zip"])

if zip_file is None:
    st.info("ارفع ملف واحد فقط: ZIP فيه Sc-*.json و WD-*.json و (اختياري) H0-*.json.")
    st.stop()

scenario, weekdatas, histories, err = load_inrc2_zip(zip_file.getvalue())
if err:
    st.error(err)
    st.stop()

nurse_ids, nurse_contract, nurse_skills, contracts, shift_types, skills, forbidden_pairs = build_maps(scenario)

wd_name = st.sidebar.selectbox("WeekData (WD-*.json)", list(weekdatas.keys()), index=0)
weekdata = weekdatas[wd_name]

hist_options = ["None (auto)"] + list(histories.keys())
hist_name = st.sidebar.selectbox("History (H0-*.json) [optional]", hist_options, index=0)
history = default_history_from_scenario(scenario) if hist_name.startswith("None") else histories[hist_name]

req_df = parse_requirements(weekdata)
off_df = parse_shift_off_requests(weekdata)

# GA params
st.sidebar.markdown("---")
st.sidebar.subheader("Genetic Algorithm")
generations = st.sidebar.slider("Generations", 50, 800, 300, 10)
pop_size = st.sidebar.slider("Population Size", 20, 400, 150, 10)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.30, 0.01)
elitism = st.sidebar.number_input("Elitism Count", min_value=0, max_value=50, value=5, step=1)

# Weights
st.sidebar.markdown("---")
st.sidebar.subheader("Weights")
w_under = st.sidebar.number_input("Under Min (hard)", value=1_000_000, step=50_000)
w_opt = st.sidebar.number_input("Missing Optimal", value=10, step=1)
w_pref = st.sidebar.number_input("Preference Violation", value=200, step=10)
w_illegal = st.sidebar.number_input("Illegal Succession", value=300, step=10)
w_cons = st.sidebar.number_input("Contract Rules (base)", value=30, step=5)

weights = Weights(
    under_min=int(w_under),
    missing_opt=int(w_opt),
    pref_viol=int(w_pref),
    illegal_succ=int(w_illegal),
    cons_rule=int(w_cons),
)

st.sidebar.success(f"Loaded: {wd_name} | Nurses: {len(nurse_ids)} | Shifts: {shift_types} | Skills: {skills}")


tab1, tab2 = st.tabs(["🧪 Data Audit", "🚀 Optimizer"])


with tab1:
    c1, c2, c3 = st.columns([1.2, 1, 1])

    with c1:
        st.subheader("Scenario Overview")
        st.write(
            {
                "Scenario ID": scenario.get("id"),
                "Weeks": scenario.get("numberOfWeeks"),
                "Nurses": len(nurse_ids),
                "Skills": skills,
                "Shift Types": shift_types,
            }
        )

        st.subheader("Contracts")
        if scenario.get("contracts"):
            st.dataframe(pd.DataFrame(scenario["contracts"]), use_container_width=True)
        else:
            st.info("No contracts found.")

    with c2:
        st.subheader("Forbidden Successions")
        if forbidden_pairs:
            st.dataframe(pd.DataFrame(sorted(list(forbidden_pairs)), columns=["preceding", "succeeding"]), use_container_width=True)
        else:
            st.info("No forbidden successions.")

        st.subheader("Shift Off Requests")
        st.write(f"#requests: {len(off_df)}")
        st.dataframe(off_df.head(20), use_container_width=True)

    with c3:
        st.subheader("Demand Summary (this WeekData)")
        total_min = int(req_df["min"].sum()) if not req_df.empty else 0
        total_opt = int(req_df["opt"].sum()) if not req_df.empty else 0
        st.metric("Total minimum demand", total_min)
        st.metric("Total optimal demand", total_opt)
        st.dataframe(req_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Sanity check (all-OFF)")
    N = len(nurse_ids)
    all_off_shift = np.zeros((N, 7), dtype=np.int16)
    all_off_skill = np.full((N, 7), -1, dtype=np.int16)

    cost0, br0, diag0 = evaluate_schedule(
        req_df, off_df, nurse_ids, nurse_contract, nurse_skills, contracts,
        shift_types, skills, forbidden_pairs, history,
        all_off_shift, all_off_skill, weights
    )
    st.write("Evaluate sanity check on all-OFF:", br0)
    st.dataframe(diag0, use_container_width=True)


with tab2:
    st.subheader("Run Optimizer")
    st.caption("اضغط Run Optimizer بعد رفع الـ ZIP. النتائج هتظهر تحت + تقدر تعمل Download للـ CSVs.")

    run_btn = st.button("🚀 Run Optimizer", type="primary")

    if run_btn:
        best_cost, best_break, best_diag, best_sol = run_ga(
            req_df=req_df,
            off_df=off_df,
            nurse_ids=nurse_ids,
            nurse_contract=nurse_contract,
            nurse_skills=nurse_skills,
            contracts=contracts,
            shift_types=shift_types,
            skills=skills,
            forbidden_pairs=forbidden_pairs,
            history=history,
            generations=int(generations),
            pop_size=int(pop_size),
            mutation_rate=float(mutation_rate),
            elitism=int(elitism),
            weights=weights,
            seed=42,
        )

        best_shift, best_skill = best_sol

        st.markdown("### ✅ BEST RESULT")
        st.write(f"BEST fitness: {-best_cost}")
        st.write(best_break)

        df_long = schedule_to_long_df(nurse_ids, shift_types, skills, best_shift, best_skill)
        df_pivot = schedule_to_pivot_df(nurse_ids, shift_types, skills, best_shift, best_skill)
        diag = best_diag.copy() if best_diag is not None else pd.DataFrame()

        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("### Schedule (Long)")
            st.dataframe(df_long, use_container_width=True, height=420)

        with c2:
            st.markdown("### Schedule (Timetable)")
            st.dataframe(df_pivot, use_container_width=True, height=420)

        st.markdown("### Coverage Diagnostics")
        st.dataframe(diag, use_container_width=True, height=420)

        # Prepare downloads
        out_long = df_long.to_csv(index=False).encode("utf-8")
        out_pivot = df_pivot.to_csv().encode("utf-8")
        out_diag = diag.to_csv(index=False).encode("utf-8")
        out_metrics = json.dumps(best_break, indent=2).encode("utf-8")

        st.markdown("### ⬇️ Download Outputs")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("Download schedule_long.csv", out_long, file_name="best_schedule_long_inrc2.csv", mime="text/csv")
        with d2:
            st.download_button("Download schedule_pivot.csv", out_pivot, file_name="best_schedule_pivot_inrc2.csv", mime="text/csv")
        with d3:
            st.download_button("Download coverage_diag.csv", out_diag, file_name="coverage_diag.csv", mime="text/csv")
        with d4:
            st.download_button("Download best_metrics.json", out_metrics, file_name="best_metrics.json", mime="application/json")

    else:
        st.info("ارفع ZIP ➜ عدّل السلايدرز/الـ weights ➜ اضغط Run Optimizer.")
