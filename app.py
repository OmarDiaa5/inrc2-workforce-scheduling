import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Page / UI Styling
# =========================
st.set_page_config(page_title="INRC-II Workforce Scheduler", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .stButton>button {border-radius: 10px; padding: 0.6rem 1rem; font-weight: 600;}
      .stDataFrame {border-radius: 12px; overflow: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_TO_IDX = {d: i for i, d in enumerate(DAYS)}
IDX_TO_DAY = {i: d for i, d in enumerate(DAYS)}


# =========================
# Helpers: JSON loading
# =========================
def load_json_file(uploaded) -> dict:
    return json.loads(uploaded.getvalue().decode("utf-8"))

def identify_inrc_kind(obj: dict) -> str:
    keys = set(obj.keys())
    if {"id", "numberOfWeeks", "skills", "shiftTypes", "contracts", "nurses"}.issubset(keys):
        return "SCENARIO"
    if {"week", "scenario", "nurseHistory"}.issubset(keys):
        return "HISTORY"
    if {"scenario", "requirements", "shiftOffRequests"}.issubset(keys):
        return "WEEKDATA"
    return "UNKNOWN"


# =========================
# Data structures
# =========================
@dataclass
class Nurse:
    id: str
    contract_id: str
    skills: List[str]

@dataclass
class Contract:
    id: str
    min_assign: int
    max_assign: int
    min_cons_work: int
    max_cons_work: int
    min_cons_off: int
    max_cons_off: int
    max_work_weekends: int
    complete_weekends: int

@dataclass
class ReqCell:
    day_idx: int
    shift: str
    skill: str
    minimum: int
    optimal: int


# =========================
# Parse INRC files
# =========================
def parse_inrc(scenario: dict, weekdata: dict, history: Optional[dict]):
    skills = [s["id"] if isinstance(s, dict) else str(s) for s in scenario["skills"]]
    shift_types = [s["id"] if isinstance(s, dict) else str(s) for s in scenario["shiftTypes"]]
    forb = {(x["precedingShiftType"], y) for x in scenario["forbiddenShiftTypeSuccessions"] for y in x["succeedingShiftTypes"]}

    contracts = {}
    for c in scenario["contracts"]:
        contracts[c["id"]] = Contract(
            id=c["id"],
            min_assign=int(c["minimumNumberOfAssignments"]),
            max_assign=int(c["maximumNumberOfAssignments"]),
            min_cons_work=int(c["minimumNumberOfConsecutiveWorkingDays"]),
            max_cons_work=int(c["maximumNumberOfConsecutiveWorkingDays"]),
            min_cons_off=int(c["minimumNumberOfConsecutiveDaysOff"]),
            max_cons_off=int(c["maximumNumberOfConsecutiveDaysOff"]),
            max_work_weekends=int(c["maximumNumberOfWorkingWeekends"]),
            complete_weekends=int(c.get("completeWeekends", 0)),
        )

    nurses = []
    for n in scenario["nurses"]:
        nurses.append(Nurse(id=n["id"], contract_id=n["contract"], skills=list(n["skills"])))

    # Requirements -> list of ReqCell
    req_cells: List[ReqCell] = []
    for r in weekdata["requirements"]:
        shift = r["shiftType"]
        skill = r["skill"]
        for d in DAYS:
            key = f"requirementOn{d}"
            minv = int(r[key]["minimum"])
            optv = int(r[key]["optimal"])
            req_cells.append(ReqCell(day_idx=DAY_TO_IDX[d], shift=shift, skill=skill, minimum=minv, optimal=optv))

    # Off requests: (nurse_id, day_idx, shiftType or "Any")
    off_requests = []
    for sr in weekdata.get("shiftOffRequests", []):
        off_requests.append((sr["nurse"], DAY_TO_IDX[sr["day"]], sr["shiftType"]))

    # History: lastAssignedShiftType (for illegal succession day0)
    last_shift = {n.id: None for n in nurses}
    hist_work = {n.id: 0 for n in nurses}
    hist_off = {n.id: 0 for n in nurses}
    if history is not None:
        for h in history.get("nurseHistory", []):
            nid = h["nurse"]
            last_shift[nid] = h.get("lastAssignedShiftType", None)
            hist_work[nid] = int(h.get("numberOfConsecutiveWorkingDays", 0))
            hist_off[nid] = int(h.get("numberOfConsecutiveDaysOff", 0))

    return skills, shift_types, forb, contracts, nurses, req_cells, off_requests, last_shift, hist_work, hist_off


# =========================
# GA representation
# Each gene = OFF or (shiftType, chosenSkill)
# =========================
OFF = ("OFF", None)

def random_gene(shift_types: List[str], nurse_skills: List[str], p_off=0.25):
    if random.random() < p_off:
        return OFF
    sh = random.choice(shift_types)
    sk = random.choice(nurse_skills)
    return (sh, sk)

def init_individual(nurses: List[Nurse], shift_types: List[str], p_off=0.25):
    # schedule[nurse_i][day_i] = (shift, skill) or OFF
    sched = []
    for n in nurses:
        row = [random_gene(shift_types, n.skills, p_off=p_off) for _ in range(7)]
        sched.append(row)
    return sched

def copy_individual(ind):
    return [row[:] for row in ind]

def mutate(ind, nurses, shift_types, rate):
    out = copy_individual(ind)
    for i, n in enumerate(nurses):
        for d in range(7):
            if random.random() < rate:
                out[i][d] = random_gene(shift_types, n.skills, p_off=0.20)
    return out

def crossover(a, b):
    # uniform crossover
    na = len(a)
    out1, out2 = [], []
    for i in range(na):
        row1, row2 = [], []
        for d in range(7):
            if random.random() < 0.5:
                row1.append(a[i][d]); row2.append(b[i][d])
            else:
                row1.append(b[i][d]); row2.append(a[i][d])
        out1.append(row1); out2.append(row2)
    return out1, out2


# =========================
# Evaluation
# =========================
def eval_schedule(
    sched,
    skills, shift_types, forb, contracts, nurses, req_cells, off_requests,
    last_shift, hist_work, hist_off,
    W
):
    # Coverage table: (day, shift, skill) -> count
    cov = {(c.day_idx, c.shift, c.skill): 0 for c in req_cells}

    skill_mismatch = 0
    illegal_succ = 0
    pref_viol = 0

    # Track for contract penalties
    assign_count = {n.id: 0 for n in nurses}
    work_flags = {n.id: [0]*7 for n in nurses}  # 1 if working that day
    shift_only = {n.id: ["OFF"]*7 for n in nurses}

    # Fill counts
    for i, n in enumerate(nurses):
        for d in range(7):
            sh, sk = sched[i][d]
            if sh == "OFF":
                continue
            assign_count[n.id] += 1
            work_flags[n.id][d] = 1
            shift_only[n.id][d] = sh

            if sk not in n.skills:
                skill_mismatch += 1

            # coverage: ONLY counts for the chosen skill
            key = (d, sh, sk)
            if key in cov:
                cov[key] += 1

    # Under min / missing opt
    under_min = 0
    missing_opt = 0
    for c in req_cells:
        k = (c.day_idx, c.shift, c.skill)
        covered = cov.get(k, 0)
        under_min += max(0, c.minimum - covered)
        missing_opt += max(0, c.optimal - covered)

    # Forbidden successions (day0 compares with history lastShift)
    for n in nurses:
        prev = last_shift.get(n.id, None)
        for d in range(7):
            cur = shift_only[n.id][d]
            if d == 0:
                if prev is not None and cur != "OFF" and (prev, cur) in forb:
                    illegal_succ += 1
            else:
                p = shift_only[n.id][d-1]
                if p != "OFF" and cur != "OFF" and (p, cur) in forb:
                    illegal_succ += 1

    # Preferences (off requests)
    for (nid, day_idx, req_shift) in off_requests:
        # find nurse index
        i = next((k for k, nn in enumerate(nurses) if nn.id == nid), None)
        if i is None:
            continue
        sh, _ = sched[i][day_idx]
        if req_shift == "Any":
            if sh != "OFF":
                pref_viol += 1
        else:
            if sh == req_shift:
                pref_viol += 1

    # Contract penalties (soft)
    cons_work = 0
    cons_off = 0
    complete_weekend = 0
    over_assign = 0
    under_assign = 0
    too_many_weekends = 0

    for n in nurses:
        c = contracts[n.contract_id]
        cnt = assign_count[n.id]
        if cnt < c.min_assign:
            under_assign += (c.min_assign - cnt)
        if cnt > c.max_assign:
            over_assign += (cnt - c.max_assign)

        # weekend completeness (Sat idx=5, Sun idx=6)
        if c.complete_weekends:
            sat = work_flags[n.id][5]
            sun = work_flags[n.id][6]
            if sat != sun:
                complete_weekend += 1

        # working weekends count (if any day worked on weekend)
        wk = 1 if (work_flags[n.id][5] or work_flags[n.id][6]) else 0
        if wk > c.max_work_weekends:
            too_many_weekends += (wk - c.max_work_weekends)

        # consecutive runs inside the week (+ history prefix)
        flags = work_flags[n.id]

        # build runs for work/off
        def count_run_violations(flags, is_work: bool, hist_prefix: int, min_len: int, max_len: int):
            viol = 0
            cur = 0
            # prefix
            if flags[0] == (1 if is_work else 0):
                cur = hist_prefix
            for d in range(7):
                if flags[d] == (1 if is_work else 0):
                    cur += 1
                else:
                    # run ended
                    if cur > 0:
                        if cur < min_len:
                            viol += (min_len - cur)
                        if cur > max_len:
                            viol += (cur - max_len)
                    cur = 0
            # last run end at week end
            if cur > 0:
                if cur < min_len:
                    viol += (min_len - cur)
                if cur > max_len:
                    viol += (cur - max_len)
            return viol

        cons_work += count_run_violations(flags, True, hist_work.get(n.id, 0), c.min_cons_work, c.max_cons_work)
        cons_off  += count_run_violations(flags, False, hist_off.get(n.id, 0), c.min_cons_off, c.max_cons_off)

    cons_cost = (
        W["under_assign"] * under_assign +
        W["over_assign"]  * over_assign +
        W["cons_work"]    * cons_work +
        W["cons_off"]     * cons_off +
        W["weekend"]      * complete_weekend +
        W["work_weekends"]* too_many_weekends
    )

    # Total cost
    cost = (
        W["under_min"]     * under_min +
        W["missing_opt"]   * missing_opt +
        W["skill_mismatch"]* skill_mismatch +
        W["illegal_succ"]  * illegal_succ +
        W["pref_viol"]     * pref_viol +
        cons_cost
    )

    breakdown = dict(
        under_min=under_min,
        skill_mismatch=skill_mismatch,
        illegal_succ=illegal_succ,
        missing_opt=missing_opt,
        pref_viol=pref_viol,
        cons_work=cons_work,
        cons_off=cons_off,
        complete_weekend=complete_weekend,
        cons_cost=cons_cost,
        cost=int(cost),
    )
    fitness = -cost
    return fitness, breakdown


def fairness_std_assignments(sched, nurses):
    counts = []
    for i, n in enumerate(nurses):
        c = sum(1 for d in range(7) if sched[i][d][0] != "OFF")
        counts.append(c)
    return float(np.std(counts))


# =========================
# Build output DataFrames
# =========================
def build_outputs(sched, nurses, req_cells):
    rows = []
    for i, n in enumerate(nurses):
        for d in range(7):
            sh, sk = sched[i][d]
            if sh == "OFF":
                continue
            rows.append({"nurse": n.id, "day": IDX_TO_DAY[d], "shiftType": sh, "skill": sk})
    df_long = pd.DataFrame(rows).sort_values(["day", "shiftType", "nurse"])

    # pivot
    pivot_rows = []
    for i, n in enumerate(nurses):
        row = {"nurse": n.id}
        for d in range(7):
            sh, sk = sched[i][d]
            row[IDX_TO_DAY[d]] = "OFF" if sh == "OFF" else f"{sh}:{sk}"
        pivot_rows.append(row)
    df_pivot = pd.DataFrame(pivot_rows).set_index("nurse")

    # coverage diag
    # cov counts per cell
    cov_map = {(c.day_idx, c.shift, c.skill): 0 for c in req_cells}
    for i, n in enumerate(nurses):
        for d in range(7):
            sh, sk = sched[i][d]
            if sh == "OFF":
                continue
            k = (d, sh, sk)
            if k in cov_map:
                cov_map[k] += 1

    diag = []
    for c in req_cells:
        k = (c.day_idx, c.shift, c.skill)
        covered = cov_map.get(k, 0)
        diag.append({
            "day": IDX_TO_DAY[c.day_idx],
            "shiftType": c.shift,
            "skill": c.skill,
            "cov": covered,
            "min": c.minimum,
            "opt": c.optimal,
            "under_min": max(0, c.minimum - covered),
            "missing_opt": max(0, c.optimal - covered),
        })
    df_cov = pd.DataFrame(diag).sort_values(["day", "shiftType", "skill"])
    return df_long, df_pivot, df_cov


# =========================
# GA Runner
# =========================
def run_ga(
    skills, shift_types, forb, contracts, nurses, req_cells, off_requests,
    last_shift, hist_work, hist_off,
    generations, pop_size, mutation_rate, elitism,
    W,
    progress_cb=None
):
    pop = [init_individual(nurses, shift_types, p_off=0.30) for _ in range(pop_size)]

    best = None
    best_fit = -1e18
    best_breakdown = None

    def tournament_select(scored, k=3):
        cand = random.sample(scored, k)
        cand.sort(key=lambda x: x[0], reverse=True)
        return cand[0][1]

    for gen in range(1, generations + 1):
        scored = []
        for ind in pop:
            fit, bd = eval_schedule(
                ind, skills, shift_types, forb, contracts, nurses, req_cells, off_requests,
                last_shift, hist_work, hist_off, W
            )
            scored.append((fit, ind, bd))

        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit = scored[0][0]
            best = copy_individual(scored[0][1])
            best_breakdown = scored[0][2]

        if progress_cb and (gen == 1 or gen % 10 == 0 or gen == generations):
            progress_cb(gen, best_fit, best_breakdown, best)

        elites = [copy_individual(x[1]) for x in scored[:elitism]]
        new_pop = elites[:]

        while len(new_pop) < pop_size:
            p1 = tournament_select(scored)
            p2 = tournament_select(scored)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, nurses, shift_types, mutation_rate)
            c2 = mutate(c2, nurses, shift_types, mutation_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop

    return best, best_fit, best_breakdown


# =========================
# Sidebar Controls (Like your screenshot)
# =========================
st.sidebar.title("⚙️ Settings")

sc_file = st.sidebar.file_uploader("Upload Scenario (Sc-*.json)", type=["json"])
wd_file = st.sidebar.file_uploader("Upload WeekData (WD-*.json)", type=["json"])
h_file  = st.sidebar.file_uploader("Upload History (H0-*.json) [optional]", type=["json"])

st.sidebar.markdown("---")
generations = st.sidebar.slider("Generations", 50, 800, 300, 10)
pop_size = st.sidebar.slider("Population Size", 20, 400, 150, 10)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.00, 1.00, 0.30, 0.01)
elitism = st.sidebar.number_input("Elitism Count", min_value=1, max_value=50, value=5, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Weights")

W = {
    "under_min": st.sidebar.number_input("Under Min (hard)", value=1_000_000, step=10_000),
    "missing_opt": st.sidebar.number_input("Missing Opt", value=10, step=1),
    "skill_mismatch": st.sidebar.number_input("Skill Mismatch", value=50_000, step=1_000),
    "illegal_succ": st.sidebar.number_input("Illegal Succession", value=200_000, step=1_000),
    "pref_viol": st.sidebar.number_input("Preference Violation", value=200, step=10),

    # contracts (soft)
    "under_assign": st.sidebar.number_input("Under Assignments", value=50, step=10),
    "over_assign": st.sidebar.number_input("Over Assignments", value=50, step=10),
    "cons_work": st.sidebar.number_input("Consecutive Work Viol", value=30, step=5),
    "cons_off": st.sidebar.number_input("Consecutive Off Viol", value=30, step=5),
    "weekend": st.sidebar.number_input("Complete Weekend Viol", value=50, step=10),
    "work_weekends": st.sidebar.number_input("Too Many Weekends", value=50, step=10),
}

# =========================
# Main Header + Layout
# =========================
st.title("🧬 INRC-II Workforce Scheduler")
st.caption("Upload INRC-II JSONs → Tune GA params/weights → Run optimizer → View schedule + diagnostics + download outputs.")

if not sc_file or not wd_file:
    st.info("ارفع ملف Scenario + WeekData من الشريط الجانبي عشان نقدر نشغّل الـ Optimizer.")
    st.stop()

scenario = load_json_file(sc_file)
weekdata = load_json_file(wd_file)
history = load_json_file(h_file) if h_file else None

skills, shift_types, forb, contracts, nurses, req_cells, off_requests, last_shift, hist_work, hist_off = parse_inrc(
    scenario, weekdata, history
)

# Show top data like your screenshot (two tables)
left, right = st.columns(2)

with left:
    st.subheader("👩‍⚕️ Nurses")
    df_n = pd.DataFrame([{"id": n.id, "contract": n.contract_id, "skills": ", ".join(n.skills)} for n in nurses])
    st.dataframe(df_n, use_container_width=True, height=320)

with right:
    st.subheader("📌 Requirements (sample)")
    # show a compact table
    req_preview = pd.DataFrame([{
        "day": IDX_TO_DAY[c.day_idx], "shiftType": c.shift, "skill": c.skill, "min": c.minimum, "opt": c.optimal
    } for c in req_cells])
    st.dataframe(req_preview.head(25), use_container_width=True, height=320)

tab_audit, tab_opt = st.tabs(["🛡️ Data Audit", "🚀 Optimizer"])

with tab_audit:
    st.write("**Scenario summary**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Nurses", len(nurses))
    c2.metric("Shift Types", len(shift_types))
    c3.metric("Skills", len(skills))

    st.write("**Demand totals**")
    total_min = sum(c.minimum for c in req_cells)
    total_opt = sum(c.optimal for c in req_cells)
    st.metric("Total minimum demand", total_min)
    st.metric("Total optimal demand", total_opt)

    st.write("**Shift off requests**")
    df_req = pd.DataFrame(off_requests, columns=["nurse", "day_idx", "shiftType"])
    if not df_req.empty:
        df_req["day"] = df_req["day_idx"].map(IDX_TO_DAY)
        st.dataframe(df_req[["nurse", "day", "shiftType"]], use_container_width=True)
    else:
        st.success("No shift-off requests in this WeekData.")

with tab_opt:
    run = st.button("🚀 Run Optimizer", type="primary")

    prog = st.progress(0)
    status = st.empty()
    best_box = st.empty()

    if run:
        def progress_cb(gen, best_fit, bd, best_sched):
            prog.progress(int(gen / generations * 100))
            fair = fairness_std_assignments(best_sched, nurses) if best_sched is not None else 0.0
            status.info(f"Gen {gen}/{generations} | Best Fitness: {int(best_fit)} | Fairness Std(assignments): {fair:.2f}")
            best_box.write(bd)

        best_sched, best_fit, best_bd = run_ga(
            skills, shift_types, forb, contracts, nurses, req_cells, off_requests,
            last_shift, hist_work, hist_off,
            generations, pop_size, mutation_rate, int(elitism),
            W,
            progress_cb=progress_cb
        )

        st.success(f"✅ BEST Fitness = {int(best_fit)}")
        st.json(best_bd)

        df_long, df_pivot, df_cov = build_outputs(best_sched, nurses, req_cells)

        st.subheader("📋 Best Schedule (Long)")
        st.dataframe(df_long, use_container_width=True, height=340)

        st.subheader("🗓️ Best Schedule (Pivot)")
        st.dataframe(df_pivot, use_container_width=True, height=260)

        st.subheader("📊 Coverage Diagnostics")
        st.dataframe(df_cov, use_container_width=True, height=320)

        # downloads
        st.download_button(
            "⬇️ Download best_schedule_long_inrc2.csv",
            df_long.to_csv(index=False).encode("utf-8"),
            file_name="best_schedule_long_inrc2.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download best_schedule_pivot_inrc2.csv",
            df_pivot.to_csv().encode("utf-8"),
            file_name="best_schedule_pivot_inrc2.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download coverage_diag.csv",
            df_cov.to_csv(index=False).encode("utf-8"),
            file_name="coverage_diag.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇️ Download best_metrics.json",
            json.dumps(best_bd, indent=2).encode("utf-8"),
            file_name="best_metrics.json",
            mime="application/json",
        )
