
import argparse, csv, json, math, random
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

# If you have errors running this
# re read the readme.md you might be missing some packages
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
SKLEARN_OK = True

#old try except remove this in the future if everything works 
#except Exception:
#    SKLEARN_OK = False

CATEGORIES = ["Network", "Account", "Printing", "Email", "Hardware", "Software"]

KEYWORDS = {
    "Network": ["wifi", "wireless", "vpn", "internet", "disconnect", "network"],
    "Account": ["password", "login", "locked", "2fa", "mfa", "reset"],
    "Printing": ["printer", "print", "jam", "toner", "duplex"],
    "Email": ["email", "outlook", "quota", "attachment"],
    "Hardware": ["laptop", "keyboard", "monitor", "projector", "fan", "hdmi"],
    "Software": ["install", "license", "matlab", "software", "update", "bug"],
}

DEFAULT_CATEGORY = "Software"

#for any additional/custom helpers add here for it to recognize it with a shortened version in the argument
KNOWN_ALIASES = {
    "helpers": {
        "easy": "helpers_easy.csv",
        "rare": "helpers_rare.csv",
    },
    "tickets": {
        "small": "tickets_small.csv",
        "medium": "tickets_medium.csv",
        "hard": "tickets_hard.csv",
        "small_noisy": "tickets_small_noisy.csv",
        "medium_noisy": "tickets_medium_noisy.csv",
        "hard_noisy": "tickets_hard_noisy.csv",
    },
}

"""
Resolves a CSV path from a short specification example being "--helpers helpers_easy.csv -> --helpers easy"
Accepts absolute/relative paths, bare names ('easy'), or short names with .csv ('easy.csv').
Searches default_dirs and uses KNOWN_ALIASES. Raises FileNotFoundError with helpful context.
"""
def resolve_csv(kind: str, spec: str, default_dirs: list[str]) -> Path:
    p = Path(spec)
    if p.exists():
        return p.resolve()

    key = spec.lower().strip()
    if key.endswith(".csv"):
        key = key[:-4]

    # look inside known aliases 
    alias = KNOWN_ALIASES.get(kind, {}).get(key)
    candidates = []
    if alias:
        candidates.append(alias)

    # adds more to canidates to look through based off the kind
    if kind == "helpers":
        candidates += [f"helpers_{key}.csv", f"{key}.csv"]
    elif kind == "tickets":
        candidates += [f"tickets_{key}.csv", f"{key}.csv"]

    # searches through provided directories
    for d in default_dirs:
        for c in candidates:
            candidate = Path(d) / c
            if candidate.exists():
                return candidate.resolve()

    # Fuzzy glob last 
    for d in default_dirs:
        dd = Path(d)
        if dd.exists():
            found = list(dd.glob(f"*{key}*.csv"))
            if found:
                return found[0].resolve()

    # helpful error that builds and prints inventory
    inventory = []
    for d in default_dirs:
        dd = Path(d)
        if dd.exists():
            inventory += [p.name for p in dd.glob("*.csv")]
    raise FileNotFoundError(
        f"Could not resolve {kind} spec '{spec}'. Searched {default_dirs}. "
        f"Try one of aliases {KNOWN_ALIASES.get(kind, {})} or existing files: {sorted(set(inventory))}"
    )


'''
Reads the arguments being parsed 
Example of how to use is:
python main.sa_py --argument1 detail --argument2 detail2
'''
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helpers", required=True, help="helpers.csv")
    ap.add_argument("--tickets", required=True, help="tickets.csv")
    ap.add_argument("--out_csv", default="assignments.csv")
    ap.add_argument("--out_json", default="run_summary.json")
    ap.add_argument("--slot_minutes", type=int, default=15)
    ap.add_argument("--time_policy", choices=["start_sla","finish_sla"], default="finish_sla")
    ap.add_argument("--scheduler", choices=["random","greedy","greedy+sa"], default="greedy")
    
    ap.add_argument("--triage", choices=["keywords","naive_bayes"], default="keywords")
    ap.add_argument("--nb_min_conf", type=float, default=0.60)
    ap.add_argument("--nb_test_size", type=float, default=0.20)
    ap.add_argument("--nb_seed", type=int, default=42) 
    ap.add_argument("--sa_steps", type=int, default=5000)
    ap.add_argument("--sa_alpha", type=float, default=100.0)  # SLA weight
    ap.add_argument("--sa_beta", type=float, default=1.0)     # variance weight
    ap.add_argument("--sa_T0", type=float, default=1.0)
    ap.add_argument("--sa_cooling", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--order", choices=["arrival","edd"], default="arrival", help="Initial ticket order for greedy seed: arrival time or earliest due date")
    return ap.parse_args()

# Bottom two are two fo the most important 
'''
Part 1 of the main input, helper csv file 
Pass the helper csv file as the argument to read it
'''
def read_helpers(path):
    skills = {}
    capacity = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            helper_id = int(row["helper_id"])
            capacity[helper_id] = int(row.get("capacity", 1))
            s = set()
            for cat in CATEGORIES:
                col = cat.lower()
                if col in row and row[col] and int(row[col]) == 1:
                    s.add(cat)
            skills[helper_id] = s
    return skills, capacity

'''
Part 2 of the main input, tickets csv file
Reads the tickets to store it in a 2d array like the csv file
'''
def read_tickets(path):
    tickets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["ticket_id"] = int(row["ticket_id"])
            row["customer_id"] = int(row["customer_id"])
            row["sla_hours"] = float(row["sla_hours"] or 0)
            row["proc_minutes"] = int(row.get("proc_minutes", 15))
            row["created_at_dt"] = datetime.fromisoformat(row["created_at"])
            tickets.append(row)
    return tickets

'''
Labels based off category, or returns a default to Software as the default category if none are found
If category is mispelled it will also go to default category which is important to note
'''
def label_category(text: str) -> str:
    t = (text or "").lower()
    for cat, words in KEYWORDS.items():
        for w in words:
            if w in t:
                return cat  
    return DEFAULT_CATEGORY

'''
Finds the first earliest available slot at or after the start date time that isn't full, 
(for note we use a range of 10,000 just as a failsafe)
'''
def earliest_slot_at_or_after(start_dt, occupied_count, slot_minutes, cap):
    dt = start_dt
    step = timedelta(minutes=slot_minutes)
    for _ in range(10000):
        if occupied_count.get(dt, 0) < cap:
            return dt
        dt += step
    return None

'''
Trains and evaluates a text classifier using the scikit package 

inputs:
 - labeled rows with text description and category
 - test size : fraction of data we want to use for testing (typical 80:20 split for now but you can change this later)
 - seed : integer random seed for reproducible results

returns:

'''
def nb_fit_eval(labeled_rows, test_size=0.2, seed=42):
    if len(labeled_rows) < 10:
        return None, {"accuracy": 0.0, "macro_f1": 0.0, "n_train": 0, "n_test": 0}
    texts = [r.get('text','') for r in labeled_rows]
    y = [r['true_category'] for r in labeled_rows]
    try:
        Xtr_text, Xte_text, ytr, yte = train_test_split(texts, y, test_size=test_size, stratify=y, random_state=seed)
    except Exception:
        Xtr_text, Xte_text, ytr, yte = train_test_split(texts, y, test_size=test_size, random_state=seed)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
    Xtr = vec.fit_transform(Xtr_text)
    Xte = vec.transform(Xte_text)
    nb = MultinomialNB()
    nb.fit(Xtr, ytr)
    yhat = nb.predict(Xte)
    try:
        acc = float(accuracy_score(yte, yhat))
        macro = float(f1_score(yte, yhat, average='macro'))
    except Exception:
        acc, macro = 0.0, 0.0
    return {"vec": vec, "nb": nb}, {"accuracy": acc, "macro_f1": macro, "n_train": len(ytr), "n_test": len(yte)}

def nb_predict_with_conf(model, text):
    X = model['vec'].transform([text or ''])
    proba = model['nb'].predict_proba(X)[0]
    j = int(proba.argmax())
    return model['nb'].classes_[j], float(proba[j])

def prepare_tickets(tickets, slot_minutes):
    for t in tickets:
        t["predicted_category"] = label_category(t["text"])
        t["arrival"] = t["created_at_dt"]
        p_slots = max(1, math.ceil(t["proc_minutes"] / slot_minutes))
        t["proc_slots"] = p_slots
        t["deadline"] = t["created_at_dt"] + timedelta(hours=t["sla_hours"])
    return sorted(tickets, key=lambda x: (x["created_at_dt"], x["ticket_id"]))

def _ok_record(t, cat, h, start, finish, meets, lateness):
    return {
        "ticket_id": t["ticket_id"],
        "predicted_category": cat,
        "helper_id": h,
        "start_time": start,
        "finish_time": finish,
        "meets_sla": "YES" if meets else "NO",
        "lateness_minutes": lateness,
        "note": ""
    }

def _miss_record(t, cat, reason):
    return {
        "ticket_id": t["ticket_id"],
        "predicted_category": cat,
        "helper_id": "",
        "start_time": "",
        "finish_time": "",
        "meets_sla": "NO",
        "lateness_minutes": "",
        "note": reason
    }

def build_initial_assignments(tickets, skills, capacity, slot_minutes, time_policy, policy="greedy"):
    helper_slots = defaultdict(lambda: defaultdict(int))
    helper_load = Counter()
    assignments = []
    rng = random.Random(0)

    for t in tickets:
        cat = t["predicted_category"]
        arrival = t["arrival"]
        deadline = t["deadline"]
        p_slots = t["proc_slots"]

        qualified = [h for h in skills if cat in skills[h]]
        if not qualified:
            assignments.append(_miss_record(t, cat, "No qualified helper"))
            continue

        if policy == "greedy":
            qualified.sort(key=lambda h: helper_load[h])
        elif policy == "random":
            rng.shuffle(qualified)

        assigned = None
        for h in qualified:
            slots = helper_slots[h]
            cap = capacity.get(h, 1)
            start = earliest_slot_at_or_after(arrival, slots, slot_minutes, cap)
            if start is None: 
                continue
            step = timedelta(minutes=slot_minutes)
            times = [start + i*step for i in range(p_slots)]
            if any(slots.get(ts, 0) >= cap for ts in times):
                continue

            finish = times[-1] + step
            meets = (finish <= deadline) if (time_policy=="finish_sla") else (start <= deadline)
            for ts in times:
                slots[ts] = slots.get(ts, 0) + 1
            helper_load[h] += 1
            lateness = 0
            if not meets:
                dt = (finish if time_policy=="finish_sla" else start) - deadline
                lateness = int(max(0, dt.total_seconds()//60))
            assigned = _ok_record(t, cat, h, start, finish, meets, lateness)
            break

        if assigned is None:
            assignments.append(_miss_record(t, cat, "No slot available"))
        else:
            assignments.append(assigned)

    return assignments

def compute_metrics(assignments):
    load = Counter()
    breaches = 0
    total = len(assignments)
    tardiness = 0
    for a in assignments:
        if a["helper_id"] != "":
            load[a["helper_id"]] += 1
        if a["meets_sla"] == "NO":
            breaches += 1
            try:
                tardiness += int(a.get("lateness_minutes") or 0)
            except:
                pass
    loads = list(load.values()) if load else [0]
    mean = sum(loads)/len(loads) if loads else 0.0
    variance = sum((x-mean)**2 for x in loads)/len(loads) if loads else 0.0
    return {
        "total_tickets": total,
        "sla_breach_count": breaches,
        "sla_breach_rate": (breaches/total) if total else 0.0,
        "total_tardiness_minutes": tardiness,
        "workload_variance": variance,
        "loads_by_helper": dict(load)
    }

def cost(assignments, alpha=100.0, beta=1.0):
    m = compute_metrics(assignments)
    return alpha*m["sla_breach_rate"] + beta*m["workload_variance"], m

def rebuild_helper_slots(assignments, slot_minutes):
    slots_by_helper = defaultdict(lambda: defaultdict(int))
    for a in assignments:
        h = a["helper_id"]
        if h == "": 
            continue
        start = a["start_time"]; finish = a["finish_time"]
        step = timedelta(minutes=slot_minutes)
        t = start
        while t < finish:
            slots_by_helper[h][t] = slots_by_helper[h].get(t, 0) + 1
            t += step
    return slots_by_helper

def try_place(ticket, helper_id, slots_by_helper, slot_minutes, capacity, time_policy, deadline, p_slots, arrival):
    slots = slots_by_helper[helper_id]
    cap = capacity.get(helper_id, 1)
    start = earliest_slot_at_or_after(arrival, slots, slot_minutes, cap)
    if start is None: 
        return None
    step = timedelta(minutes=slot_minutes)
    times = [start + i*step for i in range(p_slots)]
    if any(slots.get(ts, 0) >= cap for ts in times):
        return None
    finish = times[-1] + step
    meets = (finish <= deadline) if (time_policy=="finish_sla") else (start <= deadline)
    return start, finish, meets

def _free_occupancy(a, slots_by_helper, slot_minutes):
    h = a["helper_id"]
    if not h or not a.get("start_time") or not a.get("finish_time"):
        return
    step = timedelta(minutes=slot_minutes)
    tcur = a["start_time"]
    while tcur < a["finish_time"]:
        slots = slots_by_helper[h]
        slots[tcur] = slots.get(tcur, 0) - 1
        if slots[tcur] <= 0:
            del slots[tcur]
        tcur += step

def _apply_occupancy(h, start, finish, slots_by_helper, slot_minutes):
    step = timedelta(minutes=slot_minutes)
    tcur = start
    while tcur < finish:
        slots_by_helper[h][tcur] = slots_by_helper[h].get(tcur, 0) + 1
        tcur += step

def simulated_annealing(assignments, tickets, skills, capacity, slot_minutes, time_policy,
                        steps=5000, alpha=100.0, beta=1.0, T0=1.0, cooling=0.999, seed=1337):
    """
    Two-move SA:
      - Move A (50%): reassign one ticket to a different qualified helper
      - Move B (50%): swap two tickets of the same category across two helpers
    Objective = alpha * SLA_breach_rate + beta * workload_variance
    """
    rng = random.Random(seed)

    # Quick access maps
    id2ticket = {t["ticket_id"]: t for t in tickets}

    # Working copies
    best = [a.copy() for a in assignments]
    best_cost, best_metrics = cost(best, alpha, beta)
    current = [a.copy() for a in assignments]
    current_cost = best_cost

    # Build initial occupancy
    slots_by_helper = rebuild_helper_slots(current, slot_minutes)

    accept_improve = 0

    for step in range(steps):
        move_type = rng.random()   # <0.5 => reassign, >=0.5 => swap
        accepted = False

        if move_type < 0.5:
            # -------- Move A: reassign a single ticket --------
            # pick any currently scheduled/attempted assignment (even misses; they likely have empty helper_id)
            a = rng.choice(current)
            tid = a["ticket_id"]; t = id2ticket[tid]
            cat = t["predicted_category"]
            arrival = t["arrival"]; deadline = t["deadline"]; p_slots = t["proc_slots"]

            qualified = [h for h in skills if cat in skills[h]]
            if not qualified:
                continue

            old_h = a.get("helper_id", "")
            candidates = [h for h in qualified if h != old_h]
            if not candidates:
                continue

            # free old occupancy (if any)
            _free_occupancy(a, slots_by_helper, slot_minutes)

            new_h = rng.choice(candidates)
            placed = try_place(t, new_h, slots_by_helper, slot_minutes, capacity,
                               time_policy, deadline, p_slots, arrival)
            if placed is None:
                # restore old occupancy and skip
                _apply_occupancy(old_h, a.get("start_time"), a.get("finish_time"),
                                 slots_by_helper, slot_minutes)
                continue

            new_start, new_finish, meets = placed
            old_record = a.copy()

            # tentatively commit
            a.update({
                "helper_id": new_h,
                "start_time": new_start,
                "finish_time": new_finish,
                "meets_sla": "YES" if meets else "NO",
                "lateness_minutes": 0 if meets else int(max(
                    0,
                    ((new_finish if time_policy == "finish_sla" else new_start) - deadline
                    ).total_seconds() // 60
                ))
            })

            new_cost, _ = cost(current, alpha, beta)
            delta = new_cost - current_cost
            T = max(T0 * (cooling ** step), 1e-9)

            if (delta <= 0) or (rng.random() < math.exp(-delta / T)):
                # accept: apply new occupancy and update best if needed
                _apply_occupancy(new_h, new_start, new_finish, slots_by_helper, slot_minutes)
                if delta < 0:
                    accept_improve += 1
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost, best_metrics = new_cost, compute_metrics(current)
                    best = [x.copy() for x in current]
                accepted = True
            else:
                # reject: revert record and restore old occupancy
                a.update(old_record)
                _apply_occupancy(old_record.get("helper_id"), old_record.get("start_time"),
                                 old_record.get("finish_time"), slots_by_helper, slot_minutes)

        else:
            # -------- Move B: swap two tickets of the SAME category across different helpers --------
            # choose a first assigned ticket
            assigned = [x for x in current if x.get("helper_id")]
            if not assigned:
                continue
            a1 = rng.choice(assigned)
            t1 = id2ticket[a1["ticket_id"]]
            cat = t1["predicted_category"]

            # choose a second assigned ticket of the same category, different helper
            pool2 = [x for x in assigned if x["helper_id"] != a1["helper_id"]
                     and id2ticket[x["ticket_id"]]["predicted_category"] == cat]
            if not pool2:
                continue
            a2 = rng.choice(pool2)
            t2 = id2ticket[a2["ticket_id"]]

            # free both occupancies
            _free_occupancy(a1, slots_by_helper, slot_minutes)
            _free_occupancy(a2, slots_by_helper, slot_minutes)

            # attempt cross placement
            placed1 = try_place(t1, a2["helper_id"], slots_by_helper, slot_minutes, capacity,
                                time_policy, t1["deadline"], t1["proc_slots"], t1["arrival"])
            placed2 = try_place(t2, a1["helper_id"], slots_by_helper, slot_minutes, capacity,
                                time_policy, t2["deadline"], t2["proc_slots"], t2["arrival"])

            if placed1 and placed2:
                old1, old2 = a1.copy(), a2.copy()
                n1s, n1f, m1 = placed1
                n2s, n2f, m2 = placed2

                # tentatively commit swap
                a1.update({
                    "helper_id": a2["helper_id"],
                    "start_time": n1s, "finish_time": n1f,
                    "meets_sla": "YES" if m1 else "NO",
                    "lateness_minutes": 0 if m1 else int(max(
                        0,
                        ((n1f if time_policy == "finish_sla" else n1s) - t1["deadline"]
                        ).total_seconds() // 60
                    ))
                })
                a2.update({
                    "helper_id": old1["helper_id"],
                    "start_time": n2s, "finish_time": n2f,
                    "meets_sla": "YES" if m2 else "NO",
                    "lateness_minutes": 0 if m2 else int(max(
                        0,
                        ((n2f if time_policy == "finish_sla" else n2s) - t2["deadline"]
                        ).total_seconds() // 60
                    ))
                })

                new_cost, _ = cost(current, alpha, beta)
                delta = new_cost - current_cost
                T = max(T0 * (cooling ** step), 1e-9)

                if (delta <= 0) or (rng.random() < math.exp(-delta / T)):
                    # accept: apply both occupancies and update best
                    _apply_occupancy(a1["helper_id"], a1["start_time"], a1["finish_time"],
                                     slots_by_helper, slot_minutes)
                    _apply_occupancy(a2["helper_id"], a2["start_time"], a2["finish_time"],
                                     slots_by_helper, slot_minutes)
                    if delta < 0:
                        accept_improve += 1
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost, best_metrics = new_cost, compute_metrics(current)
                        best = [x.copy() for x in current]
                    accepted = True
                else:
                    # reject: revert both and restore occupancies
                    a1.update(old1); a2.update(old2)
                    _apply_occupancy(old1["helper_id"], old1["start_time"], old1["finish_time"],
                                     slots_by_helper, slot_minutes)
                    _apply_occupancy(old2["helper_id"], old2["start_time"], old2["finish_time"],
                                     slots_by_helper, slot_minutes)
            else:
                # Couldn’t place both; restore both occupancies
                _apply_occupancy(a1["helper_id"], a1.get("start_time"), a1.get("finish_time"),
                                 slots_by_helper, slot_minutes)
                _apply_occupancy(a2["helper_id"], a2.get("start_time"), a2.get("finish_time"),
                                 slots_by_helper, slot_minutes)

        # Light reheating if stuck for long stretches
        if (step % 5000 == 0) and (step > 0) and (not accepted):
            T0 *= 1.15  # small nudge to escape local minima

    return best, best_metrics, accept_improve

def write_assignments(path, assignments, tickets):
    ticket_map = {t["ticket_id"]: t for t in tickets}
    cols = ["ticket_id","customer_id","text","true_category","predicted_category","priority","created_at","sla_hours",
            "helper_id","start_time","finish_time","meets_sla","lateness_minutes","note"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for a in assignments:
            t = ticket_map.get(a["ticket_id"], {})
            row = {
                "ticket_id": a["ticket_id"],
                "customer_id": t.get("customer_id",""),
                "text": t.get("text",""),
                "true_category": t.get("true_category",""),
                "predicted_category": a.get("predicted_category",""),
                "priority": t.get("priority",""),
                "created_at": t.get("created_at",""),
                "sla_hours": t.get("sla_hours",""),
                "helper_id": a.get("helper_id",""),
                "start_time": a.get("start_time","").isoformat(timespec="minutes") if a.get("start_time") else "",
                "finish_time": a.get("finish_time","").isoformat(timespec="minutes") if a.get("finish_time") else "",
                "meets_sla": a.get("meets_sla",""),
                "lateness_minutes": a.get("lateness_minutes",""),
                "note": a.get("note",""),
            }
            w.writerow(row)

'''
Writes the summary to the json file based on everything passed, including more specific information for things like metrics that we care about 
'''
def write_summary_json(path, args, helpers_path, tickets_path, metrics, algo_name, sa_info=None, triage_info=None):
    summary = {
        "run_id": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00.00","Z") + f"_{algo_name}",
        "algorithm": {
            "scheduler": algo_name,
            "triage": triage_info or {"method": "keywords"}
        },
        "config": {
            "slot_minutes": args.slot_minutes,
            "time_policy": args.time_policy,
            "random_seed": args.seed
        },
        "datasets": {
            "tickets_csv": str(tickets_path),
            "helpers_csv": str(helpers_path),
            "n_tickets": metrics["total_tickets"]
        },
        "metrics_overall": {
            "sla_breach_count": metrics["sla_breach_count"],
            "sla_breach_rate": metrics["sla_breach_rate"],
            "total_tardiness_minutes": metrics["total_tardiness_minutes"],
            "workload_variance": metrics["workload_variance"]
        },
        "metrics_by_helper": [{"helper_id": int(h), "tickets": int(c)} for h, c in metrics["loads_by_helper"].items()],
        "notes": "Run complete."
    }
    if sa_info:
        summary["algorithm"]["annealing"] = sa_info
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

def main():
    args = parse_args()
    random.seed(args.seed)

    helpers_path = resolve_csv("helpers", args.helpers, [".", "helpers"])
    tickets_path = resolve_csv("tickets", args.tickets, [".", "tickets"])

    # loads from the first two input files
    skills, capacity = read_helpers(helpers_path)
    tickets = read_tickets(tickets_path)
    # TRIAGE: keywords or naive_bayes 
    # route beforehand
    triage_report = {"method": "keywords"}
    if args.triage == "naive_bayes":
        if SKLEARN_OK: #always TRUE, but we can change this to first download dependencies
            labeled = [t for t in tickets if t.get('true_category')]
            model, nbm = nb_fit_eval(labeled, test_size=args.nb_test_size, seed=args.nb_seed)
            if model:
                low = 0
                for t in tickets:
                    pred, conf = nb_predict_with_conf(model, t.get('text',''))
                    if conf >= args.nb_min_conf:
                        t['predicted_category'] = pred
                        t['triage_source'] = 'nb'
                        t['nb_confidence'] = conf
                    else:
                        t['predicted_category'] = label_category(t.get('text',''))
                        t['triage_source'] = 'keywords_fallback'
                        t['nb_confidence'] = conf
                        low += 1
                triage_report = {"method": "naive_bayes", "nb_min_conf": args.nb_min_conf, **nbm, "low_conf_pct": (low/len(tickets)) if tickets else 0.0}
            else:
                for t in tickets:
                    t['predicted_category'] = label_category(t.get('text',''))
                    t['triage_source'] = 'keywords'
                    t['nb_confidence'] = None
                triage_report = {"method": "keywords", "note": "insufficient labeled data or sklearn missing"}
        else:
            for t in tickets:
                t['predicted_category'] = label_category(t.get('text',''))
                t['triage_source'] = 'keywords'
                t['nb_confidence'] = None

            triage_report = {"method": "keywords", "note": "sklearn not available"}
    elif args.triage == "keywords":
        for t in tickets:
            t['predicted_category'] = label_category(t.get('text',''))
            t['triage_source'] = 'keywords'
            t['nb_confidence'] = None
        
    tickets = prepare_tickets(tickets, args.slot_minutes)

    # apply ordering before building initial assignments
    if args.order == "edd":
        tickets.sort(key=lambda t: (t["deadline"], t["created_at_dt"], t["ticket_id"]))
    else:
        tickets.sort(key=lambda t: (t["created_at_dt"], t["ticket_id"]))

    # initial schedule rule
    if args.scheduler == "random":
        init = build_initial_assignments(tickets, skills, capacity, args.slot_minutes, args.time_policy, policy="random")
        algo_name = "random"
        metrics = compute_metrics(init)
        out_assignments = init
        sa_info = None

    elif args.scheduler == "greedy":
        init = build_initial_assignments(tickets, skills, capacity, args.slot_minutes, args.time_policy, policy="greedy")
        algo_name = "greedy"
        metrics = compute_metrics(init)
        out_assignments = init
        sa_info = None

    else:  # greedy+sa
        init = build_initial_assignments(tickets, skills, capacity, args.slot_minutes, args.time_policy, policy="greedy")
        best, best_metrics, accept_improve = simulated_annealing(
            init, tickets, skills, capacity,
            args.slot_minutes, args.time_policy,
            steps=args.sa_steps, alpha=args.sa_alpha, beta=args.sa_beta,
            T0=args.sa_T0, cooling=args.sa_cooling, seed=args.seed
        )
        algo_name = "greedy+sa"
        metrics = best_metrics
        out_assignments = best
        sa_info = {
            "alpha_sla": args.sa_alpha,
            "beta_variance": args.sa_beta,
            "steps": args.sa_steps,
            "T0": args.sa_T0,
            "cooling": args.sa_cooling,
            "accepted_improving_moves": accept_improve
        }

    # write outputs
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_assignments(out_csv, out_assignments, tickets)
    write_summary_json(out_json, args, helpers_path, tickets_path, metrics, algo_name, sa_info=sa_info, triage_info=triage_report)

    print(f"{algo_name}: wrote {out_csv} and {out_json}.")
    print(f"SLA breach rate: {metrics['sla_breach_rate']:.2%}, workload variance: {metrics['workload_variance']:.2f}")
    if triage_report.get('method') == 'naive_bayes':
        print(f"NB accuracy={triage_report.get('accuracy',0):.3f}, macro-F1={triage_report.get('macro_f1',0):.3f}, low_conf_pct={triage_report.get('low_conf_pct',0):.2%}")
    elif triage_report.get('note'):
        print(f"Triage note: {triage_report['note']}")

if __name__ == "__main__":
    main()
