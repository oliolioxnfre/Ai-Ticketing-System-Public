import tkinter as tk
from tkinter import messagebox, scrolledtext
import subprocess
import sys
import os
import threading
import queue
from pathlib import Path

# --- Constants ---
SCHEDULERS = ["greedy", "random", "greedy+sa"]
ORDERS = ["arrival", "edd"]
TIME_POLICIES = ["finish_sla", "start_sla"]

def get_base_path():
    return Path(__file__).parent.resolve()

def scan_csv_files(subfolder):
    base = get_base_path()
    folder_path = base / subfolder
    if not folder_path.exists():
        return []
    return [f.name for f in folder_path.glob("*.csv")]

def get_script_path():
    base = get_base_path()
    script_path = base / "main_sa.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find main_sa.py at {script_path}")
    return str(script_path), str(base)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Helpdesk Configurator")
        self.root.geometry("950x900")
        
        # State variables
        self.helper_files = []
        self.ticket_files = []
        self.msg_queue = queue.Queue()
        
        # Initialize UI Variables
        self.init_vars()
        
        # Build UI
        self.setup_ui()
        
        # Load Data
        self.refresh_files()
        
        # Start background polling
        self.check_queue()
        
        # Initial State Update
        self.update_state()

    def init_vars(self):
        # File Vars
        self.helper_var = tk.StringVar(value="")
        self.ticket_var = tk.StringVar(value="")
        
        # General Vars
        self.scheduler_var = tk.StringVar(value="greedy")
        self.order_var = tk.StringVar(value="arrival")
        self.time_policy_var = tk.StringVar(value="finish_sla")
        
        # Triage Vars
        self.use_nb_var = tk.BooleanVar(value=False)
        
        # References to entry widgets for enabling/disabling
        self.sa_widgets = []
        self.nb_widgets = []

    def setup_ui(self):
        # Main Scrollable Container (Optional but good for small screens)
        # For simplicity and robustness on Mac, we'll use a fixed frame for now
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 1. File Selection ---
        file_frame = tk.LabelFrame(main_frame, text="1. Input Files", padx=10, pady=10, font=("Arial", 12, "bold"))
        file_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(file_frame, text="Helper File:").grid(row=0, column=0, sticky=tk.W)
        self.helper_menu = tk.OptionMenu(file_frame, self.helper_var, "")
        self.helper_menu.config(width=30)
        self.helper_menu.grid(row=0, column=1, padx=5)
        
        tk.Button(file_frame, text="Refresh Files", command=self.refresh_files).grid(row=0, column=2, padx=10)

        tk.Label(file_frame, text="Ticket File:").grid(row=1, column=0, sticky=tk.W)
        self.ticket_menu = tk.OptionMenu(file_frame, self.ticket_var, "")
        self.ticket_menu.config(width=30)
        self.ticket_menu.grid(row=1, column=1, padx=5)

        # --- 2. General Settings ---
        gen_frame = tk.LabelFrame(main_frame, text="2. General Settings", padx=10, pady=10, font=("Arial", 12, "bold"))
        gen_frame.pack(fill=tk.X, pady=5)

        # Scheduler
        tk.Label(gen_frame, text="Scheduler:").grid(row=0, column=0, sticky=tk.W)
        for i, s in enumerate(SCHEDULERS):
            tk.Radiobutton(gen_frame, text=s, variable=self.scheduler_var, value=s, command=self.update_state).grid(row=0, column=1+i, sticky=tk.W)

        # Order
        tk.Label(gen_frame, text="Order:").grid(row=1, column=0, sticky=tk.W)
        for i, o in enumerate(ORDERS):
            tk.Radiobutton(gen_frame, text=o, variable=self.order_var, value=o).grid(row=1, column=1+i, sticky=tk.W)

        # Time Policy
        tk.Label(gen_frame, text="Time Policy:").grid(row=2, column=0, sticky=tk.W)
        for i, t in enumerate(TIME_POLICIES):
            tk.Radiobutton(gen_frame, text=t, variable=self.time_policy_var, value=t).grid(row=2, column=1+i, sticky=tk.W)

        # Basic Params
        tk.Label(gen_frame, text="Slot Mins:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.slot_entry = tk.Entry(gen_frame, width=10); self.slot_entry.insert(0, "15"); self.slot_entry.grid(row=3, column=1, sticky=tk.W)
        
        tk.Label(gen_frame, text="Global Seed:").grid(row=3, column=2, sticky=tk.W)
        self.seed_entry = tk.Entry(gen_frame, width=10); self.seed_entry.insert(0, "1337"); self.seed_entry.grid(row=3, column=3, sticky=tk.W)

        tk.Label(gen_frame, text="Out CSV:").grid(row=4, column=0, sticky=tk.W)
        self.out_csv_entry = tk.Entry(gen_frame, width=20); self.out_csv_entry.insert(0, "assignments.csv"); self.out_csv_entry.grid(row=4, column=1, sticky=tk.W)

        tk.Label(gen_frame, text="Out JSON:").grid(row=4, column=2, sticky=tk.W)
        self.out_json_entry = tk.Entry(gen_frame, width=20); self.out_json_entry.insert(0, "run_summary.json"); self.out_json_entry.grid(row=4, column=3, sticky=tk.W)


        # --- 3. Triage Settings ---
        triage_frame = tk.LabelFrame(main_frame, text="3. Triage (AI Classification)", padx=10, pady=10, font=("Arial", 12, "bold"))
        triage_frame.pack(fill=tk.X, pady=5)

        tk.Checkbutton(triage_frame, text="Enable Naive Bayes Triage", variable=self.use_nb_var, command=self.update_state).grid(row=0, column=0, columnspan=2, sticky=tk.W)

        self.nb_widgets = []
        def add_nb_param(label, default, r, c):
            lbl = tk.Label(triage_frame, text=label)
            lbl.grid(row=r, column=c, sticky=tk.E, padx=5)
            ent = tk.Entry(triage_frame, width=10)
            ent.insert(0, default)
            ent.grid(row=r, column=c+1, sticky=tk.W)
            self.nb_widgets.extend([lbl, ent])
            return ent

        self.nb_conf_entry = add_nb_param("Min Conf:", "0.60", 1, 0)
        self.nb_test_entry = add_nb_param("Test Size:", "0.20", 1, 2)
        self.nb_seed_entry = add_nb_param("NB Seed:", "42", 1, 4)


        # --- 4. Simulated Annealing ---
        sa_frame = tk.LabelFrame(main_frame, text="4. Simulated Annealing (Greedy+SA Only)", padx=10, pady=10, font=("Arial", 12, "bold"))
        sa_frame.pack(fill=tk.X, pady=5)

        self.sa_widgets = []
        def add_sa_param(label, default, r, c):
            lbl = tk.Label(sa_frame, text=label)
            lbl.grid(row=r, column=c, sticky=tk.E, padx=5)
            ent = tk.Entry(sa_frame, width=10)
            ent.insert(0, default)
            ent.grid(row=r, column=c+1, sticky=tk.W)
            self.sa_widgets.extend([lbl, ent])
            return ent

        self.sa_steps_entry = add_sa_param("Steps:", "5000", 0, 0)
        self.sa_alpha_entry = add_sa_param("Alpha:", "100.0", 0, 2)
        self.sa_beta_entry = add_sa_param("Beta:", "1.0", 0, 4)
        self.sa_t0_entry = add_sa_param("T0:", "1.0", 1, 0)
        self.sa_cool_entry = add_sa_param("Cooling:", "0.999", 1, 2)


        # --- 5. Execution ---
        self.run_btn = tk.Button(main_frame, text="▶ RUN SIMULATION", command=self.start_simulation, height=2, bg="#d0f0c0", font=("Arial", 14, "bold"))
        self.run_btn.pack(pady=10, fill=tk.X)

        console_frame = tk.LabelFrame(main_frame, text="Console Output", padx=5, pady=5)
        console_frame.pack(fill=tk.BOTH, expand=True)

        self.output_box = scrolledtext.ScrolledText(console_frame, height=10, state='disabled', font=("Courier", 10))
        self.output_box.pack(fill=tk.BOTH, expand=True)


    def update_state(self):
        # Enable/Disable SA Widgets
        state = "normal" if self.scheduler_var.get() == "greedy+sa" else "disabled"
        for w in self.sa_widgets:
            w.config(state=state)

        # Enable/Disable NB Widgets
        state = "normal" if self.use_nb_var.get() else "disabled"
        for w in self.nb_widgets:
            w.config(state=state)

    def refresh_files(self):
        self.helper_files = scan_csv_files("helpers")
        self.ticket_files = scan_csv_files("tickets")
        
        if not self.helper_files: self.helper_files = ["No files found"]
        if not self.ticket_files: self.ticket_files = ["No files found"]
        
        # Helper Menu
        menu = self.helper_menu["menu"]
        menu.delete(0, "end")
        for f in self.helper_files:
            menu.add_command(label=f, command=tk._setit(self.helper_var, f))
        self.helper_var.set(self.helper_files[0])

        # Ticket Menu
        menu = self.ticket_menu["menu"]
        menu.delete(0, "end")
        for f in self.ticket_files:
            menu.add_command(label=f, command=tk._setit(self.ticket_var, f))
        self.ticket_var.set(self.ticket_files[0])

    def log(self, message):
        self.output_box.config(state='normal')
        self.output_box.insert(tk.END, message)
        self.output_box.see(tk.END)
        self.output_box.config(state='disabled')

    def check_queue(self):
        while not self.msg_queue.empty():
            msg_type, content = self.msg_queue.get()
            if msg_type == "stdout":
                self.log(content)
            elif msg_type == "stderr":
                self.log(f"ERROR: {content}")
            elif msg_type == "done":
                self.run_btn.config(state='normal')
                self.log(f"\n--- Simulation Finished (Exit Code: {content}) ---\n")
                if content != 0:
                    messagebox.showerror("Failed", f"Process exited with code {content}")
                else:
                    messagebox.showinfo("Success", "Simulation completed successfully!")
            elif msg_type == "error":
                self.run_btn.config(state='normal')
                messagebox.showerror("Error", content)
        
        self.root.after(100, self.check_queue)

    def start_simulation(self):
        self.run_btn.config(state='disabled')
        self.output_box.config(state='normal')
        self.output_box.delete(1.0, tk.END)
        self.output_box.config(state='disabled')
        
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        try:
            script_path, working_dir = get_script_path()
            
            cmd = [sys.executable, "-u", script_path]
            cmd.extend(["--helpers", self.helper_var.get()])
            cmd.extend(["--tickets", self.ticket_var.get()])
            cmd.extend(["--scheduler", self.scheduler_var.get()])
            cmd.extend(["--order", self.order_var.get()])
            cmd.extend(["--time_policy", self.time_policy_var.get()])
            
            cmd.extend(["--out_csv", self.out_csv_entry.get()])
            cmd.extend(["--out_json", self.out_json_entry.get()])
            cmd.extend(["--slot_minutes", self.slot_entry.get()])
            cmd.extend(["--seed", self.seed_entry.get()])

            if self.use_nb_var.get():
                cmd.extend([
                    "--triage", "naive_bayes", 
                    "--nb_min_conf", self.nb_conf_entry.get(),
                    "--nb_test_size", self.nb_test_entry.get(),
                    "--nb_seed", self.nb_seed_entry.get()
                ])
            else:
                cmd.extend(["--triage", "keywords"])

            if self.scheduler_var.get() == "greedy+sa":
                cmd.extend([
                    "--sa_steps", self.sa_steps_entry.get(),
                    "--sa_alpha", self.sa_alpha_entry.get(),
                    "--sa_beta", self.sa_beta_entry.get(),
                    "--sa_T0", self.sa_t0_entry.get(),
                    "--sa_cooling", self.sa_cool_entry.get()
                ])

            self.msg_queue.put(("stdout", f"Executing: {' '.join(cmd)}\n\n"))
            print(f"Running command: {cmd}")

            process = subprocess.Popen(
                cmd,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                self.msg_queue.put(("stdout", line))
            
            for line in process.stderr:
                self.msg_queue.put(("stderr", line))
                
            process.wait()
            self.msg_queue.put(("done", process.returncode))

        except Exception as e:
            print(f"Error in run_process: {e}")
            self.msg_queue.put(("error", str(e)))

if __name__ == "__main__":
    print("Starting Config GUI...")
    root = tk.Tk()
    app = App(root)
    root.mainloop()