import json
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DEFAULT_CONFIG_FILE = "rail_resistance_config.json"
DEFAULT_DLL_PATH = "Calculations.dll"


# =========================
# DLL bridge
# =========================

class MechanicsBridge:
    def __init__(self, dll_path: str):
        self.dll_path = os.path.abspath(dll_path)
        self._loaded = False
        self._mechanics = None
        self.load()

    def load(self):
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"DLL not found: {self.dll_path}")

        dll_dir = os.path.dirname(self.dll_path)
        if dll_dir and dll_dir not in sys.path:
            sys.path.append(dll_dir)

        try:
            from pythonnet import load
            load("coreclr")
            import clr
        except Exception as e:
            raise RuntimeError(
                "pythonnet is installed, but CoreCLR could not be loaded. "
                "Check 'dotnet --info' and verify that .NET is installed in WSL."
            ) from e

        try:
            clr.AddReference(self.dll_path)
            from Calculations import Mechanics
            self._mechanics = Mechanics
            self._loaded = True
        except Exception as e:
            raise RuntimeError(
                f"Could not load Calculations.dll from {self.dll_path}. "
                f"A dependent DLL may be missing, or the assembly may not be compatible with CoreCLR/Linux."
            ) from e

    @property
    def mechanics(self):
        if not self._loaded or self._mechanics is None:
            raise RuntimeError("Mechanics DLL is not loaded.")
        return self._mechanics

    def straight_resistance_kn(self, v_kph, a, b, c, tunnel_factor, starting_resistance):
        # Returns kN
        return float(
            self.mechanics.StraightResistance(
                float(v_kph),
                float(a),
                float(b),
                float(c),
                float(tunnel_factor),
                float(starting_resistance)
            )
        )

    def slope_resistance_kn(self, slope_per_thousand, mass_kg):
        # DLL returns N -> convert to kN
        return float(
            self.mechanics.SlopeResistance(
                float(slope_per_thousand),
                float(mass_kg)
            )
        ) / 1000.0

    def total_resistance_kn(self, straight_resistance_kn, slope_resistance_kn):
        # Returns kN
        return float(straight_resistance_kn) + float(slope_resistance_kn)

    def adherence_force_kn(self, adherence_coefficient, speed_mps, adherent_mass_kg):
        # DLL returns N -> convert to kN
        return float(
            self.mechanics.GetAdherenceForce(
                float(adherence_coefficient),
                float(speed_mps),
                float(adherent_mass_kg)
            )
        ) / 1000.0


# =========================
# Data models
# =========================

@dataclass
class ResistanceConfig:
    vehicle_name: str
    mass_t: float
    adherent_mass_t: float
    adherence_coefficient: float
    straight_resistance: dict
    traction_effort_curve: List[dict]
    calculation: dict


def default_config() -> ResistanceConfig:
    return ResistanceConfig(
        vehicle_name="Example Rollingstock",
        mass_t=400.0,
        adherent_mass_t=400.0,
        adherence_coefficient=0.33,
        straight_resistance={
            "a": 3.0,
            "b": 0.03,
            "c": 0.0009,
            "tunnelFactor": 1.0,
            "startingResistance": 4.0
        },
        traction_effort_curve=[
            {"speed": 0.0, "effort_kN": 300.0},
            {"speed": 10.0, "effort_kN": 300.0},
            {"speed": 30.0, "effort_kN": 280.0},
            {"speed": 60.0, "effort_kN": 220.0},
            {"speed": 90.0, "effort_kN": 160.0},
            {"speed": 120.0, "effort_kN": 110.0},
            {"speed": 160.0, "effort_kN": 70.0},
            {"speed": 200.0, "effort_kN": 40.0}
        ],
        calculation={
            "slope_per_thousand": 0.0,
            "max_speed_kph": 220.0,
            "speed_step_kph": 0.5,
            "y_mode": "force"
        }
    )


# =========================
# Unit helpers
# =========================

def kph_to_mps(v_kph: float) -> float:
    return v_kph / 3.6


def tonnes_to_kg(value_t):
    return float(value_t) * 1000.0


def kn_to_n(value_kn):
    return np.asarray(value_kn, dtype=float) * 1000.0


def force_kn_to_acceleration(force_kn: np.ndarray, mass_t: float) -> np.ndarray:
    mass_kg = tonnes_to_kg(mass_t)
    return kn_to_n(force_kn) / mass_kg


# =========================
# Calculation functions
# =========================

def traction_effort_from_curve_kn(v_kph: np.ndarray, curve_points: List[Tuple[float, float]]) -> np.ndarray:
    """
    curve_points: [(speed_kph, effort_kN), ...]
    returns kN
    """
    curve_points = sorted(curve_points, key=lambda x: x[0])
    x = np.array([p[0] for p in curve_points], dtype=float)
    y_kn = np.array([p[1] for p in curve_points], dtype=float)
    return np.interp(v_kph, x, y_kn, left=y_kn[0], right=y_kn[-1])


def calculate_curves(cfg: ResistanceConfig, mechanics: MechanicsBridge):
    sr = cfg.straight_resistance
    calc = cfg.calculation

    max_speed = float(calc["max_speed_kph"])
    speed_step = float(calc["speed_step_kph"])
    slope = float(calc["slope_per_thousand"])
    mass_t = float(cfg.mass_t)
    adherent_mass_t = float(cfg.adherent_mass_t)
    adherence_coefficient = float(cfg.adherence_coefficient)

    mass_kg = tonnes_to_kg(mass_t)
    adherent_mass_kg = tonnes_to_kg(adherent_mass_t)

    if speed_step <= 0:
        raise ValueError("speed_step_kph must be greater than 0.")
    if max_speed <= 0:
        raise ValueError("max_speed_kph must be greater than 0.")
    if mass_t <= 0:
        raise ValueError("mass_t must be greater than 0.")
    if adherent_mass_t <= 0:
        raise ValueError("adherent_mass_t must be greater than 0.")
    if adherence_coefficient < 0:
        raise ValueError("adherence_coefficient must be greater than or equal to 0.")

    speeds_kph = np.arange(0.0, max_speed + speed_step, speed_step)
    speeds_mps = np.array([kph_to_mps(v) for v in speeds_kph], dtype=float)

    straight_kn = np.array([
        mechanics.straight_resistance_kn(
            v_kph=v,
            a=float(sr["a"]),
            b=float(sr["b"]),
            c=float(sr["c"]),
            tunnel_factor=float(sr["tunnelFactor"]),
            starting_resistance=float(sr["startingResistance"])
        )
        for v in speeds_kph
    ], dtype=float)

    slope_kn_scalar = mechanics.slope_resistance_kn(
        slope_per_thousand=slope,
        mass_kg=mass_kg
    )
    slope_kn = np.full_like(speeds_kph, slope_kn_scalar, dtype=float)

    total_res_kn = np.array([
        mechanics.total_resistance_kn(straight_kn[i], slope_kn[i])
        for i in range(len(speeds_kph))
    ], dtype=float)

    te_curve_points = [
        (float(p["speed"]), float(p["effort_kN"]))
        for p in cfg.traction_effort_curve
    ]
    raw_traction_kn = traction_effort_from_curve_kn(speeds_kph, te_curve_points)

    if adherence_coefficient > 0:
        adherence_limit_kn = np.array([
            mechanics.adherence_force_kn(
                adherence_coefficient=adherence_coefficient,
                speed_mps=float(v_mps),
                adherent_mass_kg=adherent_mass_kg
            )
            for v_mps in speeds_mps
        ], dtype=float)

        effective_traction_kn = np.minimum(raw_traction_kn, adherence_limit_kn)
    else:
        adherence_limit_kn = np.full_like(raw_traction_kn, np.nan, dtype=float)
        effective_traction_kn = raw_traction_kn.copy()

    return {
        "speeds_kph": speeds_kph,
        "speeds_mps": speeds_mps,
        "straight_kn": straight_kn,
        "slope_kn": slope_kn,
        "total_res_kn": total_res_kn,
        "raw_traction_kn": raw_traction_kn,
        "adherence_limit_kn": adherence_limit_kn,
        "effective_traction_kn": effective_traction_kn
    }


def find_intersections(x: np.ndarray, y1: np.ndarray, y2: np.ndarray) -> List[Tuple[float, float]]:
    diff = y1 - y2
    intersections = []

    for i in range(len(x) - 1):
        d1 = diff[i]
        d2 = diff[i + 1]

        if d1 == 0:
            intersections.append((x[i], y1[i]))
            continue

        if d1 * d2 < 0:
            x1, x2 = x[i], x[i + 1]
            y1_1, y1_2 = y1[i], y1[i + 1]
            y2_1, y2_2 = y2[i], y2[i + 1]

            ratio = abs(d1) / (abs(d1) + abs(d2))
            xi = x1 + (x2 - x1) * ratio

            yi_curve1 = y1_1 + (y1_2 - y1_1) * ((xi - x1) / (x2 - x1))
            yi_curve2 = y2_1 + (y2_2 - y2_1) * ((xi - x1) / (x2 - x1))
            yi = 0.5 * (yi_curve1 + yi_curve2)

            intersections.append((xi, yi))

    cleaned = []
    for pt in intersections:
        if not cleaned or abs(pt[0] - cleaned[-1][0]) > 1e-9:
            cleaned.append(pt)

    return cleaned


# =========================
# JSON helpers
# =========================

def config_to_dict(cfg: ResistanceConfig) -> dict:
    return asdict(cfg)


def load_config_from_file(path: str) -> ResistanceConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ResistanceConfig(
        vehicle_name=data["vehicle_name"],
        mass_t=float(data["mass_t"]),
        adherent_mass_t=float(data["adherent_mass_t"]),
        adherence_coefficient=float(data["adherence_coefficient"]),
        straight_resistance=data["straight_resistance"],
        traction_effort_curve=data["traction_effort_curve"],
        calculation=data["calculation"]
    )


def save_config_to_file(cfg: ResistanceConfig, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=4, ensure_ascii=False)


def ensure_default_config(path: str):
    if not os.path.exists(path):
        save_config_to_file(default_config(), path)


# =========================
# Scrollable frame helper
# =========================

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.window_id, width=e.width)
        )

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


# =========================
# Tkinter App
# =========================

class RailResistanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rail Resistance vs Traction Effort (DLL + Adherence)")
        self.root.geometry("1600x930")

        self.config_path = DEFAULT_CONFIG_FILE
        self.dll_path = DEFAULT_DLL_PATH

        ensure_default_config(self.config_path)
        self.cfg = load_config_from_file(self.config_path)
        self.mechanics: Optional[MechanicsBridge] = None

        self.current_speeds = None
        self.current_straight = None
        self.current_slope = None
        self.current_total = None
        self.current_raw_te = None
        self.current_effective_te = None
        self.current_y_label = "Force [kN]"

        self._build_ui()
        self._load_config_into_widgets()
        self.dll_path_var.set(self.dll_path)

        self.try_load_dll()
        self.plot_curves()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left_container = ttk.Frame(main)
        left_container.pack(side="left", fill="y", padx=(0, 10))

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        self.left_scrollable = ScrollableFrame(left_container)
        self.left_scrollable.pack(fill="both", expand=True)

        left = self.left_scrollable.inner

        # DLL frame
        dll_frame = ttk.LabelFrame(left, text="DLL", padding=10)
        dll_frame.pack(fill="x", pady=(0, 10))

        self.dll_path_var = tk.StringVar()
        ttk.Entry(dll_frame, textvariable=self.dll_path_var, width=52).pack(fill="x", pady=(0, 5))

        dll_btns = ttk.Frame(dll_frame)
        dll_btns.pack(fill="x")
        ttk.Button(dll_btns, text="Browse DLL", command=self.browse_dll).pack(side="left", padx=(0, 5))
        ttk.Button(dll_btns, text="Load DLL", command=self.try_load_dll).pack(side="left", padx=(0, 5))

        self.dll_status_var = tk.StringVar(value="DLL not loaded")
        ttk.Label(dll_frame, textvariable=self.dll_status_var, justify="left").pack(anchor="w", pady=(5, 0))

        # Config file controls
        file_frame = ttk.LabelFrame(left, text="Config file", padding=10)
        file_frame.pack(fill="x", pady=(0, 10))

        self.path_var = tk.StringVar(value=self.config_path)
        ttk.Entry(file_frame, textvariable=self.path_var, width=52).pack(fill="x", pady=(0, 5))

        btns = ttk.Frame(file_frame)
        btns.pack(fill="x")
        ttk.Button(btns, text="Load JSON", command=self.load_json).pack(side="left", padx=(0, 5))
        ttk.Button(btns, text="Save JSON", command=self.save_json).pack(side="left", padx=(0, 5))
        ttk.Button(btns, text="Save As...", command=self.save_json_as).pack(side="left")

        # Vehicle data
        vehicle_frame = ttk.LabelFrame(left, text="Vehicle / Adherence / Resistance data", padding=10)
        vehicle_frame.pack(fill="x", pady=(0, 10))

        self.vehicle_name_var = tk.StringVar()
        self.mass_var = tk.StringVar()
        self.adherent_mass_var = tk.StringVar()
        self.adherence_coef_var = tk.StringVar()

        self.a_var = tk.StringVar()
        self.b_var = tk.StringVar()
        self.c_var = tk.StringVar()
        self.tunnel_var = tk.StringVar()
        self.starting_var = tk.StringVar()

        self._add_labeled_entry(vehicle_frame, "Vehicle name", self.vehicle_name_var)
        self._add_labeled_entry(vehicle_frame, "Mass [t]", self.mass_var)
        self._add_labeled_entry(vehicle_frame, "Adherent mass [t]", self.adherent_mass_var)
        self._add_labeled_entry(vehicle_frame, "Adherence coefficient", self.adherence_coef_var)
        self._add_labeled_entry(vehicle_frame, "A", self.a_var)
        self._add_labeled_entry(vehicle_frame, "B", self.b_var)
        self._add_labeled_entry(vehicle_frame, "C", self.c_var)
        self._add_labeled_entry(vehicle_frame, "Tunnel factor", self.tunnel_var)
        self._add_labeled_entry(vehicle_frame, "Starting resistance", self.starting_var)

        # Calculation params
        calc_frame = ttk.LabelFrame(left, text="Calculation", padding=10)
        calc_frame.pack(fill="x", pady=(0, 10))

        self.slope_var = tk.StringVar()
        self.max_speed_var = tk.StringVar()
        self.speed_step_var = tk.StringVar()
        self.y_mode_var = tk.StringVar()

        self._add_labeled_entry(calc_frame, "Slope [‰]", self.slope_var)
        self._add_labeled_entry(calc_frame, "Max speed [km/h]", self.max_speed_var)
        self._add_labeled_entry(calc_frame, "Speed step [km/h]", self.speed_step_var)

        row = ttk.Frame(calc_frame)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text="Y mode", width=22).pack(side="left")
        y_combo = ttk.Combobox(
            row,
            textvariable=self.y_mode_var,
            state="readonly",
            values=["force", "acceleration"]
        )
        y_combo.pack(side="left", fill="x", expand=True)

        # TE curve editor
        te_frame = ttk.LabelFrame(left, text="Traction effort curve", padding=10)
        te_frame.pack(fill="both", expand=False, pady=(0, 10))

        ttk.Label(te_frame, text="Format: speed_kph,effort_kN (one point per line)").pack(anchor="w")
        self.te_text = tk.Text(te_frame, height=12, width=42)
        self.te_text.pack(fill="both", expand=True, pady=(5, 0))

        # Action buttons
        action_frame = ttk.Frame(left)
        action_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(action_frame, text="Plot / Recalculate", command=self.plot_curves).pack(side="left", padx=(0, 5))
        ttk.Button(action_frame, text="Reset default config", command=self.reset_default).pack(side="left")

        # Cursor values
        cursor_frame = ttk.LabelFrame(left, text="Cursor values", padding=10)
        cursor_frame.pack(fill="x", pady=(0, 10))

        self.cursor_info_var = tk.StringVar(value="Move the mouse over the graph")
        ttk.Label(cursor_frame, textvariable=self.cursor_info_var, justify="left", anchor="w").pack(fill="x")

        # Results
        results_frame = ttk.LabelFrame(left, text="Results", padding=10)
        results_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.results_text = tk.Text(results_frame, height=16, width=55)
        self.results_text.pack(fill="both", expand=True)

        # Plot area
        self.figure = plt.Figure(figsize=(11, 7), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.vline = self.ax.axvline(0, visible=False)
        self.hline = self.ax.axhline(0, visible=False)
        self.cursor_annotation = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(15, 15),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
            visible=False
        )

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self.on_mouse_leave)

    def _add_labeled_entry(self, parent, label, variable):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, width=22).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

    def _load_config_into_widgets(self):
        cfg = self.cfg
        sr = cfg.straight_resistance
        calc = cfg.calculation

        self.vehicle_name_var.set(cfg.vehicle_name)
        self.mass_var.set(str(cfg.mass_t))
        self.adherent_mass_var.set(str(cfg.adherent_mass_t))
        self.adherence_coef_var.set(str(cfg.adherence_coefficient))

        self.a_var.set(str(sr["a"]))
        self.b_var.set(str(sr["b"]))
        self.c_var.set(str(sr["c"]))
        self.tunnel_var.set(str(sr["tunnelFactor"]))
        self.starting_var.set(str(sr["startingResistance"]))

        self.slope_var.set(str(calc["slope_per_thousand"]))
        self.max_speed_var.set(str(calc["max_speed_kph"]))
        self.speed_step_var.set(str(calc["speed_step_kph"]))
        self.y_mode_var.set(str(calc.get("y_mode", "force")))

        self.te_text.delete("1.0", tk.END)
        for p in cfg.traction_effort_curve:
            self.te_text.insert(tk.END, f'{p["speed"]},{p["effort_kN"]}\n')

    def _build_config_from_widgets(self) -> ResistanceConfig:
        te_lines = self.te_text.get("1.0", tk.END).strip().splitlines()
        te_curve = []

        for idx, line in enumerate(te_lines, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid traction effort point at line {idx}: '{line}'. Expected format speed_kph,effort_kN."
                )
            speed = float(parts[0])
            effort_kn = float(parts[1])
            te_curve.append({"speed": speed, "effort_kN": effort_kn})

        if not te_curve:
            raise ValueError("Traction effort curve cannot be empty.")

        te_curve = sorted(te_curve, key=lambda x: x["speed"])

        return ResistanceConfig(
            vehicle_name=self.vehicle_name_var.get().strip() or "Unnamed Vehicle",
            mass_t=float(self.mass_var.get()),
            adherent_mass_t=float(self.adherent_mass_var.get()),
            adherence_coefficient=float(self.adherence_coef_var.get()),
            straight_resistance={
                "a": float(self.a_var.get()),
                "b": float(self.b_var.get()),
                "c": float(self.c_var.get()),
                "tunnelFactor": float(self.tunnel_var.get()),
                "startingResistance": float(self.starting_var.get())
            },
            traction_effort_curve=te_curve,
            calculation={
                "slope_per_thousand": float(self.slope_var.get()),
                "max_speed_kph": float(self.max_speed_var.get()),
                "speed_step_kph": float(self.speed_step_var.get()),
                "y_mode": self.y_mode_var.get().strip().lower()
            }
        )

    def browse_dll(self):
        path = filedialog.askopenfilename(
            title="Select Calculations.dll",
            filetypes=[("DLL files", "*.dll"), ("All files", "*.*")]
        )
        if path:
            self.dll_path_var.set(path)

    def try_load_dll(self):
        dll_path = self.dll_path_var.get().strip()
        try:
            self.mechanics = MechanicsBridge(dll_path)
            self.dll_path = dll_path
            self.dll_status_var.set(f"DLL loaded correctly:\n{os.path.abspath(dll_path)}")
        except Exception as e:
            self.mechanics = None
            self.dll_status_var.set("DLL not loaded")
            messagebox.showerror("DLL Load Error", str(e))

    def load_json(self):
        path = filedialog.askopenfilename(
            title="Open JSON config",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.cfg = load_config_from_file(path)
            self.config_path = path
            self.path_var.set(path)
            self._load_config_into_widgets()
            self.plot_curves()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def save_json(self):
        try:
            self.cfg = self._build_config_from_widgets()
            path = self.path_var.get().strip() or DEFAULT_CONFIG_FILE
            save_config_to_file(self.cfg, path)
            self.config_path = path
            self.path_var.set(path)
            messagebox.showinfo("Saved", f"Config saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def save_json_as(self):
        try:
            self.cfg = self._build_config_from_widgets()
            path = filedialog.asksaveasfilename(
                title="Save JSON config as",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not path:
                return
            save_config_to_file(self.cfg, path)
            self.config_path = path
            self.path_var.set(path)
            messagebox.showinfo("Saved", f"Config saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def reset_default(self):
        try:
            self.cfg = default_config()
            self._load_config_into_widgets()
            self.plot_curves()
        except Exception as e:
            messagebox.showerror("Reset Error", str(e))

    def plot_curves(self):
        try:
            if self.mechanics is None:
                raise RuntimeError("DLL not loaded. Load Calculations.dll first.")

            self.cfg = self._build_config_from_widgets()
            res = calculate_curves(self.cfg, self.mechanics)

            speeds_kph = res["speeds_kph"]
            y_mode = self.cfg.calculation["y_mode"].strip().lower()
            mass_t = self.cfg.mass_t

            if y_mode == "acceleration":
                y_straight = force_kn_to_acceleration(res["straight_kn"], mass_t)
                y_slope = force_kn_to_acceleration(res["slope_kn"], mass_t)
                y_total = force_kn_to_acceleration(res["total_res_kn"], mass_t)
                y_raw_te = force_kn_to_acceleration(res["raw_traction_kn"], mass_t)
                y_effective_te = force_kn_to_acceleration(res["effective_traction_kn"], mass_t)
                ylabel = "Acceleration equivalent [m/s²]"
            else:
                y_straight = res["straight_kn"]
                y_slope = res["slope_kn"]
                y_total = res["total_res_kn"]
                y_raw_te = res["raw_traction_kn"]
                y_effective_te = res["effective_traction_kn"]
                ylabel = "Force [kN]"

            self.current_speeds = speeds_kph
            self.current_straight = y_straight
            self.current_slope = y_slope
            self.current_total = y_total
            self.current_raw_te = y_raw_te
            self.current_effective_te = y_effective_te
            self.current_y_label = ylabel

            intersections_effective = find_intersections(speeds_kph, y_total, y_effective_te)
            intersections_raw = find_intersections(speeds_kph, y_total, y_raw_te)

            self.ax.clear()
            self.ax.grid(True)

            self.ax.plot(speeds_kph, y_straight, label="Straight resistance")
            self.ax.plot(speeds_kph, y_slope, label="Slope resistance")
            self.ax.plot(speeds_kph, y_total, label="Total resistance", linewidth=2)
            self.ax.plot(speeds_kph, y_raw_te, label="Raw traction effort", linestyle="--", linewidth=1.8)
            self.ax.plot(speeds_kph, y_effective_te, label="Effective traction effort", linewidth=2.2)

            for i, (xi, yi) in enumerate(intersections_effective, start=1):
                self.ax.plot(xi, yi, "ro")
                self.ax.annotate(
                    f"Eff I{i}\n({xi:.2f}, {yi:.2f})",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(8, 8)
                )

            self.ax.set_title(f"{self.cfg.vehicle_name} - Resistance vs Traction (DLL + Adherence)")
            self.ax.set_xlabel("Speed [km/h]")
            self.ax.set_ylabel(ylabel)
            self.ax.legend()

            self.vline = self.ax.axvline(0, linestyle="--", linewidth=1, visible=False)
            self.hline = self.ax.axhline(0, linestyle="--", linewidth=1, visible=False)
            self.cursor_annotation = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", alpha=0.9),
                visible=False
            )

            self.canvas.draw()

            self._write_results(
                res=res,
                intersections_effective=intersections_effective,
                intersections_raw=intersections_raw
            )

        except Exception as e:
            messagebox.showerror("Plot Error", str(e))

    def _write_results(self, res, intersections_effective, intersections_raw):
        self.results_text.delete("1.0", tk.END)

        adherence_enabled = self.cfg.adherence_coefficient > 0

        self.results_text.insert(tk.END, f"Vehicle: {self.cfg.vehicle_name}\n")
        self.results_text.insert(tk.END, f"Mass: {self.cfg.mass_t:.3f} t\n")
        self.results_text.insert(tk.END, f"Adherent mass: {self.cfg.adherent_mass_t:.3f} t\n")
        self.results_text.insert(tk.END, f"Adherence coefficient: {self.cfg.adherence_coefficient:.6f}\n")
        self.results_text.insert(tk.END, f"Adherence enabled: {'Yes' if adherence_enabled else 'No'}\n")
        self.results_text.insert(tk.END, f"Slope: {float(self.cfg.calculation['slope_per_thousand']):.3f} ‰\n")
        self.results_text.insert(tk.END, f"Max speed: {float(self.cfg.calculation['max_speed_kph']):.3f} km/h\n")
        self.results_text.insert(tk.END, f"Speed step: {float(self.cfg.calculation['speed_step_kph']):.3f} km/h\n")
        self.results_text.insert(tk.END, f"Y mode: {self.cfg.calculation['y_mode']}\n\n")

        i0 = 0
        self.results_text.insert(tk.END, "At 0 km/h:\n")
        self.results_text.insert(tk.END, f"  Straight resistance: {res['straight_kn'][i0]:.4f} kN\n")
        self.results_text.insert(tk.END, f"  Slope resistance:    {res['slope_kn'][i0]:.4f} kN\n")
        self.results_text.insert(tk.END, f"  Total resistance:    {res['total_res_kn'][i0]:.4f} kN\n")
        self.results_text.insert(tk.END, f"  Raw traction effort: {res['raw_traction_kn'][i0]:.4f} kN\n")
        self.results_text.insert(tk.END, f"  Effective traction:  {res['effective_traction_kn'][i0]:.4f} kN\n\n")

        if intersections_effective:
            self.results_text.insert(tk.END, "Intersections (Total resistance vs Effective traction):\n")
            for i, (x, y) in enumerate(intersections_effective, start=1):
                unit = "m/s²" if self.cfg.calculation["y_mode"] == "acceleration" else "kN"
                self.results_text.insert(
                    tk.END,
                    f"  Eff I{i}: speed = {x:.4f} km/h, Y = {y:.4f} {unit}\n"
                )
        else:
            self.results_text.insert(tk.END, "No intersections found for Total resistance vs Effective traction.\n")

        self.results_text.insert(tk.END, "\n")

        if intersections_raw:
            self.results_text.insert(tk.END, "Intersections (Total resistance vs Raw traction):\n")
            for i, (x, y) in enumerate(intersections_raw, start=1):
                unit = "m/s²" if self.cfg.calculation["y_mode"] == "acceleration" else "kN"
                self.results_text.insert(
                    tk.END,
                    f"  Raw I{i}: speed = {x:.4f} km/h, Y = {y:.4f} {unit}\n"
                )
        else:
            self.results_text.insert(tk.END, "No intersections found for Total resistance vs Raw traction.\n")

        margin_kn = res["effective_traction_kn"] - res["total_res_kn"]
        idx_best = int(np.argmax(margin_kn))
        self.results_text.insert(tk.END, "\nAdditional info:\n")
        self.results_text.insert(tk.END, f"  Maximum Effective TE - Total Resistance margin: {margin_kn[idx_best]:.4f} kN\n")
        self.results_text.insert(tk.END, f"  At speed: {res['speeds_kph'][idx_best]:.4f} km/h\n")

        if adherence_enabled:
            adhesion_is_limiting = res["effective_traction_kn"] < res["raw_traction_kn"] - 1e-9
            limited_count = int(np.sum(adhesion_is_limiting))
            self.results_text.insert(
                tk.END,
                f"  Points where adhesion limits traction: {limited_count} / {len(res['speeds_kph'])}\n"
            )
        else:
            self.results_text.insert(tk.END, "  Adhesion is not being applied because adherence coefficient is 0.\n")

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.on_mouse_leave(None)
            return

        if self.current_speeds is None:
            return

        x = float(event.xdata)
        speeds = self.current_speeds

        if x < speeds[0] or x > speeds[-1]:
            self.on_mouse_leave(None)
            return

        straight_val = float(np.interp(x, speeds, self.current_straight))
        slope_val = float(np.interp(x, speeds, self.current_slope))
        total_val = float(np.interp(x, speeds, self.current_total))
        raw_te_val = float(np.interp(x, speeds, self.current_raw_te))
        effective_te_val = float(np.interp(x, speeds, self.current_effective_te))
        y = float(event.ydata)

        self.vline.set_xdata([x, x])
        self.hline.set_ydata([y, y])
        self.vline.set_visible(True)
        self.hline.set_visible(True)

        text = (
            f"V = {x:.2f} km/h\n"
            f"Y = {y:.2f}\n"
            f"Straight = {straight_val:.2f}\n"
            f"Slope = {slope_val:.2f}\n"
            f"Total = {total_val:.2f}\n"
            f"Raw TE = {raw_te_val:.2f}\n"
            f"Eff. TE = {effective_te_val:.2f}"
        )

        self.cursor_annotation.xy = (x, y)
        self.cursor_annotation.set_text(text)
        self.cursor_annotation.set_visible(True)

        self.cursor_info_var.set(
            f"Speed: {x:.2f} km/h\n"
            f"Straight: {straight_val:.2f}\n"
            f"Slope: {slope_val:.2f}\n"
            f"Total: {total_val:.2f}\n"
            f"Raw TE: {raw_te_val:.2f}\n"
            f"Effective TE: {effective_te_val:.2f}\n"
            f"Y axis: {self.current_y_label}"
        )

        self.canvas.draw_idle()

    def on_mouse_leave(self, event):
        if hasattr(self, "vline"):
            self.vline.set_visible(False)
        if hasattr(self, "hline"):
            self.hline.set_visible(False)
        if hasattr(self, "cursor_annotation"):
            self.cursor_annotation.set_visible(False)

        self.cursor_info_var.set("Move the mouse over the graph")
        self.canvas.draw_idle()


def main():
    root = tk.Tk()
    app = RailResistanceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()