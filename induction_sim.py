"""
Magnetic Induction Simulator: a bar magnet moving through a coil (Faraday's and Lenz's laws)
UM Perform - Physics Coding and Simulation

Run simulation:   python induction_sim.py
Run validation:   python induction_sim.py --test      (no window needed)

Requires: Python 3.9+, matplotlib  (pip install matplotlib)
Tkinter ships with Python (on Linux: sudo apt install python3-tk)
"""

import math
import sys

# ---------------------------------------------------------------------------
# 1. PHYSICS MODEL  (no graphics here, so it can be tested on its own)
# ---------------------------------------------------------------------------
# The magnet is modelled as a point magnetic dipole of moment m (A m^2) moving
# along the axis of a circular coil of radius A. z = distance from coil centre.
#
#   Flux through ONE turn:   Phi(z) = s * mu0 * m * A^2 / ( 2 * (A^2 + z^2)^(3/2) )
#   Faraday's law:           emf = -N * dPhi/dt = -N * (dPhi/dz) * v
#   Lenz's law:              the minus sign (induced current opposes the flux change)
#
# s = +1 : north pole points to the right (+x),  s = -1 : north pole points left.
# Positive emf = current counter-clockwise as seen from the right (right-hand rule about +x).

MU0 = 4 * math.pi * 1e-7      # permeability of free space (T m / A)
A = 0.04                      # coil radius (m)
R_LOAD = 5.0                  # circuit resistance (ohm)


def flux(z, m, s=1):
    """Magnetic flux through one turn (Wb)."""
    return s * MU0 * m * A ** 2 / (2.0 * (A ** 2 + z ** 2) ** 1.5)


def emf_theory(z, v, N, m, s=1):
    """THEORY: analytic derivative  emf = -N (dPhi/dz) v  (volts)."""
    dphi_dz = -s * MU0 * m / 2.0 * 3.0 * A ** 2 * z / (A ** 2 + z ** 2) ** 2.5
    return -N * dphi_dz * v


def emf_numeric(z_prev, z_now, dt, N, m, s=1):
    """SIMULATION: emf = -N * (change in flux) / (time step)  (volts).
    Uses only the flux function, like a real coil 'sees' only the flux."""
    return -N * (flux(z_now, m, s) - flux(z_prev, m, s)) / dt


def peak_emf_theory(N, v, m):
    """Largest |emf| during a pass. It occurs at z = +/- A/2."""
    return N * v * MU0 * m / 2.0 * (1.5 / 1.25 ** 2.5) / A ** 2


def run_pass(N, m, v, s=1, x0=-0.15, x1=0.15, dt=1e-4):
    """Move the magnet from x0 to x1 at constant v without drawing anything.
    Returns (peak |emf| numeric, integral of emf dt)."""
    steps = max(1, round((x1 - x0) / (v * dt)))
    peak, integral = 0.0, 0.0
    z = x0
    for _ in range(steps):
        z_next = z + v * dt
        e = emf_numeric(z, z_next, dt, N, m, s)
        peak = max(peak, abs(e))
        integral += e * dt
        z = z_next
    return peak, integral


def lenz_text(emf, d_abs_flux):
    """Words describing direction of the induced current (Lenz's law)."""
    if abs(emf) < 1e-6:
        return "No change in flux: no induced current"
    if d_abs_flux > 0:
        return "Flux INCREASING: induced field opposes it, coil REPELS the magnet"
    return "Flux DECREASING: induced field supports it, coil ATTRACTS the magnet"


# ---------------------------------------------------------------------------
# 2. VALIDATION  (Physics testing: theory vs simulation)
# ---------------------------------------------------------------------------

def run_validation():
    N, m, v = 100, 2.0, 0.3
    print(f"Coil radius = {A*100:.1f} cm, N = {N}, m = {m} A m^2, v = {v} m/s\n")
    print(f"{'z (cm)':>7} | {'theory (mV)':>12} {'simulation (mV)':>16} {'error %':>10}")
    worst = 0.0
    dt = 1e-4
    for z in (-0.12, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.12):
        th = emf_theory(z, v, N, m) * 1000
        sim = emf_numeric(z - v * dt / 2, z + v * dt / 2, dt, N, m) * 1000
        err = 0.0 if th == 0 else abs(sim - th) / abs(th) * 100
        worst = max(worst, err)
        print(f"{z*100:7.1f} | {th:12.4f} {sim:16.4f} {err:10.2e}")
        assert err < 1e-3

    print("\nScaling tests (emf should scale linearly)")
    base = emf_theory(-0.02, 0.3, 100, 2.0)
    assert abs(emf_theory(-0.02, 0.6, 100, 2.0) / base - 2) < 1e-9
    assert abs(emf_theory(-0.02, 0.3, 200, 2.0) / base - 2) < 1e-9
    assert abs(emf_theory(-0.02, 0.3, 100, 4.0) / base - 2) < 1e-9
    print("  double v -> double emf; double N -> double emf; double m -> double emf : OK")

    print("Lenz / direction tests")
    assert emf_theory(-0.02, 0.3, 100, 2.0, +1) < 0, "approaching N pole: clockwise (negative)"
    assert emf_theory(+0.02, 0.3, 100, 2.0, +1) > 0, "leaving N pole: counter-clockwise (positive)"
    assert abs(emf_theory(-0.02, 0.3, 100, 2.0, -1) + emf_theory(-0.02, 0.3, 100, 2.0, +1)) < 1e-12
    assert emf_theory(0.0, 0.3, 100, 2.0) == 0.0
    assert abs(emf_theory(-0.02, 0.3, 100, 2.0) + emf_theory(0.02, 0.3, 100, 2.0)) < 1e-12
    assert emf_theory(-0.02, 0.0, 100, 2.0) == 0.0
    print("  approach vs leave reverse sign; flipping polarity reverses sign;")
    print("  emf = 0 at coil centre and when v = 0 : OK")

    print("Peak emf and Faraday's-law integral (magnet moves from -15 cm to the coil centre)")
    for vv in (0.1, 0.3, 0.6, 1.0):
        pk, _ = run_pass(N, m, vv)
        pt = peak_emf_theory(N, vv, m)
        err = abs(pk - pt) / pt * 100
        worst = max(worst, err)
        _, integ = run_pass(N, m, vv, x0=-0.15, x1=0.0)
        dphi = -N * (flux(0.0, m) - flux(-0.15, m))          # -N * change in flux
        h = vv * 1e-4
        th_int = sum(emf_theory(-0.15 + (k + 0.5) * h, vv, N, m) * 1e-4 for k in range(round(0.15 / h)))
        ierr = abs(th_int - dphi) / abs(dphi) * 100
        worst = max(worst, ierr)
        print(f"  v = {vv:.1f} m/s: peak theory = {pt*1000:8.3f} mV, sim = {pk*1000:8.3f} mV, error = {err:.2e} %")
        print(f"             integral of emf dt = {integ*1000:.4f} mV s vs -N*dPhi = {dphi*1000:.4f} mV s "
              f"(theory-formula check error {ierr:.2e} %)")
        assert err < 1e-2 and ierr < 1e-2
        assert abs(integ - dphi) < 1e-9
    peak_z = max((abs(emf_theory(z / 10000, 0.3, N, m)), z / 10000) for z in range(-1500, 1500))[1]
    print(f"  peak |emf| occurs at z = {abs(peak_z)*100:.2f} cm  (A/2 = {A/2*100:.2f} cm)")
    assert abs(abs(peak_z) - A / 2) < 1e-3
    print(f"\nAll tests passed. Largest error = {worst:.2e} %")


# ---------------------------------------------------------------------------
# 3. GRAPHICAL USER INTERFACE
# ---------------------------------------------------------------------------

def run_gui():
    import tkinter as tk
    from tkinter import ttk
    from collections import deque
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    W, H = 640, 330
    CX, CY = W // 2, H // 2 + 10
    PX = 1800                      # pixels per metre
    XMAX = 0.15                    # magnet travels from -XMAX to +XMAX (m)
    DT = 0.03                      # time per animation frame (s)
    HALF_L, HALF_H = 54, 20        # magnet half length / half height (pixels)
    CH = int(A * PX)               # coil half height (pixels)
    CW = 16                        # coil half width (pixels)

    class App:
        def __init__(self, root):
            self.root = root
            root.title("Magnetic Induction: Faraday's and Lenz's Laws")
            self.v = tk.DoubleVar(value=0.30)
            self.m = tk.DoubleVar(value=2.0)
            self.N = tk.DoubleVar(value=100)
            self.pos = tk.DoubleVar(value=-XMAX)
            self.sign = 1
            self.x = -XMAX
            self.dir = 1
            self.running = True
            self.t = 0.0
            self.frame = 0
            self.hist_t, self.hist_phi, self.hist_emf = deque(maxlen=200), deque(maxlen=200), deque(maxlen=200)
            self.emf = 0.0
            self.lenz = ""
            self.records = []
            keys = ("v", "m", "N", "phi", "emf", "cur", "pow", "dir", "lenz", "play")
            self.read = {k: tk.StringVar() for k in keys}
            self.read["play"].set("Pause")

            nb = ttk.Notebook(root)
            nb.pack(fill="both", expand=True)
            self.nb = nb
            self.tab_intro, self.tab_sim = ttk.Frame(nb), ttk.Frame(nb)
            self.tab_exp, self.tab_res = ttk.Frame(nb), ttk.Frame(nb)
            for tab, name in ((self.tab_intro, "1. Introduction"), (self.tab_sim, "2. Simulation"),
                              (self.tab_exp, "3. Explanation"), (self.tab_res, "4. Results")):
                nb.add(tab, text=name)
            self.build_intro()
            self.build_sim()
            self.build_explanation()
            self.build_results()
            self.tick()

        # ----- Screen 1: Introduction -----
        def build_intro(self):
            f = self.tab_intro
            ttk.Label(f, text="Magnetic Induction", font=("Segoe UI", 28, "bold")).pack(pady=(50, 5))
            ttk.Label(f, text="Faraday's Law and Lenz's Law", font=("Segoe UI", 16)).pack()
            txt = ("Physics concept: a changing magnetic flux through a coil induces an emf and a\n"
                   "current. The induced current always opposes the change that caused it.\n\n"
                   "What you can do: move a magnet through a coil, change its speed, strength and\n"
                   "polarity, change the number of turns, and watch the flux, emf, current direction\n"
                   "and bulb respond.\n\n"
                   "Try: a slow pass and a fast pass, then flip the polarity and see the current reverse.")
            ttk.Label(f, text=txt, font=("Segoe UI", 12), justify="center").pack(pady=30)
            ttk.Button(f, text="Start Simulation", command=lambda: self.nb.select(self.tab_sim)).pack()

        # ----- Screen 2: Simulation -----
        def build_sim(self):
            f = self.tab_sim
            left = ttk.Frame(f)
            left.grid(row=0, column=0, rowspan=3, padx=6, pady=6)
            self.canvas = tk.Canvas(left, width=W, height=H, bg="white", highlightthickness=0)
            self.canvas.pack()
            self.fig = Figure(figsize=(6.4, 2.7), dpi=100)
            self.ax1 = self.fig.add_subplot(211)
            self.ax2 = self.fig.add_subplot(212)
            (self.l1,) = self.ax1.plot([], [], color="tab:green")
            (self.l2,) = self.ax2.plot([], [], color="tab:red")
            self.ax1.set_ylabel("flux (uWb)", fontsize=8)
            self.ax2.set_ylabel("emf (mV)", fontsize=8)
            self.ax2.set_xlabel("time (s)", fontsize=8)
            for ax in (self.ax1, self.ax2):
                ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
            self.fig.tight_layout()
            self.mpl = FigureCanvasTkAgg(self.fig, master=left)
            self.mpl.get_tk_widget().pack()

            ctl = ttk.LabelFrame(f, text="Controls")
            ctl.grid(row=0, column=1, sticky="new", padx=8, pady=6)

            def slider(row, label, var, lo, hi, key, cmd=True):
                ttk.Label(ctl, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=(6, 0))
                if key:
                    ttk.Label(ctl, textvariable=self.read[key], width=9).grid(row=row, column=1, sticky="e")
                ttk.Scale(ctl, from_=lo, to=hi, variable=var, length=230).grid(
                    row=row + 1, column=0, columnspan=2, padx=6)
            slider(0, "Magnet speed (m/s)", self.v, 0.05, 1.0, "v")
            slider(2, "Magnet strength m (A m^2)", self.m, 0.5, 10.0, "m")
            slider(4, "Number of turns N", self.N, 1, 200, "N")
            slider(6, "Magnet position (drag when paused)", self.pos, -XMAX, XMAX, None)
            bar = ttk.Frame(ctl)
            bar.grid(row=8, column=0, columnspan=2, pady=8)
            ttk.Button(bar, textvariable=self.read["play"], command=self.toggle).grid(row=0, column=0, padx=3)
            ttk.Button(bar, text="Flip polarity", command=self.flip).grid(row=0, column=1, padx=3)
            ttk.Button(bar, text="Reset", command=self.reset).grid(row=0, column=2, padx=3)

            pre = ttk.LabelFrame(f, text="Presets")
            pre.grid(row=1, column=1, sticky="new", padx=8)
            for i, (name, v, m, n) in enumerate([("Slow pass", 0.10, 2.0, 100), ("Fast pass", 0.80, 2.0, 100),
                                                 ("Few turns (N = 10)", 0.30, 2.0, 10),
                                                 ("Many turns (N = 200)", 0.30, 2.0, 200)]):
                ttk.Button(pre, text=name, command=lambda v=v, m=m, n=n: self.preset(v, m, n)
                           ).grid(row=i, column=0, sticky="ew", padx=6, pady=2)
            pre.columnconfigure(0, weight=1)

            out = ttk.LabelFrame(f, text="Output")
            out.grid(row=2, column=1, sticky="new", padx=8, pady=6)
            for i, k in enumerate(("phi", "emf", "cur", "pow", "dir")):
                ttk.Label(out, textvariable=self.read[k], font=("Consolas", 10)).grid(row=i, column=0, sticky="w", padx=6)
            ttk.Label(out, textvariable=self.read["lenz"], wraplength=270, foreground="#b00020",
                      font=("Segoe UI", 10, "bold")).grid(row=5, column=0, sticky="w", padx=6, pady=4)
            ttk.Button(out, text="Record a full pass", command=self.record).grid(row=6, column=0, pady=4)
            ttk.Button(out, text="View results", command=lambda: self.nb.select(self.tab_res)
                       ).grid(row=7, column=0, pady=(0, 4))

        # ----- Screen 3: Explanation -----
        def build_explanation(self):
            t = tk.Text(self.tab_exp, wrap="word", font=("Segoe UI", 12), padx=20, pady=15, relief="flat")
            t.pack(fill="both", expand=True)
            t.insert("1.0",
                     "PHYSICS PRINCIPLE\n"
                     "A coil only produces an emf when the magnetic flux through it CHANGES. A magnet\n"
                     "sitting still inside the coil gives no emf, however strong it is.\n\n"
                     "EQUATIONS\n"
                     "  Magnetic flux (one turn):  Phi = B A cos(theta)\n"
                     "  Faraday's law:             emf = -N dPhi/dt\n"
                     "  Ohm's law:                 I = emf / R\n"
                     "  Power in the circuit:      P = I^2 R\n"
                     "  Dipole on the coil axis:   Phi(z) = mu0 m a^2 / (2 (a^2 + z^2)^(3/2))\n\n"
                     "LENZ'S LAW (the minus sign)\n"
                     "  - Magnet approaching: flux through the coil grows, so the induced current makes a\n"
                     "    field that points against the magnet's field. The coil REPELS the magnet.\n"
                     "  - Magnet leaving: flux shrinks, so the induced current makes a field in the same\n"
                     "    direction as the magnet's field. The coil ATTRACTS the magnet.\n"
                     "  - The coil always resists the motion. That is conservation of energy: pushing the\n"
                     "    magnet costs work, and that work becomes electrical energy (the bulb).\n\n"
                     "INTERPRETATION\n"
                     "  - Faster magnet: flux changes faster, so the emf is larger (emf is proportional to v).\n"
                     "  - More turns or a stronger magnet: larger emf (proportional to N and m).\n"
                     "  - Flipping the polarity reverses the current direction.\n"
                     "  - emf is zero when the magnet is at the coil centre (flux is at its maximum, so it is\n"
                     "    momentarily not changing). The emf peaks near half a coil radius from the centre.\n\n"
                     "HOW THE SIMULATION COMPUTES IT\n"
                     "  The emf on screen comes from the CHANGE in flux between animation frames. The theory\n"
                     "  value comes from the analytic derivative. The Results screen compares the two.\n"
                     "  In the side view: a dot means current out of the page, a cross means into the page.")
            t.config(state="disabled")

        # ----- Screen 4: Results -----
        def build_results(self):
            f = self.tab_res
            self.rfig = Figure(figsize=(6.2, 3.3), dpi=100)
            self.rax = self.rfig.add_subplot(111)
            self.rmpl = FigureCanvasTkAgg(self.rfig, master=f)
            self.rmpl.get_tk_widget().pack(fill="x", padx=8, pady=6)
            cols = ("v", "N", "m", "pol", "theory", "sim", "err")
            heads = ("speed (m/s)", "turns N", "m (A m^2)", "polarity", "peak emf theory (mV)",
                     "peak emf simulated (mV)", "% error")
            self.tree = ttk.Treeview(f, columns=cols, show="headings", height=8)
            for c, h in zip(cols, heads):
                self.tree.heading(c, text=h)
                self.tree.column(c, width=105, anchor="center")
            self.tree.pack(fill="both", expand=True, padx=8)
            bar = ttk.Frame(f)
            bar.pack(fill="x", pady=6)
            ttk.Button(bar, text="Clear table", command=self.clear_records).pack(side="left", padx=8)
            ttk.Label(bar, text="Observation: peak emf rises in a straight line with speed, and the simulated "
                                "values match Faraday's law.", wraplength=650).pack(side="left", padx=8)
            self.update_results_graph()

        # ----- controls -----
        def params(self):
            return int(round(self.N.get())), round(self.m.get(), 2), self.v.get()

        def toggle(self):
            self.running = not self.running
            self.read["play"].set("Pause" if self.running else "Play")

        def flip(self):
            self.sign = -self.sign

        def reset(self):
            self.x, self.dir = -XMAX, 1
            self.pos.set(self.x)
            self.t = 0.0
            for h in (self.hist_t, self.hist_phi, self.hist_emf):
                h.clear()

        def preset(self, v, m, n):
            self.v.set(v); self.m.set(m); self.N.set(n)

        def record(self):
            N, m, v = self.params()
            pk, _ = run_pass(N, m, v, self.sign)
            th = peak_emf_theory(N, v, m)
            err = abs(pk - th) / th * 100
            self.records.append((N, m, v, pk))
            self.tree.insert("", "end", values=(f"{v:.2f}", N, f"{m:.2f}",
                                                "N right" if self.sign > 0 else "N left",
                                                f"{th*1000:.3f}", f"{pk*1000:.3f}", f"{err:.4f}"))
            self.update_results_graph()

        def clear_records(self):
            self.records.clear()
            self.tree.delete(*self.tree.get_children())
            self.update_results_graph()

        def update_results_graph(self):
            N, m, _ = self.params()
            ax = self.rax
            ax.clear()
            vs = [0.05 + 0.01 * i for i in range(96)]
            ax.plot(vs, [peak_emf_theory(N, v, m) * 1000 for v in vs], color="tab:blue",
                    label="Faraday's law (theory)")
            mine = [(r[2], r[3] * 1000) for r in self.records if r[0] == N and r[1] == m]
            if mine:
                ax.scatter([a for a, _ in mine], [b for _, b in mine], color="green", zorder=4,
                           label="recorded (simulation)")
            ax.set_xlabel("magnet speed (m/s)"); ax.set_ylabel("peak emf (mV)")
            ax.set_title(f"Peak emf vs speed   (N = {N}, m = {m} A m^2)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
            self.rfig.tight_layout()
            self.rmpl.draw_idle()

        # ----- animation loop -----
        def tick(self):
            N, m, v = self.params()
            s = self.sign
            prev = self.x
            if self.running:
                self.x += self.dir * v * DT
                if self.x >= XMAX:
                    self.x, self.dir = XMAX, -1
                elif self.x <= -XMAX:
                    self.x, self.dir = -XMAX, 1
                self.pos.set(self.x)
                self.t += DT
            else:
                self.x = self.pos.get()
            self.emf = emf_numeric(prev, self.x, DT, N, m, s)
            phi = flux(self.x, m, s)
            d_abs = abs(phi) - abs(flux(prev, m, s))
            cur = self.emf / R_LOAD
            self.cur_a = cur
            self.lenz = lenz_text(self.emf, d_abs)

            if self.running:
                self.hist_t.append(self.t)
                self.hist_phi.append(phi * 1e6)
                self.hist_emf.append(self.emf * 1000)
            self.read["v"].set(f"{v:.2f}")
            self.read["m"].set(f"{m:.2f}")
            self.read["N"].set(f"{N}")
            self.read["phi"].set(f"Flux per turn   = {phi*1e6:8.3f} uWb")
            self.read["emf"].set(f"Induced emf     = {self.emf*1000:8.3f} mV")
            self.read["cur"].set(f"Current         = {cur*1000:8.3f} mA")
            self.read["pow"].set(f"Power           = {cur*cur*R_LOAD*1000:8.4f} mW")
            if abs(self.emf) < 1e-6:
                self.read["dir"].set("Direction       = none")
            else:
                self.read["dir"].set("Direction       = " + ("counter-clockwise" if self.emf > 0 else "clockwise")
                                     + " (seen from right)")
            self.read["lenz"].set(self.lenz)
            self.draw()
            self.frame += 1
            if self.frame % 3 == 0 and self.running and len(self.hist_t) > 1:
                self.update_live_graph(N, m, v)
            self.root.after(int(DT * 1000), self.tick)

        def update_live_graph(self, N, m, v):
            ts = list(self.hist_t)
            self.l1.set_data(ts, list(self.hist_phi))
            self.l2.set_data(ts, list(self.hist_emf))
            lo, hi = max(0.0, ts[-1] - 6.0), max(6.0, ts[-1])
            for ax in (self.ax1, self.ax2):
                ax.set_xlim(lo, hi)
            p0 = abs(flux(0.0, m)) * 1e6 * 1.15
            self.ax1.set_ylim((-p0, 0) if self.sign < 0 else (0, p0))
            pk = max(0.05, peak_emf_theory(N, v, m) * 1000 * 1.2)
            self.ax2.set_ylim(-pk, pk)
            self.mpl.draw_idle()

        # ----- drawing -----
        def draw(self):
            c = self.canvas
            c.delete("all")
            c.create_rectangle(0, 0, W, H, fill="#f7f9fc", outline="")
            mx = CX + self.x * PX
            s = self.sign
            emf = self.emf
            g = min(1.0, abs(self.cur_a) / 0.02) if abs(emf) > 1e-6 else 0.0

            # circuit wires and bulb
            wire = "#8d6e63"
            for pts in ([(CX, CY - CH), (CX, 45), (W - 84, 45)],
                        [(W - 38, 45), (W - 20, 45), (W - 20, CY + CH + 30), (CX, CY + CH + 30), (CX, CY + CH)]):
                c.create_line(*[q for p in pts for q in p], fill=wire, width=2)
            r, gg, b = int(221 + 34 * g), int(221 + 12 * g), int(221 - 147 * g)
            c.create_oval(W - 82, 23, W - 38, 67, fill=f"#{r:02x}{gg:02x}{b:02x}", outline="black", width=2)
            if g > 0.25:
                for k in range(8):
                    a = k * math.pi / 4
                    c.create_line(W - 60 + 26 * math.cos(a), 45 + 26 * math.sin(a),
                                  W - 60 + 34 * math.cos(a), 45 + 34 * math.sin(a), fill="#f9a825", width=2)
            c.create_text(W - 60, 78, text="bulb", fill="gray30")

            # field lines around the magnet
            for d in (30, 60, 95):
                c.create_oval(mx - HALF_L - d, CY - HALF_H - d * 0.9, mx + HALF_L + d, CY + HALF_H + d * 0.9,
                              outline="#9fb4d6", dash=(3, 3))

            # coil (side view)
            c.create_rectangle(CX - CW, CY - CH, CX + CW, CY + CH, outline="#b26a00", width=3)
            for k in range(1, 8):
                xk = CX - CW + k * (2 * CW) / 8
                c.create_line(xk, CY - CH, xk, CY + CH, fill="#d9a066")
            c.create_text(CX, CY - CH - 12, text=f"coil  N = {int(round(self.N.get()))}", fill="#b26a00",
                          font=("Segoe UI", 10, "bold"))

            # magnet
            left, right = ("#1e66f5", "#e53935") if s > 0 else ("#e53935", "#1e66f5")
            c.create_rectangle(mx - HALF_L, CY - HALF_H, mx, CY + HALF_H, fill=left, outline="black")
            c.create_rectangle(mx, CY - HALF_H, mx + HALF_L, CY + HALF_H, fill=right, outline="black")
            c.create_text(mx - HALF_L / 2, CY, text="S" if s > 0 else "N", fill="white", font=("Segoe UI", 14, "bold"))
            c.create_text(mx + HALF_L / 2, CY, text="N" if s > 0 else "S", fill="white", font=("Segoe UI", 14, "bold"))
            if self.running:
                c.create_line(mx, CY + HALF_H + 18, mx + self.dir * 45, CY + HALF_H + 18,
                              arrow=tk.LAST, width=3, fill="#37474f")

            # induced current symbols (dot = out of page, cross = into page)
            if abs(emf) > 1e-6:
                rr = 6 + 6 * g
                for (px, py, out) in ((CX, CY - CH, emf > 0), (CX, CY + CH, emf < 0)):
                    if out:
                        c.create_oval(px - rr, py - rr, px + rr, py + rr, fill="#d50000", outline="white", width=2)
                        c.create_oval(px - 2, py - 2, px + 2, py + 2, fill="white", outline="")
                    else:
                        c.create_oval(px - rr, py - rr, px + rr, py + rr, fill="#d50000", outline="white", width=2)
                        c.create_line(px - 4, py - 4, px + 4, py + 4, fill="white", width=2)
                        c.create_line(px - 4, py + 4, px + 4, py - 4, fill="white", width=2)
                # poles induced on the coil faces
                nleft = emf < 0
                c.create_text(CX - CW - 14, CY, text="N" if nleft else "S", fill="#ef6c00", font=("Segoe UI", 13, "bold"))
                c.create_text(CX + CW + 14, CY, text="S" if nleft else "N", fill="#ef6c00", font=("Segoe UI", 13, "bold"))

            c.create_text(10, 10, anchor="nw", font=("Segoe UI", 11, "bold"),
                          text=f"emf = {emf*1000:.2f} mV")
            c.create_text(10, H - 10, anchor="sw", fill="#b00020", font=("Segoe UI", 10, "bold"), text=self.lenz)

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_validation()
    else:
        run_gui()
