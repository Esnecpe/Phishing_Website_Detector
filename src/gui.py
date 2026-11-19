# gui.py
# Tkinter GUI for the phishing detector.
# Uses the saved URL-only Random Forest and Logistic Regression models.
#
# Run from the project folder:
#   python src/gui.py

import tkinter as tk
from tkinter import ttk, messagebox
from urllib.parse import urlparse

from phishing_predictor import PhishingPredictor, LogisticPredictor


# Words that often show up in phishing links.
# Only used to show info to the user, NOT used by the model.
KEYWORDS = ["login", "signin", "verify", "account", "update", "secure",
            "bank", "password", "confirm", "billing", "suspend"]


class PhishingGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("ML - Phishing Website Detector")
        self.root.geometry("900x520")
        self.root.minsize(820, 460)

        # load the model once when the app starts
        try:
            self.predictor = PhishingPredictor()
            self.logistic_predictor = LogisticPredictor()
        except Exception as e:
            self.predictor = None
            self.logistic_predictor = None
            messagebox.showerror("Error", "Could not load model:\n" + str(e))

        self.build_widgets()

    def build_widgets(self):
        # Use colorable bar elements without changing the theme of the rest of the GUI.
        self.bar_style = ttk.Style(self.root)
        self.bar_style.element_create("Risk.trough", "from", "clam", "Horizontal.Progressbar.trough")
        self.bar_style.element_create("Risk.pbar", "from", "clam", "Horizontal.Progressbar.pbar")
        for name in ["RF.Horizontal.TProgressbar", "LR.Horizontal.TProgressbar"]:
            self.bar_style.layout(name, [
                ("Risk.trough", {"sticky": "nswe", "children": [
                    ("Risk.pbar", {"side": "left", "sticky": "ns"})
                ]})
            ])
            self.bar_style.configure(name, background="green", troughcolor="#eeeeee")

        frame = ttk.Frame(self.root, padding=12)
        frame.pack(fill="both", expand=True)

        # ---------- title ----------
        ttk.Label(frame, text="Phishing Website Detector",
                  font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        ttk.Label(frame, text=" ML Using URL Features").pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=8)

        # ---------- url input ----------
        input_row = ttk.Frame(frame)
        input_row.pack(fill="x")

        ttk.Label(input_row, text="Enter URL:").pack(side="left")
        self.url_entry = ttk.Entry(input_row)
        self.url_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.url_entry.bind("<Return>", lambda e: self.check_url())
        self.url_entry.focus()

        ttk.Button(input_row, text="Check both", command=self.check_url).pack(side="left")
        ttk.Button(input_row, text="Clear", command=self.clear).pack(side="left", padx=(5, 0))

        # ---------- result + details side by side ----------
        middle = ttk.Frame(frame)
        middle.pack(fill="x", pady=(10, 0))
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=1)

        # result box
        result_box = ttk.LabelFrame(middle, text="Model results", padding=10)
        result_box.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        models_frame = ttk.Frame(result_box)
        models_frame.pack(fill="x", expand=True)
        models_frame.columnconfigure(0, weight=1)
        models_frame.columnconfigure(1, weight=1)

        rf_frame = ttk.Frame(models_frame)
        rf_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ttk.Label(rf_frame, text="Random Forest").pack(anchor="w")
        self.prediction_lbl = ttk.Label(rf_frame, text="-", font=("TkDefaultFont", 14, "bold"))
        self.prediction_lbl.pack(anchor="w")

        ttk.Label(rf_frame, text="Phishing risk:").pack(anchor="w", pady=(8, 0))
        self.risk_bar = ttk.Progressbar(rf_frame, maximum=100, length=200, style="RF.Horizontal.TProgressbar")
        self.risk_bar.pack(anchor="w", fill="x")
        self.risk_lbl = ttk.Label(rf_frame, text="-")
        self.risk_lbl.pack(anchor="w")

        self.conf_lbl = ttk.Label(rf_frame, text="Confidence: -")
        self.conf_lbl.pack(anchor="w", pady=(6, 0))

        lr_frame = ttk.Frame(models_frame)
        lr_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        ttk.Label(lr_frame, text="Logistic Regression").pack(anchor="w")
        self.logistic_prediction_lbl = ttk.Label(lr_frame, text="-", font=("TkDefaultFont", 14, "bold"))
        self.logistic_prediction_lbl.pack(anchor="w")

        ttk.Label(lr_frame, text="Phishing risk:").pack(anchor="w", pady=(8, 0))
        self.logistic_risk_bar = ttk.Progressbar(lr_frame, maximum=100, length=200, style="LR.Horizontal.TProgressbar")
        self.logistic_risk_bar.pack(anchor="w", fill="x")


        self.logistic_risk_lbl = ttk.Label(lr_frame, text="Estimated phishing probability: -")
        self.logistic_risk_lbl.pack(anchor="w", pady=(6, 0))

        # url details box
        details_box = ttk.LabelFrame(middle, text="URL details", padding=10)
        details_box.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.length_lbl = self.add_row(details_box, 0, "Length:")
        self.domain_lbl = self.add_row(details_box, 1, "Domain:")
        self.https_lbl = self.add_row(details_box, 2, "HTTPS:")
        self.sub_lbl = self.add_row(details_box, 3, "Subdomains:")
        self.ip_lbl = self.add_row(details_box, 4, "IP address:")
        self.kw_lbl = self.add_row(details_box, 5, "Keywords:")

        # ---------- warning signs ----------
        warn_box = ttk.LabelFrame(frame, text="Warning signs (not used by the model)", padding=8)
        warn_box.pack(fill="x", pady=(10, 0))
        self.warn_lbl = ttk.Label(warn_box, text="-", wraplength=600, justify="left")
        self.warn_lbl.pack(anchor="w")

        # ---------- history of checked urls ----------
        history_box = ttk.LabelFrame(frame, text="History", padding=8)
        history_box.pack(fill="both", expand=True, pady=(10, 0))
        self.history = tk.Listbox(history_box, height=5)
        self.history.pack(fill="both", expand=True)

        # ---------- status bar ----------
        self.status = ttk.Label(frame, text="Model loaded." if self.predictor else "Model not loaded.")
        self.status.pack(anchor="w", pady=(6, 0))

    def add_row(self, parent, row, text):
        # makes a "name: value" row in a grid and returns the value label
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=(0, 8))
        value = ttk.Label(parent, text="-", wraplength=200, justify="left")
        value.grid(row=row, column=1, sticky="w")
        return value

    def check_url(self):
        if self.predictor is None:
            messagebox.showerror("Error", "Model is not loaded.")
            return

        url = self.url_entry.get().strip()
        if url == "":
            messagebox.showwarning("Warning", "Please enter a URL.")
            return
        if " " in url:
            messagebox.showwarning("Warning", "URL can't have spaces.")
            return

        try:
            result = self.predictor.predict(url)
            # Use the same normalized URL as Random Forest.
            logistic_result = self.logistic_predictor.predict(result["url"])
        except Exception as e:
            messagebox.showerror("Error", "Something went wrong:\n" + str(e))
            return

        f = result["features"]
        domain = urlparse(result["url"]).hostname or ""

        # Display categories based on phishing risk; these do not change the trained models.
        risk = result["risk_score"]
        if risk >= 70:
            status = "PHISHING"
            self.prediction_lbl.config(text=status, foreground="red")
        elif risk >= 30:
            status = "SUSPICIOUS"
            self.prediction_lbl.config(text=status, foreground="#c9a000")
        else:
            status = "LEGITIMATE"
            self.prediction_lbl.config(text=status, foreground="green")
        # Each bar takes its own label's color, so the two models can disagree.
        self.bar_style.configure("RF.Horizontal.TProgressbar", background=self.prediction_lbl.cget("foreground"))
        self.risk_bar["value"] = result["risk_score"]
        self.risk_lbl.config(text=f"{result['risk_score']:.2f}%")
        self.conf_lbl.config(text=f"Confidence: {result['confidence']:.2f}%")

        # Apply the same display ranges to Logistic Regression's own probability.
        logistic_risk = logistic_result["risk_score"]
        if logistic_risk >= 70:
            logistic_status = "PHISHING"
            self.logistic_prediction_lbl.config(text=logistic_status, foreground="red")
        elif logistic_risk >= 30:
            logistic_status = "SUSPICIOUS"
            self.logistic_prediction_lbl.config(text=logistic_status, foreground="#c9a000")
        else:
            logistic_status = "LEGITIMATE"
            self.logistic_prediction_lbl.config(text=logistic_status, foreground="green")
        self.bar_style.configure("LR.Horizontal.TProgressbar", background=self.logistic_prediction_lbl.cget("foreground"))
        self.logistic_risk_bar["value"] = logistic_result["risk_score"]
        self.logistic_risk_lbl.config(
            text=f"Estimated phishing probability: {logistic_result['risk_score']:.2f}%"
        )

        # url details section
        found = [k for k in KEYWORDS if k in result["url"].lower()]

        self.length_lbl.config(text=str(f["URLLength"]) + " chars")
        self.domain_lbl.config(text=domain)
        self.https_lbl.config(text="Yes" if f["IsHTTPS"] == 1 else "No")
        self.sub_lbl.config(text="-" if f["IsDomainIP"] == 1 else str(f["NoOfSubDomain"]))
        self.ip_lbl.config(text="Yes" if f["IsDomainIP"] == 1 else "No")
        self.kw_lbl.config(text=", ".join(found) if found else "none")

        # simple warning signs (just for the user, the model decides the prediction)
        warnings = []
        if f["IsHTTPS"] == 0:
            warnings.append("no HTTPS")
        if f["IsDomainIP"] == 1:
            warnings.append("uses an IP address instead of a domain")
        if f["URLLength"] > 75:
            warnings.append("long URL")
        if f["NoOfDegitsInURL"] >= 5:
            warnings.append("many digits")
        if "@" in result["url"]:
            warnings.append("has @ symbol")
        if found:
            warnings.append("suspicious keywords")
        self.warn_lbl.config(text=", ".join(warnings) if warnings else "none found")

        # add to history (newest on top)
        self.history.insert(
            0,
            f"RF: {status} ({result['risk_score']:.2f}%) | "
            f"LR: {logistic_status} ({logistic_result['risk_score']:.2f}%) | "
            f"{result['url']}",
        )

        self.status.config(text="Done.")

    def clear(self):
        self.url_entry.delete(0, "end")
        for lbl in [self.prediction_lbl, self.risk_lbl, self.length_lbl,
                    self.domain_lbl, self.https_lbl, self.sub_lbl,
                    self.ip_lbl, self.kw_lbl, self.warn_lbl, self.logistic_prediction_lbl]:
            lbl.config(text="-", foreground="")
        self.conf_lbl.config(text="Confidence: -")
        self.logistic_risk_lbl.config(text="Estimated phishing probability: -")
        self.risk_bar["value"] = 0
        self.logistic_risk_bar["value"] = 0
        self.status.config(text="Model loaded." if self.predictor else "Model not loaded.")


if __name__ == "__main__":
    root = tk.Tk()
    app = PhishingGUI(root)
    root.mainloop()
