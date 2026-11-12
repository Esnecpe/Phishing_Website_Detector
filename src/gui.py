"""
Phishing Website Detector - GUI

Tkinter front end for the URL-only Random Forest model.

Flow when the user clicks "Analyze":
    URL entered
        -> PhishingPredictor.predict(url)       (phishing_predictor.py)
        -> 15 URL features -> url_random_forest.pkl
        -> PHISHING / LEGITIMATE + risk score + confidence
        -> results shown in this window

Run from the project root:
    python src/gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
from urllib.parse import urlparse

from phishing_predictor import PhishingPredictor


# ============================================================
# DISPLAY-ONLY URL CHECKS
# ============================================================
# These lists are used only to describe the URL to the user in the
# "URL Information" and "Observations" panels. They are NOT model
# inputs. The prediction comes only from the 15-feature Random Forest.

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "bank", "password", "confirm",
    "wallet", "billing", "invoice", "suspend", "unlock", "webscr",
]

URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
]

LONG_URL_THRESHOLD = 75      # characters
MANY_SUBDOMAINS_THRESHOLD = 2

COLOR_PHISHING = "#c0392b"   # red
COLOR_LEGITIMATE = "#1e8449" # green
COLOR_NEUTRAL = "#333333"


class PhishingDetectorGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ML - Phishing Website Detector")
        self.geometry("1280x720")
        self.minsize(1000, 620)

        self._build_gui()
        self._load_predictor()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_predictor(self):
        """
        Load the URL-only Random Forest once at startup.
        If the model file is missing, disable the Analyze button.
        """
        try:
            self.predictor = PhishingPredictor()
        except Exception as error:
            self.predictor = None
            self.analyze_button.state(["disabled"])
            self.status_label.config(
                text="Model failed to load. See error message.",
                foreground=COLOR_PHISHING,
            )
            messagebox.showerror(
                "Model Error",
                "Could not load the phishing detection model.\n\n"
                f"{error}\n\n"
                "Run 'python src/train_url_deployment_model.py' to create it.",
            )
            return

        self.status_label.config(
            text="Model loaded: URL-only Random Forest (15 features). "
                 "Enter a URL and click Analyze.",
            foreground=COLOR_NEUTRAL,
        )

    # ========================================================
    # LAYOUT
    # ========================================================

    def _build_gui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Frame(self, padding=16)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(3, weight=1)

        # Header
        header = ttk.Frame(main, padding=(0, 0, 0, 10))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title = ttk.Label(
            header,
            text="Phishing Website Detector",
            font=("Segoe UI", 22, "bold")
        )
        title.grid(row=0, column=0, sticky="n", pady=(0, 4))

        # URL input card
        input_group = ttk.LabelFrame(main, text="Enter a URL to analyze", padding=16)
        input_group.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        input_group.columnconfigure(0, weight=1)

        input_row = ttk.Frame(input_group)
        input_row.grid(row=0, column=0, sticky="ew")
        input_row.columnconfigure(0, weight=1)

        self.url_var = tk.StringVar(value="")

        self.url_entry = ttk.Entry(input_row, textvariable=self.url_var, font=("Segoe UI", 12))
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 12), ipady=6)
        self.url_entry.bind("<Return>", lambda event: self.on_analyze_clicked())
        self.url_entry.focus_set()

        self.analyze_button = ttk.Button(
            input_row,
            text="Analyze",
            command=self.on_analyze_clicked
        )
        self.analyze_button.grid(row=0, column=1, sticky="e", ipadx=16, ipady=4)

        self.status_label = ttk.Label(input_group, text="Loading model...", font=("Segoe UI", 9))
        self.status_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Prediction result card
        result_frame = ttk.Frame(main, padding=12, relief="groove", borderwidth=1)
        result_frame.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        result_frame.columnconfigure(1, weight=1)

        self.prediction_label = ttk.Label(
            result_frame,
            text="Prediction:  —",
            font=("Segoe UI", 28, "bold")
        )
        self.prediction_label.grid(row=0, column=1, sticky="w", pady=(12, 4))

        self.risk_label = ttk.Label(
            result_frame,
            text="Risk Score:  —",
            font=("Segoe UI", 24, "bold")
        )
        self.risk_label.grid(row=1, column=1, sticky="w", pady=(0, 4))

        self.confidence_label = ttk.Label(
            result_frame,
            text="Confidence:  —",
            font=("Segoe UI", 14)
        )
        self.confidence_label.grid(row=2, column=1, sticky="w", pady=(0, 12))

        # Bottom panels
        bottom = ttk.Frame(main)
        bottom.grid(row=3, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        # Observations panel (filled after each analysis)
        self.reasons_group = ttk.LabelFrame(bottom, text="URL Observations", padding=16)
        self.reasons_group.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.reasons_group.columnconfigure(0, weight=1)
        self.reason_labels = []
        self._show_reasons(["•  No URL analyzed yet."])

        # URL information panel
        info_group = ttk.LabelFrame(bottom, text="URL Information", padding=16)
        info_group.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        info_group.columnconfigure(2, weight=1)

        info_rows = [
            "URL",
            "Length",
            "Protocol",
            "Domain",
            "Path",
            "Subdomains",
            "Suspicious Keywords",
            "IP Address",
            "URL Shortened",
        ]

        self.info_value_labels = {}

        for r, label in enumerate(info_rows):
            key = ttk.Label(info_group, text=label, font=("Segoe UI", 10, "bold"))
            key.grid(row=r, column=0, sticky="w", padx=(0, 16), pady=4)

            colon = ttk.Label(info_group, text=":")
            colon.grid(row=r, column=1, sticky="w", padx=(0, 12), pady=4)

            val = ttk.Label(info_group, text="—", font=("Segoe UI", 10), wraplength=420, justify="left")
            val.grid(row=r, column=2, sticky="w", pady=4)
            self.info_value_labels[label] = val

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    def on_analyze_clicked(self):
        """
        Backend integration:
        read the URL, run the Random Forest, and update every panel.
        """
        if self.predictor is None:
            return

        url = self.url_var.get().strip()

        # --- Input validation ---
        if not url:
            messagebox.showwarning("No URL", "Please enter a URL to analyze.")
            return

        if " " in url:
            messagebox.showwarning("Invalid URL", "A URL cannot contain spaces.")
            return

        # --- Prediction ---
        try:
            result = self.predictor.predict(url)
        except Exception as error:
            messagebox.showerror("Analysis Error", f"Could not analyze this URL.\n\n{error}")
            return

        parsed = urlparse(result["url"])
        if not parsed.hostname:
            messagebox.showwarning("Invalid URL", "Could not find a domain in this URL.")
            return

        self._show_prediction(result)
        self._show_url_information(result, parsed)
        self._show_reasons(self._build_observations(result, parsed))

        self.status_label.config(
            text=f"Analyzed: {result['url']}",
            foreground=COLOR_NEUTRAL,
        )

    # ========================================================
    # RESULT DISPLAY HELPERS
    # ========================================================

    def _show_prediction(self, result):
        """Prediction, risk score, and confidence."""
        is_phishing = result["prediction"] == 0
        color = COLOR_PHISHING if is_phishing else COLOR_LEGITIMATE

        self.prediction_label.config(
            text=f"Prediction:  {result['label']}",
            foreground=color,
        )
        self.risk_label.config(
            text=f"Risk Score:  {result['risk_score']:.2f}%",
            foreground=color,
        )
        self.confidence_label.config(
            text=f"Confidence:  {result['confidence']:.2f}%",
        )

    def _show_url_information(self, result, parsed):
        """Fill the URL Information panel from the analyzed URL."""
        features = result["features"]
        url = result["url"]
        domain = parsed.hostname or ""
        keywords = self._find_keywords(url)

        values = {
            "URL": url,
            "Length": f"{features['URLLength']} characters",
            "Protocol": parsed.scheme.upper(),
            "Domain": domain,
            "Path": parsed.path or "/",
            "Subdomains": (
                "N/A (IP address)" if features["IsDomainIP"] == 1
                else str(features["NoOfSubDomain"])
            ),
            "Suspicious Keywords": ", ".join(keywords) if keywords else "None found",
            "IP Address": "Used" if features["IsDomainIP"] == 1 else "Not used",
            "URL Shortened": "Yes" if self._is_shortened(domain) else "No",
        }

        for label, value in values.items():
            self.info_value_labels[label].config(text=value)

    def _build_observations(self, result, parsed):
        """
        Human-readable notes about the URL.

        These describe common phishing warning signs found in the URL.
        They help the user understand the URL, but they are not the
        model's internal reasoning.
        """
        features = result["features"]
        domain = parsed.hostname or ""
        notes = []

        if features["IsHTTPS"] == 0:
            notes.append("•  Does not use HTTPS")
        if features["IsDomainIP"] == 1:
            notes.append("•  Uses an IP address instead of a domain name")
        if features["URLLength"] > LONG_URL_THRESHOLD:
            notes.append(f"•  Unusually long URL ({features['URLLength']} characters)")
        if (features["IsDomainIP"] == 0
                and features["NoOfSubDomain"] >= MANY_SUBDOMAINS_THRESHOLD):
            notes.append(f"•  Multiple subdomains ({features['NoOfSubDomain']})")
        if features["NoOfDegitsInURL"] >= 5:
            notes.append(f"•  Many digits in the URL ({features['NoOfDegitsInURL']})")
        if "@" in result["url"]:
            notes.append('•  Contains "@" (can hide the real destination)')
        path_specials = self._count_path_specials(parsed)
        if path_specials >= 4:
            notes.append(f"•  Many special characters in the path/query ({path_specials})")
        if self._is_shortened(domain):
            notes.append("•  Uses a URL shortener (real destination is hidden)")

        keywords = self._find_keywords(result["url"])
        if keywords:
            quoted = ", ".join(f'"{k}"' for k in keywords)
            notes.append(f"•  Suspicious keyword(s): {quoted}")

        if not notes:
            notes.append("•  No common phishing warning signs found in the URL")

        notes.append(
            f"•  Model: {result['phishing_probability']:.2f}% phishing / "
            f"{result['legitimate_probability']:.2f}% legitimate"
        )
        return notes

    def _show_reasons(self, reasons):
        """Replace the lines in the Observations panel."""
        for lbl in self.reason_labels:
            lbl.destroy()
        self.reason_labels = []

        for i, reason in enumerate(reasons):
            lbl = ttk.Label(
                self.reasons_group,
                text=reason,
                font=("Segoe UI", 11),
                wraplength=520,
                justify="left",
            )
            lbl.grid(row=i, column=0, sticky="w", pady=6)
            self.reason_labels.append(lbl)

    # ========================================================
    # SMALL UTILITIES
    # ========================================================

    @staticmethod
    def _find_keywords(url):
        url_lower = url.lower()
        return [word for word in SUSPICIOUS_KEYWORDS if word in url_lower]

    @staticmethod
    def _count_path_specials(parsed):
        """
        Count special characters after the domain (path + query),
        ignoring "/" separators. The "https://" and the dots in the
        domain are not counted, since every URL has them.
        """
        text = parsed.path + parsed.query
        return sum(1 for ch in text if not ch.isalnum() and ch != "/")

    @staticmethod
    def _is_shortened(domain):
        domain = domain.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain in URL_SHORTENERS


if __name__ == "__main__":
    app = PhishingDetectorGUI()
    app.mainloop()
