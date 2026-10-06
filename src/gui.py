import tkinter as tk
from tkinter import ttk


class PhishingDetectorGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ML - Phishing Website Detector")
        self.geometry("1280x720")
        self.minsize(1000, 620)

        self._build_gui()

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

        self.url_var = tk.StringVar(value="https://secure-login-example.com/account/verify")

        self.url_entry = ttk.Entry(input_row, textvariable=self.url_var, font=("Segoe UI", 12))
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 12), ipady=6)

        self.analyze_button = ttk.Button(
            input_row,
            text="Analyze",
            command=self.on_analyze_clicked
        )
        self.analyze_button.grid(row=0, column=1, sticky="e", ipadx=16, ipady=4)

        # Prediction result card
        result_frame = ttk.Frame(main, padding=12, relief="groove", borderwidth=1)
        result_frame.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        result_frame.columnconfigure(1, weight=1)

        self.prediction_label = ttk.Label(
            result_frame,
            text="Prediction:  PHISHING",
            font=("Segoe UI", 28, "bold")
        )
        self.prediction_label.grid(row=0, column=1, sticky="w", pady=(12, 4))

        self.risk_label = ttk.Label(
            result_frame,
            text="Risk Score:  91%",
            font=("Segoe UI", 24, "bold")
        )
        self.risk_label.grid(row=1, column=1, sticky="w", pady=(0, 12))

        # Bottom panels
        bottom = ttk.Frame(main)
        bottom.grid(row=3, column=0, sticky="nsew")
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.rowconfigure(0, weight=1)

        # Reasons panel
        reasons_group = ttk.LabelFrame(bottom, text="Why this URL is suspicious?", padding=16)
        reasons_group.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        reasons_group.columnconfigure(0, weight=1)

        reasons = [
            "•  Unusually long URL",
            '•  Suspicious keyword: "verify"',
            "•  Multiple subdomains",
            "•  Domain structure resembles known phishing patterns",
        ]

        self.reason_labels = []
        for i, reason in enumerate(reasons):
            lbl = ttk.Label(reasons_group, text=reason, font=("Segoe UI", 11))
            lbl.grid(row=i, column=0, sticky="w", pady=8)
            self.reason_labels.append(lbl)

        # URL information panel
        info_group = ttk.LabelFrame(bottom, text="URL Information", padding=16)
        info_group.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        info_group.columnconfigure(1, weight=1)

        info_rows = [
            ("URL", "https://secure-login-example.com/account/verify"),
            ("Length", "52 characters"),
            ("Protocol", "HTTPS"),
            ("Domain", "secure-login-example.com"),
            ("Path", "/account/verify"),
            ("Subdomains", "0"),
            ("Suspicious Keywords", "login, secure, account, verify"),
            ("IP Address", "Not used"),
            ("URL Shortened", "No"),
        ]

        self.info_value_labels = {}

        for r, (label, value) in enumerate(info_rows):
            key = ttk.Label(info_group, text=label, font=("Segoe UI", 10, "bold"))
            key.grid(row=r, column=0, sticky="w", padx=(0, 16), pady=4)

            colon = ttk.Label(info_group, text=":")
            colon.grid(row=r, column=1, sticky="w", padx=(0, 12), pady=4)

            val = ttk.Label(info_group, text=value, font=("Segoe UI", 10), wraplength=420, justify="left")
            val.grid(row=r, column=2, sticky="w", pady=4)
            self.info_value_labels[label] = val

    def on_analyze_clicked(self):
        """
        Backend integration hook.
        Replace this with your real ML/backend call later.
        """
        pass


if __name__ == "__main__":
    app = PhishingDetectorGUI()
    app.mainloop()
