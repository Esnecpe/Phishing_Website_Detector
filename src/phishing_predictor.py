from pathlib import Path
from urllib.parse import urlparse
import ipaddress
import re

import joblib
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "url_random_forest.pkl"
)


# ============================================================
# 2. URL FEATURE EXTRACTION
# ============================================================

def normalize_url(url):
    """
    Normalize the URL so feature extraction matches
    the training script.
    """

    url = str(url).strip()

    # Accept uppercase prefixes without changing case-sensitive paths or query values.
    if url.lower().startswith(("http://", "https://")):
        scheme, address = url.split("://", 1)
        url = scheme.lower() + "://" + address
    else:
        url = "https://" + url

    return url


# The 12 features used by both URL-only models.
# They are calculated from the domain name only (plus HTTPS).
#
# Why domain-only: every legitimate URL in the PhiUSIIL dataset is a
# homepage written as https://www.domain with nothing after it. A model
# trained on the full URL learned that "no www." or "has a path" means
# phishing. Removing "www." and ignoring the path/query stops that
# shortcut, so google.com, www.google.com and google.com/maps all give
# the same result.
DOMAIN_FEATURES = [
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfLettersInDomain",
    "LetterRatioInDomain",
    "NoOfDigitsInDomain",
    "DigitRatioInDomain",
    "NoOfHyphensInDomain",
    "SpecialCharRatioInDomain",
    "LongestLabelLength",
    "IsHTTPS",
]


def extract_url_features(url):
    """
    Extract the 12 domain-based features used when training
    url_random_forest.pkl and url_logistic_model.pkl.
    """

    url = normalize_url(url)
    parsed = urlparse(url)

    # domain name only, lowercase, without a leading "www."
    domain = (parsed.hostname or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]

    # IP address instead of a domain name?
    try:
        ipaddress.ip_address(domain)
        is_domain_ip = 1
    except ValueError:
        is_domain_ip = 0

    parts = domain.split(".")
    length = len(domain)
    safe_length = length if length > 0 else 1

    # top-level domain length (com = 3, uk = 2)
    if len(parts) > 1 and not is_domain_ip:
        tld_length = len(parts[-1])
    else:
        tld_length = 0

    # subdomains after removing www. (mail.google.com = 1)
    if is_domain_ip:
        subdomain_count = 0
    else:
        subdomain_count = max(len(parts) - 2, 0)

    letters = sum(ch.isalpha() for ch in domain)
    digits = sum(ch.isdigit() for ch in domain)
    hyphens = domain.count("-")
    specials = len(re.findall(r"[^a-z0-9.]", domain))

    features = {
        "DomainLength": length,
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": subdomain_count,
        "NoOfLettersInDomain": letters,
        "LetterRatioInDomain": letters / safe_length,
        "NoOfDigitsInDomain": digits,
        "DigitRatioInDomain": digits / safe_length,
        "NoOfHyphensInDomain": hyphens,
        "SpecialCharRatioInDomain": specials / safe_length,
        "LongestLabelLength": max(len(part) for part in parts),
        "IsHTTPS": 1 if parsed.scheme == "https" else 0,
    }
    return features


# ============================================================
# 3. PHISHING PREDICTOR
# ============================================================

class PhishingPredictor:

    def __init__(self):

        # ----------------------------------------------------
        # Verify model exists
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"URL deployment model not found:\n"
                f"{MODEL_PATH}"
            )


        # ----------------------------------------------------
        # Load model package
        # ----------------------------------------------------

        self.model_package = joblib.load(
            MODEL_PATH
        )

        self.model = self.model_package[
            "model"
        ]

        self.feature_names = self.model_package[
            "feature_names"
        ]

        self.label_mapping = self.model_package[
            "label_mapping"
        ]


        # ----------------------------------------------------
        # Verify deployment model
        # ----------------------------------------------------

        deployment_type = (
            self.model_package.get(
                "deployment_type"
            )
        )

        if deployment_type != "URL_ONLY":

            raise ValueError(
                "Incorrect model loaded. "
                "Expected URL_ONLY deployment model."
            )


        if len(self.feature_names) != len(DOMAIN_FEATURES):

            raise ValueError(
                "URL model does not contain "
                f"{len(DOMAIN_FEATURES)} feature names."
            )


        if self.model.n_features_in_ != len(DOMAIN_FEATURES):

            raise ValueError(
                "Random Forest does not expect "
                f"{len(DOMAIN_FEATURES)} features."
            )


    # ========================================================
    # EXTRACT FEATURES
    # ========================================================

    def extract_features(
        self,
        url
    ):

        return extract_url_features(
            url
        )


    # ========================================================
    # VALIDATE FEATURES
    # ========================================================

    def validate_features(
        self,
        features
    ):

        extracted_names = set(
            features.keys()
        )

        required_names = set(
            self.feature_names
        )


        missing_features = (
            required_names
            - extracted_names
        )


        extra_features = (
            extracted_names
            - required_names
        )


        if missing_features:

            raise ValueError(
                "Missing required features:\n"
                + "\n".join(
                    sorted(
                        missing_features
                    )
                )
            )


        if extra_features:

            raise ValueError(
                "Unexpected extra features:\n"
                + "\n".join(
                    sorted(
                        extra_features
                    )
                )
            )


    # ========================================================
    # BUILD MODEL INPUT
    # ========================================================

    def build_model_input(
        self,
        features
    ):

        self.validate_features(
            features
        )


        ordered_features = {
            feature_name:
                features[
                    feature_name
                ]

            for feature_name
            in self.feature_names
        }


        model_input = pd.DataFrame(
            [ordered_features]
        )


        return model_input


    # ========================================================
    # MAKE PREDICTION
    # ========================================================

    def predict(
        self,
        url
    ):

        # ----------------------------------------------------
        # Normalize URL
        # ----------------------------------------------------

        normalized_url = normalize_url(
            url
        )


        # ----------------------------------------------------
        # Extract URL features
        # ----------------------------------------------------

        features = self.extract_features(
            normalized_url
        )


        # ----------------------------------------------------
        # Build DataFrame in exact saved feature order
        # ----------------------------------------------------

        model_input = (
            self.build_model_input(
                features
            )
        )


        # ----------------------------------------------------
        # Predict class
        # ----------------------------------------------------

        prediction = int(
            self.model.predict(
                model_input
            )[0]
        )


        # ----------------------------------------------------
        # Predict class probabilities
        # ----------------------------------------------------

        probabilities = (
            self.model.predict_proba(
                model_input
            )[0]
        )


        class_probabilities = {}

        for (
            class_value,
            probability
        ) in zip(
            self.model.classes_,
            probabilities
        ):

            class_probabilities[
                int(class_value)
            ] = float(
                probability
            )


        phishing_probability = (
            class_probabilities.get(
                0,
                0.0
            )
        )


        legitimate_probability = (
            class_probabilities.get(
                1,
                0.0
            )
        )


        # ----------------------------------------------------
        # Convert prediction to readable label
        # ----------------------------------------------------

        prediction_label = (
            self.label_mapping[
                prediction
            ]
        )


        # ----------------------------------------------------
        # Prediction confidence
        # ----------------------------------------------------

        confidence = (
            class_probabilities.get(
                prediction,
                0.0
            )
        )


        # ----------------------------------------------------
        # Risk score
        #
        # Risk = probability of phishing class.
        # ----------------------------------------------------

        risk_score = (
            phishing_probability
            * 100
        )


        # ----------------------------------------------------
        # Build result
        # ----------------------------------------------------

        result = {

            "url":
                normalized_url,

            "prediction":
                prediction,

            "label":
                prediction_label,

            "confidence":
                confidence * 100,

            "risk_score":
                risk_score,

            "phishing_probability":
                phishing_probability * 100,

            "legitimate_probability":
                legitimate_probability * 100,

            "feature_count":
                len(features),

            "features":
                features,
        }


        return result


# ============================================================
# 4. TERMINAL TEST
# ============================================================

class LogisticPredictor:
    def __init__(self):
        path = Path(__file__).resolve().parent.parent / "models" / "url_logistic_model.pkl"
        self.model = joblib.load(path)

    def predict(self, url):
        url = normalize_url(url)
        features = extract_url_features(url)
        model_input = pd.DataFrame([features])
        prediction = int(self.model.predict(model_input)[0])
        probabilities = self.model.predict_proba(model_input)[0]
        classes = list(self.model.classes_)
        phishing = float(probabilities[classes.index(0)])
        confidence = float(probabilities[classes.index(prediction)])
        return {
            "url": url,
            "label": "PHISHING" if prediction == 0 else "LEGITIMATE",
            "prediction": prediction,
            "risk_score": phishing * 100,
            "confidence": confidence * 100,
            "features": features,
        }


if __name__ == "__main__":

    print(
        "\n============================================"
    )

    print(
        "       PHISHING WEBSITE DETECTOR"
    )

    print(
        "============================================"
    )


    try:

        # ----------------------------------------------------
        # Load predictor
        # ----------------------------------------------------

        predictor = PhishingPredictor()


        print(
            "\nURL deployment model "
            "loaded successfully."
        )


        print(
            "Deployment type:",
            predictor.model_package[
                "deployment_type"
            ]
        )


        print(
            "Expected features:",
            len(
                predictor.feature_names
            )
        )


        # ----------------------------------------------------
        # Get URL
        # ----------------------------------------------------

        url = input(
            "\nEnter URL to analyze: "
        ).strip()


        if not url:

            print(
                "\nNo URL entered."
            )

            raise SystemExit


        print(
            "\nAnalyzing URL..."
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        result = predictor.predict(
            url
        )


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print(
            "\n============================================"
        )

        print(
            "             ANALYSIS RESULT"
        )

        print(
            "============================================"
        )


        print(
            "\nURL:"
        )

        print(
            result["url"]
        )


        print(
            "\nPrediction:"
        )

        print(
            result["label"]
        )


        print(
            "\nRisk Score:"
        )

        print(
            f'{result["risk_score"]:.2f}%'
        )


        print(
            "\nPrediction Confidence:"
        )

        print(
            f'{result["confidence"]:.2f}%'
        )


        print(
            "\nPhishing Probability:"
        )

        print(
            f'{result["phishing_probability"]:.2f}%'
        )


        print(
            "\nLegitimate Probability:"
        )

        print(
            f'{result["legitimate_probability"]:.2f}%'
        )


        print(
            "\nFeatures Extracted:"
        )

        print(
            result["feature_count"]
        )


        print(
            "\n============================================"
        )


        # ----------------------------------------------------
        # Optional feature inspection
        # ----------------------------------------------------

        show_features = input(
            "\nShow all extracted features? (y/n): "
        ).strip().lower()


        if show_features == "y":

            print(
                "\n===== EXTRACTED FEATURES ====="
            )


            for feature_name in (
                predictor.feature_names
            ):

                print(
                    f"{feature_name}: "
                    f'{result["features"][feature_name]}'
                )


    except FileNotFoundError as error:

        print(
            "\nMODEL ERROR"
        )

        print(
            error
        )


    except ValueError as error:

        print(
            "\nMODEL / FEATURE ERROR"
        )

        print(
            error
        )


    except Exception as error:

        print(
            "\nUNEXPECTED ERROR"
        )

        print(
            error
        )