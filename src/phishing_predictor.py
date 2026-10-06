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

    if not url.startswith(
        ("http://", "https://")
    ):
        url = "https://" + url

    return url


def extract_url_features(url):
    """
    Extract the exact same 15 URL-only features
    used when training url_random_forest.pkl.
    """

    url = normalize_url(url)

    parsed = urlparse(url)

    domain = (
        parsed.hostname
        or ""
    )


    # --------------------------------------------------------
    # Basic URL values
    # --------------------------------------------------------

    url_length = len(url)

    domain_length = len(domain)


    # --------------------------------------------------------
    # IsDomainIP
    # --------------------------------------------------------

    try:
        ipaddress.ip_address(
            domain
        )

        is_domain_ip = 1

    except ValueError:
        is_domain_ip = 0


    # --------------------------------------------------------
    # TLDLength
    # --------------------------------------------------------

    domain_parts = domain.split(".")

    if len(domain_parts) > 1:

        tld_length = len(
            domain_parts[-1]
        )

    else:

        tld_length = 0


    # --------------------------------------------------------
    # NoOfSubDomain
    # --------------------------------------------------------

    if len(domain_parts) <= 2:

        subdomain_count = 0

    else:

        subdomain_count = (
            len(domain_parts)
            - 2
        )


    # --------------------------------------------------------
    # Letter features
    # --------------------------------------------------------

    letter_count = sum(
        character.isalpha()
        for character in url
    )

    if url_length > 0:

        letter_ratio = (
            letter_count
            / url_length
        )

    else:

        letter_ratio = 0.0


    # --------------------------------------------------------
    # Digit features
    # --------------------------------------------------------

    digit_count = sum(
        character.isdigit()
        for character in url
    )

    if url_length > 0:

        digit_ratio = (
            digit_count
            / url_length
        )

    else:

        digit_ratio = 0.0


    # --------------------------------------------------------
    # URL character counts
    # --------------------------------------------------------

    equals_count = (
        url.count("=")
    )

    question_mark_count = (
        url.count("?")
    )

    ampersand_count = (
        url.count("&")
    )


    # --------------------------------------------------------
    # Special characters
    # --------------------------------------------------------

    special_characters = re.findall(
        r"[^A-Za-z0-9]",
        url
    )

    other_special_count = len(
        special_characters
    )


    if url_length > 0:

        special_char_ratio = (
            other_special_count
            / url_length
        )

    else:

        special_char_ratio = 0.0


    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    is_https = (
        1
        if parsed.scheme == "https"
        else 0
    )


    # --------------------------------------------------------
    # Final feature dictionary
    # --------------------------------------------------------

    features = {
        "URLLength":
            url_length,

        "DomainLength":
            domain_length,

        "IsDomainIP":
            is_domain_ip,

        "TLDLength":
            tld_length,

        "NoOfSubDomain":
            subdomain_count,

        "NoOfLettersInURL":
            letter_count,

        "LetterRatioInURL":
            letter_ratio,

        "NoOfDegitsInURL":
            digit_count,

        "DegitRatioInURL":
            digit_ratio,

        "NoOfEqualsInURL":
            equals_count,

        "NoOfQMarkInURL":
            question_mark_count,

        "NoOfAmpersandInURL":
            ampersand_count,

        "NoOfOtherSpecialCharsInURL":
            other_special_count,

        "SpacialCharRatioInURL":
            special_char_ratio,

        "IsHTTPS":
            is_https,
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


        if len(self.feature_names) != 15:

            raise ValueError(
                "URL model does not contain "
                "15 feature names."
            )


        if self.model.n_features_in_ != 15:

            raise ValueError(
                "Random Forest does not expect "
                "15 features."
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
        # Extract 15 URL features
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
                "\n===== 15 EXTRACTED FEATURES ====="
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