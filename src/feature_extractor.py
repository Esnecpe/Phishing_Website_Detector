from urllib.parse import urlparse
import ipaddress
import re

import requests
from bs4 import BeautifulSoup


class FeatureExtractor:
    def __init__(self, url):
        self.url = url

        if not self.url.startswith(("http://", "https://")):
            self.url = "https://" + self.url

        self.parsed = urlparse(self.url)
        self.domain = self.parsed.hostname or ""

        self.html = ""
        self.soup = None
        self.response = None

    # ========================================================
    # DOWNLOAD WEBSITE
    # ========================================================

    def fetch_page(self):
        try:
            self.response = requests.get(
                self.url,
                timeout=5,
                allow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 PhishingDetector/1.0"
                }
            )

            self.html = self.response.text
            self.soup = BeautifulSoup(self.html, "html.parser")
            return True

        except requests.RequestException:
            self.response = None
            self.html = ""
            self.soup = BeautifulSoup("", "html.parser")
            return False

    # ========================================================
    # URL FEATURES
    # ========================================================

    def get_url_length(self):
        return len(self.url)

    def get_domain_length(self):
        return len(self.domain)

    def get_is_domain_ip(self):
        try:
            ipaddress.ip_address(self.domain)
            return 1
        except ValueError:
            return 0

    def get_tld_length(self):
        parts = self.domain.split(".")
        return len(parts[-1]) if len(parts) > 1 else 0

    def get_subdomain_count(self):
        parts = self.domain.split(".")
        return 0 if len(parts) <= 2 else len(parts) - 2

    def get_letter_count(self):
        return sum(character.isalpha() for character in self.url)

    def get_letter_ratio(self):
        return 0 if len(self.url) == 0 else self.get_letter_count() / len(self.url)

    def get_digit_count(self):
        return sum(character.isdigit() for character in self.url)

    def get_digit_ratio(self):
        return 0 if len(self.url) == 0 else self.get_digit_count() / len(self.url)

    def get_equals_count(self):
        return self.url.count("=")

    def get_question_mark_count(self):
        return self.url.count("?")

    def get_ampersand_count(self):
        return self.url.count("&")

    def get_other_special_char_count(self):
        return len(re.findall(r"[^A-Za-z0-9]", self.url))

    def get_special_char_ratio(self):
        return 0 if len(self.url) == 0 else self.get_other_special_char_count() / len(self.url)

    def get_is_https(self):
        return 1 if self.parsed.scheme == "https" else 0

    # ========================================================
    # HTML FEATURES
    # ========================================================

    def get_line_count(self):
        return 0 if not self.html else len(self.html.splitlines())

    def get_largest_line_length(self):
        if not self.html:
            return 0

        lines = self.html.splitlines()
        return 0 if not lines else max(len(line) for line in lines)

    def get_has_title(self):
        return 1 if (
            self.soup
            and self.soup.title
            and self.soup.title.string
        ) else 0

    def get_image_count(self):
        return 0 if not self.soup else len(self.soup.find_all("img"))

    def get_js_count(self):
        return 0 if not self.soup else len(self.soup.find_all("script"))

    def get_css_count(self):
        return 0 if not self.soup else len(
            self.soup.find_all("link", rel="stylesheet")
        )

    def get_has_password_field(self):
        if not self.soup:
            return 0

        password = self.soup.find(
            "input",
            attrs={"type": "password"}
        )

        return 1 if password else 0

    def get_has_submit_button(self):
        if not self.soup:
            return 0

        button = self.soup.find(
            ["button", "input"],
            attrs={"type": "submit"}
        )

        return 1 if button else 0

    def get_has_description(self):
        if not self.soup:
            return 0

        description = self.soup.find(
            "meta",
            attrs={"name": "description"}
        )

        return 1 if description else 0

    def get_has_favicon(self):
        if not self.soup:
            return 0

        favicon = self.soup.find(
            "link",
            rel=lambda value: value and "icon" in str(value).lower()
        )

        return 1 if favicon else 0

    def get_has_iframe(self):
        if not self.soup:
            return 0

        return 1 if self.soup.find("iframe") else 0

    def get_has_hidden_fields(self):
        if not self.soup:
            return 0

        hidden = self.soup.find(
            "input",
            attrs={"type": "hidden"}
        )

        return 1 if hidden else 0

    def get_has_social_network(self):
        if not self.soup:
            return 0

        social_domains = [
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "x.com",
            "linkedin.com",
            "youtube.com",
            "tiktok.com",
        ]

        for tag in self.soup.find_all(["a", "link", "script"]):
            value = (
                tag.get("href")
                or tag.get("src")
                or ""
            ).lower()

            if any(domain in value for domain in social_domains):
                return 1

        return 0

    def get_bank_keyword(self):
        return 1 if self.html and "bank" in self.html.lower() else 0

    def get_pay_keyword(self):
        if not self.html:
            return 0

        payment_words = ["pay", "payment", "paypal"]
        html = self.html.lower()

        return 1 if any(word in html for word in payment_words) else 0

    def get_crypto_keyword(self):
        if not self.html:
            return 0

        crypto_words = [
            "crypto",
            "bitcoin",
            "ethereum",
            "wallet",
        ]

        html = self.html.lower()

        return 1 if any(word in html for word in crypto_words) else 0

    def get_has_copyright(self):
        if not self.html:
            return 0

        html = self.html.lower()

        return 1 if (
            "copyright" in html
            or "©" in self.html
        ) else 0

    # ========================================================
    # REFERENCE / REDIRECT FEATURES
    # ========================================================

    def get_robots(self):
        try:
            robots_url = (
                f"{self.parsed.scheme}://"
                f"{self.domain}/robots.txt"
            )

            response = requests.get(
                robots_url,
                timeout=5,
                headers={
                    "User-Agent": "Mozilla/5.0 PhishingDetector/1.0"
                }
            )

            return 1 if response.status_code == 200 else 0

        except requests.RequestException:
            return 0

    def get_is_responsive(self):
        if not self.soup:
            return 0

        viewport = self.soup.find(
            "meta",
            attrs={"name": "viewport"}
        )

        return 1 if viewport else 0

    def get_url_redirect_count(self):
        if self.response is None:
            return 0

        return len(self.response.history)

    def get_self_redirect_count(self):
        if self.response is None:
            return 0

        count = 0

        for redirect in self.response.history:
            location = redirect.headers.get("Location")

            if not location:
                continue

            redirect_url = urlparse(location)
            redirect_domain = redirect_url.hostname or self.domain

            if redirect_domain == self.domain:
                count += 1

        return count

    def get_popup_count(self):
        if not self.html:
            return 0

        html = self.html.lower()

        popup_patterns = [
            "window.open(",
            "alert(",
            "confirm(",
            "prompt(",
        ]

        return sum(html.count(pattern) for pattern in popup_patterns)

    def get_has_external_form_submit(self):
        if not self.soup:
            return 0

        for form in self.soup.find_all("form"):
            action = form.get("action")

            if not action:
                continue

            parsed_action = urlparse(action)

            if not parsed_action.hostname:
                continue

            if parsed_action.hostname != self.domain:
                return 1

        return 0

    def get_reference_counts(self):
        if not self.soup:
            return 0, 0, 0

        self_refs = 0
        empty_refs = 0
        external_refs = 0

        tags_and_attributes = [
            ("a", "href"),
            ("img", "src"),
            ("script", "src"),
            ("link", "href"),
        ]

        for tag_name, attribute in tags_and_attributes:
            for tag in self.soup.find_all(tag_name):
                reference = tag.get(attribute)

                if (
                    reference is None
                    or reference.strip() == ""
                    or reference.strip() == "#"
                ):
                    empty_refs += 1
                    continue

                reference = reference.strip()

                if reference.startswith(("/", "./", "../")):
                    self_refs += 1
                    continue

                parsed_reference = urlparse(reference)

                if not parsed_reference.hostname:
                    self_refs += 1
                    continue

                if parsed_reference.hostname == self.domain:
                    self_refs += 1
                else:
                    external_refs += 1

        return self_refs, empty_refs, external_refs

    # ========================================================
    # BUILD FEATURE DICTIONARY
    # ========================================================

    def extract_features(self):
        self.fetch_page()

        (
            self_refs,
            empty_refs,
            external_refs,
        ) = self.get_reference_counts()

        features = {
            # URL features
            "URLLength": self.get_url_length(),
            "DomainLength": self.get_domain_length(),
            "IsDomainIP": self.get_is_domain_ip(),
            "TLDLength": self.get_tld_length(),
            "NoOfSubDomain": self.get_subdomain_count(),
            "NoOfLettersInURL": self.get_letter_count(),
            "LetterRatioInURL": self.get_letter_ratio(),
            "NoOfDegitsInURL": self.get_digit_count(),
            "DegitRatioInURL": self.get_digit_ratio(),
            "NoOfEqualsInURL": self.get_equals_count(),
            "NoOfQMarkInURL": self.get_question_mark_count(),
            "NoOfAmpersandInURL": self.get_ampersand_count(),
            "NoOfOtherSpecialCharsInURL": self.get_other_special_char_count(),
            "SpacialCharRatioInURL": self.get_special_char_ratio(),
            "IsHTTPS": self.get_is_https(),

            # HTML / webpage features
            "LineOfCode": self.get_line_count(),
            "LargestLineLength": self.get_largest_line_length(),
            "HasTitle": self.get_has_title(),
            "HasFavicon": self.get_has_favicon(),
            "Robots": self.get_robots(),
            "IsResponsive": self.get_is_responsive(),
            "NoOfURLRedirect": self.get_url_redirect_count(),
            "NoOfSelfRedirect": self.get_self_redirect_count(),
            "HasDescription": self.get_has_description(),
            "NoOfPopup": self.get_popup_count(),
            "NoOfiFrame": self.get_has_iframe(),
            "HasExternalFormSubmit": self.get_has_external_form_submit(),
            "HasSocialNet": self.get_has_social_network(),
            "HasSubmitButton": self.get_has_submit_button(),
            "HasHiddenFields": self.get_has_hidden_fields(),
            "HasPasswordField": self.get_has_password_field(),
            "Bank": self.get_bank_keyword(),
            "Pay": self.get_pay_keyword(),
            "Crypto": self.get_crypto_keyword(),
            "HasCopyrightInfo": self.get_has_copyright(),
            "NoOfImage": self.get_image_count(),
            "NoOfCSS": self.get_css_count(),
            "NoOfJS": self.get_js_count(),
            "NoOfSelfRef": self_refs,
            "NoOfEmptyRef": empty_refs,
            "NoOfExternalRef": external_refs,
        }

        return features


if __name__ == "__main__":
    url = input("Enter URL: ")

    extractor = FeatureExtractor(url)
    features = extractor.extract_features()

    print("\n===== EXTRACTED FEATURES =====")

    for name, value in features.items():
        print(f"{name}: {value}")

    print(
        "\nNumber of extracted features:",
        len(features)
    )
