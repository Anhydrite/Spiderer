import argparse
from enum import Enum
import json
from pprint import pprint
import string
from typing import Any, Dict, Iterator, List, Optional, Set, cast
from attr import asdict
import bs4
import requests
import re
import urllib3
import validators
import validators.uri
from attrs import define, field

from logger import Logger

urllib3.disable_warnings()


"""
TODO:
 - check avec le full domain (tester decrawler des osus domaines et vérif uqe ca sorte pas)
 - prendre d'autres tags que href et src
 - validate, check le startswith / 
 - multi thread le bouzin
 - option pour trier par extensio et status code
 - bug avec ca
"""


class EnumEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Enum):
            return o.name
        return json.JSONEncoder.default(self, o)


class MatchTypeEnum(Enum):
    HTTP = "http"
    SCHEME_SEPARATOR = "://"
    RELATIVE_URL = "./"
    RELATIVE_URL_WITHOUT_POINT = "/"
    WITHOUT_SLASH = ""
    STARTING_URL = "starting_url"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if str(value).startswith("http"):
            return MatchTypeEnum.HTTP

        if value is None:
            return MatchTypeEnum.WITHOUT_SLASH

        return super()._missing_(value)


@define(unsafe_hash=True)
class Url:

    ALLOWED_URL_CHARACTERS = string.ascii_lowercase + string.digits + "-" + "_" + "." + "~" + "%"

    path: str = field(hash=True)
    base_path: str = field(init=False)
    domain: str = field(init=False)
    protocol: Optional[str] = field(init=False)
    tld: str = field(init=False)
    status_code: Optional[int] = field(default=None, init=False)

    def __attrs_post_init__(self):

        self.path: str = Url._clean_path_from_parameters(self.path)
        self.tld = Url._parse_tld(self.path)
        self.base_path: str = Url._parse_base_path(self.path, self.tld)
        self.domain = Url._parse_domain(self.base_path)

        temp_protocol = Url.parse_protocol(self.path)

        self.protocol: Optional[str] = temp_protocol
        self.tld: str = self._parse_tld(self.path)

    @staticmethod
    def _parse_tld(path: str) -> str:
        if "//" not in path:
            raise ValueError(f"Could not guess the TLD of {path}, initial")

        cleaned_string = "".join(path.split("//")[1])
        if "/" in cleaned_string:
            cleaned_string = "".join(cleaned_string).split("/")[0]

        if "." not in cleaned_string:
            raise ValueError(f"Could not guess the TLD of {path}, no .")

        tld: str = cleaned_string.split(".")[-1]
        return tld

    @staticmethod
    def _clean_path_from_parameters(path: str) -> str:
        clean_path: str = path.split("?")[0]
        return clean_path

    @staticmethod
    def _parse_base_path(path: str, tld: str) -> str:
        base_url_array = path.partition(tld)
        if len(base_url_array) > 2:
            base_url_array = base_url_array[:2]

        base_url: str = "".join(base_url_array)
        return base_url

    @staticmethod
    def _parse_domain(base_url: str) -> str:
        domain: str = base_url.split("://")[1]

        domain_regex: re.Pattern[str] = re.compile(r"[a-z\.]*")
        domain = domain_regex.match(domain).string  # type: ignore
        return domain

    @staticmethod
    def parse_protocol(path: str) -> Optional[str]:
        if "://" not in path:
            return None
        protocol: str = path.split("://")[0]
        if not protocol:
            return None
        return protocol

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return self.__str__()

    # def __hash__(self) -> int:
    #     path_without_slash: str = self.path.replace("/", "")
    #     """Because we can have www.google.fr and www.google.fr/ that are not differents links but www.google.fr/test is different from www.google.fr/test/"""
    #     return hash(path_without_slash)


@define(unsafe_hash=True)
class Finding:

    raw_path: str = field(hash=True)
    url: Url = field(hash=True)
    match_type: Optional[MatchTypeEnum] = field(default=None)
    from_url: Optional[Url] = field(default=None)
    scrapped: bool = field(default=False)
    status_code: Optional[int] = field(init=False, default=None)

    def __str__(self) -> str:
        return str(self.url)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Finding):
            return False

        return self.url == __value.url  # type: ignore

    def __ne__(self, __value: object) -> bool:
        if not hasattr(__value, "path"):
            return False

        return self.path == __value.path  # type: ignore


class FindingsManager:
    findings: Dict[int, Finding] = {}

    def __init__(self, reset=False) -> None:
        if not reset:
            Logger.warning(
                "FindingsManager must be called with reset=True to reinitialize all findings. Otherwise consider removing parenthese"
            )
            return

        self.findings = {}

    @staticmethod
    def create_finding(raw_path: str, match_type: MatchTypeEnum, from_url: Optional[Url] = None) -> Optional[Finding]:
        # Check raw path is valid
        # Reconstruct full path
        if from_url is None and match_type != MatchTypeEnum.STARTING_URL:
            Logger.warning(
                f"Error creating {raw_path} from url is None and match type is not starting url. Finding will not be created."
            )
            return None

        if match_type == MatchTypeEnum.STARTING_URL:
            corresponding_url = Url(raw_path)
        else:
            from_url = cast(Url, from_url)
            clean_path = FindingsManager._reconstruct_and_clean_full_url(raw_path, from_url, match_type)
            corresponding_url = FindingsManager.create_url(clean_path)

        if corresponding_url is None:
            Logger.warning(f"Invalid url {raw_path} provided. Finding will not be created.")
            return None

        finding = Finding(raw_path, url=corresponding_url, match_type=match_type, from_url=from_url)

        finding_hash = hash(finding)
        if finding_hash in FindingsManager.findings:
            return None

        FindingsManager.findings[hash(finding)] = finding
        return finding

    @staticmethod
    def create_url(raw_path: str, source_url: Optional[Url] = None) -> Optional[Url]:
        if not FindingsManager.validate_url(raw_path):
            return None

        return Url(raw_path)

    @staticmethod
    def _reconstruct_and_clean_full_url(raw_path: str, from_url: Url, match_type: MatchTypeEnum) -> str:
        full_path: str = raw_path

        if from_url.domain in raw_path:
            raw_path = raw_path.split(from_url.domain)[1]

        if raw_path.startswith("./") or match_type == MatchTypeEnum.RELATIVE_URL:
            clean_raw_path: str = raw_path.split("./")[1]
            full_path = f"{from_url.base_path}/{clean_raw_path}"
        elif raw_path.startswith("/") or match_type == MatchTypeEnum.RELATIVE_URL_WITHOUT_POINT:
            full_path = f"{from_url.base_path}{raw_path}"
        elif match_type == MatchTypeEnum.WITHOUT_SLASH:
            full_path = f"{from_url.base_path}/{raw_path}"

        while full_path.endswith("//") or full_path.endswith(r"\\"):
            full_path = full_path[:-1]

        if full_path.endswith("/") and full_path[:-1] == from_url.domain:
            full_path = full_path[:-1]

        return full_path

    @staticmethod
    def validate_url(path: str) -> bool:
        path = path.strip("")

        if path == "/":
            return False

        validated = validators.url(path)  # type: ignore
        if validated is True:
            return validated

        if "/" in path:
            split_string: List[str] = path.split("/")
            url_path: str
            if len(split_string) > 1:
                url_path = split_string[1]
            else:
                url_path = split_string[0]

            if all(char.lower() in Url.ALLOWED_URL_CHARACTERS for char in url_path.strip("/")):
                return True

        # Mystique
        if not path.startswith("/") and path:
            return FindingsManager.validate_url(f"/{path}")

        return False


@define
class ScrapResult:
    scrapped_links: List[Finding]
    not_scrapped_links: List[Finding]
    iteration_count: int

    @property
    def links_count(self) -> int:
        return len(self.scrapped_links) + len(self.not_scrapped_links)


class Spiderer:
    MASTER_REGEX = r"(?P<url>(?<!<)(?P<match_type>[a-z]*\:\/\/|\.\/|\/)[a-zA-Z1-9][0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\.\-\_\~\!\+\,\*\:\@\/]*)"
    r"""
    MASTER_REGEX = (
        r"(?P<url>(?P<match_type>[a-z]*\:\/\/|\.\/|\/)?[0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\.\-\_\~\!\+\,\*\:\@\/]*)"
    )
    """

    def __init__(self, website_url: str, maximum_depth: int = 0, verbose: bool = False) -> None:
        temp_protocol = Url.parse_protocol(website_url)
        if not temp_protocol:
            website_url = f"https://{website_url}"
            Logger.warning(f"No scheme/protocol provided, defaulting to {website_url}")

        initial_link = FindingsManager.create_finding(website_url, match_type=MatchTypeEnum.STARTING_URL)
        if initial_link is None:
            Logger.error(f"Could not create initial link from {website_url}")
            raise ValueError(f"Could not create initial link from {website_url}")

        self.initial_link: Finding = initial_link
        self.session: requests.Session = requests.Session()
        self.maximum_depth: int = maximum_depth
        self.verbose: bool = verbose

    @property
    def initial_full_domain(self) -> str:
        return self.initial_link.url.domain

    def scrap(self) -> ScrapResult:
        scrapped_link: List[Finding] = list()
        to_be_scrapped: List[Finding] = [self.initial_link]
        depth_counter: int = 1
        while len(to_be_scrapped) > 0:
            if self.maximum_depth > 0 and depth_counter > self.maximum_depth:
                break
            depth_counter += 1

            current_finding: Finding = to_be_scrapped.pop()
            current_finding.scrapped = True
            scrapped_link.append(current_finding)

            if current_finding.url is None:
                Logger.warning(f"Could not parse {current_finding.path}, could not identify url.")
                continue

            if current_finding.url.domain != self.initial_full_domain:
                continue

            found_findings: List[Finding] = self.parse(current_finding)
            for found_finding in found_findings:

                if found_finding in scrapped_link:
                    continue

                if found_finding in to_be_scrapped:
                    continue

                to_be_scrapped.append(found_finding)

        scrap_result = ScrapResult(scrapped_link, to_be_scrapped, depth_counter - 1)

        return scrap_result

    def parse(self, scrapped_finding: Finding) -> List[Finding]:

        Logger.info(f"parsing: {scrapped_finding}")
        try:
            response: requests.Response = self.session.get(scrapped_finding.url.path, verify=False)
        except Exception as e:
            Logger.error(f"The target is not accessible: {e}")
            return []

        scrapped_finding.status_code = response.status_code

        # response.encoding = "utf-8"
        # Filters image, weird files
        if response.apparent_encoding is None:
            return []

        if response.status_code != 200:
            Logger.warning(f"Status code {response.status_code} for {scrapped_finding.url}")
            # return []

        urls: List[Finding] = self._find_urls(response, scrapped_finding.url)
        return list(urls)

    def _find_urls(self, response: requests.Response, source_url: Url) -> List[Finding]:
        links: List[Finding] = list()
        links.extend(self._parse_with_tag(response, source_url))
        # return links
        new_regex_findings: List[Finding] = self._parse_with_regex(response, source_url)

        # Give priority to tag findings
        for new_regex_finding in new_regex_findings:
            if new_regex_finding in links:
                continue
            links.append(new_regex_finding)

        # Filter by exts
        return links

    def _parse_with_tag(self, response: requests.Response, source_url: Url) -> List[Finding]:
        findings: List[Finding] = []

        bs4_response = bs4.BeautifulSoup(response.text, "html.parser")
        tags = bs4_response.find_all()
        raw_links: List[str] = []

        for tag in tags:

            if tag.has_attr("href"):
                raw_links.append(tag["href"])
            if tag.has_attr("src"):
                raw_links.append(tag["src"])

        for raw_link in raw_links:
            regex: re.Pattern[str] = re.compile(Spiderer.MASTER_REGEX, re.MULTILINE)
            regexed_link: re.Match[str] | None = regex.match(raw_link)

            # if regexed_link is None:
            #     raw_link = raw_link
            # else:
            if regexed_link is None:
                Logger.warning(f"Regex could not match raw_link {raw_link}")
                continue

            raw_path: str = regexed_link.group("url")
            raw_matched_type = MatchTypeEnum(regexed_link.group("match_type"))

            new_finding = FindingsManager.create_finding(raw_path, raw_matched_type, from_url=source_url)

            if new_finding is None or new_finding in findings:
                Logger.debug(f"New finding {raw_path} else invalid or already in existing findings")
                continue

            findings.append(new_finding)

        return findings

    def _parse_with_regex(self, response: requests.Response, source_url: Url) -> List[Finding]:
        regex: re.Pattern[str] = re.compile(Spiderer.MASTER_REGEX, re.MULTILINE)
        response_text = response.text
        response_text = re.sub(r".*?(?:<\!DOCTYPE).*(?:>)", "", response_text)

        findings: List[Finding] = list()

        raw_links_list: Iterator[re.Match[str]] = regex.finditer(response_text)

        for raw_link in raw_links_list:
            if not raw_link.group("url"):
                continue
            raw_path: str = raw_link.group("url")
            raw_matched_type: MatchTypeEnum = MatchTypeEnum(raw_link.group("match_type"))

            new_finding = FindingsManager.create_finding(raw_path, raw_matched_type, from_url=source_url)

            if new_finding is None or new_finding in findings:
                Logger.debug(f"New finding {raw_path} else invalid or already in existing findings")
                continue

            findings.append(new_finding)

        return findings


def setup_arguments_parser():
    parser = argparse.ArgumentParser(prog="Spiderer", description="Scrap each single url without pity")
    parser.add_argument("url", help="Starting url or domain.")
    parser.add_argument(
        "-d", "--depth", action="store", default="0", help="Maximum scanning depth, defaults to 0 which is unlimited."
    )
    parser.add_argument("-v", "--verbose", help="Adds more logs")
    parser.add_argument(
        "-j",
        "--json",
        default="",
        nargs="?",
        action="store",
        help="Outputs the result as JSON. Also stored inside a file if the path is provided.",
    )
    return parser


def parse_arguments(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    target: str = args.url

    user_max_depth: str = args.depth
    if not user_max_depth.isnumeric():
        raise ValueError(f"The maximum depth must be a number, got {user_max_depth}")

    max_depth: int = int(user_max_depth)
    verbose: bool = args.verbose
    scrap_result = Spiderer(target, max_depth, verbose).scrap()

    if args.json != "":
        dict_res = asdict(scrap_result)
        if args.json is not None:
            with open(args.json, "w") as f:
                json.dump(asdict(dict_res), cls=EnumEncoder, indent=4, fp=f)
        Logger.info(json.dumps(dict_res, cls=EnumEncoder, indent=4))
    else:
        # Removes duplicates among scrapped and not scapped links
        all_links = set(scrap_result.scrapped_links)
        all_links.update(scrap_result.not_scrapped_links)

        # filters links by source url
        smart_from_dict = dict()
        for link in all_links:
            if link.from_url is None:
                continue
            smart_from_dict.setdefault(link.from_url, []).append(link)

        # Default print by url source
        for key, value in smart_from_dict.items():
            Logger.info(key)
            for link in value:
                Logger.info(" " * 4, link)

    Logger.result(f"Found {scrap_result.links_count} urls in {scrap_result.iteration_count} iteration(s).")


def main():
    parser = setup_arguments_parser()
    parse_arguments(parser)
    # links = Spiderer("https://robinzmuda.fr/").scrap()
    # for link in links:
    #     print(link, "from", link.from_url)


if __name__ == "__main__":
    main()
