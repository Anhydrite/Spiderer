import argparse
from calendar import c
from enum import Enum
import json
from pprint import pprint
import string
from typing import Any, Dict, Iterator, List, Optional, Set
from attr import asdict
import bs4
import requests
import re
import urllib3
import validators
import validators.uri
from attrs import define, field

from logger import Logger

"""
TODO:
 - Parser les balises <a> dans le site
 - Gérer les errurs 404 etc
"""

urllib3.disable_warnings()


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


@define
class Url:

    ALLOWED_URL_CHARACTERS = string.ascii_lowercase + string.digits + "-" + "_" + "." + "~" + "%"

    raw_path: str = field()
    path: str = field(init=False)
    protocol: Optional[str] = field(init=False)
    tld: str = field(init=False)
    status_code: Optional[int] = field(default=None, init=False)

    def __attrs_post_init__(self):
        if not Url.validate_url(self.raw_path):
            Logger.error(f"The url {self.raw_path} is not valid.")

        self.path: str = self._clean_path_from_parameters(self.raw_path)

        while self.path.endswith("//") or self.path.endswith(r"\\"):
            self.path = self.path[:-1]

        temp_protocol = Url.parse_protocol(self.path)

        self.protocol: Optional[str] = temp_protocol
        self.tld: str = self._parse_tld(self.path)

    def _parse_tld(self, path: str) -> str:
        if "//" not in path:
            raise ValueError(f"Could not guess the TLD of {path}, initial")

        cleaned_string = "".join(path.split("//")[1])
        if "/" in cleaned_string:
            cleaned_string = "".join(cleaned_string).split("/")[0]

        if "." not in cleaned_string:
            raise ValueError(f"Could not guess the TLD of {path}, no .")

        tld: str = cleaned_string.split(".")[-1]
        return tld

    def _clean_path_from_parameters(self, path: str) -> str:
        clean_path: str = path.split("?")[0]
        return clean_path

    @property
    def base_url(self) -> str:
        base_url_array = self.path.partition(self.tld)
        if len(base_url_array) > 2:
            base_url_array = base_url_array[:2]

        base_url: str = "".join(base_url_array)
        return base_url

    @property
    def domain(self) -> str:
        domain: str = self.base_url.split("://")[1]

        domain_regex: re.Pattern[str] = re.compile(r"[a-z\.]*")
        domain = domain_regex.match(domain).string  # type: ignore
        return domain

    @staticmethod
    def validate_url(url: str) -> bool:
        url = url.strip("")

        if url == "/":
            return False

        validated = validators.url(url)  # type: ignore
        if validated is True:
            return validated
        if "/" in url:
            split_string: List[str] = url.split("/")
            url_path: str
            if len(split_string) > 1:
                url_path = split_string[1]
            else:
                url_path = split_string[0]

            if all(char.lower() in Url.ALLOWED_URL_CHARACTERS for char in url_path.strip("/")):
                return True

        if not url.startswith("/") and url:
            return Url.validate_url(f"/{url}")

        return False

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

    def __hash__(self) -> int:
        path_without_slash: str = self.path.replace("/", "")
        """Because we can have www.google.fr and www.google.fr/ that are not differents links but www.google.fr/test is different from www.google.fr/test/"""
        return hash(path_without_slash)

    def __eq__(self, __value) -> bool:
        if not hasattr(__value, "path"):
            return False

        return self.path == __value.path  # type: ignore

    def __ne__(self, __value: object) -> bool:
        if not hasattr(__value, "path"):
            return False

        return self.path != __value.path  # type: ignore


@define
class FoundLink:

    raw_path: str
    url: Url = field(default=None)
    match_type: Optional[MatchTypeEnum] = field(default=None)
    from_url: Optional[Url] = field(default=None)
    scrapped: bool = field(default=False)

    # def __attrs_post_init__(self):
    #     if self.from_url is not None:
    #         self.path = self.reconstruct_full_url(self.raw_path, self.from_url)
    #     self.url = Url(self.path if self.path else self.raw_path)

    def reconstruct_full_url(self, path: str, from_url: Url) -> str:
        full_path: str = path

        if from_url.domain in path:
            path = path.split(from_url.domain)[1]

        if path.startswith("./") or self.match_type == MatchTypeEnum.RELATIVE_URL:
            clean_path: str = path.split("./")[1]
            full_path = f"{from_url.base_url}/{clean_path}"
        elif path.startswith("/") or self.match_type == MatchTypeEnum.RELATIVE_URL_WITHOUT_POINT:
            full_path = f"{from_url.base_url}{path}"
        elif self.match_type == MatchTypeEnum.WITHOUT_SLASH:
            full_path = f"{from_url.base_url}/{path}"

        return full_path

    # @classmethod
    # def from_parsing_result(cls, parsing_result: ParsingResult) -> "FoundLink":
    #     match_type = MatchTypeEnum(parsing_result.raw_matched_type)
    #     url: str = parsing_result.raw_link
    #     from_url = Url(parsing_result.from_url) if parsing_result.from_url else None
    #     return cls(url, match_type=match_type, from_url=from_url)

    def __str__(self) -> str:
        return str(self.url)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __value: object) -> bool:
        if not hasattr(__value, "path"):
            return False

        return self.path == __value.path  # type: ignore

    def __ne__(self, __value: object) -> bool:
        if not hasattr(__value, "path"):
            return False

        return self.path == __value.path  # type: ignore

    def __hash__(self) -> int:
        return hash(f"{self.from_url}{self.raw_path}")


@define
class ScrapResult:
    scrapped_links: List[FoundLink]
    not_scrapped_links: List[FoundLink]
    iteration_count: int

    @property
    def links_count(self) -> int:
        return len(self.scrapped_links) + len(self.not_scrapped_links)


class Spiderer:
    MASTER_REGEX = r"(?P<url>(?<!<)(?P<match_type>[a-z]*\:\/\/|\.\/|\/)[0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\.\-\_\~\!\+\,\*\:\@\/]*)"
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

        self.initial_link: FoundLink = FoundLink(website_url, Url(website_url), match_type=MatchTypeEnum.STARTING_URL)
        self.session: requests.Session = requests.Session()
        self.maximum_depth: int = maximum_depth
        self.verbose: bool = verbose
        self.found_links: Dict[str, FoundLink] = dict()

    def get_or_create_link(self, url: str):
        found_link = self.found_links.get(url)
        if found_link is None:
            found_link = FoundLink(url, Url(url))
            self.found_links[url] = found_link
        return found_link

    @property
    def domain(self) -> str:
        return self.initial_link.url.domain

    def scrap(self) -> ScrapResult:
        scrapped_link: List[FoundLink] = list()
        to_be_scrapped: List[FoundLink] = [self.initial_link]
        depth_counter: int = 1
        while len(to_be_scrapped) > 0:
            if self.maximum_depth > 0 and depth_counter > self.maximum_depth:
                break
            depth_counter += 1

            current_link: FoundLink = to_be_scrapped.pop()
            current_link.scrapped = True
            scrapped_link.append(current_link)

            if current_link.url is None:
                Logger.warning(f"Could not parse {current_link.path}, could not identify url.")
                continue

            if current_link.url.domain != self.domain:
                continue

            found_links: List[FoundLink] = self.parse(current_link.url)
            for link in found_links:

                if link in scrapped_link:
                    continue

                if link in to_be_scrapped:
                    continue

                to_be_scrapped.append(link)

        scrap_result = ScrapResult(scrapped_link, to_be_scrapped, depth_counter - 1)

        return scrap_result

    def parse(self, website_url: Url) -> List[FoundLink]:

        Logger.info(f"parsing: {website_url}")
        try:
            response: requests.Response = self.session.get(website_url.path, verify=False)
        except ConnectionError:
            Logger.error("The target is not accessible")
            return []

        website_url.status_code = response.status_code

        # response.encoding = "utf-8"
        if response.apparent_encoding is None:
            return []

        if response.status_code != 200:
            return []

        urls: Set[FoundLink] = self._find_urls(response)
        return list(urls)

    def _find_urls(self, response: requests.Response) -> Set[FoundLink]:
        links: Set[FoundLink] = set()
        links.update(self._parse_with_tag(response))
        links.update(self._parse_with_regex(response))
        # Filter by exts
        return links

    def _parse_with_tag(self, response: requests.Response) -> List[FoundLink]:
        links: List[FoundLink] = []
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

            raw_link: str
            raw_matched_type: str = ""

            if regexed_link is None:
                raw_link = raw_link
            else:
                raw_link = regexed_link.group("url")
                raw_matched_type = regexed_link.group("match_type")

            if not raw_link or not Url.validate_url(raw_link):
                Logger.debug(f"Url {raw_link} not valid.")
                continue
            # en pleine refonte de ca
            # self.get_or_create_link(raw_link)
            final_link = FoundLink(raw_link, Url(raw_link), match_type=raw_matched_type, from_url=Url(response.url))

            links.append(final_link)
        return links

    def _parse_with_regex(self, response: requests.Response) -> List[FoundLink]:
        regex: re.Pattern[str] = re.compile(Spiderer.MASTER_REGEX, re.MULTILINE)
        response_text = response.text
        response_text = re.sub(r".*?(?:<\!DOCTYPE).*(?:>)", "", response_text)
        raw_links_list: Iterator[re.Match[str]] = regex.finditer(response_text)
        parsing_result_list: List[ParsingResult] = []

        for raw_link in raw_links_list:
            if not raw_link.group("url") or not Url.validate_url(raw_link.group("url")):
                continue

            try:
                found_link = ParsingResult.from_master_regex(raw_link, response.url)
                parsing_result_list.append(found_link)
            except Exception:
                Logger.error(f"An error occured parsing {raw_link} from object {response.url}")

        links_obj: List[FoundLink] = []
        for raw_link in parsing_result_list:
            try:
                new_link = FoundLink.from_parsing_result(raw_link)
            except Exception:
                continue
            links_obj.append(new_link)

        return links_obj


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
