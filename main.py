from enum import Enum
import os
from pathlib import Path
from shutil import ExecError
import string
from typing import Any, Iterator, List, Optional, Self, Set, Tuple, TypeAlias, cast
from attr import validate
import bs4
from outcome import Value
import requests
import re
import urllib3
import validators
import validators.uri

"""
TODO:
 - Parser les balises <a> dans le site
 - Gérer les errurs 404 etc
"""

urllib3.disable_warnings()


class MatchTypeEnum(Enum):
    HTTP = "http"
    SCHEME_SEPARATOR = "://"
    RELATIVE_URL = "./"
    RELATIVE_URL_WITHOUT_POINT = "/"
    WITHOUT_SLASH = ""

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if str(value).startswith("http"):
            return MatchTypeEnum.HTTP

        if value is None:
            return MatchTypeEnum.WITHOUT_SLASH

        return super()._missing_(value)


class Url:

    ALLOWED_URL_CHARACTERS = string.ascii_lowercase + string.digits + "-" + "_" + "." + "~" + "%"

    def __init__(self, path: str) -> None:
        if not Url.validate_url(path):
            raise ValueError(f"The url {path} is not valid.")
        self.raw_path: str = path

        self.path: str = self._clean_path_from_parameters(path)
        self.protocol: str = self._parse_protocol(self.path)
        self.tld: str = self._parse_tld(self.path)
        self.status_code: int = 0

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

    def _parse_protocol(self, path: str) -> str:
        protocol: str = path.split("://")[0]
        if not protocol:
            raise ValueError("Protocol is None, URL is not valid.")
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


class ParsingResult:
    def __init__(self, raw_link: str, raw_matched_type: str, from_url: str) -> None:
        self.raw_link: str = raw_link
        self.raw_matched_type: str = raw_matched_type
        self.from_url: str = from_url

    @classmethod
    def from_master_regex(cls, raw_parsing_result: re.Match[str], from_url: str):
        obj = cls(raw_parsing_result.group("url"),raw_parsing_result.group("match_type"), from_url)
        return obj


class Link:
    def __init__(
        self,
        path: str,
        match_type: Optional[MatchTypeEnum] = None,
        from_url: Optional[str] = None,
    ) -> None:
        self.path: str = path
        self.from_url = from_url  # type: ignore
        self.match_type: Optional[MatchTypeEnum] = match_type

        if self.from_url is not None:
            if isinstance(self.from_url, str):
                self.from_url: Url = Url(self.from_url)

            path = self.reconstruct_full_url(path, self.from_url)

        else:
            if match_type is MatchTypeEnum.RELATIVE_URL:
                raise ValueError("from_url parameter cannot be set to None when using a relative URL")

        self.url: Url = Url(path)

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

    @classmethod
    def from_parsing_result(cls, parsing_result: ParsingResult) -> Self:
        match_type = MatchTypeEnum(parsing_result.raw_matched_type)
        url: str = parsing_result.raw_link
        return cls(url, match_type, parsing_result.from_url)

    @classmethod
    def from_full_url(cls, url: Url) -> Self:
        return cls(url.path)

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
        return hash(f"{self.from_url}{self.path}")


class Spiderer:
    MASTER_REGEX = (
        r"(?P<url>(?<!<)(?P<match_type>[a-z]*\:\/\/|\.\/|\/)[0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\.\-\_\~\!\+\,\*\:\@\/]*)"
    )
    r"""
    MASTER_REGEX = (
        r"(?P<url>(?P<match_type>[a-z]*\:\/\/|\.\/|\/)?[0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\.\-\_\~\!\+\,\*\:\@\/]*)"
    )
    """

    def __init__(self, website_url: str) -> None:
        self.website_url: Url = Url(website_url)
        self.session: requests.Session = requests.Session()

    @property
    def domain(self) -> str:
        return self.website_url.domain

    def scrap(self) -> set[Url]:
        scrapped_url: List[Url] = list()
        to_be_scrapped: List[Link] = list(set([Link.from_full_url(self.website_url)]))
        while len(to_be_scrapped) > 0:
            current_link: Link = to_be_scrapped.pop()
            scrapped_url.append(current_link.url)

            if current_link.url.domain != self.domain:
                continue

            found_links: List[Link] = self.parse(current_link.url)
            print("J'ai trouvé", len(found_links), "via", current_link.url, found_links)
            sorted_list = sorted(scrapped_url, key=lambda a: a.path)
            # print("Voici la liste des scrappés", sorted_list)
            for link in found_links:
                url: Url = link.url

                print(f"url in scrapped ul {url} ? ", url in scrapped_url)
                if url in scrapped_url:
                    continue

                print(f"link in to be scrapped {link}?", link in to_be_scrapped)
                if link in to_be_scrapped:
                    continue

                to_be_scrapped.append(link)

        print(f"Parsing done. Found {len(scrapped_url)} urls !\n")

        return scrapped_url

    def parse(self, website_url: Url) -> List[Link]:
        print(f"parsing: {website_url}")
        try:
            response: requests.Response = self.session.get(website_url.path, verify=False)
        except ConnectionError:
            raise ValueError("The target is not accessible")

        response.encoding = "utf-8"
        response.url = Url(response.url)  # type: ignore
        print(response.status_code, response.status_code != 200)
        if response.status_code != 200:
            website_url.status_code = response.status_code
            response.url.status_code = response.status_code
            return []
        urls: list[Link] = self._find_urls(response)
        return urls

    def _find_urls(self, response: requests.Response) -> List[Link]:
        links: List[Link] = []
        links: List[Link] = self._parse_with_regex(response)
        links.extend(self._parse_with_tag(response))
        return links

    def _parse_with_tag(self, response: requests.Response) -> List[Link]:
        links: List[Link] = []
        bs4_response = bs4.BeautifulSoup(response.text, "html.parser")
        a_tags = bs4_response.find_all()
        raw_links: List[str] = []
        for a_tag in a_tags:
            if a_tag.has_attr("href"):
                raw_links.append(a_tag["href"])
            if a_tag.has_attr("src"):
                raw_links.append(a_tag["src"])

        for raw_link in raw_links:
            regex: re.Pattern[str] = re.compile(Spiderer.MASTER_REGEX, re.MULTILINE)
            regexed_link: re.Match[str] | None = regex.match(raw_link)

            raw_link: str
            raw_matched_type: str = ""

            if regexed_link is None:
                print("ALED", raw_link, regexed_link)
                raw_link = raw_link
            else:
                raw_link = regexed_link.group("url")
                raw_matched_type = regexed_link.group("match_type")

            if not raw_link or not Url.validate_url(raw_link):
                print("ALED mais là ", raw_link, regexed_link, raw_link, Url.validate_url(raw_link))
                continue

            parsing_result = ParsingResult(raw_link, raw_matched_type, response.url)
            final_link = Link.from_parsing_result(parsing_result)
            # input(f"Nouveau lien {final_link.url} from {response.url}")
            links.append(final_link)

        return links

    def _parse_with_regex(self, response: requests.Response) -> List[Link]:
        regex: re.Pattern[str] = re.compile(Spiderer.MASTER_REGEX, re.MULTILINE)
        response_text = response.text
        response_text = re.sub(r".*?(?:<\!DOCTYPE).*(?:>)", "", response_text)
        raw_links_list: Iterator[re.Match[str]] = regex.finditer(response_text)
        parsing_result_list: List[ParsingResult] = []

        for raw_link in raw_links_list:
            if not raw_link.group("url") or not Url.validate_url(raw_link.group("url")):
                continue
            try:
                parsing_result_list.append(ParsingResult.from_master_regex(raw_link, response.url))
            except Exception:
                input(f"Probleme avec {raw_link} depuis {response.url}")

        links_obj: List[Link] = []
        for raw_link in parsing_result_list:
            try:
                new_link = Link.from_parsing_result(raw_link)
            except Exception:
                continue
            # input(f"Nouveau lien {new_link.url} from {response.url} {raw_link.raw_link}")
            links_obj.append(new_link)

        return links_obj


def main():
    urls = Spiderer("https://robinzmuda.fr/").scrap()
    for url in urls:
        print(url)
    # urls dupliques, des /
    # des urls scannes deux fois
    # faut faire un putain de manager


if __name__ == "__main__":
    main()
