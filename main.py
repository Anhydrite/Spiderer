from enum import Enum
from typing import Any, List, Optional, Self, Set, Tuple, TypeAlias, cast
import requests
import re
import urllib3
import validators


urllib3.disable_warnings()

URL: TypeAlias = str
RawMatchType = str
RawLinkParsingResult: TypeAlias = Tuple[URL, RawMatchType]


class MatchTypeEnum(Enum):
    HTTP = "http"
    SCHEME_SEPARATOR = "://"
    RELATIVE_URL = "./"

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if str(value).startswith("http"):
            return MatchTypeEnum.HTTP

        return super()._missing_(value)


class Url:
    def __init__(self, path: str) -> None:
        if not Url.validate_url(path):
            raise ValueError(f"The url {path} is not valid.")
        self.raw_path: str = path

        self.path: str = self._clean_path_from_parameters(path)
        self.protocol: str = self._parse_protocol(self.path)
        self.tld: str = self._parse_tld(self.path)

    def _parse_tld(self, path: str) -> str:
        temp_tld: str = path.split(".")[-1]
        small_temp_tld: str = temp_tld.lower()
        text_regex: re.Match[str] | None = re.match(r"[a-z]*", small_temp_tld, re.NOFLAG)
        if not text_regex:
            raise ValueError(f"TLD is not valid: {small_temp_tld}")

        tld: str = text_regex.group()

        return tld

    def _clean_path_from_parameters(self, path: str) -> str:
        clean_path: str = path.split("?")[0]
        return clean_path

    @property
    def base_url(self) -> str:
        base_url: str = "".join(self.path.partition(self.tld)[:2])
        return base_url

    @property
    def domain(self) -> str:
        domain: str = self.base_url.split("://")[1]

        domain_regex: re.Pattern[str] = re.compile(r"[a-z\.]*")
        domain = domain_regex.match(domain).string  # type: ignore
        return domain

    @staticmethod
    def validate_url(url: str) -> bool:
        return validators.url(url)  # type: ignore

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
    def __init__(self, raw_parsing_result: RawLinkParsingResult, from_url: str) -> None:
        self.raw_parsing_result: RawLinkParsingResult = raw_parsing_result
        self.raw_link: str = raw_parsing_result[0]
        self.raw_matched_type: str = raw_parsing_result[1]
        self.from_url: str = from_url


class Link:
    def __init__(self, path: str, match_type: Optional[MatchTypeEnum] = None, from_url: Optional[str] = None) -> None:
        self.path: str = path
        self.from_url = from_url  # type: ignore

        if self.from_url is not None:
            if isinstance(self.from_url, str):
                self.from_url: Url = Url(self.from_url)

            path = Link.reconstruct_full_url(path, self.from_url)

        else:
            if match_type is MatchTypeEnum.RELATIVE_URL:
                raise ValueError("from_url parameter cannot be set to None when using a relative URL")

        self.url: Url = Url(path)
        self.match_type: Optional[MatchTypeEnum] = match_type

    @staticmethod
    def reconstruct_full_url(path: str, from_url: Url) -> str:
        full_path: str = path

        if path.startswith("./"):
            clean_path: str = path.split("./")[1]
            full_path = f"{from_url.base_url}/{clean_path}"
        elif path.startswith("/"):
            full_path = f"{from_url.base_url}{path}"

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
        r"(([a-z]*\:\/\/|\.\/)[0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\.\-\_\~\!\+\,\*\:\@\/]*)"
    )

    def __init__(self, website_url: str) -> None:
        self.website_url: Url = Url(website_url)
        self.session: requests.Session = requests.Session()

    @property
    def domain(self) -> str:
        return self.website_url.domain

    def scrap(self) -> set[Url]:
        scrapped_url: Set[Url] = set()
        to_be_scrapped: Set[Link] = set([Link.from_full_url(self.website_url)])
        while len(to_be_scrapped) > 0:
            current_link: Link = to_be_scrapped.pop()
            scrapped_url.add(current_link.url)

            if current_link.url.domain != self.domain:
                continue

            found_links: List[Link] = self.parse(current_link.url)

            for link in found_links:
                url: Url = link.url

                if url in scrapped_url:
                    continue

                if link in to_be_scrapped:
                    continue

                to_be_scrapped.add(link)

        print(f"Parsing done. Found {len(scrapped_url)} urls !\n")

        return scrapped_url

    def parse(self, website_url: Url) -> list[Link]:
        print(f"parsing: {website_url}")
        try:
            response: requests.Response = self.session.get(website_url.path, verify=False)
        except ConnectionError:
            raise ValueError("The target is not accessible")

        response.encoding = "utf-8"
        response.url = Url(response.url)  # type: ignore
        urls: list[Link] = self._find_urls(response)
        return urls

    def _find_urls(self, response: requests.Response) -> list[Link]:
        links: list[ParsingResult] = self._parse_with_regex(response)

        links_obj: List[Link] = []
        for raw_link in links:
            links_obj.append(Link.from_parsing_result(raw_link))

        return links_obj

    def _parse_with_regex(self, response: requests.Response) -> list[ParsingResult]:
        regex: re.Pattern[str] = re.compile(Spiderer.MASTER_REGEX, re.MULTILINE)
        raw_links_list: List[RawLinkParsingResult] = regex.findall(response.text)
        parsing_result_list: List[ParsingResult] = [
            ParsingResult(raw_link, response.url) for raw_link in raw_links_list
        ]

        return parsing_result_list


def main():
    urls = Spiderer("https://twitter.com").scrap()
    for url in urls:
        print(url)
    # urls dupliques, des /
    # des urls scannes deux fois
    # faut faire un putain de manager


if __name__ == "__main__":
    main()
