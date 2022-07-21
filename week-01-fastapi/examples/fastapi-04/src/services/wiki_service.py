from typing import Optional

from bs4 import BeautifulSoup

from src.repos.wiki_repo import WikiRepo
from src.services.scrapper import Scrapper


def get_first_paragraph(html_doc: str):
    soup = BeautifulSoup(html_doc, 'html.parser')
    tags = soup.find_all('p')
    result = None
    if len(tags):
        result = tags[0].text
    return result


class WikiService:
    def __init__(self, scrapper: Scrapper, wiki_repo: WikiRepo):
        self._wiki_repo = wiki_repo
        self._scrapper = scrapper

    def process_description(self, title: str) -> Optional[str]:
        html_text = self._scrapper.get_desc_by_title(title)

        description = get_first_paragraph(html_text) if html_text else None

        if description:
            self._wiki_repo.save(title, description)
        return description

    def get_description(self, title: str):
        return self._wiki_repo.get(title)
