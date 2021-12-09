#
#
#   Utils
#
#

import markdown

from bs4 import BeautifulSoup


def md_to_text(md):
    html = markdown.markdown(str(md))
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()
