import sys
import os
import re
from bs4 import BeautifulSoup

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

info_dict: dict = dict()
with open(os.path.join(os.path.dirname(__file__), "../nesvor/version.py")) as fp:
    exec(fp.read(), info_dict)

print("post processing starts")
BASE = sys.argv[1]

cmd_set = {
    "nesvor",
    "reconstruct",
    "register",
    "svr",
    "correct-bias-field",
    "segment-stack",
    "segment-volume",
    "assess",
    "sample-volume",
    "sample-slices",
}

if os.path.exists(BASE):
    for root, dirs, files in os.walk(BASE):
        for f in files:
            if f.endswith(".html"):
                p = os.path.join(root, f)
                print(p)
                with open(p, "r") as read_file:
                    html_source = read_file.read()

                soup = BeautifulSoup(html_source, "html.parser")
                html_tags = soup.find_all(["kbd"])
                for tag in html_tags:
                    l = re.sub("[^0-9a-zA-Z]+", "-", tag.text).split("-")
                    l = [it for it in l if it]
                    tag.attrs["id"] = "-".join(l)
                    # h = soup.new_tag("h4")
                    # a = soup.new_tag(
                    #     "a",
                    #     attrs={
                    #         "class": "headerlink",
                    #         "href": "#" + tag.attrs["id"],
                    #         "title": "Permalink to this heading",
                    #     },
                    # )
                    # a.string = "#"  # "¶"
                    # h.string = tag.string
                    # h.append(a)
                    # tag.clear()
                    # tag.append(h)

                html_tags = soup.find_all("a")
                for tag in html_tags:
                    if (
                        (tag.parent is not None)
                        and tag.parent.has_attr("class")
                        and ("toctree" in tag.parent["class"][0])
                    ):
                        continue
                    s = tag.string
                    if s:
                        s = str(s)
                        code_tag = None
                        if s.startswith("--"):
                            code_tag = soup.new_tag(
                                "code",
                                attrs={
                                    "class": "xref py py-attr docutils literal notranslate"
                                },
                            )
                        if s in cmd_set:
                            code_tag = soup.new_tag(
                                "code",
                                attrs={
                                    "class": "xref py py-class docutils literal notranslate"
                                },
                            )
                        if code_tag is not None:
                            tag.clear()
                            span_tag = soup.new_tag(
                                "span",
                                attrs={"class": "pre", "style": "white-space:nowrap;"},
                            )
                            span_tag.string = s
                            code_tag.append(span_tag)
                            tag.append(code_tag)
                            # print(tag)

                html_string = str(soup)
                html_string = html_string.replace(
                    "-version-placeholder-", info_dict["__version__"]
                )

                with open(p, "w") as save_file:
                    save_file.write(html_string)


print("post processing finished")
