import re

PAGE_NUMBER_LINE = re.compile(r"^\d{1,3}$")
CLAUSE_MARKER = re.compile(r"^\(([a-zA-Z]{1,4}|\d{1,2})\)$")
DOT_LEADER = re.compile(r"\.{4,}")


def clean_page_text(text: str) -> str:
    """Return cleaned text for one page, or "" if the page has no real content."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]  # drop blank lines

    # 1. the first line is often just the page number
    if lines and PAGE_NUMBER_LINE.match(lines[0]):
        lines = lines[1:]

    # 2. drop dotted table-of-contents lines
    lines = [line for line in lines if not DOT_LEADER.search(line)]

    # 3. glue clause markers like "(2)" onto the sentence that follows them
    merged = []
    prefix = ""
    for line in lines:
        if CLAUSE_MARKER.match(line):
            prefix += line + " "
        else:
            merged.append(prefix + line)
            prefix = ""
    if prefix:
        merged.append(prefix.strip())
    result = "\n".join(merged)

    # 4. pages that only say "section deleted" carry no information
    if "Deleted with effect" in result and len(result) < 200:
        return ""
    return result