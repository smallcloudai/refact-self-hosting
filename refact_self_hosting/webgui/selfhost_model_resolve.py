import re
from typing import Tuple


def resolve_model(model_name: str, cursor_file: str, function: str) -> Tuple[str, str]:
    """
    Allow client to specify less in the model string, including an empty string.
    """
    m_everything = model_name.split("/")
    m_company, m_size, m_specialization, m_version = tuple(m_everything + ["", "", "", ""])[:4]

    if m_company == "CONTRASTcode":
        if function == "":  # true for plain completion (not diff)
            pass
        else:
            regex = r"^(highlight|infill|diff-anywhere|diff-atcursor|diff-selection|edit-chain)$"
            m_match = re.fullmatch(regex, function)
            if not m_match:
                return "", "function must match %s" % regex
        if not m_specialization and cursor_file:
            # file extension -> specialization here
            pass
        if not m_size:
            m_size = "3b"
        if not m_specialization:
            m_specialization = "multi"

    elif m_company == "starcoder":
        if not m_size:
            m_size = "15b"

    elif m_company == "codet5":
        if not m_size:
            m_size = "plus-770M-python"

    else:
        m_company = "CONTRASTcode"
        m_size = "3b"
        m_specialization = "multi"

    result = "/".join([m_company, m_size, m_specialization, m_version])
    result = result.rstrip("/")
    return result, ""
