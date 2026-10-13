"""Deterministic security checks on resume text. Flags only — never a rejection.

- Injection phrases: regexes in 8 categories (override: "ignore (all|any|the)? (previous|
  prior|above) instructions"; addressed_to_ai: "(note|message) to (the )?(AI|LLM|model|
  screener|ATS)"; role_play: "you are now", "act as"; score_demand: "(rate|score|rank)
  (this|the|me)? ?candidate .{0,20}(100|10/10|highest|top)"; verdict_claim: "this
  candidate is (the )?(perfect|ideal|best)"; all_requirements: "(meets|satisfies) all (the )?
  (requirements|criteria)"; chat_markup: "<\\|im_start\\|>", "[INST]", "### (system|
  instruction)", "</?system>"; concealment: "do not (mention|reveal|disclose) this").
  Case-insensitive, run on the visible text, the hidden text and the tag payload
  separately. Benign lines such as "Built an LLM evaluation harness to score model
  outputs" or "Designed system prompts and guardrails" must NOT match (both are tested).
  Any match -> ``injection_phrase`` flag, severity "review", detail quoting <= 120 chars.
- Tag payload non-empty -> ``unicode_tag_payload`` (review).
- Invisible characters removed by normalize (zero-width/bidi/blank fillers) >= 10 ->
  ``invisible_characters`` (info); >= 50 -> review.
- Mixed script: a single word mixing Latin with Cyrillic, Greek, Armenian, Cherokee or
  Lisu letters (homoglyph spoofing, e.g. "Kubernеtes" with Cyrillic е) -> ``mixed_script``
  (review when >= 3 words, else info). Whole words in a single non-Latin script are fine.

How the patterns stay quiet on ordinary resumes (they are flags, so recall matters more
than precision, but a recruiter's or ML engineer's resume must not trip them):
- Verbs are matched in their base form only ("Ignored flaky tests", "Ranked first",
  "Acted as Scrum Master" never match); score demands need a target ("this candidate",
  "me", "my resume"); verdict and all-requirements claims need a third-person subject
  ("this candidate ...", or "The candidate ..." starting a clause), so first-person
  cover-letter self-promotion ("I meet all the requirements") is not flagged.
- "act as" only counts when followed by an AI role ("act as an unbiased AI recruiter
  that ..."); "Act as primary point of contact", "Act as AI champion" and "act as if you
  own it" don't match.
- A chat-style label ("System prompt:", "Developer message:") counts only at the start of
  a line and when an instruction follows the colon ("System note: you must ..."), so a
  GenAI engineer's "Designed system prompts: few-shot templates" and "Developer notes:
  documented API usage" pass; "system override:" counts anywhere. Angle-bracket tags
  (<system>, <instructions>) do not count after Built/Designed/Implemented/Used/Wrote/
  Added or on a line that also mentions <user>, <assistant>, templates, tags or blocks.
  An override quoted as a red-team target ("against 'ignore previous instructions'
  jailbreaks") does not count, and "ignore prior context" is not an override.
- "Candidate is pre-screened by phone" is a recruiter's bullet: pre-screened/pre-vetted/
  verified count as a verdict only at the end of the clause ("This candidate is
  pre-screened.").
- Before matching, text is folded: format/private-use characters dropped, NFKC,
  look-alike letters (Cyrillic, Greek, Armenian, Cherokee, Lisu, Latin small capitals)
  mapped to Latin, letter-spaced words joined ("I g n o r e" -> "Ignore"), and a second
  copy with leetspeak digits mapped to letters ("1gn0re" -> "ignore") is matched too.
  Snippets are quoted from the folded text (digits kept).
- Mixed-script words count only when a non-Latin letter is a Latin look-alike (Cyrillic
  "е", Greek "ο", ...), so "μs" or "μServices" are not flagged. Severity counts every
  occurrence (one spoofed keyword repeated three times is a review); ``mixed_script_words``
  lists distinct words.
- The two core phrases (ignore previous instructions; rate this candidate) are also
  matched in Spanish, French, German, Portuguese, Italian, Chinese and Hindi.
- One ``injection_phrase`` flag per source text (visible, hidden, tag payload) quoting the
  first match; ``injection_phrases`` lists every distinct matched snippet (<= 120 chars).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from functools import cache

from applicant_ranking.domain import Flag, FlagCode, SecurityScan

MAX_QUOTE_CHARS = 120
MAX_PHRASES = 20
MAX_LISTED_WORDS = 5
INVISIBLE_INFO_THRESHOLD = 10
INVISIBLE_REVIEW_THRESHOLD = 50
MIXED_SCRIPT_REVIEW_THRESHOLD = 3
INVISIBLE_CATEGORIES = ("zero_width", "bidi_controls", "blank_fillers")

# Letters of other scripts that render like Latin letters (subset of Unicode
# confusables.txt that matters for English keywords), plus Latin small capitals.
LOOKALIKES: Mapping[str, str] = {
    # Cyrillic lower
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x", "ѕ": "s", "і": "i", "ї": "i",
    "ј": "j", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ӏ": "l", "ɡ": "g",
    # Cyrillic upper
    "А": "A", "В": "B", "Е": "E", "Ё": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "Ѕ": "S", "І": "I", "Ї": "I",
    "Ј": "J", "Ԛ": "Q", "Ԝ": "W", "Ӏ": "I",
    # Greek lower
    "α": "a", "ο": "o", "ρ": "p", "ι": "i", "κ": "k", "ν": "v", "τ": "t", "υ": "u",
    "χ": "x", "γ": "y",
    # Greek upper
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    # Armenian
    "օ": "o", "ո": "n", "ս": "u", "ց": "g", "հ": "h",
    "Օ": "O", "Ս": "U", "Տ": "S", "Լ": "L",
    # Cherokee
    "Ꭰ": "D", "Ꭱ": "R", "Ꭲ": "T", "Ꭺ": "A", "Ꭶ": "G", "Ꭹ": "Y",
    "Ꭻ": "J", "Ꭼ": "E", "Ꮍ": "H", "Ꮃ": "W", "Ꮇ": "M", "Ꮋ": "H",
    "Ꮐ": "G", "Ꮒ": "h", "Ꮓ": "Z", "Ꮢ": "R", "Ꮪ": "S", "Ꮯ": "C",
    "Ꮲ": "P", "Ꮩ": "V", "Ꮶ": "K", "Ꮮ": "L",
    # Lisu
    "ꓮ": "A", "ꓐ": "B", "ꓚ": "C", "ꓓ": "D", "ꓰ": "E", "ꓝ": "F",
    "ꓖ": "G", "ꓧ": "H", "ꓲ": "I", "ꓙ": "J", "ꓗ": "K", "ꓡ": "L",
    "ꓟ": "M", "ꓠ": "N", "ꓳ": "O", "ꓑ": "P", "ꓣ": "R", "ꓢ": "S",
    "ꓔ": "T", "ꓴ": "U", "ꓦ": "V", "ꓪ": "W", "ꓫ": "X", "ꓬ": "Y",
    "ꓜ": "Z",
    # Latin small capitals and phonetic letters (NFKC leaves them alone)
    "ᴀ": "a", "ʙ": "b", "ᴄ": "c", "ᴅ": "d", "ᴇ": "e", "ꜰ": "f",
    "ɢ": "g", "ʜ": "h", "ɪ": "i", "ᴊ": "j", "ᴋ": "k", "ʟ": "l",
    "ᴍ": "m", "ɴ": "n", "ᴏ": "o", "ᴘ": "p", "ᴩ": "p", "ʀ": "r",
    "ꜱ": "s", "ᴛ": "t", "ᴜ": "u", "ᴠ": "v", "ᴡ": "w", "ʏ": "y",
    "ᴢ": "z",
}
_FOLD_TABLE = str.maketrans(LOOKALIKES)
_SPOOF_SCRIPTS = frozenset({"CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE", "LISU"})
_LETTER_SPACED_RE = re.compile(r"\b(?:[A-Za-z][^\S\n]){2,}[A-Za-z]\b")
_LEET_TOKEN_RE = re.compile(r"\b[A-Za-z0-9]{4,}\b")
_LEET_TABLE = str.maketrans("013457", "oieast")

# --- injection patterns --------------------------------------------------------------------

# Unambiguous references to an automated screener.
_AI = (
    r"(?:a\.i\.|ai|llms?|chat\s?gpt|gpt(?:-?\d[\w.]*)?|gemini|copilot|ats|screeners?|"
    r"(?:large\s+)?language\s+models?|chatbots?|"
    r"ai[\s-](?:screeners?|recruiters?|reviewers?|assistants?|models?|systems?|agents?|"
    r"evaluators?|tools?|bots?)|"
    r"(?:resume|cv|applicant)\s+(?:screeners?|screening\s+(?:tools?|systems?|bots?)|"
    r"parsers?|tracking\s+systems?)|"
    r"screening\s+(?:tools?|systems?|bots?|software|ai)|hiring\s+(?:ai|bots?|algorithms?)|"
    r"recruit(?:er|ing)\s+(?:ai|bots?|tools?|software))(?![\w'’-])"
)
# Generic words that only count in an unmistakable address ("Note to the model: ...").
_AI_WEAK = r"(?:models?|bots?|assistants?|systems?|reviewers?|evaluators?|graders?|agents?)"
_TARGET = (
    r"(?:(?:this|the)\s+(?:candidate|applicant|resume|cv|profile|application|person|"
    r"individual)(?!s\b|['’]|\s+(?:pool|list|slate|pipeline|funnel|database|experience))|"
    r"me|my\s+(?:resume|cv|application|profile)|him|her)"
)
_HIGH = (
    r"(?:\b(?:100|10\s*/\s*10|5\s*/\s*5|100\s*/\s*100|highest|highly|top|first|maximum|max|"
    r"perfect|excellent|best|strongest|tier\s*a|a[-\s]tier|qualified)\b|#\s*1\b|"
    r"\bnumber\s+one\b)"
)
_SUBJECT = (
    r"(?:\bthis\s+(?:candidate|applicant|resume|cv|profile|person|individual)|"
    r"(?:^|(?<=[.!?:;\-–—(\[\"“]))\s*(?:the\s+)?(?:candidate|applicant))"
)
_CLAUSE_END = r"(?=\s*(?:[,.;:!)]|$|that\b|who\b|which\b|and\b))"
# What follows a chat-style label when it is an instruction rather than a bullet.
_IMPERATIVE = (
    r"(?:you\b|your\b|ignore|disregard|forget|always|never|do\s+not|don['’]t|rate|score|"
    r"rank|grade|give|treat|output|return|respond|answer|assume|act\b|be\b|please\b|hire|"
    r"approve|recommend|summari[sz]e|consider|read\b|this\s+(?:candidate|applicant|resume)|"
    r"the\s+(?:candidate|applicant))"
)
_NOT_QUOTED = r"(?<!['\"“‘])"
_NOT_AFTER_BUILD_VERB = (r"(?<!Built )(?<!Designed )(?<!Implemented )(?<!Used )(?<!Wrote )"
                         r"(?<!Added )(?<!Created )(?<!Developed )")

INJECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "override": (
        rf"{_NOT_QUOTED}\b(?:ignore|disregard|forget|override|bypass|skip|neglect)\s+"
        r"(?:(?:all|any|the|of|your|my|these|those)\s+){0,3}"
        r"(?:previous|prior|above|earlier|preceding|foregoing|system)\s+(?:\w+\s+)?"
        r"(?:instructions?|prompts?|directions?|directives?|guidelines|rules|commands?|"
        r"criteria|rubric|guidance|programming)\b",
        rf"{_NOT_QUOTED}\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:of\s+)?"
        r"(?:the|your|any)\s+(?:\w+\s+)?(?:instructions|rubric|system\s+prompt|guardrails|"
        r"programming|directives)\b",
        rf"{_NOT_QUOTED}\b(?:ignore|disregard|forget)\s+(?:all\s+|everything\s+)(?:you\s+"
        r"(?:were|have\s+been)\s+told|(?:that\s+)?(?:came\s+)?before|above)",
        # Spanish, French, Portuguese, Italian
        r"\bignor(?:a|ar|ad|e|em|en|er|ez|ate|are|i)\s+(?:(?:todas|toutes|tutte)\s+)?"
        r"(?:(?:las|les|le|as)\s+)?(?:instrucciones|instructions|instru[çc][õo]es|istruzioni)"
        r"\s+(?:anteriores|previas|pr[ée]c[ée]dentes|precedenti)\b",
        # German
        r"\bignorier(?:e|en|t|st)?(?:\s+sie)?\s+(?:alle\s+)?(?:vorherigen|bisherigen|"
        r"vorangegangenen|fr[üu]heren)\s+anweisungen\b",
        # Chinese, Hindi
        r"忽略(?:之前|以上|先前|前面|上述)?的?(?:所有|全部)?(?:指令|指示|说明|提示)",
        r"निर्देशों\s+को\s+(?:अनदेखा|नज़रअंदाज़|नजरअंदाज)\s+कर",
    ),
    "addressed_to_ai": (
        rf"\b(?:note|message|memo)\s+to\s+(?:the\s+|any\s+|all\s+|an?\s+)?{_AI}",
        rf"\b(?:note|message|memo|instructions?|attention|reminder|notice)\s+(?:to|for)\s+"
        rf"(?:the\s+|any\s+|all\s+|an?\s+)?(?:{_AI}(?:\s+\w+){{0,2}}|{_AI_WEAK})\s*:",
        rf"\b(?:dear|hey|hi|hello|greetings|attention|attn)\s*,?\s+(?:the\s+)?{_AI}",
        rf"\bto\s+(?:the|any|all|every)\s+(?:{_AI}|{_AI_WEAK})\s+(?:that\s+is\s+|who\s+is\s+)?"
        r"(?:reading|reviewing|evaluating|screening|parsing|processing|scoring|analy[sz]ing)"
        r"\s+(?:this|my)\b",
        rf"\bif\s+you\s+are\s+(?:an?\s+)?(?:{_AI}|automated\s+(?:system|screener|reviewer)|"
        r"bot|machine)",
        rf"<!--\s*(?:{_AI}|{_AI_WEAK})\b[^\n]{{0,80}}-->",
    ),
    "role_play": (
        r"\byou\s+are\s+now\b",
        r"\byou['’]re\s+now\b",
        r"\bfrom\s+now\s+on\s*,?\s+you\b",
        r"\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b",
        rf"\bact\s+as\s+(?:an?\s+|the\s+|my\s+)?(?:unbiased\s+|neutral\s+|helpful\s+|"
        rf"friendly\s+|expert\s+)?{_AI}{_CLAUSE_END}",
        r"\bact\s+as\s+(?:if|though)\s+(?:you\s+(?:are|were)\s+(?:an?|the)\s+|"
        r"this\s+candidate|the\s+candidate)",
        r"^[^\S\n]*new\s+(?:system\s+)?(?:instructions?|rules|prompt)\s*:",
        rf"\bnew\s+(?:system\s+)?(?:instructions?|rules|prompt)\s*:\s*(?={_IMPERATIVE})",
    ),
    "score_demand": (
        rf"\b(?:rate|score|rank|grade|mark)\s+{_TARGET}\b[^.\n]{{0,20}}?{_HIGH}",
        rf"\b(?:give|assign|award|output|return|respond\s+with)\s+"
        rf"(?:{_TARGET}\s+)?(?:an?\s+|the\s+)?(?:(?:maximum|max|highest|perfect|top|full|best)"
        r"\s+(?:possible\s+)?(?:score|rating|marks|points|tier|ranking|rank|grade)\b|"
        r"(?:fit[_\s]?)?(?:score|rating)\s*(?:of|to|=|:|as)?\s*(?:100|10\s*/\s*10|5\s*/\s*5)\b)",
        r"\b(?:this|the)\s+(?:candidate|applicant|resume|cv)\s+(?:should|must|deserves\s+to|"
        r"needs\s+to|has\s+to)\s+(?:be\s+|receive\s+|get\s+)?(?:rated|ranked|scored|given|"
        r"placed|marked|moved|advanced|shortlisted|hired|selected|prioriti[sz]ed|"
        r"fast[-\s]?tracked|approved|recommended)\b",
        rf"\b(?:mark|rank|rate|score|classify|label|tag|flag|list)\s+(?:{_TARGET}\s+)?as\s+"
        r"(?:an?\s+|the\s+)?(?:(?:top|best|strongest|ideal|perfect|excellent|highest[-\s]rated|"
        r"first)[-\s]?(?:candidate|applicant|choice|pick|match|fit|hire)\b|"
        r"must[-\s]hire\b)",
        rf"\b(?:assign|give|award|grant)\s+(?:{_TARGET}\s+)?(?:an?\s+|the\s+)?"
        r"(?:tier\s*a|a[-\s]tier|top\s+tier)\b",
        r"\b(?:overall_fit|fit_score|overall\s+fit)\b\s*[:=]\s*[\"']?\s*(?:100|9\d|exceeds)\b",
        r"\b(?:rating|result|verdict)s?\b\s*(?:[:=]\s*)?[\"'“]exceeds[\"'”]",
        r"\b(?:rating|result|verdict)s?\s*[:=]\s*exceeds\b",
        # Spanish, French, German, Portuguese, Italian, Chinese
        r"\b(?:califica|califique|punt[úu]a|punt[úu]e|eval[úu]a|eval[úu]e)\s+a\s+est[ea]\s+"
        r"candidat[oa]\b",
        r"\bnote[zr]?\s+ce\s+candidat\b",
        r"\bbewerte(?:n\s+sie)?\s+diesen\s+kandidaten\b",
        r"\bavalie\s+este\s+candidato\b",
        r"\bvaluta(?:re|te)?\s+quest[oa]\s+candidat[oa]\b",
        r"给(?:这位|该|此|这个)候选人(?:打|评|给)",
    ),
    "verdict_claim": (
        rf"{_SUBJECT}\s+(?:is|was|has\s+been|would\s+be|will\s+be)\s+(?:clearly\s+|truly\s+|"
        r"definitely\s+|absolutely\s+)?(?:an?\s+|the\s+)?(?:perfect|ideal|best|exceptional|"
        r"excellent|outstanding|top|strongest|most\s+qualified|highly\s+qualified|"
        r"exceptionally\s+(?:well[-\s])?qualified|pre[-\s]?approved|must[-\s]hire|"
        r"(?:great|strong)\s+(?:fit|match))\b",
        rf"{_SUBJECT}\s+(?:is|was|has\s+been)\s+(?:already\s+)?(?:pre[-\s]?screened|"
        rf"pre[-\s]?vetted|verified)(?:\s+for\s+this\s+(?:role|position|job))?{_CLAUSE_END}",
        r"\bthis\s+is\s+an?\s+(?:exceptionally\s+|extremely\s+|highly\s+|very\s+|truly\s+)?"
        r"(?:well[-\s])?(?:qualified|perfect|ideal|excellent|outstanding|exceptional|top|"
        r"strong)\s+(?:candidate|applicant)\b",
        r"\b(?:hire|shortlist|advance|interview)\s+(?:this\s+(?:candidate|applicant|person)|"
        r"him|her)\s+(?:immediately|now|right\s+away|without\s+(?:further\s+)?(?:review|"
        r"delay|question))",
        rf"\b(?:hiring\s+manager|recruiter|hr|ceo|cto|the\s+team|the\s+committee)\s+"
        rf"(?:has\s+|have\s+)?(?:already\s+)?(?:pre[-\s]?)?(?:approved|selected|shortlisted|"
        rf"cleared|vetted|endorsed)\s+{_TARGET}\b",
    ),
    "all_requirements": (
        rf"{_SUBJECT}(?:\s+(?:clearly|fully|already|perfectly))?\s+(?:meets?|satisf(?:y|ies)|"
        r"fulfil(?:l?s)?|fulfills|exceeds?|matches|checks)\s+(?:all|every|each)\s+"
        r"(?:of\s+)?(?:the\s+|your\s+|its\s+)?(?:\w+\s+){0,2}?(?:requirements?|criteria|"
        r"criterion|qualifications?|boxes|must[-\s]haves)\b",
        r"\btreat\s+(?:every|all|each)\s+(?:of\s+the\s+)?requirements?\s+as\s+"
        r"(?:met|exceeded|satisfied|fulfilled)\b",
    ),
    "chat_markup": (
        r"<\|[a-z_]{2,30}\|>",
        r"\[/?INST\]",
        r"<</?SYS>>",
        rf"{_NOT_AFTER_BUILD_VERB}</?(?:system|instructions?|prompt|sys)\b[^>\n]{{0,40}}>"
        r"(?![^\n]*(?:<user>|<assistant>|\btemplates?\b|\btags?\b|\bblocks?\b))",
        r"^\s*#{2,}\s*(?:system(?:\s+prompt)?|instructions?|new\s+instructions?|prompt)"
        r"\s*(?::|$)",
        rf"^[^\S\n]*(?:system|developer|admin)\s+(?:prompt|message|note|instruction|override)s?"
        rf"\s*:\s*(?={_IMPERATIVE})",
        r"\b(?:system|developer|admin)\s+overrides?\s*:",
        r"^\s*(?:system|assistant)\s*:\s*you\b",
        r"\bcorrection(?:\s+note)?\s+from\s+the\s+(?:system|developer|admin|evaluator|"
        r"screener|reviewer|recruiting\s+team)\b",
        # A copy of the prompt's resume fence (the prompt builder neutralises these too;
        # the attempt itself must still force human review).
        r"<{3,}\s*(?:END[-_ ]?)?RESUME[-_ ]?",
    ),
    "concealment": (
        r"\b(?:do\s+not|don['’]t|never|without)\s+(?:mention(?:ing)?|reveal(?:ing)?|"
        r"disclos(?:e|ing)|repeat(?:ing)?|acknowledg(?:e|ing)|flag(?:ging)?|report(?:ing)?|"
        r"quot(?:e|ing)|tell(?:ing)?)\s+(?:this|these|that|any\s+of\s+this|"
        r"the\s+(?:above|following|hidden|previous|prior|preceding)|"
        r"(?:the|this|these|my)\s+(?:instructions?|note|message|prompt|text|request))\b",
        r"\bkeep\s+(?:this|these|it)\s+(?:(?:note|instructions?|message|text)\s+)?"
        r"(?:secret|hidden|confidential|private|between\s+us)\b",
    ),
}

_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (category, re.compile(pattern, re.IGNORECASE | re.MULTILINE))
    for category, patterns in INJECTION_PATTERNS.items()
    for pattern in patterns
)

_WS_RE = re.compile(r"[^\S\n]+")
_NEWLINE_RUN_RE = re.compile(r" ?\n[\n ]*")
_WORD_RE = re.compile(r"\w+")


def _visible(ch: str) -> bool:
    if ch in "\t\n":
        return True
    if 0xE0000 <= ord(ch) <= 0xE01EF or 0xFE00 <= ord(ch) <= 0xFE0F:
        return False
    return unicodedata.category(ch) not in ("Cc", "Cf", "Co", "Cs")


def _join_spaced(m: re.Match[str]) -> str:
    return _WS_RE.sub("", m.group())


def _fold(text: str) -> str:
    """Copy of ``text`` for pattern matching: no invisible characters, NFKC, look-alike
    letters mapped to Latin, letter-spaced words joined, whitespace runs collapsed (blank
    lines included)."""
    kept = "".join(ch for ch in text.replace("\r", "\n") if _visible(ch))
    folded = unicodedata.normalize("NFKC", kept).translate(_FOLD_TABLE)
    folded = _LETTER_SPACED_RE.sub(_join_spaced, folded)
    # Every whitespace run becomes one character, so no pattern's \s* can go quadratic.
    return _NEWLINE_RUN_RE.sub("\n", _WS_RE.sub(" ", folded))


def _deleet(token: re.Match[str]) -> str:
    """"1gn0re" -> "ignore" for tokens that are mostly letters; same length as the input."""
    word = token.group()
    letters = sum(1 for ch in word if ch.isalpha())
    digits = len(word) - letters
    if digits and letters >= 4 and letters >= digits:
        return word.translate(_LEET_TABLE)
    return word


def _clip(text: str, limit: int = MAX_QUOTE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def find_injection_phrases(text: str) -> list[tuple[str, str]]:
    """(category, snippet) for every pattern match, in text order. The snippet runs from
    the match to the end of its line, clipped to 120 characters."""
    folded = _fold(text)
    variants = [folded]
    deleeted = _LEET_TOKEN_RE.sub(_deleet, folded)
    if deleeted != folded:
        variants.append(deleeted)  # same length: offsets map back onto ``folded``
    hits: dict[tuple[int, str], str] = {}  # two patterns of a category can match one phrase
    for variant in variants:
        for category, rx in _COMPILED:
            for m in rx.finditer(variant):
                line_end = folded.find("\n", m.end())
                end = len(folded) if line_end < 0 else line_end
                hits.setdefault((m.start(), category),
                                _clip(folded[m.start():max(end, m.end())]))
    return [(category, snippet) for (_, category), snippet in sorted(hits.items(),
                                                                     key=lambda h: h[0][0])]


@cache
def _script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    for script in ("LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE", "LISU"):
        if name.startswith(script):
            return script
    return "OTHER"


def _is_spoofed(word: str) -> bool:
    scripts = {_script(ch) for ch in word}
    if "LATIN" not in scripts or not scripts & _SPOOF_SCRIPTS:
        return False
    return any(ch in LOOKALIKES for ch in word if _script(ch) in _SPOOF_SCRIPTS)


def mixed_script_occurrences(text: str) -> list[str]:
    """Every word (with repeats) mixing Latin with look-alike letters of another script."""
    text = unicodedata.normalize("NFKC", text)
    return [w for w in _WORD_RE.findall(text) if not w.isascii() and _is_spoofed(w)]


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def mixed_script_words(text: str) -> list[str]:
    """Distinct mixed-script words, in order of first appearance."""
    return _dedupe(mixed_script_occurrences(text))


def scan_text(visible_text: str, *, hidden_text: str = "", tag_payload: str = "",
              removed_counts: dict[str, int] | None = None) -> SecurityScan:
    flags: list[Flag] = []
    phrases: list[str] = []
    for label, text in (("visible", visible_text), ("hidden", hidden_text),
                        ("tag payload", tag_payload)):
        if not text or not text.strip():
            continue
        hits = find_injection_phrases(text)
        if not hits:
            continue
        phrases.extend(snippet for _, snippet in hits)
        category, snippet = hits[0]
        more = len(hits) - 1
        detail = f'{label} text, {category}: "{snippet}"' + (f" (+{more} more)" if more else "")
        flags.append(Flag(code=FlagCode.INJECTION_PHRASE, severity="review", detail=detail))

    if tag_payload.strip():
        flags.append(Flag(code=FlagCode.UNICODE_TAG_PAYLOAD, severity="review",
                          detail=f'decoded hidden characters: "{_clip(tag_payload)}"'))

    counts = removed_counts or {}
    invisible = {k: counts.get(k, 0) for k in INVISIBLE_CATEGORIES if counts.get(k, 0)}
    total = sum(invisible.values())
    if total >= INVISIBLE_INFO_THRESHOLD:
        breakdown = ", ".join(f"{k}={v}" for k, v in invisible.items())
        flags.append(Flag(
            code=FlagCode.INVISIBLE_CHARACTERS,
            severity="review" if total >= INVISIBLE_REVIEW_THRESHOLD else "info",
            detail=f"{total} invisible characters removed ({breakdown})",
        ))

    occurrences = mixed_script_occurrences(visible_text)
    words = _dedupe(occurrences)
    if words:
        listed = ", ".join(words[:MAX_LISTED_WORDS])
        flags.append(Flag(
            code=FlagCode.MIXED_SCRIPT,
            severity="review" if len(occurrences) >= MIXED_SCRIPT_REVIEW_THRESHOLD else "info",
            detail=f"{len(occurrences)} word(s) mixing Latin with look-alike letters: "
                   f"{_clip(listed)}",
        ))

    return SecurityScan(
        injection_phrases=_dedupe(phrases)[:MAX_PHRASES],
        mixed_script_words=words,
        flags=flags,
    )
