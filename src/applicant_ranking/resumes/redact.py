"""Redact identity and protected-class proxies from resume text before the LLM sees it.

The text after redaction is the ONLY resume text sent for scoring, and the evidence
quotes the model returns are checked against it. Placeholders are bracketed and stable:

- [CANDIDATE]: each token (len >= 2) of the candidate's name(s) from the Lever record,
  whole-word, case- and accent-insensitive (also hyphenated parts).
- [EMAIL], [PHONE] (>= 7 digits; must not eat year ranges like 2019-2021 or version
  numbers), [LINKEDIN_URL], [GITHUB_URL], [URL] (any other http(s)/www URL or bare
  domain with a path), [HANDLE] (@social handles at word start).
- [ADDRESS]: US-style street addresses ("123 N Main St, Apt 4"), PO boxes; [ZIP]: 5/9
  digit ZIP after a state; [LOCATION]: "City, ST" / "City, State" pairs using US state
  names/abbreviations (Illinois HB 3773 bars zip codes as a proxy).
- [REDACTED PERSONAL DETAIL]: whole lines labelled date of birth/DOB/birth date, age,
  gender/sex, pronouns, marital status, nationality, citizenship, visa/work permit,
  religion, ethnicity/race, disability, health, height/weight, photo.
- [YEAR]: 4-digit years on education lines only (lines mentioning a degree, university,
  college, school, B.S./M.S./PhD/MBA/bachelor/master/diploma, or "graduated"), when
  ``redact_education_years``. Employment date ranges stay (tenure is job-relevant).
- Honorifics Mr/Mrs/Ms/Miss/Mx/Dr. before a [CANDIDATE] -> removed.
Keep tech tokens intact ("Node.js", "ASP.NET", "C#", "v2.3", "AWS us-east-1").
``extra_identifiers``: other literal strings to replace with [CANDIDATE] (e.g. emails
and phone numbers from the Lever record, matched as literals).
``counts`` maps placeholder name -> number of replacements (count replacements, not
pattern hits). ``version`` is REDACTION_VERSION; bump it whenever the rules change.

Further rules (bump REDACTION_VERSION when changing them):
- ``counts`` keys are the placeholder names without brackets ("CANDIDATE", "EMAIL", ...,
  "REDACTED PERSONAL DETAIL", "YEAR"); see ``PLACEHOLDERS``.
- Order: extra identifiers, labelled personal lines, pronoun sets, emails, URLs,
  handles, addresses, ZIPs, locations, phones, names, honorifics, education years. URLs
  and emails go before names so "jose.alvarez@x.com" becomes [EMAIL], not a mangled
  address.
- Extra identifiers that look like phone numbers match on their digits with any
  separators (the resume may format the number differently); URLs match with or
  without scheme/"www."; anything else is a case-insensitive literal.
- Name tokens: suffixes/honorifics in the record (Jr, III, PhD, Dr ...) are not treated
  as name tokens; particles written in lower case ("van", "do") are skipped unless the
  name has nothing else ("Van Le" and "Đỗ Minh" keep them). Format characters in the
  record (a ZWNJ inside a Persian name, a ZWSP) are dropped before tokenising, so the
  tokens match the normalised resume text. Possessives ("Alvarez's") and digits glued to
  a name ("alvarez99") are covered, and so are handle-like concatenations of the name
  tokens ("josealvarez", "jalvarez"). CJK name tokens match as substrings (no spaces
  between words). Name tokens that are English function words ("An", "To", "Do") only
  match capitalised words, so "built an API" survives.
- Word-like name tokens (``WORD_LIKE_NAME_TOKENS``: months, technologies and common
  English words that are also given names or surnames: "Ruby", "Kafka", "Jan", "Ai",
  "Grant", "Brown", "Martin", ...) are redacted only where they are used as a name:
  written in title case (or exactly as in the Lever record) AND next to another name
  token, an initial, an honorific or a [CANDIDATE]; or on a header/contact line (one of
  the first ``HEADER_LINES`` non-empty lines, or a line holding a contact placeholder);
  or possessive ("Ruby's"); or starting a sentence followed by a third-person summary
  verb ("Ruby is a data engineer who ..."). So "Ruby on Rails", "Kafka and Flink",
  "Jan 2019 - Dec 2021", "AI/ML" (for Ai), "Lockheed Martin" and "Brown University"
  keep their meaning for every candidate, while full-name sequences, digit-glued and
  handle forms are still redacted everywhere. Residual: a lone word-name first name
  mid-bullet ("contact Ruby for references") is not caught; the model prompt and the
  fairness counterfactuals are the control for it. Every other token is fully
  case-insensitive and redacted everywhere: privacy wins over fidelity.
- Personal-detail lines: an unambiguous label ("Date of birth", "Nationality", "Marital
  status", ...) at the line start or after a field separator ("|", "•", "·", ";")
  followed by ":" replaces the whole line (one-line "Personal details" blocks). The same
  label mid-sentence (after a space, comma, bracket or slash) replaces only its own
  field, "label: value" up to the next separator, so "Masked PII (name, DOB: dates, SSN)"
  keeps the rest of the bullet. A label that is also an ordinary word (age, sex, health,
  weight, photo, ...) counts only at the line start or after a field separator, so
  "System health: 99.9%" stays; a weak label at the line start followed by " - " or
  " | " counts only when the value looks personal (short, no year range, no job title,
  no further field), so "Age - 34" is redacted but "Health - Lead Data Engineer (Optum)"
  and the employer "Visa | Senior Software Engineer | 2019 - 2022" are not. Also
  covered: "DOB 1/2/1990", "I am 34 years old", lines of at most 60 characters (or
  starting with Personal/About/Profile) containing "NN years old", a separator-delimited
  field that is just a marital status ("Married, two children"; never the bare word
  "single", which collides with single sign-on), unlabelled citizenship/immigration
  lines ("US Citizen", "Green card holder", "authorized to work in the US", "H-1B
  visa holder"), and ID numbers without a colon ("Passport No. X1234567"). Labels are
  also recognised in German, Spanish, French, Italian and Portuguese ("Geburtsdatum",
  "Nacionalidad", "Date de naissance", "Stato civile", "Data de nascimento"). Inline
  pronoun sets such as "(she/her)" become [REDACTED PERSONAL DETAIL] as well. A matched
  line longer than ``MAX_PERSONAL_LINE_CHARS`` is redacted field by field instead of as a
  whole, so a single label can never empty a resume whose layout put several fields on
  one line.
- Bare domains without a path are redacted only when they contain the candidate's name
  ("josealvarez.com"). Bare-domain detection needs a lower-case TLD from a list of real,
  common TLDs, so "ASP.NET/C#", "Node.js/React" and "Socket.io" stay; "java.io" is a
  known package, and a bare domain whose path is a single capitalised word
  ("Hotels.com/Expedia") is a brand list, not a profile URL.
- Handles: "@Word" is a social handle unless it is a code annotation (@Override) or a
  capitalised alphabetic word that follows a role or event word within three words
  ("Senior Engineer @Google", "Speaker @PyCon"; a further @Word within six words of a
  kept one is kept too: "@PyCon US 2023 and @KubeCon").
- IP addresses are kept: they are technical content, not a candidate identifier.
  Version numbers, ISO and dotted dates, numeric ranges with fewer than 10 digits
  ("1000-5000 users") and space-separated small numbers are not phones; nor is a
  space-grouped quantity in thousands followed by a unit or count noun ("250 000 000
  events") or preceded by a currency sign or "over/about/nearly". International numbers
  with a country code and short groups ("+33 6 12 34 56 78", "+91 9876543210"), labelled
  numbers in any grouping ("Tel. +49 30 12345678", "Mobile: 06 12 34 56 78") and a final
  6-digit group ("07700 900123") are phones.
- [LOCATION] also covers "City, Country" for common countries (national origin is a
  protected class); a "city" containing words such as University, Engineer, Science or
  Inc, or a field of study (Statistics, Psychology, Nursing ...), is not a city, so
  "Boston University, MA", "Computer Science, MS" and "Statistics, MS - University of
  Chicago" stay; "MS" followed by a Microsoft product ("MS Access", "MS Teams") or by
  " - <University|College|Institute|School>" is a degree, not Mississippi. "City ST
  ZIP" without the comma is a location when the ZIP follows; "City ST" at the end of a
  line or field is one when ST is not also an English word or abbreviation (IN, OR, ME,
  MA, MS, CO, ...).
- Education years: besides lines with education keywords, years are redacted on every
  line of an EDUCATION section (until the next recognised heading, at most 25 lines)
  and on date-only lines next to an education line that has no year of its own (the
  common two-line layout). Short ranges ("2015-19") and apostrophe years ("Class of
  '15") are covered. Inside another recognised section (EXPERIENCE, EMPLOYMENT, ...) a
  line is education only when it names a degree, never on a bare "University",
  "College", "School" or "Graduate"; outside any section a line with such a word but no
  degree and a job title ("Graduate Research Assistant, NLP Lab, Jan 2019 - May 2021",
  "Software Engineer, Harvard University IT, 2016 - 2021") is employment and keeps its
  dates.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from itertools import permutations

from applicant_ranking.domain import RedactionResult

REDACTION_VERSION = "redact-v3"

CANDIDATE = "CANDIDATE"
EMAIL = "EMAIL"
PHONE = "PHONE"
LINKEDIN_URL = "LINKEDIN_URL"
GITHUB_URL = "GITHUB_URL"
URL = "URL"
HANDLE = "HANDLE"
ADDRESS = "ADDRESS"
ZIP = "ZIP"
LOCATION = "LOCATION"
PERSONAL_DETAIL = "REDACTED PERSONAL DETAIL"
YEAR = "YEAR"
PLACEHOLDERS = (CANDIDATE, EMAIL, PHONE, LINKEDIN_URL, GITHUB_URL, URL, HANDLE, ADDRESS, ZIP,
                LOCATION, PERSONAL_DETAIL, YEAR)

# A matched personal-detail line longer than this is redacted field by field.
MAX_PERSONAL_LINE_CHARS = 200
# Lines (non-empty) from the top that count as the resume header for name matching.
HEADER_LINES = 6
# A weak personal label followed by " - " / " | " counts only with a value this short.
MAX_PERSONAL_VALUE_CHARS = 40
MAX_AGE_LINE_CHARS = 60


def placeholder(name: str) -> str:
    return f"[{name}]"


# --- personal-detail lines ------------------------------------------------------------------

# Unambiguous labels count anywhere at a word start; words that also occur in ordinary
# resume text ("System health:", "Big picture:") only at a line start or after a field
# separator.
_STRONG_LABELS = (
    r"(?:date\s+of\s+birth|d\.?\s?o\.?\s?b\.?|birth\s*date|birthday|place\s+of\s+birth|"
    r"gender|pronouns?|marital\s+status|civil\s+status|family\s+status|relationship\s+status|"
    r"nationality|citizenship|citizen\s+of|visa\s+(?:status|type)|work\s+visa|"
    r"work\s+(?:permit|authori[sz]ation)|"
    r"immigration\s+status|residency\s+status|religion|religious\s+affiliation|ethnicity|"
    r"ethnic\s+origin|caste|disability|disabilities|health\s+status|"
    r"medical\s+(?:condition|history)|passport(?:\s+(?:no|number))?|"
    r"social\s+security(?:\s+(?:no|number))?|ssn|national\s+id|spouse['’]?s?\s+name|"
    r"father['’]?s?\s+name|mother['’]?s?\s+name|veteran\s+status|military\s+status|"
    r"sexual\s+orientation|"
    # German, Spanish, French, Italian, Portuguese (accented and folded spellings)
    r"geburtsdatum|geburtstag|geboren(?:\s+am)?|geburtsort|staatsangeh(?:ö|oe|o)rigkeit|"
    r"nationalit(?:ä|ae|a)t|familienstand|konfession|fecha\s+de\s+nacimiento|"
    r"lugar\s+de\s+nacimiento|nacionalidad|estado\s+civil|religi(?:ó|o)n|"
    r"date\s+de\s+naissance|lieu\s+de\s+naissance|nationalit(?:é|e)|situation\s+familiale|"
    r"(?:é|e)tat\s+civil|data\s+di\s+nascita|nazionalit(?:à|a)|stato\s+civile|"
    r"data\s+de\s+nascimento|nacionalidade)"
)
_WEAK_LABELS = (r"(?:age|sex|born|visa|race|faith|health|height|weight|photo(?:graph)?|"
                r"dependents|children|spouse)")
# "Visa - Staff Engineer" is an employer line, so the bare word is not a dash label.
_WEAK_DASH_LABELS = (r"(?:age|sex|born|race|faith|health|height|weight|photo(?:graph)?|"
                     r"dependents|children|spouse)")
# Bounded runs: input is normalised (single spaces), and unbounded quantifiers next to a
# lazy prefix would go quadratic on hostile raw text.
_WS = r"[^\S\n]{0,8}"
_LABEL_TAIL = rf"{_WS}(?:\([^)\n]{{0,30}}\))?{_WS}:[^\n]*$"
_SEP = r"[|•·;]"
_NATIONALITY_ADJ = (
    r"(?:U\.?S\.?|United\s+States|American|Canadian|British|UK|EU|European|Australian|"
    r"Indian|German|French|Mexican|Brazilian|Chinese|Japanese|Korean|Irish|Italian|Spanish|"
    r"Dutch|Swedish|Polish|Nigerian|Pakistani|Filipino|Vietnamese|Turkish|Israeli|Russian|"
    r"Ukrainian|dual|naturali[sz]ed)"
)
_PERSONAL_LINE_RES = (
    re.compile(rf"^[^\n]*?(?:^|(?<={_SEP})){_WS}{_STRONG_LABELS}{_LABEL_TAIL}", re.I | re.M),
    re.compile(rf"^[^\n]*?(?:^|(?<={_SEP})){_WS}(?:[-*•]{_WS})?{_WEAK_LABELS}{_LABEL_TAIL}",
               re.I | re.M),
    re.compile(rf"^{_WS}(?:[-*•]{_WS})?{_STRONG_LABELS}[^\S\n]{{1,8}}[-–—|][^\S\n][^\n]*$",
               re.I | re.M),
    re.compile(r"^[^\n]*\b(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date|born(?:\s+on)?)"
               r"[^\S\n]+\d[^\n]*$", re.I | re.M),
    re.compile(r"^[^\S\n]*(?:[-*•][^\S\n]*)?age[^\S\n]+\d{2}\b[^\n]*$", re.I | re.M),
    re.compile(r"^[^\n]*\b(?:I\s+am|I['’]m)\s+\d{2}\s+(?:years?|yrs?)\s+old\b[^\n]*$",
               re.I | re.M),
    re.compile(
        rf"^[^\n]*\b(?:{_NATIONALITY_ADJ}\s+citizen(?:ship)?|citizen\s+of\s+(?:the\s+)?\w|"
        r"green\s+card|permanent\s+resident|visa\s+sponsorship|"
        r"(?:no|not|without|requires?|needs?|will\s+need)\s+sponsorship|"
        r"sponsorship\s+(?:is\s+)?(?:required|needed|not\s+required)|"
        r"(?:legally\s+)?(?:authori[sz]ed|eligible)\s+to\s+work|work\s+authori[sz]ation|"
        r"(?:H-?1B|H-?4|L-?1|F-?1|J-?1|O-?1|OPT|CPT|EAD|TN|GC)\s+(?:visa|status|holder))"
        r"\b[^\n]*$", re.I | re.M),
)
# A weak label at the line start followed by " - " / " | ": whole line only when the
# value looks personal (checked in ``_personal_value``).
_WEAK_DASH_LINE_RE = re.compile(
    rf"^{_WS}(?:[-*•]{_WS})?{_WEAK_DASH_LABELS}[^\S\n]{{1,8}}[-–—|][^\S\n](?P<value>[^\n]*)$",
    re.I | re.M)
_JOB_TITLE_RE = re.compile(
    r"\b(?:engineer|developer|analyst|scientist|instructor|lecturer|assistant|researcher|"
    r"fellow|postdoc|manager|intern|consultant|administrator|teacher|professor|specialist|"
    r"architect|director|coordinator|technician|designer|lead|officer|head|associate|"
    r"vp|president|founder|owner|partner)\b", re.I)
_YEAR_RANGE_RE = re.compile(r"\b(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|now|current)\b",
                            re.I)
# A strong label mid-sentence replaces only its own field.
_LABEL_INLINE_RE = re.compile(
    rf"(?<=[\s,(/]){_STRONG_LABELS}{_WS}(?:\([^)\n]{{0,30}}\))?{_WS}:[^\n|;,)]*", re.I)
_AGE_LINE_RE = re.compile(r"^[^\n]*\b\d{2}\s+(?:years?|yrs?)\s+old\b[^\n]*$", re.I | re.M)
_PERSONAL_HEADED_RE = re.compile(r"^[^\S\n]*(?:[-*•][^\S\n]*)?(?:personal|about|profile)\b", re.I)
_MARITAL_FIELD_RE = re.compile(
    rf"(?:^|(?<={_SEP}))[^\S\n]*(?:-[^\S\n])?(?:married|unmarried|divorced|widowed)"
    rf"(?:\s*,\s*(?:\w+\s+)?(?:children|kids|child|dependents))?[^\S\n]*(?={_SEP}|$)",
    re.I | re.M)
_ID_NUMBER_RE = re.compile(
    r"\b(?:passport|national\s+id|ssn|social\s+security)\s*(?:no\.?|number|#|:)?\s*"
    r"[A-Z0-9][A-Z0-9-]{5,11}\b", re.I)
_PRONOUN_SET = (
    r"(?:she|he|they|ze|xe)\s*/\s*(?:her|him|them|they|hir|zir|xem)"
    r"(?:\s*/\s*(?:hers|his|theirs|hirs|zirs|xyrs|them|her|him))?"
)
_PRONOUN_RE = re.compile(rf"\(\s*{_PRONOUN_SET}\s*\)|\b{_PRONOUN_SET}\b", re.I)
_FIELD_SPLIT_RE = re.compile(rf"({_SEP})")

# --- contact details ------------------------------------------------------------------------

_EMAIL_RES = (
    re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    re.compile(r"(?<![\w.+-])[\w.+-]+\s{0,3}[\[(]\s{0,3}at\s{0,3}[\])]\s{0,3}[\w-]+"
               r"(?:\s{0,3}[\[(]\s{0,3}dot\s{0,3}[\])]\s{0,3}[\w-]+)+", re.I),
    re.compile(r"(?<![\w.+-])[\w.+-]+\s{1,3}at\s{1,3}[\w-]+\s{1,3}dot\s{1,3}"
               r"(?:com|net|org|edu|io|co|me|dev)\b", re.I),
)

_URL_STOP = r"\s<>\"'`|"
_URL_RE = re.compile(
    rf"\b(?:https?://|www\.)[^{_URL_STOP}]+"
    rf"|(?<![\w@./-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{{0,61}}[A-Za-z0-9])?\.)+(?P<tld>[a-z]{{2,24}})"
    rf"(?P<path>/[^{_URL_STOP}]*)?(?![\w@-])"
)
_COMMON_TLDS = frozenset(
    "com net org edu gov mil int io dev me co ai app info biz us uk ca de fr es it nl in au "
    "br mx ch se no dk fi ie be at nz za sg hk jp kr tw cn ru ua eu tech site online page "
    "blog xyz design codes tv ly gg fm studio space website art name pro cloud digital "
    "works one link live world today email ninja rocks wiki portfolio is la ms".split()
)
_TECH_HOSTS = frozenset({"asp.net", "ado.net", "vb.net", "dot.net", "socket.io", "draw.io",
                         "chart.js", "node.js", "java.io"})
_BRAND_PATH_RE = re.compile(r"/[A-Z][a-z]+")
_TRAILING_PUNCT = ".,;:!?'\")]}"
_LINKEDIN_HOSTS = ("linkedin.com", "lnkd.in")
_GITHUB_HOSTS = ("github.com",)

_HANDLE_RE = re.compile(r"(?<![\w@./+\-])@([A-Za-z0-9_](?:[A-Za-z0-9_.]{0,38}[A-Za-z0-9_])?)"
                        r"(?![\w/@-])")
_CODE_ANNOTATIONS = frozenset(
    "Override Autowired Component Service Repository Controller RestController RequestMapping "
    "GetMapping PostMapping PutMapping DeleteMapping Transactional Entity Table Id Column Test "
    "Before After BeforeEach AfterEach BeforeAll AfterAll Bean Configuration Inject Injectable "
    "NgModule Input Output Deprecated FunctionalInterface SuppressWarnings Value Data Builder "
    "Getter Setter Slf4j SpringBootApplication EnableAutoConfiguration Query Param Path "
    "property staticmethod classmethod dataclass abstractmethod cached_property lru_cache "
    "cache pytest app router fixture task shared_task Published State Binding ObservedObject "
    "Environment MainActor objc IBOutlet IBAction media import keyframes font-face mixin "
    "include extend apply tailwind types babel here channel everyone".split()
)
_ORG_HANDLE_RE = re.compile(r"[A-Z][a-zA-Z]{2,}")
_ROLE_WORD_RE = re.compile(
    r"\b(?:engineer|developer|scientist|analyst|manager|lead|intern|architect|consultant|"
    r"director|founder|cto|ceo|coo|swe|sde|pm|speaker|talk|keynote|presented|presenter|spoke|"
    r"panelist|workshop|researcher|designer|contractor|volunteer|mentor|advisor|at)\b", re.I)
_ROLE_WINDOW_WORDS = 3
_HANDLE_CHAIN_WORDS = 6

_PHONE_RE = re.compile(
    r"(?<![\w.+/\-])(?:\+\d{1,3}[\s.\-]?|\d{1,3}[\s.\-])?(?:\(\d{1,4}\)[\s.\-]?)?"
    r"\d{2,5}(?:[\s.\-]\d{2,5}){0,3}[\s.\-]\d{2,6}(?![\w/\-]|\.\d)"
)
_PHONE_LABELLED_RE = re.compile(
    r"(\b(?:phone|mobile|mob|cell|tel|telephone|whatsapp|contact(?:\s+no)?|ph|handy|"
    r"t(?:é|e)l(?:é|e)phone|telefon|tel(?:é|e)fono|telefone|celular|m(?:ó|o)vil)\b\.?\s*"
    r"(?:no\.?|number|#)?\s*[:\-]?\s*)(\+?\d(?:[\d\s().\-]{5,20}\d)?)(?![\w/\-]|\.\d)", re.I
)
# Country code plus up to five groups of any length ("+33 6 12 34 56 78", "+91 9876543210").
_PHONE_INTL_RE = re.compile(r"(?<![\w+])\+\d{1,3}(?:[\s.\-]?\(?\d{1,5}\)?){1,5}(?![\w/\-]|\.\d)")
_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_MIN_PHONE_DIGITS, _MAX_PHONE_DIGITS = 7, 15
_QUANTITY_AFTER_RE = re.compile(
    r"^\s*(?:events?|rows?|records?|users?|requests?|transactions?|messages?|queries|"
    r"customers?|documents?|files?|images?|emails?|calls?|visits?|views?|items?|orders?|"
    r"products?|devices?|sensors?|nodes?|lines?|entries|samples?|per\b|daily|monthly|annual|"
    r"usd|eur|gbp|k\b|m\b|mm\b|bn\b|million|billion|thousand|points?|clicks?|impressions?|"
    r"downloads?|installs?|sessions?|pageviews?|tickets?|incidents?|alerts?|jobs?|tasks?|"
    r"builds?|deployments?|commits?|tests?|rpm|rps|qps|tps|ms\b|seconds?|minutes?|hours?|"
    r"days?|percent|%|\+)", re.I)
_QUANTITY_BEFORE_RE = re.compile(
    r"(?:[$€£¥]|\b(?:over|about|nearly|approx\.?|approximately|around|up\s+to|roughly|"
    r"processed|handled|served|generated|managed|ingested|stored|analy[sz]ed)|[~>])\s*$", re.I)

# --- addresses and locations ------------------------------------------------------------

_STREET_SUFFIXES = ("Street", "St", "Avenue", "Ave", "Road", "Rd", "Boulevard", "Blvd",
                    "Lane", "Ln", "Drive", "Dr", "Court", "Ct", "Way", "Place", "Terrace",
                    "Parkway", "Pkwy", "Circle", "Highway", "Hwy", "Square", "Trail", "Loop",
                    "Alley", "Plaza", "Crescent", "Pike", "Turnpike", "Expressway", "Freeway")
_SUFFIX = "(?:" + "|".join(f"{s}|{s.upper()}" for s in _STREET_SUFFIXES) + ")"
_STREET_RE = re.compile(
    r"\b\d{1,6}[A-Za-z]?\s{1,3}(?:(?:[A-Z][\w'.-]{0,30}|\d{1,4}(?:st|nd|rd|th))\s{1,3}){1,4}?"
    rf"{_SUFFIX}\b\.?(?:\s+(?:N|S|E|W|NE|NW|SE|SW)\b\.?)?"
    r"(?:\s*,?\s*(?:Apt|Apartment|Suite|Ste|Unit|Fl|Floor|Rm|Room|Bldg|Building)\.?\s*#?\s*"
    r"[\w-]+|\s*,?\s*#\s*[\w-]+)?"
)
_PO_BOX_RE = re.compile(r"\b(?:P\.?\s*O\.?\s*Box|Post\s+Office\s+Box)\s*#?\s*\d+\b", re.I)

_STATE_ABBR = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH "
    "NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR GU VI"
).split()
# Two-letter codes that are also English words or common abbreviations: without a comma
# they count as a state only when a ZIP code follows.
_AMBIGUOUS_ABBR = frozenset("IN OR ME HI OK OH DE LA PA MA MS MD CO AL AR ID MT GA MO NE UT "
                            "VA".split())
_STATE_NAMES = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia",
    "Wisconsin", "Wyoming", "District of Columbia", "Puerto Rico",
)
_COUNTRIES = (
    "United States", "USA", "U.S.A.", "U.S.", "US", "United Kingdom", "UK", "U.K.", "England",
    "Scotland", "Wales", "Ireland", "Canada", "Mexico", "Brazil", "Argentina", "Chile",
    "Colombia", "Peru", "Venezuela", "Cuba", "Jamaica", "India", "Pakistan", "Bangladesh",
    "Sri Lanka", "Nepal", "China", "Japan", "South Korea", "Korea", "Taiwan", "Hong Kong",
    "Vietnam", "Philippines", "Indonesia", "Malaysia", "Singapore", "Thailand", "Australia",
    "New Zealand", "Germany", "France", "Spain", "Portugal", "Italy", "Netherlands",
    "Belgium", "Switzerland", "Austria", "Poland", "Ukraine", "Russia", "Romania", "Greece",
    "Turkey", "Sweden", "Norway", "Denmark", "Finland", "Israel", "Egypt", "Nigeria",
    "Kenya", "Ghana", "Ethiopia", "South Africa", "Morocco", "Iran", "Iraq", "Lebanon",
    "Saudi Arabia", "UAE", "United Arab Emirates", "Qatar", "Ontario", "Quebec",
    "British Columbia", "Alberta",
)


def _alternation(items: Sequence[str]) -> str:
    return "|".join(re.escape(i) for i in sorted(items, key=len, reverse=True))


_STATE_ABBR_ALT = f"(?:{_alternation(_STATE_ABBR)})"
_STATE_ALT = f"(?:{_STATE_ABBR_ALT}|{_alternation(_STATE_NAMES)}|D\\.C\\.)"
_ZIP_RE = re.compile(rf"\b({_STATE_ALT})(\s{{0,3}},?\s{{1,3}})\d{{5}}(?:-\d{{4}})?(?![\d-])")
_PLACE_ALT = f"(?:{_STATE_ALT}|{_alternation(_COUNTRIES)})"
_CITY_WORD = r"(?:[A-Z][a-zA-Z.'’-]{0,30}|St\.)"
_LOCATION_RE = re.compile(
    rf"(?<![\w\[])(?P<city>{_CITY_WORD}(?:[ \t]{{1,3}}(?:{_CITY_WORD}|of|de|la|del)){{0,3}})"
    rf"(?:,[ \t]{{0,3}}(?P<place>{_PLACE_ALT})(?![\w])"
    rf"|[ \t]{{1,3}}(?P<abbr>{_STATE_ABBR_ALT})(?=[ \t]{{0,3}}(?P<zip>\d{{5}}|\[ZIP\])|{_WS}$"
    rf"|{_WS}{_SEP}|{_WS},))"
)
_NON_CITY_WORDS = frozenset(
    "university college institute school academy science sciences engineering engineer "
    "developer manager analyst scientist architect consultant intern lead director "
    "administration arts studies management technology technologies inc llc ltd corp "
    "corporation company co group labs systems solutions services health bank hospital "
    "center centre computer data software business mathematics physics economics finance "
    "marketing design research degree bachelor master remote hybrid onsite python java "
    "certified certification "
    # fields of study ("Statistics, MS" is a degree, not Mississippi)
    "statistics psychology sociology economics accounting nursing pharmacy linguistics "
    "philosophy history english journalism architecture bioinformatics robotics "
    "cybersecurity communications anthropology geography geology kinesiology neuroscience "
    "biochemistry biostatistics informatics analytics education law medicine dentistry "
    "theology music learning intelligence interaction relations policy biology chemistry "
    "security networking".split()
)
_FIELD_SUFFIXES = ("ics", "ology", "istry", "ematics", "ance", "ment")
_DEGREE_STATES = frozenset({"MS", "MA"})
_MS_PRODUCT_RE = re.compile(
    r"^[ \t]+(?:Access|Excel|Word|Office|Project|Teams|Visio|Outlook|PowerPoint|SharePoint|"
    r"Dynamics|SQL|Azure|Publisher|Exchange|Paint|Fabric|Power)\b")
_DEGREE_SCHOOL_RE = re.compile(r"^[ \t]*[-–—][ \t]*(?:University|College|Institute|School)\b")

# --- education years ------------------------------------------------------------------------

_INSTITUTION_RE = re.compile(
    r"(?i:\b(?:universit(?:y|ies|ät|é|à|ario|aria|ad|e)|college|school|polytechnic|"
    r"institute\s+of\s+technology|graduat\w*|alumn\w*)\b)"
)
_DEGREE_RE = re.compile(
    r"(?i:\b(?:degree|bachelor(?:['’]?s)?|master['’]?s|masters|master\s+of|master\s+in|"
    r"diploma|doctorate|doctoral|gpa|class\s+of|cum\s+laude|magna|summa|dean['’]?s\s+list)\b)"
    r"|\b(?:B|M)\.\s?(?:S|A|E|Sc|Eng|Tech|Com|Ed|Arch|Phil|F\.?A)\.?(?!\w)"
    r"|\bPh\.?\s?D\b\.?|\bJ\.D\.|\bM\.B\.A\.?"
    r"|\b(?:BSc|MSc|BEng|MEng|BTech|MTech|BCom|MCom|MBA|MCA|BCA|BBA|MPhil|MPH|MFA|BFA|LLB|"
    r"LLM|PhD|DPhil|EdD|AAS|BSN|MSN)\b"
    r"|\b(?:BS|BA|MS|MA|BE|AA|AS)\b(?=\s+(?:in|of)\s|\s*,\s*[A-Z]|\s+[A-Z][a-z]+\s+"
    r"(?:Science|Engineering|Studies|Arts|Administration)\b)"
)
_EDU_RE = re.compile(f"{_INSTITUTION_RE.pattern}|{_DEGREE_RE.pattern}")
_EDU_HEADINGS = frozenset({
    "education", "education and training", "education & training", "academic background",
    "academic qualifications", "educational background", "educational qualifications",
    "academics", "education & certifications", "education and certifications",
    "academic history", "academic profile", "qualifications and education",
})
_OTHER_HEADINGS = (
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history", "skills", "technical skills",
    "core competencies", "projects", "certifications", "certificates", "licenses", "awards",
    "honors", "publications", "summary", "professional summary", "profile", "objective",
    "volunteer", "volunteering", "interests", "languages", "references", "activities",
    "leadership", "research", "patents", "training", "courses", "achievements",
    "memberships", "affiliations", "contact", "personal details", "additional information",
)
_EDU_SECTION_MAX_LINES = 25
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SHORT_RANGE_RE = re.compile(r"(\[YEAR\]\s*[-–—/]\s*)\d{2}\b(?!\d)")
_APOSTROPHE_YEAR_RE = re.compile(r"(?<!\w)['’]\d{2}\b")
_MONTHS = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
_DATE_ONLY_RE = re.compile(
    rf"(?:[\s\d/.,:()\[\]'’\-–—]|\b(?:{_MONTHS}|present|current|now|today|expected|"
    r"anticipated|graduat\w*|class|of|to|since|from|until|till|spring|summer|fall|autumn|"
    r"winter|semester|term)\b\.?)*", re.I
)
_HONORIFIC_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Mx|Dr|Prof)\.?\s+(?=\[CANDIDATE\])", re.I)

# --- names ----------------------------------------------------------------------------------

_NAME_SPLIT_RE = re.compile(r"[\s,.\-‐‑–—_/()\"]+")
_NAME_SUFFIXES = frozenset("jr sr ii iii iv phd md mba esq cpa pe pmp dds rn mr mrs ms mx "
                           "miss dr prof".split())
_NAME_PARTICLES = frozenset("de da di del della der den van von la le du dos das do bin binti "
                            "al el ibn ben st".split())
# Name tokens that are also English function words ("An", "To", "Do") only match
# capitalised words, so "built an API" keeps its article.
_STOPWORD_TOKENS = frozenset("a an and as at be by do go he if in is it me my no of on or so to "
                             "up us we".split())
# Name tokens that are also months, technologies or ordinary words: redacted only where
# they are used as a name (see the module docstring).
WORD_LIKE_NAME_TOKENS = frozenset(
    # months
    "jan january feb february mar march apr april may jun june jul july aug august sep sept "
    "september oct october nov november dec december "
    # technologies and products
    "ruby swift rust kafka ray jenkins cassandra athena spark flink dart julia scala kotlin "
    "python java rails django flask spring hadoop hive storm beam presto druid redis mongo "
    "oracle sun solaris linux apple google amazon azure docker helm ansible chef puppet "
    "terraform jira slack teams excel word access office tableau looker airflow dagster "
    "prefect luigi celery glue redshift snowflake databricks figma sketch notion asana trello "
    "sage nova lambda aurora atlas mesos tornado falcon iris jasmine karma mocha chai jest "
    "ember express gatsby hugo bash perl lua elm elixir erlang haskell lisp clojure racket "
    "nim zig crystal pascal ada cobol fortran basic delphi groovy forth gin echo fiber vue "
    "react angular svelte pandas keras torch bert gemini llama alpaca mistral bison "
    "palm bard siri alexa cortana watson einstein newton tesla edison ford lincoln rover "
    "amber pearl jade opal onyx "
    # common words and romanised given names that are words
    "ai an can long min max sun fan ha ma ping ran lan wan bin tan li grant price mark will "
    "bill art gene brown white green young black martin davis hope faith grace joy rose summer "
    "autumn winter dawn river forest reed stone rock hill wood field lake bay dale heath moss "
    "ash oak holly olive page knight king queen prince dean marshall judge major chip buck red "
    "blue gray grey jack pat sam tom don ted ben jay kit kay rex guy rich chuck wade drew lane "
    "cole clay frank fox wolf bear crane hart robin wren ruth charity mercy patience sharp "
    "strong little small old best true noble worth lean agile".split()
)
# A word may contain apostrophes and the joiners some scripts write inside words (ZWNJ in
# Persian, ZWJ in Devanagari conjuncts), which ``normalize`` keeps there.
_TEXT_TOKEN_RE = re.compile(r"\[[A-Z][A-Z_ ]*\]|[^\W_]+(?:(?:['’]|[\u200c\u200d]+)[^\W_]+)*")
_MIN_HANDLE_FORM = 6
_CONTACT_PLACEHOLDER_RE = re.compile(
    r"\[(?:EMAIL|PHONE|LINKEDIN_URL|GITHUB_URL|URL|HANDLE|ADDRESS|LOCATION|ZIP)\]")
_HONORIFIC_TOKENS = frozenset("mr mrs ms miss mx dr prof".split())
_SUMMARY_VERBS = frozenset("is has was led leads joined brings holds works worked specialises "
                           "specializes".split())
_SENTENCE_BREAK_RE = re.compile(r"[.!?;:•·|]|^\s*[-*]\s")


# Letters that NFKD does not decompose to a base letter plus an accent.
_BASE_LETTERS = str.maketrans({"đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "ł": "l", "Ł": "L",
                               "ø": "o", "Ø": "O", "ħ": "h", "Ħ": "H", "ı": "i", "ŧ": "t",
                               "Ŧ": "T", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
                               "þ": "th", "Þ": "TH", "’": "'"})


def _fold(s: str) -> str:
    """Accent-, case- and format-character-insensitive form of a word."""
    decomposed = unicodedata.normalize("NFKD", s.translate(_BASE_LETTERS))
    return "".join(c for c in decomposed if not unicodedata.combining(c)
                   and unicodedata.category(c) != "Cf").casefold()


def _is_cjk(s: str) -> bool:
    return any(
        0x3040 <= ord(c) <= 0x30FF or 0x3400 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7AF
        or 0xF900 <= ord(c) <= 0xFAFF or 0x20000 <= ord(c) <= 0x2FA1F
        for c in s
    )


def _clean_name(name: str) -> str:
    """NFKC and drop format characters (ZWNJ, ZWSP, bidi marks) so the record's tokens
    match the normalised resume text."""
    return "".join(ch for ch in unicodedata.normalize("NFKC", name or "")
                   if unicodedata.category(ch) != "Cf")


def _raw_name_parts(candidate_names: Iterable[str]) -> list[str]:
    return [p.strip("'’") for name in candidate_names
            for p in _NAME_SPLIT_RE.split(_clean_name(name)) if len(p.strip("'’")) >= 2]


def name_tokens(candidate_names: Sequence[str]) -> set[str]:
    """Folded name tokens that are redacted as whole words. A particle ("van", "do") is
    skipped when it is written in lower case and the name has other tokens."""
    tokens: set[str] = set()
    for name in candidate_names:
        raw = _raw_name_parts([name])
        parts = [(p, _fold(p)) for p in raw]
        parts = [(p, f) for p, f in parts if f not in _NAME_SUFFIXES]
        core = [f for p, f in parts if not (f in _NAME_PARTICLES and p.islower())]
        tokens.update(core or [f for _, f in parts])
    return tokens


def _handle_forms(tokens: set[str]) -> set[str]:
    """Concatenations people use as handles: "josealvarez", "alvarezjose", "jalvarez"."""
    plain = sorted({re.sub(r"[^a-z]", "", t) for t in tokens if not _is_cjk(t)} - {""})
    forms: set[str] = set()
    for a, b in permutations(plain, 2):
        forms.update({a + b, a[0] + b, a + b[0]})
    if len(plain) > 2:
        forms.add("".join(plain))
    return {f for f in forms if len(f) >= _MIN_HANDLE_FORM}


class _NameMatcher:
    """Whole-word name matching shared by ``redact`` and ``scrub_identity``."""

    def __init__(self, candidate_names: Sequence[str]) -> None:
        self.tokens = name_tokens(candidate_names)
        self.cjk_tokens = {t for t in self.tokens if _is_cjk(t)}
        self.word_tokens = self.tokens - self.cjk_tokens
        self.word_like = {t for t in self.word_tokens
                          if t in WORD_LIKE_NAME_TOKENS and t not in _STOPWORD_TOKENS}
        self.raw_forms = set(_raw_name_parts(candidate_names))
        self.handle_forms = _handle_forms(self.tokens)

    def is_plain_token(self, word: str, folded: str) -> bool:
        return (folded in self.word_tokens and folded not in self.word_like
                and (folded not in _STOPWORD_TOKENS or word[:1].isupper()))

    def _neighbour_is_name(self, word: str | None) -> bool:
        if word is None:
            return False
        if word == placeholder(CANDIDATE):
            return True
        if len(word) == 1:
            return word.isalpha() and word.isupper()  # an initial
        folded = _fold(word)
        return folded in self.word_tokens or folded in _HONORIFIC_TOKENS

    def _used_as_name(self, words: list[str], i: int, line: str, starts: list[int],
                      ends: list[int], header: bool, possessive: bool) -> bool:
        word = words[i]
        base = word[:-2] if possessive else word
        if not (base.istitle() or base in self.raw_forms):
            return False
        if header or possessive:
            return True
        if self._neighbour_is_name(words[i - 1] if i else None) or self._neighbour_is_name(
                words[i + 1] if i + 1 < len(words) else None):
            return True
        sentence_start = i == 0 or _SENTENCE_BREAK_RE.search(line[ends[i - 1]:starts[i]])
        return bool(sentence_start and i + 1 < len(words)
                    and _fold(words[i + 1]) in _SUMMARY_VERBS)

    def replacement(self, words: list[str], i: int, line: str, starts: list[int],
                    ends: list[int], header: bool) -> str | None:
        """[CANDIDATE] (with a possessive suffix kept) when token ``i`` is the name."""
        word = words[i]
        if word.startswith("["):
            return None
        folded = _fold(word)
        possessive = folded.endswith("'s")
        base = folded[:-2] if possessive else folded
        suffix = word[-2:] if possessive else ""
        if base in self.word_like:
            if self._used_as_name(words, i, line, starts, ends, header, possessive):
                return placeholder(CANDIDATE) + suffix
        elif self.is_plain_token(word, base):
            return placeholder(CANDIDATE) + suffix
        letters = folded.strip("0123456789")
        if letters != folded and letters in self.word_tokens:
            return placeholder(CANDIDATE)
        if re.sub(r"[^a-z]", "", folded) in self.handle_forms:
            return placeholder(CANDIDATE)
        return None

    def redact_words(self, text: str) -> tuple[str, int]:
        """Replace name words line by line; returns (text, replacements)."""
        if not self.word_tokens:
            return text, 0
        out: list[str] = []
        replaced = 0
        non_empty = 0
        for line in text.split("\n"):
            if not line.strip():
                out.append(line)
                continue
            header = non_empty < HEADER_LINES or bool(_CONTACT_PLACEHOLDER_RE.search(line))
            non_empty += 1
            matches = list(_TEXT_TOKEN_RE.finditer(line))
            words = [m.group() for m in matches]
            starts = [m.start() for m in matches]
            ends = [m.end() for m in matches]
            pieces: list[str] = []
            last = 0
            for i, m in enumerate(matches):
                repl = self.replacement(words, i, line, starts, ends, header)
                if repl is None:
                    continue
                pieces.append(line[last:m.start()])
                pieces.append(repl)
                last = m.end()
                replaced += 1
            pieces.append(line[last:])
            out.append("".join(pieces))
        return "\n".join(out), replaced


class _Redactor:
    def __init__(self, candidate_names: Sequence[str], extra_identifiers: Sequence[str]) -> None:
        self.counts: Counter[str] = Counter()
        self.names_ = _NameMatcher(candidate_names)
        self.tokens = self.names_.tokens
        self.handle_forms = self.names_.handle_forms
        idents = sorted({i.strip() for i in extra_identifiers if i and i.strip()},
                        key=len, reverse=True)
        self.identifier_patterns = [p for p in map(_identifier_pattern, idents) if p]

    def sub(self, rx: re.Pattern[str], name: str, text: str,
            repl: Callable[[re.Match[str]], str] | None = None) -> str:
        """Replace matches with [name] (or ``repl``'s result) and count real replacements."""
        token = placeholder(name)

        def wrapped(m: re.Match[str]) -> str:
            out = repl(m) if repl else token
            self.counts[name] += out.count(token) - m.group().count(token)
            return out

        return rx.sub(wrapped, text)

    # -- stages ------------------------------------------------------------------------

    def identifiers(self, text: str) -> str:
        for rx in self.identifier_patterns:
            text = self.sub(rx, CANDIDATE, text)
        return text

    def _whole_line(self, m: re.Match[str]) -> str:
        """The placeholder, or a field-by-field redaction of an over-long line."""
        line = m.group()
        if len(line) <= MAX_PERSONAL_LINE_CHARS:
            return placeholder(PERSONAL_DETAIL)
        parts = _FIELD_SPLIT_RE.split(line)
        for i in range(0, len(parts), 2):
            field = parts[i]
            if field.strip() and _is_personal_line(field.strip()):
                lead = field[:len(field) - len(field.lstrip())]
                tail = field[len(field.rstrip()):]
                parts[i] = f"{lead}{placeholder(PERSONAL_DETAIL)}{tail}"
        return "".join(parts)

    def personal_lines(self, text: str) -> str:
        for rx in _PERSONAL_LINE_RES:
            text = self.sub(rx, PERSONAL_DETAIL, text, self._whole_line)
        text = self.sub(_WEAK_DASH_LINE_RE, PERSONAL_DETAIL, text,
                        lambda m: placeholder(PERSONAL_DETAIL)
                        if _personal_value(m.group("value")) else m.group())
        text = self.sub(_AGE_LINE_RE, PERSONAL_DETAIL, text,
                        lambda m: placeholder(PERSONAL_DETAIL)
                        if len(m.group()) <= MAX_AGE_LINE_CHARS
                        or _PERSONAL_HEADED_RE.match(m.group()) else m.group())
        text = self.sub(_LABEL_INLINE_RE, PERSONAL_DETAIL, text)
        text = self.sub(_ID_NUMBER_RE, PERSONAL_DETAIL, text)
        text = self.sub(_MARITAL_FIELD_RE, PERSONAL_DETAIL, text,
                        lambda m: m.group()[:len(m.group()) - len(m.group().lstrip())]
                        + placeholder(PERSONAL_DETAIL))
        return self.sub(_PRONOUN_RE, PERSONAL_DETAIL, text)

    def emails(self, text: str) -> str:
        for rx in _EMAIL_RES:
            text = self.sub(rx, EMAIL, text)
        return text

    def urls(self, text: str) -> str:
        def repl(m: re.Match[str]) -> str:
            raw = url = m.group()
            while url and url[-1] in _TRAILING_PUNCT and not (
                    url[-1] == ")" and url.count("(") >= url.count(")")):
                url = url[:-1]
            host = re.sub(r"^(?:https?://)?(?:www\.)?", "", url, flags=re.I)
            host = re.split(r"[/:?#]", host, maxsplit=1)[0].lower()
            if m.group("tld") is not None:  # bare domain, no scheme or www.
                path = (m.group("path") or "").rstrip(_TRAILING_PUNCT)
                if (m.group("tld") not in _COMMON_TLDS or host in _TECH_HOSTS
                        or (not path and not self._contains_name(host))
                        or (_BRAND_PATH_RE.fullmatch(path) and not self._contains_name(host))):
                    return raw
            if host.endswith(_LINKEDIN_HOSTS):
                kind = LINKEDIN_URL
            elif host.endswith(_GITHUB_HOSTS):
                kind = GITHUB_URL
            else:
                kind = URL
            self.counts[kind] += 1
            return placeholder(kind) + raw[len(url):]

        return _URL_RE.sub(repl, text)

    def _contains_name(self, host: str) -> bool:
        label = re.sub(r"[^a-z]", "", host.rsplit(".", 1)[0])
        return any(f in label for f in self.handle_forms) or any(
            len(t) >= 4 and t in label for t in self.tokens)

    def _is_org_handle(self, m: re.Match[str], last_kept: int | None) -> bool:
        """"@Google" after a role/event word (or chained to a kept one) is an employer or
        event, not a social handle."""
        name = m.group(1)
        if not _ORG_HANDLE_RE.fullmatch(name):
            return False
        folded = _fold(name)
        if folded in self.tokens or re.sub(r"[^a-z]", "", folded) in self.handle_forms or any(
                len(t) >= 4 and t in folded for t in self.tokens):
            return False
        line_start = m.string.rfind("\n", 0, m.start()) + 1
        before = m.string[line_start:m.start()]
        preceding = before.split()
        if _ROLE_WORD_RE.search(" ".join(preceding[-_ROLE_WINDOW_WORDS:])):
            return True
        if last_kept is not None and last_kept >= line_start:
            between = m.string[last_kept:m.start()].split()
            return len(between) <= _HANDLE_CHAIN_WORDS
        return False

    def handles(self, text: str) -> str:
        last_kept: int | None = None

        def repl(m: re.Match[str]) -> str:
            nonlocal last_kept
            if m.group(1) in _CODE_ANNOTATIONS:
                return m.group()
            if self._is_org_handle(m, last_kept):
                last_kept = m.end()
                return m.group()
            return placeholder(HANDLE)

        return self.sub(_HANDLE_RE, HANDLE, text, repl)

    def addresses(self, text: str) -> str:
        text = self.sub(_PO_BOX_RE, ADDRESS, text)
        return self.sub(_STREET_RE, ADDRESS, text)

    def zips(self, text: str) -> str:
        return self.sub(_ZIP_RE, ZIP, text, lambda m: f"{m.group(1)}{m.group(2)}[{ZIP}]")

    def locations(self, text: str) -> str:
        return self.sub(_LOCATION_RE, LOCATION, text,
                        lambda m: placeholder(LOCATION) if _is_location(m) else m.group())

    def phones(self, text: str) -> str:
        def labelled(m: re.Match[str]) -> str:
            number = _trim_phone(m.group(2))
            if number is None:
                return m.group()
            return m.group(1) + placeholder(PHONE) + m.group(2)[len(number):]

        text = self.sub(_PHONE_LABELLED_RE, PHONE, text, labelled)
        text = self.sub(_PHONE_INTL_RE, PHONE, text,
                        lambda m: placeholder(PHONE) if _looks_like_phone(m.group())
                        else m.group())
        return self.sub(_PHONE_RE, PHONE, text,
                        lambda m: placeholder(PHONE) if _looks_like_phone(
                            m.group(), before=m.string[max(0, m.start() - 16):m.start()],
                            after=m.string[m.end():m.end() + 24]) else m.group())

    def names(self, text: str) -> str:
        if not self.tokens:
            return text
        for token in self.names_.cjk_tokens:
            n = text.count(token)
            if n:
                text = text.replace(token, placeholder(CANDIDATE))
                self.counts[CANDIDATE] += n
        text, replaced = self.names_.redact_words(text)
        self.counts[CANDIDATE] += replaced
        return text

    def honorifics(self, text: str) -> str:
        return _HONORIFIC_RE.sub("", text)

    def education_years(self, text: str) -> str:
        lines = text.split("\n")
        sections = _section_kinds(lines)
        is_edu = [_is_education_line(line, section)
                  for line, section in zip(lines, sections, strict=True)]
        has_year = [bool(_YEAR_RE.search(line)) for line in lines]
        date_only = [y and bool(_DATE_ONLY_RE.fullmatch(line)) for line, y in
                     zip(lines, has_year, strict=True)]
        targets: set[int] = set()
        for i in range(len(lines)):
            if is_edu[i] or sections[i] == "education":
                targets.add(i)
            if is_edu[i] and not has_year[i]:
                targets.update(_dates_of_entry(i, lines, date_only))
        for i in sorted(targets):
            line = self.sub(_YEAR_RE, YEAR, lines[i])
            line = self.sub(_SHORT_RANGE_RE, YEAR, line, lambda m: m.group(1) + "[YEAR]")
            lines[i] = self.sub(_APOSTROPHE_YEAR_RE, YEAR, line)
        return "\n".join(lines)


def _is_personal_line(line: str) -> bool:
    """Would ``line`` on its own be replaced by one of the personal-detail rules?"""
    if any(rx.search(line) for rx in _PERSONAL_LINE_RES):
        return True
    m = _WEAK_DASH_LINE_RE.search(line)
    if m is not None and _personal_value(m.group("value")):
        return True
    return bool(_AGE_LINE_RE.search(line)) and (len(line) <= MAX_AGE_LINE_CHARS
                                                or bool(_PERSONAL_HEADED_RE.match(line)))


def _personal_value(value: str) -> bool:
    """A short value with no year range, job title or further field ("34", "excellent")."""
    value = value.strip()
    return (len(value) <= MAX_PERSONAL_VALUE_CHARS and not _YEAR_RANGE_RE.search(value)
            and not _JOB_TITLE_RE.search(value) and not re.search(_SEP, value))


def _is_location(m: re.Match[str]) -> bool:
    words = re.split(r"[ \t]+", m.group("city"))
    if any(w.strip(".,'’").lower() in _NON_CITY_WORDS for w in words):
        return False
    following = m.string[m.end():m.end() + 40]
    abbr = m.group("abbr")
    if abbr is not None:  # "City ST" with no comma
        if not all(w == "St." or (w[:1].isupper() and w[1:].islower()) for w in words):
            return False
        return m.group("zip") is not None or abbr not in _AMBIGUOUS_ABBR
    if m.group("place") in _DEGREE_STATES:
        if len(words) == 1 and words[0].lower().endswith(_FIELD_SUFFIXES):
            return False  # "Cybersecurity, MS": a field of study
        if _MS_PRODUCT_RE.match(following) or _DEGREE_SCHOOL_RE.match(following):
            return False
    return True


def _identifier_pattern(ident: str) -> re.Pattern[str] | None:
    if len(ident) < 3:
        return None
    digits = re.sub(r"\D", "", ident)
    if re.fullmatch(r"[\d\s().+\-/]+", ident):
        if len(digits) < 7:
            return None
        core = digits[-10:]
        sep = r"[\s().\-/]{0,3}"
        body = sep.join(core)
        return re.compile(rf"(?<![\d+])(?:\+?\d{{1,3}}[\s.\-]{{0,3}})?\(?{body}(?!\d)")
    if "://" in ident or ident.lower().startswith("www."):
        bare = re.sub(r"^(?:https?://)?(?:www\.)?", "", ident, flags=re.I).rstrip("/")
        if len(bare) < 4:
            return None
        return re.compile(rf"(?:https?://)?(?:www\.)?{re.escape(bare)}/?", re.I)
    body = r"\s+".join(re.escape(part) for part in ident.split())
    left = r"(?<!\w)" if re.match(r"\w", ident) else ""
    right = r"(?!\w)" if re.search(r"\w$", ident) else ""
    return re.compile(left + body + right, re.I)


def _is_year(group: str) -> bool:
    return len(group) == 4 and 1900 <= int(group) <= 2099


def _trim_phone(number: str) -> str | None:
    """The longest prefix of a labelled number (cut at group boundaries) with 7-15 digits,
    or None when even the first group is too short."""
    groups = list(re.finditer(r"\d+", number))
    end = len(groups)
    while end and sum(len(g.group()) for g in groups[:end]) > _MAX_PHONE_DIGITS:
        end -= 1
    # "5550102030 2019": a trailing year after a complete number is not part of it.
    while end > 1 and _is_year(groups[end - 1].group()) and sum(
            len(g.group()) for g in groups[:end - 1]) >= 10:
        end -= 1
    if not end or sum(len(g.group()) for g in groups[:end]) < _MIN_PHONE_DIGITS:
        return None
    return number[:groups[end - 1].end()]


def _looks_like_phone(s: str, *, before: str = "", after: str = "") -> bool:
    digits = re.sub(r"\D", "", s)
    if not _MIN_PHONE_DIGITS <= len(digits) <= _MAX_PHONE_DIGITS or _IPV4_RE.fullmatch(s):
        return False
    groups = re.findall(r"\d+", s)
    marked = s.startswith("+") or "(" in s
    if all(_is_year(g) for g in groups) or (
            not marked and len(groups[0]) <= 3 and all(_is_year(g) for g in groups[1:])):
        return False
    if len(groups) == 3 and not marked:
        a, b, c = (int(g) for g in groups)
        if _is_year(groups[0]) and 1 <= b <= 12 and 1 <= c <= 31:
            return False  # ISO date
        if _is_year(groups[2]) and 1 <= a <= 31 and 1 <= b <= 31:
            return False  # dd.mm.yyyy / mm-dd-yyyy
    if len(groups) == 2 and not marked and len(digits) < 10:
        return False  # a numeric range ("1000-5000"), not a phone
    separators = set(re.sub(r"[\d()+]", "", s))
    if separators <= {" "} and not marked:
        if min(len(g) for g in groups) < 3:
            return False
        if _is_quantity(groups, before, after):
            return False
    return True


def _is_quantity(groups: list[str], before: str, after: str) -> bool:
    """"250 000 000 events": a lead group of 1-3 digits then exact thousands, followed by
    a unit or count noun or preceded by a currency sign or an approximation word."""
    if len(groups) < 2 or len(groups[0]) > 3 or any(len(g) != 3 for g in groups[1:]):
        return False
    return bool(_QUANTITY_AFTER_RE.match(after) or _QUANTITY_BEFORE_RE.search(before))


def _dates_of_entry(i: int, lines: Sequence[str], date_only: Sequence[bool]) -> list[int]:
    """Date-only lines that belong to the year-less education line ``i``: the following
    one or two (skipping a blank line), else the line just before it."""
    after: list[int] = []
    for j in (i + 1, i + 2):
        if j >= len(lines):
            break
        if date_only[j]:
            after.append(j)
        elif lines[j].strip():
            break
    if after:
        return after
    return [i - 1] if i > 0 and date_only[i - 1] else []


def _is_education_line(line: str, section: str | None) -> bool:
    """Education keywords mark a line, except that inside another section only a degree
    counts, and outside any section an institution word next to a job title is a job."""
    if _DEGREE_RE.search(line):
        return True
    if not _INSTITUTION_RE.search(line) or section == "other":
        return False
    return not _JOB_TITLE_RE.search(line)


def _heading_kind(line: str) -> str | None:
    stripped = line.strip().rstrip(":").strip()
    if (not stripped or len(stripped) > 40 or any(ch.isdigit() for ch in stripped)
            or _EDU_RE.search(stripped)):
        return None
    norm = re.sub(r"\s+", " ", re.sub(r"[^a-z& ]", " ", stripped.lower())).strip()
    if norm in _EDU_HEADINGS:
        return "education"
    if any(norm == h or norm.startswith(h + " ") or norm.endswith(" " + h)
           for h in _OTHER_HEADINGS):
        return "other"
    letters = [c for c in stripped if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(stripped.split()) <= 4:
        return "other"
    return None


def _section_kinds(lines: Sequence[str]) -> list[str | None]:
    """Per line: "education" (within ``_EDU_SECTION_MAX_LINES`` of an education heading),
    "other" (after any other recognised heading) or None (before any heading)."""
    kinds: list[str | None] = [None] * len(lines)
    current: str | None = None
    remaining = 0
    for i, line in enumerate(lines):
        kind = _heading_kind(line)
        if kind == "education":
            current, remaining = "education", _EDU_SECTION_MAX_LINES
            continue
        if kind == "other":
            current, remaining = "other", 0
        if current == "education":
            if remaining > 0:
                kinds[i] = "education"
                remaining -= 1
        else:
            kinds[i] = current
    return kinds


def scrub_identity(text: str, *, candidate_names: Sequence[str] = (),
                   extra_identifiers: Sequence[str] = ()) -> tuple[str, int]:
    """Backstop after ``redact``: replace any name token (by the same rules as ``redact``)
    or literal identifier still present with [CANDIDATE]. Returns (text, replacements); 0
    is the expected case."""
    matcher = _NameMatcher(candidate_names)
    replaced = 0
    literals = [i.strip() for i in extra_identifiers if i and len(i.strip()) >= 3]
    literals += sorted(matcher.cjk_tokens)
    for literal in sorted(literals, key=len, reverse=True):
        text, n = re.subn(re.escape(literal), placeholder(CANDIDATE), text, flags=re.I)
        replaced += n
    text, n = matcher.redact_words(text)
    return text, replaced + n


def redact(text: str, *, candidate_names: Sequence[str] = (),
           extra_identifiers: Sequence[str] = (),
           redact_education_years: bool = True) -> RedactionResult:
    r = _Redactor(candidate_names, extra_identifiers)
    t = r.identifiers(text)
    t = r.personal_lines(t)
    t = r.emails(t)
    t = r.urls(t)
    t = r.handles(t)
    t = r.addresses(t)
    t = r.zips(t)
    t = r.locations(t)
    t = r.phones(t)
    t = r.names(t)
    t = r.honorifics(t)
    if redact_education_years:
        t = r.education_years(t)
    return RedactionResult(text=t, counts={k: v for k, v in r.counts.items() if v},
                           version=REDACTION_VERSION)
