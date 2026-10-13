from __future__ import annotations

import re

import pytest

from applicant_ranking.resumes.redact import (
    PLACEHOLDERS,
    REDACTION_VERSION,
    name_tokens,
    redact,
    scrub_identity,
)

SAMPLE = """\
José Álvarez-Núñez (he/him)
jose.alvarez@example.com | +1 (555) 010-2030 | 415.555.0199
linkedin.com/in/jose-alvarez | https://github.com/jalvarez | jose.dev/portfolio
742 Evergreen Terrace, Springfield, OR 97477
Chicago, IL
Date of Birth: 03/04/1979
Marital status: married

SUMMARY
Jose has 12 years building Node.js and ASP.NET services; led the GCP migration in 2019-2021.

EXPERIENCE
Staff Engineer, Northwind (2020 - present)
- Cut p99 latency 45%; on-call for 30M req/day.
Data Engineer, Beta Corp (2016-2019)

EDUCATION
B.Sc. Computer Science, State University, 1997 - 2001
"""

EXPECTED = """\
[CANDIDATE] [CANDIDATE]-[CANDIDATE] [REDACTED PERSONAL DETAIL]
[EMAIL] | [PHONE] | [PHONE]
[LINKEDIN_URL] | [GITHUB_URL] | [URL]
[ADDRESS], [LOCATION] [ZIP]
[LOCATION]
[REDACTED PERSONAL DETAIL]
[REDACTED PERSONAL DETAIL]

SUMMARY
[CANDIDATE] has 12 years building Node.js and ASP.NET services; led the GCP migration in 2019-2021.

EXPERIENCE
Staff Engineer, Northwind (2020 - present)
- Cut p99 latency 45%; on-call for 30M req/day.
Data Engineer, Beta Corp (2016-2019)

EDUCATION
B.Sc. Computer Science, State University, [YEAR] - [YEAR]
"""


def _red(text: str, *names: str, **kwargs: object) -> str:
    return redact(text, candidate_names=names, **kwargs).text  # type: ignore[arg-type]


def test_reference_sample() -> None:
    r = redact(SAMPLE, candidate_names=["José Álvarez-Núñez"])
    assert r.text == EXPECTED
    assert r.counts == {
        "CANDIDATE": 4, "REDACTED PERSONAL DETAIL": 3, "EMAIL": 1, "PHONE": 2,
        "LINKEDIN_URL": 1, "GITHUB_URL": 1, "URL": 1, "ADDRESS": 1, "ZIP": 1, "LOCATION": 2,
        "YEAR": 2,
    }
    assert r.version == REDACTION_VERSION


def test_counts_use_placeholder_names() -> None:
    r = redact(SAMPLE, candidate_names=["José Álvarez-Núñez"])
    assert set(r.counts) <= set(PLACEHOLDERS)


def test_counts_are_replacements_not_pattern_hits() -> None:
    r = redact("Ana Ana ana ANA, and Ana's work", candidate_names=["Ana"])
    assert r.text == "[CANDIDATE] [CANDIDATE] [CANDIDATE] [CANDIDATE], and [CANDIDATE]'s work"
    assert r.counts == {"CANDIDATE": 5}
    # a match that is left alone is not counted
    r = redact("Released 2021-03-15, cluster 192.168.100.200", candidate_names=[])
    assert r.counts == {}


def test_nothing_to_redact() -> None:
    r = redact("Built data pipelines in Python.")
    assert r.text == "Built data pipelines in Python."
    assert r.counts == {}


# --- tech tokens and numbers that must survive ---------------------------------------------


@pytest.mark.parametrize("text", [
    "Node.js, ASP.NET, C#, .NET 8, F#, Vue.js, D3.js, Socket.io",
    "ASP.NET/C# and Node.js/Express on AWS us-east-1",
    "Upgraded from v2.3.1 to v10.0.4",
    "VPC 10.0.0.1 and 192.168.100.200",
    "Scaled from 1000-5000 users",
    "Latency buckets 40 50 60 70 ms",
    "Released 2021-03-15 and 15.03.2021",
    "Led migration 2019-2021, then 2021 - 2023",
    "Grew revenue $1,200,000 (40%)",
    "SAT 1520, GPA-free",
    "e.g. U.S./Canada teams",
    "Engineer @ Google; SWE at Stripe",
    "Java: @Override, @Autowired; npm @angular/core, @types/node",
])
def test_tech_tokens_and_numbers_are_kept(text: str) -> None:
    assert _red(text) == text


# --- contact details ----------------------------------------------------------------------------


@pytest.mark.parametrize("phone", [
    "+1 (555) 010-2030", "415.555.0199", "555-010-2030", "(555) 010 2030", "+44 20 7946 0958",
    "+91 98765 43210", "98765 43210", "1-800-555-0199", "+15550102030", "020 7946 0958",
])
def test_phone_formats(phone: str) -> None:
    assert _red(f"Call {phone} today") == "Call [PHONE] today"


def test_labelled_bare_digit_phone() -> None:
    assert _red("Mobile: 5550102030") == "Mobile: [PHONE]"


@pytest.mark.parametrize("email", ["jose.alvarez@example.com", "a+b@mail.co.uk",
                                   "jose [at] example [dot] com", "jose at gmail dot com"])
def test_emails(email: str) -> None:
    assert _red(f"Email {email}.") == "Email [EMAIL]."


@pytest.mark.parametrize(("url", "placeholder"), [
    ("https://www.linkedin.com/in/jose-alvarez/", "[LINKEDIN_URL]"),
    ("linkedin.com/in/jose", "[LINKEDIN_URL]"),
    ("www.linkedin.com/in/jose", "[LINKEDIN_URL]"),
    ("https://github.com/jalvarez", "[GITHUB_URL]"),
    ("github.com/jalvarez/repo", "[GITHUB_URL]"),
    ("jose.dev/portfolio", "[URL]"),
    ("http://example.org", "[URL]"),
    ("www.behance.net/jose", "[URL]"),
    ("https://medium.com/@jose/post?x=1", "[URL]"),
])
def test_urls(url: str, placeholder: str) -> None:
    assert _red(f"See {url}, thanks") == f"See {placeholder}, thanks"


def test_url_in_parentheses_keeps_the_parenthesis() -> None:
    assert _red("(https://jose.dev/work)") == "([URL])"


def test_bare_domain_without_path_kept_unless_it_contains_the_name() -> None:
    assert _red("Grew example.com traffic", "Jose Alvarez") == "Grew example.com traffic"
    assert _red("Site: josealvarez.com", "Jose Alvarez") == "Site: [URL]"


def test_handles() -> None:
    assert _red("Twitter @jalvarez_dev, IG @jose.alvarez") == "Twitter [HANDLE], IG [HANDLE]"


# --- addresses and locations --------------------------------------------------------------------


@pytest.mark.parametrize(("text", "expected"), [
    ("123 N Main St, Apt 4", "[ADDRESS]"),
    ("350 5th Avenue, Suite 300", "[ADDRESS]"),
    ("1600 Pennsylvania Ave NW", "[ADDRESS]"),
    ("10 Downing Street", "[ADDRESS]"),
    ("PO Box 1234", "[ADDRESS]"),
    ("P.O. Box 77", "[ADDRESS]"),
])
def test_street_addresses(text: str, expected: str) -> None:
    assert _red(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("Chicago, IL", "[LOCATION]"),
    ("Chicago, IL 60601", "[LOCATION] [ZIP]"),
    ("Springfield, OR 97477-1234", "[LOCATION] [ZIP]"),
    ("Austin, Texas", "[LOCATION]"),
    ("New York, NY", "[LOCATION]"),
    ("St. Louis, Missouri", "[LOCATION]"),
    ("Winston-Salem, NC", "[LOCATION]"),
    ("Washington, D.C.", "[LOCATION]"),
    ("Bengaluru, India", "[LOCATION]"),
    ("Toronto, Ontario", "[LOCATION]"),
    ("Acme Corp, Mountain View, CA", "Acme Corp, [LOCATION]"),
    ("Remote, US", "Remote, US"),
])
def test_locations_and_zips(text: str, expected: str) -> None:
    assert _red(text) == expected


@pytest.mark.parametrize("text", [
    "Boston University, MA",
    "Computer Science, MS",
    "Senior Software Engineer, CA",
    "Python, Go, Rust",
    "Joined IN 2019",
])
def test_non_locations_are_kept(text: str) -> None:
    assert _red(text, redact_education_years=False) == text


# --- personal-detail lines -----------------------------------------------------------------------


@pytest.mark.parametrize("line", [
    "Date of Birth: 03/04/1979", "DOB: 1979-04-03", "D.O.B. : 3 April 1979", "DOB 01/02/1990",
    "Born on 1 May 1990 in Pune", "Age: 34", "Age - 34", "Age 34", "I am 34 years old.",
    "Gender: Female", "Sex: M", "Pronouns: they/them", "Marital Status: Married",
    "Nationality: Indian", "Citizenship: US", "Visa status: H-1B", "Work Authorization: GC",
    "Work permit: yes", "Religion: Hindu", "Ethnicity: Hispanic", "Race: Asian",
    "Disability: none", "Health: excellent", "Height: 180 cm", "Weight: 75 kg",
    "Photo: attached", "Father's Name: Ramesh", "Passport No: X123",
    "• Marital status: single",
    "Personal details | Nationality: Canadian | Languages: English",
    "Personal: Languages English; Age: 34",
    "Contact | Sex: M | Remote",
])
def test_personal_detail_lines_are_replaced_whole(line: str) -> None:
    out = _red(f"Header\n{line}\nPython developer")
    assert out == "Header\n[REDACTED PERSONAL DETAIL]\nPython developer"


@pytest.mark.parametrize("line", [
    "Usage: 40% CPU", "Stage: Series B", "Coverage: 95%", "Essex: field office",
    "Agile: Scrum, Kanban", "Image: Docker", "Raced cars as a hobby",
    "Born and raised a builder", "System health: 99.9% uptime dashboards",
    "Big picture: aligned 4 teams", "Model weight: 7B parameters", "Payments: Visa, Amex",
])
def test_similar_looking_lines_are_kept(line: str) -> None:
    assert _red(line) == line


@pytest.mark.parametrize("pronouns", ["(she/her)", "she/her/hers", "(they/them)", "he/they"])
def test_inline_pronouns(pronouns: str) -> None:
    assert _red(f"Jordan Lee {pronouns}", "Jordan Lee") == (
        "[CANDIDATE] [CANDIDATE] [REDACTED PERSONAL DETAIL]")


def test_he_slash_she_is_not_a_pronoun_set() -> None:
    assert _red("so that he/she can deploy") == "so that he/she can deploy"


# --- names -----------------------------------------------------------------------------------


def test_names_are_accent_and_case_insensitive_whole_words() -> None:
    text = "JOSE ALVAREZ, jose alvarez-nunez, José Núñez; Joseph and Alvarezsoft stay"
    assert _red(text, "José Álvarez-Núñez") == (
        "[CANDIDATE] [CANDIDATE], [CANDIDATE] [CANDIDATE]-[CANDIDATE], [CANDIDATE] "
        "[CANDIDATE]; Joseph and Alvarezsoft stay")


@pytest.mark.parametrize(("record", "written"), [
    ("Łukasz Wałęsa", "Lukasz Walesa"),
    ("Søren Kierkegaard", "Soren Kierkegaard"),
    ("Đặng Thị Hương", "Dang Thi Huong"),
    ("Zoë Æthelstan", "Zoe Aethelstan"),
])
def test_letters_without_decomposition_are_folded(record: str, written: str) -> None:
    assert _red(written, record) == " ".join(["[CANDIDATE]"] * len(written.split()))


def test_lever_name_written_last_first() -> None:
    assert _red("Kim Lee", "Lee, Kim") == "[CANDIDATE] [CANDIDATE]"


def test_single_letter_initials_and_suffixes_are_not_tokens() -> None:
    assert name_tokens(["J. Robert Smith Jr.", "Dr. Ana Lee III"]) == {
        "robert", "smith", "ana", "lee"}
    assert _red("Level II support, Jr developer, Robert", "J. Robert Smith Jr.") == (
        "Level II support, Jr developer, [CANDIDATE]")


def test_lower_case_particles_only_count_when_nothing_else_is_left() -> None:
    assert name_tokens(["Ludwig van Beethoven"]) == {"ludwig", "beethoven"}
    assert name_tokens(["Maria do Carmo"]) == {"maria", "carmo"}
    assert name_tokens(["Van Le"]) == {"van", "le"}
    assert name_tokens(["Đỗ Minh"]) == {"do", "minh"}
    assert name_tokens(["de la"]) == {"de", "la"}


def test_function_word_name_tokens_only_match_capitalised_words() -> None:
    text = "AN NGUYEN\nAn built an API to do what Do asked; An's code"
    assert _red(text, "An Nguyen", "Đỗ Minh") == (
        "[CANDIDATE] [CANDIDATE]\n[CANDIDATE] built an API to do what [CANDIDATE] asked; "
        "[CANDIDATE]'s code")


def test_possessives_digits_and_handle_forms() -> None:
    text = "Jose's team; skype jose_alvarez99; user josealvarez; login jalvarez; ALVAREZJ"
    assert _red(text, "Jose Alvarez") == (
        "[CANDIDATE]'s team; skype [CANDIDATE]_[CANDIDATE]; user [CANDIDATE]; "
        "login [CANDIDATE]; [CANDIDATE]")


def test_apostrophe_names() -> None:
    assert _red("Sean O’Brien and O'Brien's code", "Seán O'Brien") == (
        "[CANDIDATE] [CANDIDATE] and [CANDIDATE]'s code")


def test_cjk_names_match_as_substrings() -> None:
    assert _red("李明是数据工程师", "李明") == "[CANDIDATE]是数据工程师"


def test_several_names() -> None:
    assert _red("Bob Smith (née Robert Jones)", "Bob Smith", "Robert Jones") == (
        "[CANDIDATE] [CANDIDATE] (née [CANDIDATE] [CANDIDATE])")


def test_name_inside_placeholder_is_not_touched() -> None:
    assert _red("[EMAIL] [YEAR] Email Year", "Email Year") == (
        "[EMAIL] [YEAR] [CANDIDATE] [CANDIDATE]")


@pytest.mark.parametrize("title", ["Mr.", "Mrs", "Ms.", "Miss", "Mx", "Dr.", "dr"])
def test_honorifics_before_candidate_are_removed(title: str) -> None:
    assert _red(f"Contact {title} Ana Lee", "Ana Lee") == "Contact [CANDIDATE] [CANDIDATE]"


def test_honorific_not_before_candidate_is_kept() -> None:
    assert _red("Worked with Dr. Smith", "Ana Lee") == "Worked with Dr. Smith"


# --- extra identifiers -------------------------------------------------------------------------


def test_extra_identifiers_are_literals_replaced_with_candidate() -> None:
    r = redact("Reach me: JOSE@Example.com or (555) 010-2030, ref ABC-123-XYZ.",
               extra_identifiers=["jose@example.com", "+1 555-010-2030", "abc-123-xyz", "ab"])
    assert r.text == "Reach me: [CANDIDATE] or [CANDIDATE], ref [CANDIDATE]."
    assert r.counts == {"CANDIDATE": 3}


def test_extra_identifier_urls_match_with_or_without_scheme() -> None:
    r = redact("Profile: linkedin.com/in/jose-a and https://www.linkedin.com/in/jose-a/",
               extra_identifiers=["https://www.linkedin.com/in/jose-a"])
    assert r.text == "Profile: [CANDIDATE] and [CANDIDATE]"


def test_empty_and_short_identifiers_are_ignored() -> None:
    assert _red("a b c", extra_identifiers=["", "  ", "a", "12345"]) == "a b c"


# --- education years -----------------------------------------------------------------------


@pytest.mark.parametrize(("line", "expected"), [
    ("B.S. Computer Science, State University, 2015 - 2019",
     "B.S. Computer Science, State University, [YEAR] - [YEAR]"),
    ("MBA, 2010", "MBA, [YEAR]"),
    ("PhD in Physics (2012)", "PhD in Physics ([YEAR])"),
    ("MS in Data Science, 2018", "MS in Data Science, [YEAR]"),
    ("Bachelor's degree, 2015-19", "Bachelor's degree, [YEAR]-[YEAR]"),
    ("Graduated May 2016", "Graduated May [YEAR]"),
    ("Westfield High School, Class of '09", "Westfield High School, Class of [YEAR]"),
    ("Diploma in Networking, 2004", "Diploma in Networking, [YEAR]"),
    ("Master of Engineering (2001-2003)", "Master of Engineering ([YEAR]-[YEAR])"),
])
def test_years_on_education_lines(line: str, expected: str) -> None:
    assert _red(line) == expected


@pytest.mark.parametrize("line", [
    "Scrum Master, Acme (2018-2020)",
    "Proficient in MS Office 2019 and MS SQL Server 2019",
    "Senior BA at Acme (2019-2021)",
    "Windows Server 2016 administration since 2017",
])
def test_years_on_other_lines_are_kept(line: str) -> None:
    assert _red(line) == line


def test_education_section_redacts_years_until_next_heading() -> None:
    text = ("EDUCATION\nState University\nComputer Science, 2015 - 2019\nThesis award 2019\n\n"
            "Experience\nData Engineer, Acme (2019 - 2022)")
    assert _red(text) == (
        "EDUCATION\nState University\nComputer Science, [YEAR] - [YEAR]\nThesis award [YEAR]\n\n"
        "Experience\nData Engineer, Acme (2019 - 2022)")


def test_all_caps_school_name_does_not_end_the_section() -> None:
    text = "EDUCATION\nSTATE UNIVERSITY\nComputer Science\n2015 - 2019"
    assert _red(text) == "EDUCATION\nSTATE UNIVERSITY\nComputer Science\n[YEAR] - [YEAR]"


def test_date_line_after_a_yearless_education_line() -> None:
    text = "Data Engineer, Beta Corp\n2018 - 2021\nB.S. Computer Science, State University\n" \
           "Aug 2014 - May 2018"
    assert _red(text) == ("Data Engineer, Beta Corp\n2018 - 2021\n"
                          "B.S. Computer Science, State University\nAug [YEAR] - May [YEAR]")


def test_date_line_before_a_yearless_education_line_when_nothing_follows() -> None:
    text = "2014 - 2018\nM.Sc. Statistics, Tech University"
    assert _red(text) == "[YEAR] - [YEAR]\nM.Sc. Statistics, Tech University"


def test_education_years_can_be_kept() -> None:
    r = redact(SAMPLE, candidate_names=["José Álvarez-Núñez"], redact_education_years=False)
    assert "1997 - 2001" in r.text
    assert "YEAR" not in r.counts


# --- the identity guarantee ---------------------------------------------------------------------


TRICKY = """\
JOSÉ ÁLVAREZ-NÚÑEZ
Jose.Alvarez@Example.com · jose_alvarez99 · @josealvarez · josealvarez.com
José's projects: github.com/jalvarez, gitlab.com/jose.alvarez/tools
Mentored by Dr. Núñez; Alvarez-Nunez family business
"""


def test_names_and_identifiers_never_survive() -> None:
    names = ["José Álvarez-Núñez"]
    email = "jose.alvarez@example.com"
    out = redact(TRICKY, candidate_names=names, extra_identifiers=[email]).text
    words = {w.casefold() for w in re.findall(r"[^\W\d_]+", out)}
    assert not words & {"jose", "josé", "alvarez", "álvarez", "nunez", "núñez", "josealvarez",
                        "jalvarez"}
    assert email not in out.casefold()


def test_scrub_identity_backstop() -> None:
    text, n = scrub_identity("Ana met ANA at [CANDIDATE] with ana@x.io",
                             candidate_names=["Ana"], extra_identifiers=["ana@x.io"])
    assert text == "[CANDIDATE] met [CANDIDATE] at [CANDIDATE] with [CANDIDATE]"
    assert n == 3
    assert scrub_identity("clean text", candidate_names=["Ana"]) == ("clean text", 0)


def test_redaction_version_bumped_for_v3_rules() -> None:
    assert REDACTION_VERSION == "redact-v3"


# --- word-like name tokens ---------------------------------------------------------------------

HEADER = "{name}\n{email}\nSenior Data Engineer\n[LOCATION]\n\nSUMMARY\nTen years of experience.\n"


def _body(name: str, line: str) -> str:
    """``line`` placed well below the header, so only the name rules decide."""
    first, _, last = name.partition(" ")
    text = HEADER.format(name=name, email=f"{first.lower()}.{last.lower()}@example.com")
    text += "\nEXPERIENCE\nStaff Engineer, Northwind (2020 - present)\n" + line + "\n"
    out = redact(text, candidate_names=[name]).text
    assert out.startswith("[CANDIDATE] [CANDIDATE]\n[EMAIL]\n")  # header is always redacted
    scrubbed, leaked = scrub_identity(out, candidate_names=[name])
    assert leaked == 0 and scrubbed == out  # the backstop agrees with the main pass
    return out.splitlines()[-1]


@pytest.mark.parametrize(("name", "line"), [
    ("Ruby Chen", "- Built APIs in Ruby on Rails and Swift"),
    ("Eva Kafka", "- Streaming with Kafka and Flink"),
    ("Jan Novak", "Data Engineer, Tailspin, Jan 2019 - Dec 2021"),
    ("Ai Tanaka", "- Built AI agents; AI/ML platform lead"),
    ("Can Yilmaz", "- I can lead migrations; can-do attitude"),
    ("Martin Shaw", "Software Engineer, Lockheed Martin"),
    ("Emily Brown", "B.S. Computer Science, Brown University"),
    ("Ha Nguyen", "- HA Postgres cluster"),
    ("Min Park", "- min/max latency 5 min"),
    ("Long Tran", "- long-running jobs"),
    ("Sun Li", "- Sun Solaris admin"),
    ("Ma Wei", "- MA in Economics"),
    ("Fan Zhang", "- fan-out pattern"),
    ("Jun Sato", "- Shipped in Jun 2021"),
    ("Ping Wu", "- Added ping checks"),
    ("Ran Levi", "- Ran the on-call rotation"),
    ("Grant Price", "- Grant writing; Price optimisation"),
    ("Taylor Swift", "- Swift and Rust developer"),
    ("Jenkins Athena", "- Jenkins pipelines; Cassandra and Athena"),
    ("May Davis", "- Started in May 2020 at Davis Instruments"),
    ("Mark Hill", "- Mark releases; Hill-climbing search"),
])
def test_word_like_name_tokens_keep_their_meaning_in_the_body(name: str, line: str) -> None:
    assert _body(name, line) == line


@pytest.mark.parametrize(("name", "line", "expected"), [
    ("Ruby Chen", "Ruby is a data engineer who loves Rails.",
     "[CANDIDATE] is a data engineer who loves Rails."),
    ("Ruby Chen", "- Ruby's team shipped Ruby on Rails services",
     "- [CANDIDATE]'s team shipped Ruby on Rails services"),
    ("Ruby Chen", "References: Ruby Chen, Dr. Ruby, R. Ruby and Ruby",
     "References: [CANDIDATE] [CANDIDATE], [CANDIDATE], R. [CANDIDATE] and Ruby"),
    ("Ai Tanaka", "- Ai led the AI platform team", "- [CANDIDATE] led the AI platform team"),
    ("Grant Price", "Grant Price holds a PMP.", "[CANDIDATE] [CANDIDATE] holds a PMP."),
    ("Jan Novak", "- Reach Jan Novak at [EMAIL]", "- Reach [CANDIDATE] [CANDIDATE] at [EMAIL]"),
    ("Ruby Chen", "- login rubychen, ruby99", "- login [CANDIDATE], [CANDIDATE]"),
])
def test_word_like_name_tokens_used_as_a_name_are_redacted(name: str, line: str,
                                                          expected: str) -> None:
    assert _body(name, line) == expected


def test_word_like_tokens_on_header_and_contact_lines_are_redacted() -> None:
    text = "Ruby Chen\nruby.chen@example.com | Chicago, IL\nRuby\n"
    assert redact(text, candidate_names=["Ruby Chen"]).text == (
        "[CANDIDATE] [CANDIDATE]\n[EMAIL] | [LOCATION]\n[CANDIDATE]\n")


def test_name_with_format_characters_still_matches() -> None:
    # A Persian given name written with an orthographic ZWNJ, and a ZWSP typo in a record.
    assert name_tokens(["علی\u200cرضا احمدی"]) == {"علیرضا", "احمدی"}
    out = _red("علی\u200cرضا built pipelines; علیرضا too", "علی\u200cرضا احمدی")
    assert out == "[CANDIDATE] built pipelines; [CANDIDATE] too"
    assert scrub_identity("علی\u200cرضا", candidate_names=["علی\u200cرضا احمدی"])[1] == 1
    assert _red("AnaMaria Lopez", "Ana\u200bMaria Lopez") == "[CANDIDATE] [CANDIDATE]"


# --- locations vs degrees and products --------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Statistics, MS - University of Chicago",
    "Machine Learning, MS - Carnegie Mellon University",
    "Psychology, MA - Boston University",
    "Cybersecurity, MS",
    "Nursing, MS - Johns Hopkins",
    "Bioinformatics, MS - Northwestern University, Evanston",
    "Finance, MS - Tech Institute",
    "Databases: Oracle, SQL Server, MS Access, Postgres",
    "Tools: Tableau, Power BI, MS Excel, MS Project",
    "Jira, Confluence, MS Teams",
    "Corporate Communications, PR and investor relations",
])
def test_degree_fields_and_microsoft_products_are_not_locations(text: str) -> None:
    assert _red(text, redact_education_years=False) == text


@pytest.mark.parametrize(("text", "expected"), [
    ("Cambridge, MA", "[LOCATION]"),
    ("Boston, MA 02115", "[LOCATION] [ZIP]"),
    ("Jackson, MS", "[LOCATION]"),
    ("Reading, MA", "[LOCATION]"),
    ("Naperville IL 60540", "[LOCATION] [ZIP]"),
    ("Naperville IL", "[LOCATION]"),
    ("Chicago IL | Mar 2020 - Present", "[LOCATION] | Mar 2020 - Present"),
])
def test_city_state_forms_are_redacted(text: str, expected: str) -> None:
    assert _red(text) == expected


@pytest.mark.parametrize("text", [
    "Terraform OR Ansible", "PROFICIENT IN", "Docker Swarm OR", "Senior Developer IN",
    "Go 1.22 and EAD file format",
])
def test_ambiguous_state_codes_without_a_comma_are_kept(text: str) -> None:
    assert _red(text) == text


# --- personal labels inside job lines -----------------------------------------------------------


@pytest.mark.parametrize("line", [
    "Visa | Senior Software Engineer | 2019 - 2022",
    "Visa - Staff Data Engineer, Foster City (2018 - 2023)",
    "Visa | Austin, TX | Senior Software Engineer",
    "VISA - Senior Engineer (2019-2022)",
    "Visa – Staff Engineer",
    "Health - Lead Data Engineer (Optum)",
    "Race - Senior Engineer",
    "Age | Data Scientist | 2020",
    "Faith - Product Manager (2019 - 2021)",
    "Children's Hospital - Software Engineer",
])
def test_employer_and_title_lines_with_label_words_are_kept(line: str) -> None:
    assert _red(line, redact_education_years=False).replace("[LOCATION]", "Austin, TX") == line


@pytest.mark.parametrize(("line", "expected"), [
    ("Audited model bias across age, race, gender: equalized-odds gap cut 40%",
     "Audited model bias across age, race, [REDACTED PERSONAL DETAIL]"),
    ("Built tokenization for PII fields (SSN: format-preserving encryption)",
     "Built tokenization for PII fields ([REDACTED PERSONAL DETAIL])"),
    ("Masked PII (name, DOB: dates, SSN) before analytics",
     "Masked PII (name, [REDACTED PERSONAL DETAIL], SSN) before analytics"),
    ("Validated the customer schema (fields: address, nationality: ISO code)",
     "Validated the customer schema (fields: address, [REDACTED PERSONAL DETAIL])"),
    ("Personal: Nationality: Indian; Age: 34", "[REDACTED PERSONAL DETAIL]"),
])
def test_strong_label_mid_sentence_redacts_only_its_field(line: str, expected: str) -> None:
    assert _red(line) == expected


def test_over_long_personal_line_is_redacted_field_by_field() -> None:
    sidebar = "CONTACT | Chicago area | Nationality: Brazilian | SKILLS | Python | SQL | "
    line = sidebar + "Senior Data Engineer, Northwind, Mar 2020 - Present | " + "dbt, " * 40
    out = _red(line)
    assert out.startswith("CONTACT | Chicago area | [REDACTED PERSONAL DETAIL] | SKILLS | Python")
    assert "Mar 2020 - Present" in out


# --- employment tenure on academic lines --------------------------------------------------------


@pytest.mark.parametrize("line", [
    "Software Engineer, Harvard University IT, 2016 - 2021",
    "Instructor, Lambda School, 2019 - 2020",
    "Graduate Research Assistant, NLP Lab, Jan 2019 - May 2021",
    "Graduate Software Engineer, Arm Ltd, Sep 2018 - Aug 2019",
    "Data Analyst, Miami Dade College, 2015 - 2019",
    "Research Engineer, Stanford University (2017 - 2022)",
    "Teaching Assistant, School of Engineering, 2018",
])
def test_job_lines_at_academic_employers_keep_their_dates(line: str) -> None:
    assert _red(line) == line
    assert _red(f"EXPERIENCE\n{line}\nBuilt things") == f"EXPERIENCE\n{line}\nBuilt things"


def test_university_line_in_experience_section_without_a_degree_keeps_dates() -> None:
    text = "EXPERIENCE\nUniversity of Chicago\n2016 - 2021\nSoftware Engineer\n\n" \
           "EDUCATION\nUniversity of Chicago\n2012 - 2016"
    assert _red(text) == ("EXPERIENCE\nUniversity of Chicago\n2016 - 2021\nSoftware Engineer\n\n"
                          "EDUCATION\nUniversity of Chicago\n[YEAR] - [YEAR]")


def test_degree_line_inside_experience_section_is_still_redacted() -> None:
    text = "EXPERIENCE\nEngineer, Acme (2019 - 2022)\nB.S. Computer Science, State University, 2015"
    assert _red(text) == ("EXPERIENCE\nEngineer, Acme (2019 - 2022)\n"
                          "B.S. Computer Science, State University, [YEAR]")


# --- international phones, labels, age and marital status --------------------------------------


@pytest.mark.parametrize("phone", [
    "+91 9876543210", "+92 300 1234567", "+33 6 12 34 56 78", "07700 900123",
    "+49 30 12345678", "+1 (555) 010-2030",
])
def test_international_phone_formats(phone: str) -> None:
    assert _red(f"Reach me on {phone} any time") == "Reach me on [PHONE] any time"


@pytest.mark.parametrize(("line", "expected"), [
    ("Ph: +91 9876543210", "Ph: [PHONE]"),
    ("Mobile: +91-9876543210", "Mobile: [PHONE]"),
    ("Tel. +49 30 12345678", "Tel. [PHONE]"),
    ("Mobile: 06 12 34 56 78", "Mobile: [PHONE]"),
    ("Mobile: 5550102030 2019 - 2021", "Mobile: [PHONE] 2019 - 2021"),
    ("Telefon: 030 1234567", "Telefon: [PHONE]"),
])
def test_labelled_phones_in_any_grouping(line: str, expected: str) -> None:
    assert _red(line) == expected


@pytest.mark.parametrize("line", [
    "Geburtsdatum: 04.09.1991", "Nacionalidad: Mexicana", "Married, two children",
    "33 years old, married", "Personal: 34 years old, Married",
    "Geburtsdatum: 12.03.1990 | Staatsangehoerigkeit: Deutsch", "Date de naissance : 4/9/1991",
    "Stato civile: coniugato", "Data de nascimento: 04/09/1991", "Familienstand: ledig",
    "US Citizen", "U.S. citizen; no sponsorship required", "Green card holder",
    "Permanent Resident of the United States", "Passport No. X1234567",
    "Passport Number X1234567", "Authorized to work in the US for any employer",
    "H-1B visa holder", "SSN 123-45-6789", "Contact | Divorced | Remote",
])
def test_unlabelled_and_non_english_personal_lines_are_redacted(line: str) -> None:
    out = _red(f"Header\n{line}\nPython developer")
    assert "[REDACTED PERSONAL DETAIL]" in out
    assert "Python developer" in out
    for word in ("1991", "Mexicana", "children", "married", "Citizen", "card", "X1234567",
                 "Authorized", "H-1B", "6789", "Divorced", "Deutsch", "coniugato", "ledig"):
        assert word not in out.split("\n")[1]


@pytest.mark.parametrize("line", [
    "Senior Citizen Services (client project)", "Secured sponsorship for the conference",
    "Go 1.22 and EAD file format", "Single sign-on with Okta", "Separated the ETL concerns",
    "Cleaned, separated and loaded the data", "Worked 60 hours a week for 2 years old systems",
    "A married couple of services share one database (joke from the retro)",
])
def test_personal_lookalikes_are_kept(line: str) -> None:
    assert _red(line) == line


# --- handles, URLs and quantities ---------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Senior Software Engineer @Google (2019 - 2023)",
    "Speaker @PyCon US 2023 and @KubeCon EU 2024",
    "Software Engineer @Nubank (2020 - Present)",
    "Rewrote java.io/nio file handling",
    "Product Analyst, Hotels.com/Expedia Group",
    "Booking.com/Agoda",
    "Processed 250 000 000 events per day",
    "Spent $250 000 000 on infrastructure",
])
def test_job_content_that_looks_like_contact_details_is_kept(text: str) -> None:
    assert _red(text) == text


@pytest.mark.parametrize(("text", "expected"), [
    ("Twitter @jalvarez_dev, IG @jose.alvarez", "Twitter [HANDLE], IG [HANDLE]"),
    ("@Google follow me", "[HANDLE] follow me"),
    ("Engineer @RubyChen", "Engineer [HANDLE]"),  # a handle holding the name is never kept
    ("Call 612 345 678 today", "Call [PHONE] today"),
    ("See jose.dev/portfolio", "See [URL]"),
])
def test_real_handles_phones_and_profile_urls_are_still_redacted(text: str,
                                                                 expected: str) -> None:
    assert _red(text, "Ruby Chen") == expected
