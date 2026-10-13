from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from applicant_ranking.config import ConfigError
from applicant_ranking.scoring.lexicon import (
    REQUIRED_CATEGORIES,
    LexiconError,
    LexiconHit,
    ProtectedTermLexicon,
)

ZWSP = chr(0x200B)
NB_HYPHEN = chr(0x2011)
RSQUO = chr(0x2019)

# Ordinary technical resume / assessment sentences. None may produce a hit: a false
# positive costs a corrective retry and, if repeated, a needless human review.
BENIGN_SENTENCES = [
    "Built production data pipelines on GCP using BigQuery and Dataflow for 12 clients.",
    "The candidate wrote a white paper on streaming architectures and two whitepapers.",
    "Designed black-box and white-box test suites for the payments API.",
    "Migrated the blacklist and whitelist rules to a policy engine.",
    "Delivered a white-label analytics product for retail clients.",
    "Led the cloud-native re-architecture of a monolith onto Kubernetes.",
    "Shipped React Native and native mobile apps for iOS and Android.",
    "Optimized native code paths in the C++ extension module.",
    "Integrated the parent company's identity provider and implemented single sign-on.",
    "Built a single-page application in React with TypeScript.",
    "Merged feature branches into the master branch and ran master data management.",
    "Scaled the order pipeline for Black Friday traffic peaks.",
    "Fixed race conditions and data races in the Go scheduler.",
    "Oversampled the minority class with SMOTE to handle class imbalance.",
    "Read the man pages and mitigated man-in-the-middle attacks with mutual TLS.",
    "Estimated 400 man-hours for the migration project.",
    "Lean Six Sigma Black Belt with process improvement experience.",
    "Formatted the codebase with black and isort, and linted it with ruff.",
    "Implemented stochastic gradient descent and variants of gradient descent in NumPy.",
    "Built citizen developer tooling on Power Platform for citizen-facing portals.",
    "Migrated older Java services to Kotlin and retired the legacy queue.",
    "Disabled the legacy endpoints after the cutover and enabled feature flags.",
    "The candidate has 6 years of experience building batch and streaming pipelines.",
    "Strong Python and SQL; limited evidence of Terraform or CI/CD for data workloads.",
    "The resume does not show orchestration with Airflow or Dagster.",
    "Gap: no evidence of production streaming with Pub/Sub or Kafka.",
    "Significant gap in data quality tooling such as Great Expectations.",
    "Led client workshops to gather requirements and presented designs to executives.",
    "Reduced p99 latency from 900 ms to 120 ms by caching hot keys in Redis.",
    "Owned the deployment pipeline and deployed services to production weekly.",
    "Executive sponsor for the data platform program; reported to the project sponsor.",
    "Built fraud models to combat chargebacks using gradient boosting.",
    "Created a proof of concept (POC) for a vector search service.",
    "Each node in the tree stores references to its child nodes and parent node.",
    "Designed foreign key constraints and normalized the schema to third normal form.",
    "Configured accent colors and dark mode for the design system.",
    "Implemented SQL injection protections and ran security reviews.",
    "Worked on L1 cache optimizations and SIMD vectorization.",
    "Opt-in telemetry and opt-out flows were implemented in the consent service.",
    "Ran double-blind reviews and identified blind spots in the metrics.",
    "Built a genetic algorithm for scheduling and used genetic programming for features.",
    "Replaced military-grade encryption claims with standard AES-256 wording.",
    "Hands-on SQL experience writing window functions and partitioned tables.",
    "Mentored junior engineers and led a team of five senior data engineers.",
    "The candidate's experience with Spark and Dataproc meets the requirement.",
    "Professional Python development with pytest, packaging and code review.",
    "Deployed models on Vertex AI and monitored drift with custom metrics.",
    "Aged receivables reporting was automated in Looker.",
    "Visa and Mastercard payment integrations were built on Stripe.",
    "Native integration with Snowflake and native support for Parquet files.",
    "Built a young startup's first data warehouse from scratch.",
    "Implemented white-space handling in the tokenizer.",
    "Configured male and female connectors for the sensor harness.",
    "Fine-tuned OPT and LLaMA models for summarization.",
    "Ran Scrum ceremonies as Scrum Master and owned the product backlog.",
    "Managed the agency relationship and the management reporting pages.",
    "Wrote the history module and the theme engine for the shell.",
    "The resume shows strong ownership of the ingestion layer.",
    # Domain vocabulary (employers, datasets, products, fields of work) that cannot
    # describe a person; healthcare, fintech, HR-tech and public-sector roles need it.
    "Built zip code level demand forecasts for 1,200 stores.",
    "Enriched customer records with census tract data.",
    "Built credit-risk models on credit bureau data with a credit scoring platform.",
    "Built the FMLA case tracking module and parental leave accrual rules in Workday.",
    "HIPAA-compliant pipelines for a mental health app.",
    "Seizure detection from EEG signals with CNNs.",
    "Data engineer for the Department of Veterans Affairs.",
    "Software engineer at Navy Federal Credit Union.",
    "Research engineer at the Air Force Research Laboratory.",
    "Built case management software for an immigration law firm.",
    "IoT platform work at Stanley Black & Decker.",
    "Processed unemployment claims data for the state agency.",
    "Built injury claims models for an insurer.",
    "Data pipelines for cancer research at a hospital; worked with cancer registry data.",
    "Deployed the payments platform at Arab Bank and in the United Arab Emirates.",
    "Modelled claims for mom-and-pop retailers.",
    # Everyday words the person-context patterns must leave alone.
    "Built a MOM layer on RabbitMQ for message-oriented middleware.",
    "The platform is mature and the team is senior.",
    "Grew the team to around 30. Cut p99 latency by about 40. Led a team of around 30.",
    "Supported over 50 clients across 12 regions; over 50 clients onboarded.",
    "The design is blind to message ordering; ran blind SQL injection tests.",
    "The candidate came from Google and comes from a background in finance.",
    "Rebuilt the warehouse from scratch; was from the start a strong contributor.",
    "The candidate is likely a strong fit and appears to be a senior engineer.",
    "The candidate is English-speaking and the documentation is French.",
    "The candidate is American Express's platform lead.",
]

# At least one sentence per category, including the categories beyond the required set.
TRUE_POSITIVES: dict[str, list[str]] = {
    "age": ["The candidate is 45 years old.", "Looking for young engineers.",
            "Digital native preferred.", "Recent graduate with a CS degree.",
            "Too old for a startup.", "Graduated in 1998.", "A 52-year-old engineer.",
            "older workers adapt slowly", "a millennial mindset", "seems overqualified",
            "The candidate appears to be older than the rest of the team.",
            "The candidate seems older.", "They are probably quite mature.",
            "Over 50, so may not adapt.", "Probably close to 60.", "Around 65 years of age."],
    "gender": ["He has strong SQL skills.", "Her Kafka experience is limited.",
               "She led the migration.", "A female engineer.", "His resume is short.",
               "Looking for a salesman.", "Shared their pronouns in the header."],
    "race_ethnicity": ["The candidate appears to be Black.", "A Hispanic applicant.",
                       "Asian American engineer.", "A person of color.",
                       "Racial background is noted.", "The candidate attended an HBCU.",
                       "Studied at a historically Black university."],
    "national_origin": ["Native English speaker required.", "A foreign national.",
                        "Has a strong accent.", "English is not their first language.",
                        "Born in India.", "An Indian national.", "Of Mexican descent.",
                        "A non-native speaker.", "The candidate is Indian.",
                        "The candidate is Mexican.", "The applicant appears to be Chinese.",
                        "The candidate appears to be from India.",
                        "Originally from Nigeria.", "Comes from abroad.",
                        "The candidate's name sounds Arab.", "The name looks foreign.",
                        "An Arab engineer."],
    "citizenship": ["Must be a U.S. citizen.", "Requires visa sponsorship.",
                    "Holds a green card.", "Currently on H-1B.",
                    "Authorized to work in the US.", "Will need sponsorship later.",
                    "Currently on OPT until 2027.", "A work visa holder."],
    "religion": ["Attends church every Sunday.", "A devout Muslim.",
                 "Needs time off for Ramadan.", "Jewish heritage."],
    "disability_health": ["The candidate has a disability.", "Mentions ADHD.",
                          "Uses a wheelchair.", "Took medical leave in 2022.",
                          "A deaf candidate.", "Mental health issues.",
                          "The candidate is blind.", "The candidate is diabetic.",
                          "The candidate has cancer.", "Diagnosed with diabetes.",
                          "Suffers from depression."],
    "pregnancy": ["The candidate is pregnant.", "Took maternity leave.",
                  "On parental leave last year."],
    "marital_family": ["Married with two children.", "A single mother.",
                       "Marital status: divorced.", "A caregiver for a parent.",
                       "The spouse relocated.", "Likely a mom returning to work.",
                       "A dad of three."],
    "sexual_orientation": ["Member of the LGBTQ+ network.", "The candidate is gay.",
                           "Has a same-sex partner."],
    "veteran_military": ["A veteran of the Army.", "Served in the Marine Corps.",
                         "Military service from 2005 to 2010."],
    "genetic": ["Genetic testing results were mentioned.", "Family history of heart disease."],
    "employment_gap": ["Has an employment gap of two years.",
                       "A gap in employment between 2019 and 2021.", "Was unemployed for a year.",
                       "Took a career break.", "No gaps allowed.", "A job hopper."],
    "location_proxy": ["Lives in a low-income neighborhood.", "Zip code 60601.",
                       "Local candidates only.", "A long commute from the suburbs.",
                       "Must be within 30 miles of the office."],
    "appearance": ["The headshot shows the candidate.", "The candidate is overweight."],
    "criminal_record": ["Has a criminal record.", "Was arrested in 2015."],
    "financial_history": ["Their credit history is poor.", "Asked about salary history."],
    "other_status": ["A survivor of domestic violence.", "Political affiliation noted."],
}

# Allow phrases the lexicon must accept, each in context: never a hit.
ALLOW_PHRASE_SENTENCES = [
    "Published a white paper.", "Published a whitepaper.", "White-box testing.",
    "Black-box testing.", "Updated the blacklist.", "Updated the whitelist.",
    "A white-label product.", "Cloud-native services.", "React Native app.",
    "Native app development.", "Native apps for Android.", "Native mobile UI.",
    "Native code in Rust.", "The parent company.", "The master branch.",
    "Master data management.", "Black Friday sales.", "Single sign-on with SAML.",
    "A single-page app.", "Use gender-neutral language in job ads.",
]


def test_packaged_default_has_every_required_category(lexicon: ProtectedTermLexicon) -> None:
    assert lexicon.source.startswith("package:applicant_ranking.data/")
    assert lexicon.version == 2
    for category in REQUIRED_CATEGORIES:
        assert lexicon.categories[category], category
    assert lexicon.term_count > 300


def test_false_positive_suite_has_no_hits(lexicon: ProtectedTermLexicon) -> None:
    assert len(BENIGN_SENTENCES) >= 30
    hits = {s: [h.label for h in lexicon.scan(s)] for s in BENIGN_SENTENCES}
    assert {s: labels for s, labels in hits.items() if labels} == {}


def test_every_category_has_true_positive_cases(lexicon: ProtectedTermLexicon) -> None:
    assert set(TRUE_POSITIVES) == set(lexicon.categories)


@pytest.mark.parametrize(
    ("category", "sentence"),
    [(c, s) for c, sentences in TRUE_POSITIVES.items() for s in sentences],
)
def test_true_positive(lexicon: ProtectedTermLexicon, category: str, sentence: str) -> None:
    categories = {h.category for h in lexicon.scan(sentence)}
    assert category in categories, f"{sentence!r} -> {categories}"


@pytest.mark.parametrize("sentence", ALLOW_PHRASE_SENTENCES)
def test_allow_phrases_neutralise_tech_terms(lexicon: ProtectedTermLexicon,
                                             sentence: str) -> None:
    assert lexicon.scan(sentence) == []


def test_allow_phrase_does_not_hide_a_real_hit_next_to_it(
        lexicon: ProtectedTermLexicon) -> None:
    labels = [h.label for h in lexicon.scan("A white paper by a young engineer; he wrote it.")]
    assert labels == ["age:young", "gender:he"]


@pytest.mark.parametrize("text", ["he", "He", "HE", "her.", "(she)", "his,", "hers?"])
def test_pronouns_are_gender_hits_in_any_case(lexicon: ProtectedTermLexicon, text: str) -> None:
    assert [h.category for h in lexicon.scan(text)] == ["gender"]


@pytest.mark.parametrize("text", ["the theme", "a shell script", "history", "Hermes",
                                  "agent", "page", "management", "HER2 testing pipeline"])
def test_terms_match_whole_words_only(lexicon: ProtectedTermLexicon, text: str) -> None:
    assert lexicon.scan(text) == []


@pytest.mark.parametrize("text", ["45 years old", "45 years-old", "45 years   old",
                                  "45 years\nold", f"45 years{NB_HYPHEN}old"])
def test_phrase_whitespace_matches_any_whitespace_or_hyphen_run(
        lexicon: ProtectedTermLexicon, text: str) -> None:
    assert [h.label for h in lexicon.scan(text)] == ["age:years old"]


def test_hyphenated_entry_matches_spaced_text(lexicon: ProtectedTermLexicon) -> None:
    assert [h.label for h in lexicon.scan("must be able bodied")] == [
        "disability_health:able-bodied"]


def test_invisible_and_fullwidth_characters_do_not_evade_the_scan(
        lexicon: ProtectedTermLexicon) -> None:
    assert [h.label for h in lexicon.scan(f"h{ZWSP}er skills")] == ["gender:her"]
    fullwidth_her = "".join(chr(0xFF41 + ord(c) - ord("a")) for c in "her")
    assert [h.label for h in lexicon.scan(f"{fullwidth_her} skills")] == ["gender:her"]
    assert [h.label for h in lexicon.scan(f"asperger{RSQUO}s")] == [
        "disability_health:asperger's"]


def test_contained_hit_of_same_category_is_dropped(lexicon: ProtectedTermLexicon) -> None:
    labels = [h.label for h in lexicon.scan("Maternity leave, then maternity leave again.")]
    assert labels == ["pregnancy:maternity leave"]


def test_each_term_reported_once(lexicon: ProtectedTermLexicon) -> None:
    hits = lexicon.scan("He said he would; he did.")
    assert [h.label for h in hits] == ["gender:he"]


def test_hit_context_and_offsets(lexicon: ProtectedTermLexicon) -> None:
    text = "x" * 200 + " prefers hiring young engineers and " + "y" * 200
    [hit] = lexicon.scan(text)
    assert isinstance(hit, LexiconHit)
    assert hit.label == "age:young"
    assert "young" in hit.context
    assert len(hit.context) <= 60
    assert text[hit.start:hit.end] == "young"


def test_person_context_hit_absorbs_the_bare_term(lexicon: ProtectedTermLexicon) -> None:
    # "candidate is young" (the predicate pattern) contains "young"; one hit, the wider.
    assert [h.label for h in lexicon.scan("The candidate is young.")] == [
        "age:candidate is young"]


@pytest.mark.parametrize(("text", "label"), [
    ("Likely a mom returning to work.", "marital_family:mom"),
    ("Mom of two.", "marital_family:mom"),
    ("A proud Dad.", "marital_family:dad"),
])
def test_mom_and_dad_hit_in_lower_or_title_case(lexicon: ProtectedTermLexicon, text: str,
                                                label: str) -> None:
    assert [h.label for h in lexicon.scan(text)] == [label]


def test_mom_acronym_is_not_a_hit(lexicon: ProtectedTermLexicon) -> None:
    assert lexicon.scan("Replaced the MOM broker; the DAD protocol stayed.") == []


@pytest.mark.parametrize("text", [
    "Over 50, so may not adapt.", "Probably close to 60.", "Under 30, which helps.",
])
def test_bare_age_range_number_is_a_hit(lexicon: ProtectedTermLexicon, text: str) -> None:
    assert [h.category for h in lexicon.scan(text)] == ["age"]


@pytest.mark.parametrize("text", [
    "Over 50 clients.", "Grew the team to around 30.", "Cut p99 latency by about 40.",
    "Led a team of around 30.", "Reduced cost by about 40%.", "Processed close to 60 TB.",
])
def test_number_with_a_noun_or_measure_is_not_an_age(lexicon: ProtectedTermLexicon,
                                                     text: str) -> None:
    assert lexicon.scan(text) == []


def test_regex_entry_reports_matched_text() -> None:
    lex = ProtectedTermLexicon({"age": [r"re:\bborn[\s-]+in[\s-]+(?:19|20)\d{2}\b"]}, [])
    [hit] = lex.scan("Born  in 1980, apparently.")
    assert hit.label == "age:born in 1980"
    assert lex.scan("born in-house") == []


def test_regex_entries_are_case_insensitive_unless_scoped() -> None:
    lex = ProtectedTermLexicon({"citizenship": [r"re:(?-i:\bOPT\b)"]}, [])
    assert [h.term for h in lex.scan("on OPT")] == ["opt"]
    assert lex.scan("opt in") == []


def test_blanked_allow_phrase_cannot_join_a_phrase_across_it() -> None:
    lex = ProtectedTermLexicon({"national_origin": ["native speaker"]}, ["white paper"])
    assert lex.scan("native white paper speaker") == []
    assert [h.term for h in lex.scan("native  speaker")] == ["native speaker"]


def test_regex_allow_phrase() -> None:
    lex = ProtectedTermLexicon({"gender": ["male"]}, [r"re:\bmale[\s-]+connectors?\b"])
    assert lex.scan("male connectors") == []
    assert [h.term for h in lex.scan("male engineer")] == ["male"]


def test_empty_text(lexicon: ProtectedTermLexicon) -> None:
    assert lexicon.scan("") == []
    assert lexicon.scan("   \n ") == []


def test_categories_property_is_a_copy(lexicon: ProtectedTermLexicon) -> None:
    lexicon.categories["age"].append("anything")
    lexicon.categories.pop("gender")
    assert "anything" not in lexicon.categories["age"]
    assert "gender" in lexicon.categories


def _write(tmp_path: Path, data: object) -> Path:
    path = tmp_path / "terms.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _full_categories() -> dict[str, list[str]]:
    return {name: [f"term-{name}"] for name in REQUIRED_CATEGORIES}


def test_load_custom_file(tmp_path: Path) -> None:
    categories = _full_categories() | {"extra": ["re:\\bfoo\\d+\\b"]}
    path = _write(tmp_path, {"version": 3, "allow_phrases": ["term-age group"],
                             "categories": categories, "metadata": {"owner": "Legal"}})
    lex = ProtectedTermLexicon.load(path)
    assert lex.version == 3
    assert lex.source == str(path)
    assert [h.label for h in lex.scan("foo12 and term-age")] == ["extra:foo12", "age:term-age"]
    assert lex.scan("the term-age group") == []


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "expected a mapping"),
        ({"version": 1}, "'categories' must be a mapping"),
        ({"categories": {"age": ["x"]}}, "missing required categories"),
        ({"categories": _full_categories(), "allowphrases": []}, "unknown top-level keys"),
        ({"categories": _full_categories() | {"age": []}}, "non-empty list"),
        ({"categories": _full_categories() | {"age": "young"}}, "non-empty list"),
        ({"categories": _full_categories() | {"age": [""]}}, "non-empty strings"),
        ({"categories": _full_categories() | {"age": [3]}}, "non-empty strings"),
        ({"categories": _full_categories() | {"age": ["re:(unclosed"]}}, "invalid regex"),
        ({"categories": _full_categories() | {"age": ["re:x*"]}}, "matches the empty string"),
        ({"categories": _full_categories() | {"age": ["re:  "]}}, "no characters to match"),
        ({"categories": _full_categories() | {"age": ["- -"]}}, "no characters to match"),
        ({"categories": _full_categories() | {"Bad Name": ["x"]}}, "invalid category name"),
        ({"categories": _full_categories(), "version": True}, "positive integer"),
        ({"categories": _full_categories(), "version": 0}, "positive integer"),
        ({"categories": _full_categories(), "allow_phrases": "x"}, "allow_phrases"),
        ({"categories": _full_categories(), "allow_phrases": ["re:("]}, "invalid regex"),
    ],
)
def test_invalid_files_are_rejected(tmp_path: Path, data: object, message: str) -> None:
    path = _write(tmp_path, data)
    with pytest.raises(LexiconError, match=message) as excinfo:
        ProtectedTermLexicon.load(path)
    assert str(path) in str(excinfo.value)


def test_invalid_yaml_and_missing_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("categories: [unclosed", encoding="utf-8")
    with pytest.raises(LexiconError, match="invalid YAML"):
        ProtectedTermLexicon.load(bad)
    with pytest.raises(LexiconError, match="cannot read"):
        ProtectedTermLexicon.load(tmp_path / "missing.yaml")


def test_lexicon_error_is_a_config_error() -> None:
    assert issubclass(LexiconError, ConfigError)
    with pytest.raises(ConfigError):
        ProtectedTermLexicon({}, [])
