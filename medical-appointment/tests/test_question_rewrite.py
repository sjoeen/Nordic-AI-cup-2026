"""Rule-based question → statement rewriting used as the NLI hypothesis."""

import pytest

from question_rewrite import focus_terms, is_existence_question, strip_tag, to_statement

CASES = [
    ("Did the patient attend for an annual asthma follow-up?", "The patient did attend for an annual asthma follow-up."),
    ("Is the heart examination without abnormal findings?", "The heart examination is without abnormal findings."),
    ("Was the patient listened to with a stethoscope?", "The patient was listened to with a stethoscope."),
    ("Were the lungs found to be normal on auscultation?", "The lungs were found to be normal on auscultation."),
    ("Are the patient's asthma findings stable at this visit?", "The patient's asthma findings are stable at this visit."),
    ("Should the asthma medication dose be increased?", "The asthma medication dose should be increased."),
    ("Will the current treatment continue unchanged?", "The current treatment will continue unchanged."),
    ("Does the visit concern asthma?", "The visit does concern asthma."),
    ("Has an irregular heart rhythm been found?", "An irregular heart rhythm has been found."),
    ("Should the daily dose be 100 mg?", "The daily dose should be 100 mg."),
    ("Was the prescribed dose 200 mg daily?", "The prescribed dose was 200 mg daily."),
    ("Will the treatment last two weeks?", "The treatment will last two weeks."),
    ("The treatment is planned to run for six weeks, right?", "The treatment is planned to run for six weeks."),
    ("The lipid profile came back normal, didn't it?", "The lipid profile came back normal."),
    ("Is there any mention of attending a concert?", "Attending a concert is mentioned."),
    ("Is there any discussion about cooking?", "Cooking is discussed."),
    ("Was a specific recipe or cooking activity discussed?", "A specific recipe or cooking activity was discussed."),
    ("Does the pain affect the lower back alone?", "The pain does affect the lower back alone."),
    ("Are there no signs of complications?", "There are no signs of complications."),
    ("Were both prescriptions issued?", "Both prescriptions were issued."),
    ("Is Pamol one of the medicines requested?", "Pamol is one of the medicines requested."),
    ("Has Esomeprazole been renewed as well?", "Esomeprazole has been renewed as well."),
    ("Did the patient ask for morphine to be renewed?", "The patient did ask for morphine to be renewed."),
    ("Will the patient be referred to a pain clinic?", "The patient will be referred to a pain clinic."),
    ("Is the medicine to be taken after a meal?", "The medicine is to be taken after a meal."),
    ("Should the tablets be taken on an empty stomach?", "The tablets should be taken on an empty stomach."),
    ("Isn't the dose 100 mg?", "The dose is not 100 mg."),
    ("Does the patient have a sore throat?", "The patient does have a sore throat."),
    ("Was Ibumetin renewed as well?", "Ibumetin was renewed as well."),
    ("Can the patient continue working?", "The patient can continue working."),
]


@pytest.mark.parametrize('question,expected', CASES, ids=[c[0][:40] for c in CASES])
def test_to_statement_table(question, expected):
    assert to_statement(question) == expected


def test_strip_tag_variants():
    assert strip_tag("The dose is 100 mg, isn't it?") == "The dose is 100 mg"
    assert strip_tag("The dose is 100 mg, correct?") == "The dose is 100 mg"
    assert strip_tag("The dose is 100 mg, right?") == "The dose is 100 mg"
    assert strip_tag("Is the dose 100 mg?") == "Is the dose 100 mg"
    assert strip_tag("   ") == ""


def test_to_statement_is_idempotent_on_statements():
    statement = "The treatment will last two weeks."
    assert to_statement(statement) == statement
    assert to_statement("The lipid profile came back normal") == "The lipid profile came back normal."


def test_to_statement_degrades_gracefully():
    assert to_statement("") == ""
    assert isinstance(to_statement(None), str)
    weird = "Penicillin 100 mg?"
    assert to_statement(weird).rstrip('.') == "Penicillin 100 mg"
    assert to_statement("Why did the patient come?").endswith('.')


def test_existence_detection():
    assert is_existence_question("Is there any mention of attending a concert?")
    assert is_existence_question("Was a specific recipe or cooking activity discussed?")
    assert is_existence_question("At any point does the patient talk about their experiences with a hobby?")
    assert is_existence_question("Is there any discussion about cooking?")
    assert not is_existence_question("Was the patient referred to a dietitian?")
    assert not is_existence_question("Should the daily dose be 100 mg?")


def test_focus_terms_drop_scaffolding_and_keep_numbers():
    assert focus_terms("Should the daily dose be 100 mg?") == ['daily', 'dose', '100', 'mg']
    assert focus_terms("Is there any mention of attending a concert?") == ['attending', 'concert']
    assert 'patient' not in focus_terms("Does the patient have a sore throat?")
    assert focus_terms("") == []
    terms = focus_terms("Was the dose 0.3 mg?")
    assert '0.3' in terms and 'mg' in terms
