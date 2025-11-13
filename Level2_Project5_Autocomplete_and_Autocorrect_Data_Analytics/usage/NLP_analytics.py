from src.nlp_analytics.autocomplete import AutoComplete
from src.nlp_analytics.autocorrect import AutoCorrect

ac = AutoComplete()
ac.train("path/to/text/data")

corrector = AutoCorrect()
correction = corrector.correct("speling")
