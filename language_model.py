# this file is an adaptation from the work at mozilla deepspeech github.com/mozilla/DeepSpeech


import re
from heapq import heapify

import kenlm


class LanguageModel():
    def __init__(self, path='./lm/libri-timit-lm.klm'):
        self.model = kenlm.Model(path)
        with open('./lm/words.txt') as f:
            self.words = set(self.words(f.read()))

    def wer(self, original, result):
        """
        The WER is defined as the editing/Levenshtein distance on word level
        divided by the amount of words in the original text.
        In case of the original having more words (N) than the result and both
        being totally different (all N words resulting in 1 edit operation each),
        the WER will always be 1 (N / N = 1).
        """
        # The WER ist calculated on word (and NOT on character) level.
        # Therefore we split the strings into words first:
        original = original.split()
        result = result.split()
        return self.levenshtein(original, result) / float(len(original))

    def wers(self, originals, results):
        count = len(originals)
        try:
            assert count > 0
        except:
            print(originals)
            raise ("ERROR assert count>0 - looks like data is missing")
        rates = []
        mean = 0.0
        assert count == len(results)
        for i in range(count):
            rate = self.wer(originals[i], results[i])
            mean = mean + rate
            rates.append(rate)
        return rates, mean / float(count)

    def lers(self, originals, results):
        count = len(originals)
        assert count > 0
        rates = []
        norm_rates = []

        mean = 0.0
        norm_mean = 0.0

        assert count == len(results)
        for i in range(count):
            rate = self.levenshtein(originals[i], results[i])
            mean = mean + rate

            normrate = (float(rate) / len(originals[i]))

            norm_mean = norm_mean + normrate

            rates.append(rate)
            norm_rates.append(round(normrate, 4))

        return rates, (mean / float(count)), norm_rates, (norm_mean / float(count))

    # The following code is from: http://hetland.org/coding/python/levenshtein.py
    def levenshtein(self, a, b):
        """Calculates the Levenshtein distance between a and b."""
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a, b = b, a
            n, m = m, n

        current = list(range(n + 1))
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    def tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def log_probability(self, sentence):
        """Log base 10 probability of `sentence`, a list of words"""
        return self.model.score(' '.join(sentence), bos=False, eos=False)

    def correction(self, sentence, beam_width=1024):
        """Most probable spelling correction for sentence."""
        layer = [(0, [])]
        for word in self.tokenize(sentence):
            layer = [(-self.log_probability(node + [cword]), node + [cword]) for cword in self.candidate_words(word) for
                     priority, node in layer]
            heapify(layer)
            layer = layer[:beam_width]
        return ' '.join(layer[0][1])

    def candidate_words(self, word):
        """Generate possible spelling corrections for word."""
        return self.known_words([word]) or self.known_words(self.edit_distance(word)) or self.known_words(
            self.edit_distance(word, distance=2)) or [word]

    def known_words(self, words):
        return set(w for w in words if w in self.words)

    def _edit_distance(self, word):
        """All edits that are one edit away from `word`."""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edit_distance(self, word, distance=1):
        if distance > 2 or distance < 1:
            distance = 1

        if distance == 1:
            return self._edit_distance(word)
        else
            return (edit2 for edit1 in self._edit_distance(word) for edit2 in self._edit_distance(edit1))

