import os

from src.get_data_eicu import Vocabulary
import dill as pickle
import logging
import pdb


def load_pickle(filename):
    logging.info(f"Data loaded from {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)


def dump_pickle(data, filename):
    logging.info(f"Data saved to {filename}")
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def build_vocab(all_words, output_filename):
    vocab = Vocabulary()
    for word in all_words:
        vocab.add_word(word)
    print(f"Vocab len:", len(vocab))

    # sanity check
    assert set(vocab.word2idx.keys()) == set(vocab.idx2word.values())
    assert set(vocab.word2idx.values()) == set(vocab.idx2word.keys())
    for word in vocab.word2idx.keys():
        assert word == vocab.idx2word[vocab(word)]

    dump_pickle(vocab, output_filename)
    return


def main():
    processed_data_path = "/cis/home/xhan56/code/clinical-highmmt/src/datasets"
    all_icu_stay_dict = load_pickle(os.path.join(processed_data_path, "eicu/processed/icu_stay_dict.pkl"))
    all_codes = []
    for icu_id in all_icu_stay_dict.keys():
        for code in all_icu_stay_dict[icu_id].trajectory[2]:
            all_codes.append(code)
    all_codes = list(set(all_codes))
    build_vocab(all_codes, os.path.join(processed_data_path, f"eicu/processed/vocab.pkl"))


if __name__ == "__main__":
    main()