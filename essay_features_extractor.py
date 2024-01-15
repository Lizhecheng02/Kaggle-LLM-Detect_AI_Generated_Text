import torch
import pandas as pd
import regex as re
import numpy as np


def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


class EssayProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        def kurtosis_func(x): return x.kurt()

        self.SENT_AGGREGATIONS = [
            'count', 'mean', 'std', q1, 'median', q3, 'max'
        ]

        self.PARA_AGGREGATIONS = [
            'count', 'mean', 'std', q1, 'median', q3, 'min', 'max'
        ]

        self.WORD_AGGREGATIONS = [
            'count', 'mean', 'std', q1, 'median', q3, 'max'
        ]

    def split_essays_into_words(self, df):
        essay_df = df
        essay_df['word'] = essay_df['text'].apply(
            lambda x: re.split(' |\\n|\\.|\\?|\\!', x))
        essay_df = essay_df.explode('word')
        essay_df['word_len'] = essay_df['word'].apply(lambda x: len(x))
        essay_df = essay_df[essay_df['word_len'] != 0]
        return essay_df

    def compute_word_aggregations(self, word_df):
        word_agg_df = word_df[['id', 'word_len']].groupby(
            ['id']).agg(self.WORD_AGGREGATIONS)
        word_agg_df.columns = ['_'.join(x) for x in word_agg_df.columns]
        word_agg_df['id'] = word_agg_df.index
        for word_l in [5, 6, 7, 8, 9, 10, 11, 12]:
            word_agg_df[f'word_len_ge_{word_l}_count'] = \
                word_df[word_df['word_len'] >= word_l].groupby(
                    ['id']).count().iloc[:, 0]
            word_agg_df[f'word_len_ge_{word_l}_count'] = \
                word_agg_df[f'word_len_ge_{word_l}_count'].fillna(0)
        word_agg_df = word_agg_df.reset_index(drop=True)
        return word_agg_df

    def split_essays_into_sentences(self, df):
        essay_df = df
        essay_df['id'] = essay_df.index
        essay_df['sent'] = essay_df['text'].apply(
            lambda x: re.split('\\.|\\?|\\!', x))
        essay_df = essay_df.explode('sent')
        essay_df['sent'] = essay_df['sent'].apply(
            lambda x: x.replace('\n', '').strip())
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
        essay_df['sent_word_count'] = essay_df['sent'].apply(
            lambda x: len(x.split(' ')))
        essay_df = essay_df[essay_df.sent_len != 0].reset_index(drop=True)
        return essay_df

    def compute_sentence_aggregations(self, df):
        sent_agg_df = pd.concat([
            df[['id', 'sent_len']].groupby(['id']).agg(self.SENT_AGGREGATIONS),
            df[['id', 'sent_word_count']].groupby(
                ['id']).agg(self.SENT_AGGREGATIONS)
        ], axis=1)
        sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
        sent_agg_df['id'] = sent_agg_df.index
        for sent_l in [20, 40, 60, 80]:
            sent_agg_df[f'sent_len_ge_{sent_l}_count'] = \
                df[df['sent_len'] >= sent_l].groupby(
                ['id']).count().iloc[:, 0]
            sent_agg_df[f'sent_len_ge_{sent_l}_count'] = \
                sent_agg_df[f'sent_len_ge_{sent_l}_count'].fillna(0)
        sent_agg_df = sent_agg_df.reset_index(drop=True)
        sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
        sent_agg_df = sent_agg_df.rename(
            columns={"sent_len_count": "sent_count"})
        return sent_agg_df

    def split_essays_into_paragraphs(self, df):
        essay_df = df
        essay_df['id'] = essay_df.index
        essay_df['paragraph'] = essay_df['text'].apply(
            lambda x: x.split('\n'))
        essay_df = essay_df.explode('paragraph')
        essay_df['paragraph_len'] = essay_df['paragraph'].apply(
            lambda x: len(x))
        essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(
            lambda x: len(x.split(' ')))
        essay_df = essay_df[essay_df.paragraph_len != 0].reset_index(drop=True)
        return essay_df

    def compute_paragraph_aggregations(self, df):
        paragraph_agg_df = pd.concat([
            df[['id', 'paragraph_len']].groupby(
                ['id']).agg(self.PARA_AGGREGATIONS),
            df[['id', 'paragraph_word_count']].groupby(
                ['id']).agg(self.PARA_AGGREGATIONS)
        ], axis=1)
        paragraph_agg_df.columns = [
            '_'.join(x) for x in paragraph_agg_df.columns]
        paragraph_agg_df['id'] = paragraph_agg_df.index
        paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
        paragraph_agg_df.drop(
            columns=["paragraph_word_count_count"], inplace=True)
        paragraph_agg_df = paragraph_agg_df.rename(
            columns={"paragraph_len_count": "paragraph_count"})
        return paragraph_agg_df

    def word_processor(self, df):
        word_df = self.split_essays_into_words(df)
        word_agg_df = self.compute_word_aggregations(word_df)
        print("The shape of word agg:", word_agg_df.shape)
        return word_agg_df

    def sentence_processor(self, df):
        sent_df = self.split_essays_into_sentences(df)
        sent_agg_df = self.compute_sentence_aggregations(sent_df)
        print("The shape of sent agg:", sent_agg_df.shape)
        return sent_agg_df

    def paragraph_processor(self, df):
        paragraph_df = self.split_essays_into_paragraphs(df)
        paragraph_agg_df = self.compute_paragraph_aggregations(paragraph_df)
        print("The shape of paragraph agg:", paragraph_agg_df.shape)
        return paragraph_agg_df
