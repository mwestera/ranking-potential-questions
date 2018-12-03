#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AM: Questions and Models of Discourse
author: Luise Schricker

python-3.4


This file contains methods to extract the relevant data for my project "Ranking potential
Questions" from the classes QUD annotations and the Switchboard corpus.
"""
from collections import defaultdict
import json
import os
import queue
import random
import re

from data.swda import swda

# Define sets of assertion and question tags for switchboard corpus
ASSERTION = ['s', 'sd', 'sv']
QUESTION = ['qy', 'qw', '^d', 'qo', 'qr', 'qw^d']

def extract_dev_set(path):
    """
    Method for extracting the dev-set from the classes annotations of the Snowden
    interview.  I did some manual preprocessing of the files to ensure basic compatibility. One document was deleted
    because it was deemed unusable. For every assertion that is followed by a question, all questions following the
    same assertion are extracted from the set of annotations, as well as the three next questions and the three predecing
    questions, which serve as an approximation to alternative potential questions. The questions immediately following the
    assertion are labeled as correct. The extracted data is saved as a json-File

    :param path: Path to directory with annotations
    """
    print("Extracting dev set from the classes' Snowden annotations...")
    # Save data in dictionary, assertion is key, value is a tuple of two sets: all potential questions and the
    # correct questions
    data = defaultdict(lambda: defaultdict(lambda: set()))

    # Iterate over files in directory
    for filename in os.listdir(path):

        # Iterate over lines in file and save the relevant, cleaned lines in a list of questions and assertions
        document = []
        with open(os.path.join(path, filename), "r", encoding='utf-8') as f:

            for line in f:

                # Jump over metadata lines and lines I marked as non-usable with #
                if line.startswith("#") or line.startswith("Snowden:") or line.startswith("Interviewer:") or line=="\n":
                    continue

                else:

                    # Clean metadata within lines, i.e. "<"
                    if line.startswith(">"):
                        line = line.split(">")[-1]

                    # Clean whitespace
                    line = line.lstrip()
                    line = line.strip()
                    line = line.rstrip()

                    # Check whether the line is a question or an assertion
                    if line.startswith("Q"):

                        # Strip the Question enumeration
                        line_split = line.split()[1:]
                        if line_split[0]=="-":
                            line_split = line_split[1:]

                        line = " ".join(line_split)
                        document.append(("Q", line))

                        # Assertion
                    else:
                        if line.startswith("A"):
                            # Strip the Answer enumeration
                            line = " ".join(line.split()[1:])

                        document.append(("A", line))

            # Find and prepare the relevant datapoints

            # keep the last three questions in a queue
            last_three = queue.Queue(maxsize=3)

            for i,line in enumerate(document):

                if line[0]=="Q":
                    if last_three.full():
                        last_three.get()
                    last_three.put(line[1])

                # Only start extracting datapoints after the first three questions to enable full datapoints
                elif line[0]=="A" and last_three.full():

                    # Check that the next line is a question
                    if i<(len(document)-2):
                        if document[i+1][0]=="Q":
                            # Extract next three questions
                            next_three = []
                            for j in range(i+2, len(document)):
                                if len(next_three)==3:
                                    break
                                if document[j][0]=="Q":
                                    next_three.append(document[j][1])

                            if len(next_three)<3:
                                # Break if less than three questions follow this question
                                break

                            else:
                                # Update the set of "potential questions" (approximation) for this assertion
                                data[line[1]]["potential_qs"].update(list(last_three.queue))
                                data[line[1]]["potential_qs"].update(next_three)
                                data[line[1]]["potential_qs"].update([document[i + 1][1]])
                                # Update set of labels (i.e. question following the assertion directly)
                                data[line[1]]["labels"].update([document[i + 1][1]])

    # Convert sets to lists to make them JSON serializable
    for assertion in data:
        data[assertion]["potential_qs"] = list(data[assertion]["potential_qs"])
        data[assertion]["labels"] = list(data[assertion]["labels"])

    # Write data to file
    with open("snowden_qud_dev.json", "w", encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4)

    print("Done. Data written to {}.\n".format("snowden_qud_dev.json"))

def extract_train_set(path):
    """
    Method for extracting training data (i.e. the data for computing feature weights) from
    the Switchboard corpus. The data can only be used as "approximate" training data
    because it only consists of positive examples: assertions and questions following them.
    To approximate a set of other, unrealized potential questions, three random questions
    are picked from the total set of questions in the corpus.
    Assertion-question pairs are extracted, as well as the question's type.
    The data is saved as a json file.
    """
    print("Extracting training data from Switchboard corpus...")
    corpus = swda.CorpusReader(path)

    # At first go throught the corpus and collect all one-utterance questions, use these
    # for alternative potential questions
    all_questions = []

    # Iterate over the corpus
    for utterance in corpus.iter_utterances():

        # Chec whether the utterance is a question
        for q_tag in QUESTION:
            if utterance.act_tag.startswith(q_tag):

                # Check whether the question is complete
                if utterance.text.endswith("/"):
                    cleaned = clean_text(utterance.text)
                    if cleaned == "":
                        continue
                    all_questions.append(clean_text(utterance.text))

    data = []
    last_utterance = None
    incomplete_questions = []
    current_transcript = None

    # Iterate over the corpus
    for utterance in corpus.iter_utterances():

        known_tag = False

        # Is this a continuing question?
        if utterance.act_tag=="+" and len(incomplete_questions) > 0:

            known_tag = True

            # Check if this is the correct speaker and deal with overlapping incomplete questions
            # of the form A: <Q1> B: <Q2> A: <Q1> B: <Q2>
            for inc_q in incomplete_questions:
                if utterance.caller==inc_q[0].caller:
                    inc_q.append(utterance)

                    if utterance.text.endswith("/"):
                        # Clean and join utterances
                        question_text = clean_text("".join([utt.text for utt in inc_q]))

                        # Save datapoint
                        data.append({
                            "assertion": clean_text(last_utterance.text),
                            "question": question_text,
                            "question_type": inc_q[0].act_tag
                        })

                        incomplete_questions.remove(inc_q)
                        # Check if any incomplete questions left, else don't delete the utterance yet
                        if len(incomplete_questions)==0:
                            last_utterance = None
                    break

            continue

        # Chec whether the utterance is a question
        for q_tag in QUESTION:
            if utterance.act_tag.startswith(q_tag):

                if utterance.conversation_no != current_transcript:
                    last_utterance=None

                known_tag = True

                # Check whether the last utterance was an assertion
                if last_utterance:
                    # Check whether the question is complete, if not collect missing parts
                    if not utterance.text.endswith("/"):
                        incomplete_questions.append([utterance])
                        continue

                    else:
                        question_text = clean_text(utterance.text)

                    # Save datapoint
                    data.append({
                        "assertion": clean_text(last_utterance.text),
                        "question": question_text,
                        "question_type": utterance.act_tag
                    })

                    # Check that there are no more open (overlapping) incomplete questions
                    if len(incomplete_questions)==0:
                        last_utterance = None

        # If assertion, save utterance to check whether the next utterance is a question
        for a_tag in ASSERTION:
            if utterance.act_tag.startswith(a_tag):
                # Only save finished assertions, overlapping question-assertion patterns should not be saved
                if utterance.text.endswith("/"):
                    known_tag=True
                    last_utterance = utterance
                    # For checking that assertion and question are from the same conversation
                    current_transcript = utterance.conversation_no

        if not known_tag and len(incomplete_questions)==0:
            # Else delete last utterance
            last_utterance = None

    # Add a set of three alternative potential question to each datapoint
    random.seed(5)

    # Save datapoints to be removed in list
    to_be_removed = []

    for datapoint in data:
        # Check that neither question not assertion are empty strings
        if datapoint["assertion"] == "" or datapoint["question"] == "":
            to_be_removed.append(datapoint)
            continue

        alter_questions = []

        for i in range(3):
            alter_questions.append(all_questions[random.randint(0,len(all_questions)-1)])

        datapoint["alternative_questions"] = alter_questions

    for datapoint in to_be_removed:
        data.remove(datapoint)

    # Write data to file
    with open("swda_train.json", "w", encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4)

    print("Done. {} datapoints extracted. Data written to {}.\n".format(len(data), "swda_train.json"))

def clean_text(text):
    """
    Method for cleaning an utterance from the Switchboard corpus from dysfluency annotations.
    Symbols are listed here: https://catalog.ldc.upenn.edu/docs/LDC99T42/dflguide.ps

    :param text: the utterance (string) to be cleaned
    :return: the cleaned utterance
    """
    # Chop off end-of-turn symbol
    if text.endswith("/"):
        text = text.split("/")[0]

    # Deal with specially marked parts of the text
    if "{" in text:
        # Delete fillers like 'uh'
        text_split = text.split("{F")
        text_split[1:] = ["}".join(slice.split("}")[1:]) for slice in text_split[1:]]
        text = "".join(text_split)

        # Clean the rest of the text from curly brackets
        # These are editing terms, discourse markers, coordinate conjunctions and asides
        text = "".join(re.split('{E|{D|{C|{A|}',text))

    # Deal with restarts by keeping the restart and deleting the discontinued part
    text_split = text.split(" ")

    # To deal with nested restarts, save opening brackets in a stack
    stack = []
    last_tokens = []
    tokens = []
    restart = False
    # By counting the plus signs, keep track of whether to discard tokens or not
    # An even number of opening brackets and plus signs means, this is a restart to be kept
    pluses = 0

    for token in text_split:

        # There should be a space between several opening brackets but there are some mistakes of the form "[["
        if token.startswith("["):
            for char in token:
                if char == "[":
                    stack.append("[")
            restart = True

        elif token.startswith("]"):
            for char in token:
                if char == "]":
                    # Ignore errors because of missing opening brackets in annotation
                    try:
                        stack.pop()
                    except:
                        pass

                    # Plus signs are also "popped" off
                    pluses -= 1

            # Was this the last closing bracket?
            if len(stack)==0:
                tokens += last_tokens
                last_tokens = []
                restart = False

        # Keep restart tokens
        elif restart and len(stack) <= pluses:
            if not ("<" in token or token in ["#", "((", "))", "----", "--", "-", "+"] or ">>" in token):
                last_tokens.append(token)

        elif token == "+":
            pluses += 1

        # Clean meta comments like <noise>
        elif "<" in token or token in ["#", "((", "))", "----", "--", "-", "+"] or ">>" in token:
            continue

        # All tokens that are not part of a restart are kept
        elif not restart:
            tokens.append(token)

    text = " ".join(tokens)

    # Clean trailing whitespaces
    text = text.rstrip()
    # Substitute mutliple whitespaces which are artifacts after the cleaning process with single whitespaces
    text = ' '.join(text.split())

    return text

# Execution
if __name__ == "__main__":
    extract_dev_set("data/snowden-qud-annotations")
    extract_train_set('data/swda/swda_data')





