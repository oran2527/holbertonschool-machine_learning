#!/usr/bin/env python3
""" Answer """
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """ Answer """
    while True:
        question = input("Q: ")
        bye = ['exit', 'quit', 'goodbye', 'bye']
        if question.lower() in bye:
            print("A: Goodbye")
            break
        answer = question_answer(question, reference)
        error = "Sorry, I do not understand your question."
        print("A: {}".format(answer if answer else error))
