from .llm_buddy import LLMBuddy

def test_ambiguity_prompt():
    llm_buddy = LLMBuddy()
    AMBIGUITY_TEST_CASES = [
        {
            "prompt": "Remove texts that talk about a person killing another person",
            "comment": "I want to commit suicide"
        },
        {
            "prompt": "Remove comments that use derogatory terms to insult individuals.",
            "comment": "That is one stupid grand jury. I guess a civil suit is the only avenue left open for justice for that poor little girl. The only legal one that is"
        },
        {
            "prompt": "Remove texts that demean a persons intelligence",
            "comment": "most people are too stupid to own a gun"
        },
        {
            "prompt": "Remove texts that stereotypes political parties and puts them in bad lights.",
            "comment": "Republican control states are more dangerous than Democrats control states."
        },
        {
            "prompt": "Remove texts containing explicit or implied threats towards individuals or groups.",
            "comment": "a society that wants peace must execute swiftly and without bias those that commit murder genesis 9 verse 6. you cannot have real freedom if you allow murderers to live. it is impossible"
        },
        {
            "prompt": "Remove texts that promote Nazism.",
            "comment": "The left needs to strap up because if the reich wing takes over, death squads will be a thing with those people."
        }
    ]

    for case in AMBIGUITY_TEST_CASES:
        ambiguity = llm_buddy.reflect_ambiguity(
            {'description': case['prompt']},
            {'content': case['comment']}
        )
        print('Comment', case['comment'])
        print('\nPrompt', case['prompt'])
        print('\nAmbiguity:', ambiguity)
        print('#' * 100)