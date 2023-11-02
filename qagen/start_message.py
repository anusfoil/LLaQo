START_MESSAGE = [
    {
        "role": "system",
        "content": "You are an assistant that can standardize my data in piano performance assessment. \
I have a framework of performance assessment and list of teacher's feedback, you can standardize \
teacher's feedback information into QA pairs so that I can use to train models. \
I will give you my evaluation framework and some examples. "
    },
    {
        "role": "user",
        "content": "Framework - There are two kinds of question, one is more objective, \
addressing individual performance dimensions such as dynamics, tempo. The other one is more descriptive, \
general feedback about the performance.  \n \
            Objective measure (classification / regression tasks) \n \
                \t Note: Is the performed note accurate? \n \
                \t Integrity: Is the performance hesitant or smooth? \n \
                \t Tempo: Is the performed tempo stable? \n \
                       Is there reasonalbe tempo contrast? \n \
                       How is the overall tempo? \n \
                \t Rhythm: Is the rhythm accurate? \n \
                \t Articulation: Is the articulation appropriate?  \n \
                \t Pedal: Is the pedal change clean? \n \
                       Is the pedal usage appropriate? \n \
                \t Phrasing: Is the phrasing or melodic shaping clear? \n \
                \t Timbre (tone production): How is the tone production?  \n \
                \t Dynamics: Is there reasonalbe dynamics contrast?  \n \
                \t Balance (between hands): Is there reasonable balance between hands?  \n \
                \t Score-agreement: Is the performance following score instructions? \n \n \
            Description (similar to captioning & reasoning)  \n \
                \t How would you describe the emotional intent of the performance?  \n \
                \t What is the overall score you would assign to the performance, in a scale 5?  \n \
                \t In which skill level would you say the performer is in? What's the reasoning of that?  \n \
                \t How would you rate the difficulty level of this piece?  \n \
                \t Is the interpretation appropriate to the style of the composition?  \n \
                \t What does the student needs to work on?"
    },
    {
        "role": "user",
        "content": "Here is an expected example, note that you only need to include the QA pairs where \n \
                    information is given by teacher: \n \
                    Teacher: a little too beat-by-beat; needed more waltz feel.  But good accuracy and rhythm overall. \n \
                          Left hand a bit choppy. \n \
                    Generate: \n \
                    1. Q: Is there reasonalbe tempo contrast? A: Tempo a little too beat-by-beat. \n \
                    2. Q: Is the performed note accurate? A: Note accurate. \n \
                    3. Q: Is the performed rhythm accurate? A: good accuracy and rhythm overall. \n \
                    4. Q: Is the articulation appropriate? A: Left hand a bit choppy (staccato). \n \
                    5. Q: Is the interpretation appropriate to the style of the composition? A: Needs more waltz feel."
    },
    {
        "role": "user",
        "content": "Another example: \n \
                    Teacher: LH too detached. Good tempo. No rit. at end. No dynamics as indicated lines 3-4. Wrong LH note LH line 3. \n \
                    Generate: \n \
                    1. Q: How is the overall tempo? A: Good tempo. \n \
                    2. Q: Is there reasonalbe tempo contrast? A: No rit. \n \
                    3. Q: Is there reasonalbe dynamics contrast? A: No dynamics as indicated lines 3-4. \n \
                    4. Q: Is the articulation appropriate? A: LH too detached (staccato). \n \
                    5. Q: Is the performed note accurate? A: Wrong LH note LH line 3."
    },
    {
        "role": "user",
        "content": "Another example: \n \
                    Teacher: Good playing. Left hand could have been softer and staccatos more emphasized. \n \
                    Generate: \n \
                    1. Q: In which skill level would you say the performer is in?  A: Good playing. \n \
                    2. Q: What is the overall score you would assign to the performance, in a scale 5? A: 4 \n \
                    3. Q: What does the student needs to work on? A: Left hand could have been softer and staccatos more emphasized."
    }
]