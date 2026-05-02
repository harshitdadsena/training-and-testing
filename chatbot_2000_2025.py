"""
Simple Q&A + ML-based History Chatbot (2000-2025)
- Uses TF-IDF for text vectorization
- Uses Cosine Similarity for finding best answer match
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa_data = [
    # Early 2000s Events
    {"question": "What was Y2K?", "answer": "Y2K was the concern that computer systems would fail on January 1, 2000, due to the year 2000 problem. Fortunately, the crisis was largely avoided."},
    {"question": "When did the iPhone release?", "answer": "The first iPhone was released by Apple on June 29, 2007."},
    {"question": "What was the dot-com bubble?", "answer": "The dot-com bubble (2000-2002) was a stock market crash caused by excessive speculation in internet companies. Many companies went bankrupt."},
    
    # September 11 Attacks
    {"question": "When did 9/11 happen?", "answer": "The September 11 attacks occurred on September 11, 2001."},
    {"question": "What happened on September 11 2001?", "answer": "Terrorists hijacked four commercial airplanes and carried out coordinated attacks on the United States. Two planes hit the World Trade Center, one hit the Pentagon, and one crashed in Pennsylvania."},
    {"question": "What was the War on Terror?", "answer": "The War on Terror was a global military campaign launched by the US after 9/11, primarily targeting al-Qaeda and Taliban in Afghanistan."},
    
    # Iraq War
    {"question": "When did the Iraq War start?", "answer": "The Iraq War began in 2003 when the US-led coalition invaded Iraq."},
    {"question": "When did the Iraq War end?", "answer": "The Iraq War ended in 2011 when US forces withdrew from Iraq."},
    {"question": "Who was Saddam Hussein?", "answer": "Saddam Hussein was the dictator of Iraq from 1979 to 2003. He was captured in 2003 and executed in 2006."},
    
    # Hurricane Katrina
    {"question": "When did Hurricane Katrina happen?", "answer": "Hurricane Katrina struck in August 2005, devastating New Orleans and the Gulf Coast."},
    
    # Apple & Tech
    {"question": "When was YouTube created?", "answer": "YouTube was created in 2005 and launched in 2005."},
    {"question": "When was Facebook created?", "answer": "Facebook was created by Mark Zuckerberg in 2004."},
    {"question": "When was Twitter created?", "answer": "Twitter was created in 2006."},
    {"question": "Who founded Apple?", "answer": "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976."},
    
    # Global Financial Crisis
    {"question": "What was the 2008 financial crisis?", "answer": "The 2008 financial crisis was a global economic downturn caused by the subprime mortgage crisis. It led to the Great Recession."},
    {"question": "What caused the 2008 financial crisis?", "answer": "The crisis was caused by risky mortgage lending, excessive debt, and the collapse of the housing market."},
    {"question": "What was the Great Recession?", "answer": "The Great Recession (2007-2009) was the worst economic crisis since the Great Depression."},
    
    # Barack Obama
    {"question": "Who was the first African American president?", "answer": "Barack Obama was the first African American President of the United States, serving from 2009 to 2017."},
    {"question": "When did Barack Obama become president?", "answer": "Barack Obama became president in January 2009."},
    {"question": "What was Obamacare?", "answer": "Obamacare (Affordable Care Act) was healthcare reform passed in 2010 under President Obama."},
    
    # Arab Spring
    {"question": "What was the Arab Spring?", "answer": "The Arab Spring (2010-2012) was a wave of protests and revolutions across the Middle East and North Africa."},
    
    # Syria Civil War
    {"question": "When did the Syrian Civil War start?", "answer": "The Syrian Civil War started in 2011."},
    
    # Snowden & NSA
    {"question": "Who was Edward Snowden?", "answer": "Edward Snowden was a former NSA contractor who leaked classified documents about government surveillance in 2013."},
    
    # Climate & Environment
    {"question": "What is climate change?", "answer": "Climate change refers to long-term changes in global temperature and weather patterns, largely driven by human activities."},
    {"question": "What is the Paris Agreement?", "answer": "The Paris Agreement (2015) is an international treaty on climate change, aiming to limit global warming to 1.5 degrees Celsius."},
    
    # COVID-19
    {"question": "When did COVID-19 start?", "answer": "COVID-19 was first identified in Wuhan, China in late 2019 and declared a pandemic in March 2020."},
    {"question": "What is COVID-19?", "answer": "COVID-19 is a respiratory disease caused by the SARS-CoV-2 coronavirus."},
    {"question": "When was the COVID-19 vaccine released?", "answer": "The first COVID-19 vaccines were released in late 2020."},
    
    # Donald Trump
    {"question": "Who was the 45th US President?", "answer": "Donald Trump was the 45th US President, serving from 2017 to 2021."},
    {"question": "What was January 6th?", "answer": "January 6, 2021 was the day supporters of Donald Trump attacked the US Capitol building."},
    
    # Joe Biden
    {"question": "Who is the current US President?", "answer": "Joe Biden is the 46th and current US President, serving since 2021."},
    
    # Russia & Ukraine
    {"question": "When did the Russia Ukraine war start?", "answer": "Russia invaded Ukraine in February 2022."},
    {"question": "What is the war in Ukraine about?", "answer": "The Russia-Ukraine war began when Russia launched a full-scale invasion of Ukraine in February 2022."},
    
    # AI & Technology
    {"question": "What is artificial intelligence?", "answer": "Artificial Intelligence (AI) is the simulation of human intelligence by machines, including learning and problem-solving."},
    {"question": "What is ChatGPT?", "answer": "ChatGPT is an AI chatbot created by OpenAI, first released in 2022."},
    {"question": "When was ChatGPT released?", "answer": "ChatGPT was released in November 2022."},
    
    # Space & Science
    {"question": "What is SpaceX?", "answer": "SpaceX is a space company founded by Elon Musk in 2002, known for reusable rockets."},
    {"question": "Who founded SpaceX?", "answer": "SpaceX was founded by Elon Musk in 2002."},
    {"question": "What is the James Webb Space Telescope?", "answer": "The James Webb Space Telescope is a powerful space telescope launched in 2021 to observe the early universe."},
    
    # Other Notable Events
    {"question": "Who was Meghan Markle?", "answer": "Meghan Markle married Prince Harry in 2018 and became the Duchess of Sussex."},
    {"question": "What is Bitcoin?", "answer": "Bitcoin is a cryptocurrency created in 2009, the first decentralized digital currency."},
    {"question": "Who created Bitcoin?", "answer": "Bitcoin was created by an unknown person or group using the name Satoshi Nakamoto in 2009."},
    {"question": "What is Brexit?", "answer": "Brexit is the departure of the United Kingdom from the European Union, which officially happened in 2020."},
    {"question": "When did Brexit happen?", "answer": "The UK formally left the European Union on January 31, 2020."},
    {"question": "Who is Elon Musk?", "answer": "Elon Musk is a businessman who founded SpaceX and Tesla, and owns X (Twitter)."},
    {"question": "Who is Vladimir Putin?", "answer": "Vladimir Putin has been the President of Russia since 2000, with breaks in between."},
]

df = pd.DataFrame(qa_data)

print("=" * 60)
print("  HISTORY CHATBOT (2000-2025)")
print("  Q&A + TF-IDF Based Learning")
print("=" * 60)
print(f"\nKnowledge base loaded with {len(df)} Q&A pairs.\n")

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    lowercase=True
)

question_vectors = vectorizer.fit_transform(df['question'])

print("TF-IDF Vectorizer trained on questions!")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)} words")
print(f"Question matrix shape: {question_vectors.shape}")
print("\n" + "-" * 60)

def find_best_answer(user_question):
    question_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(question_vector, question_vectors)
    best_match_index = np.argmax(similarities)
    best_similarity = similarities[0][best_match_index]
    best_answer = df.iloc[best_match_index]['answer']
    best_question = df.iloc[best_match_index]['question']
    
    return {
        'answer': best_answer,
        'similarity': best_similarity,
        'matched_question': best_question
    }

def chat():
    print("CHATBOT: Hello! I'm a history bot for the 21st century.")
    print("         Ask me anything about history between 2000-2025.")
    print("         Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("CHATBOT: Goodbye! Thanks for chatting!")
            break
        
        if not user_input:
            print("CHATBOT: Please ask me a question!\n")
            continue
        
        result = find_best_answer(user_input)
        
        print(f"\nCHATBOT: {result['answer']}")
        print(f"         [Confidence: {result['similarity']:.2%}]")
        
        if result['similarity'] < 0.3:
            print("         (I'm not very confident in this answer)")
        print()

if __name__ == "__main__":
    chat()
