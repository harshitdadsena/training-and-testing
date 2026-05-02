"""
Simple Q&A + ML-based History Chatbot (1900-2000)
- Uses TF-IDF for text vectorization
- Uses Cosine Similarity for finding best answer match
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa_data = [
    # World War I (1914-1918)
    {"question": "When did World War I start?", "answer": "World War I started in 1914."},
    {"question": "When did World War I end?", "answer": "World War I ended in 1918."},
    {"question": "What caused World War I?", "answer": "World War I was caused by a complex web of alliances, militarism, and nationalism. The assassination of Archduke Franz Ferdinand was the immediate trigger."},
    {"question": "Who were the main allies in World War I?", "answer": "The Allies included France, Britain, Russia, and later the United States. The Central Powers included Germany, Austria-Hungary, and Ottoman Empire."},
    
    # Roaring Twenties
    {"question": "What was the Roaring Twenties?", "answer": "The Roaring Twenties (1920-1929) was a period of economic prosperity and cultural dynamism in the US and Europe."},
    {"question": "What happened on Wall Street in 1929?", "answer": "The Wall Street Crash of 1929 triggered the Great Depression. Stock prices collapsed dramatically in October 1929."},
    
    # World War II (1939-1945)
    {"question": "When did World War II start?", "answer": "World War II started in 1939 when Germany invaded Poland."},
    {"question": "When did World War II end?", "answer": "World War II ended in 1945 with Japan's surrender after atomic bombs were dropped."},
    {"question": "What was D-Day?", "answer": "D-Day (June 6, 1944) was the Allied invasion of Normandy, France, the largest naval invasion in history."},
    {"question": "Who were the Allied powers in World War II?", "answer": "The main Allied powers were Britain, the Soviet Union, and the United States. The Axis powers were Germany, Italy, and Japan."},
    
    # Cold War (1947-1991)
    {"question": "What was the Cold War?", "answer": "The Cold War (1947-1991) was a period of geopolitical tension between the US-led Western bloc and Soviet-led Eastern bloc."},
    {"question": "When did the Cold War start?", "answer": "The Cold War started around 1947, after World War II ended."},
    {"question": "When did the Cold War end?", "answer": "The Cold War ended in 1991 with the collapse of the Soviet Union."},
    {"question": "What was the Cuban Missile Crisis?", "answer": "The Cuban Missile Crisis (1962) was a confrontation between the US and Soviet Union over nuclear missiles in Cuba. It brought the world close to nuclear war."},
    {"question": "What was the Berlin Wall?", "answer": "The Berlin Wall (1961-1989) was a barrier dividing East and West Berlin. It became a symbol of the Cold War division."},
    
    # Korean War (1950-1953)
    {"question": "When was the Korean War?", "answer": "The Korean War lasted from 1950 to 1953."},
    
    # Vietnam War (1955-1975)
    {"question": "When was the Vietnam War?", "answer": "The Vietnam War lasted from 1955 to 1975."},
    
    # Civil Rights Movement
    {"question": "Who was Martin Luther King Jr.?", "answer": "Martin Luther King Jr. (1929-1968) was a leader of the American Civil Rights Movement who advocated nonviolent protest."},
    {"question": "What was the Civil Rights Act?", "answer": "The Civil Rights Act of 1964 outlawed discrimination based on race, color, religion, sex, or national origin."},
    {"question": "What was the March on Washington?", "answer": "The March on Washington (1963) was a historic protest where Martin Luther King Jr. gave his 'I Have a Dream' speech."},
    
    # Space Race
    {"question": "What was the Space Race?", "answer": "The Space Race was a 20th-century competition between the US and Soviet Union for space exploration dominance."},
    {"question": "Who was the first man in space?", "answer": "Yuri Gagarin from the Soviet Union was the first man in space in 1961."},
    {"question": "Who was the first man on the moon?", "answer": "Neil Armstrong was the first man on the moon in 1969, as part of NASA's Apollo 11 mission."},
    {"question": "When did Apollo 11 land on the moon?", "answer": "Apollo 11 landed on the moon on July 20, 1969."},
    
    # Key Inventions & Technology
    {"question": "When was the computer invented?", "answer": "Modern computers evolved throughout the 1900s. ENIAC (1945) is often considered the first electronic computer."},
    {"question": "When was the internet created?", "answer": "The internet's precursor ARPANET was created in 1969. The World Wide Web was invented in 1989 by Tim Berners-Lee."},
    
    # Other Major Events
    {"question": "What was the Great Depression?", "answer": "The Great Depression (1929-1939) was the worst economic downturn in modern history, starting with the stock market crash."},
    {"question": "What was the Holocaust?", "answer": "The Holocaust was the systematic genocide of European Jews by Nazi Germany during World War II. Approximately 6 million Jews were killed."},
    {"question": "When did India gain independence?", "answer": "India gained independence from Britain on August 15, 1947."},
    {"question": "When did the Titanic sink?", "answer": "The Titanic sank on April 15, 1912, on its maiden voyage."},
    {"question": "Who was Winston Churchill?", "answer": "Winston Churchill (1874-1965) was Britain's Prime Minister during World War II, known for his leadership and speeches."},
    {"question": "What was the Marshall Plan?", "answer": "The Marshall Plan (1948) was a US program to rebuild Western Europe after World War II."},
    {"question": "What was NATO?", "answer": "NATO (North Atlantic Treaty Organization) was founded in 1949 as a military alliance between Western countries."},
    {"question": "What was the United Nations?", "answer": "The United Nations was founded in 1945 after World War II to promote international cooperation and peace."},
    {"question": "Who was Adolf Hitler?", "answer": "Adolf Hitler (1889-1945) was the leader of Nazi Germany who started World War II and orchestrated the Holocaust."},
    {"question": "Who was Albert Einstein?", "answer": "Albert Einstein (1879-1955) was a physicist who developed the theory of relativity. He won the Nobel Prize in 1921."},
    {"question": "Who was Mahatma Gandhi?", "answer": "Mahatma Gandhi (1869-1948) was an Indian leader who advocated nonviolent resistance and led India to independence."},
]

df = pd.DataFrame(qa_data)

print("=" * 60)
print("  HISTORY CHATBOT (1900-2000)")
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
    print("CHATBOT: Hello! I'm a history bot born in the 1900s.")
    print("         Ask me anything about history between 1900-2000.")
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
