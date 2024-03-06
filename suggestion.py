import pandas as pd
import numpy as np
import re
import openpyxl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#pip install nltk
#python -m spacy download en_core_web_sm
import nltk
nltk.download('punkt')
#pip install openpyxl
import re
import spacy
from nltk.tokenize import word_tokenize
nlp = spacy.load('en_core_web_sm')
from difflib import SequenceMatcher
from colorama import Fore, Back, Style

def algo(problem):
        df= pd.read_excel(r"https://vscocorp.sharepoint.com/:x:/r/teams/MATIssuesHelpDesk/Shared%20Documents/Comments2.xlsx?d=w48c47ec102034471bd3578f850832cd2&csf=1&web=1&e=m0mu77",'main')
    #df = pd.DataFrame(df)
    #response = input("Comment: ")
    #problem = "Your problem description"
        new_data =  pd.DataFrame({'ID': [""], 'Category': [""], 'Comment': problem, 'Description': [""]})
        f_df = pd.concat([df, new_data], ignore_index=True)
        df =f_df
    #df = new_df
        corpus = list(set(df["Comment"].values))
        corpus2 = list(set(df["Comment"].values))

        corpus = list(filter(lambda x: str(x) != 'nan', corpus))
        corpus2 = list(filter(lambda x: str(x) != 'nan', corpus2))




        # Load the English language model


        def extract_important_words(sentence):
            # Process the sentence using spaCy
            doc = nlp(sentence)

            # Extract nouns and adjectives
            important_words = [token.lemma_.lower() for token in doc if token.pos_ in ('NOUN', 'ADJ')]

            return important_words

        new_corpus2 = []
        corpus_backup =[]
        for i in range (len(corpus)):
            try:
                review = re.sub("&" , "and" , corpus[i])
                review = re.sub('[^a-zA-Z]',' ',review)
                review = review.lower()
                new_corpus2.append(review)
                corpus_backup.append(corpus[i])
            except:
                new_corpus2.append(review)
                corpus_backup.append(corpus[i])


        new_corpus = []
        my_stop_words = ['an','the','is','a','the', 'not', '?', 'on', 'have', 'has', 'having', 'any', 'in', 'docs','documents','document',
                         'data','datas',
                         'now', 'getting','get', 'got','receive','received', 'what', 'how', 'when','zip','.','pdf',"_","/",'text','txt'

                         'i','ii','iii','iv','v','vi','vii','viii','ix'
                             'x','xi','xii','xiii','xiv','xv']
        for i in range (len(new_corpus2)):
              review = new_corpus2[i]
              review = [w for w in word_tokenize(review)  if not w in my_stop_words]
              review = ' '.join(review)
        
              review = re.sub(' +', ' ', review)
              review = re.sub(' +', ' ', review)
              review = review.strip()
              review2 = extract_important_words(review)
              #print(review2)
              review = ' '.join(review2)
              #print(review)
              new_corpus.append(review)


        corpus_p1 = []
        corpus_p2 = []
        corpus_p1_m = []
        corpus_p2_m = []

        for i in range (len(new_corpus)):
            if len(new_corpus[i].split(' '))>3:
                corpus_p1.append(new_corpus[i])
                corpus_p1_m.append(corpus_backup[i])
            else:
                corpus_p2.append(new_corpus[i])
                corpus_p2_m.append(corpus_backup[i])


        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus_p1)
        list1_1 = []
        list1_2 = []
        threshold = 0.2
        for x in range(0,X.shape[0]):
          for y in range(x,X.shape[0]):
            if(x!=y):
              if(cosine_similarity(X[x],X[y])>threshold) and (corpus_p1_m[x] != corpus_p1_m[y] and corpus_p1[x][1]==corpus_p1[y][1]):
                #print((corpus[x],' | | ',corpus[y])) 
                list1_1.append(corpus_p1_m[x])
                list1_2.append(corpus_p1_m[y])




        def similar(a, b):
            return SequenceMatcher(None, a, b).ratio()


        list2_1 = []
        list2_2 = []

        for x in range(0,len(corpus_p2)):
          for y in range(x,len(corpus_p2)):
            if(x!=y):
              if((similar(corpus_p2[x],corpus_p2[y])>.70)):

                list2_1.append(corpus_p2_m[x])
                list2_2.append(corpus_p2_m[y])


        list1 = list1_1 + list2_1
        list2 = list1_2 + list2_2


        unique_id = []
        for i in range (len(list1)):
            unique_id.append(i)


        for z in range (10):

            for i in range (0,len(list1)):
                for j in range (i+1,len(list1)):
                    if list1[j] == list1[i]:
                        if (unique_id[j]>unique_id[i]):
                          unique_id[j]=unique_id[i]
                        else:
                            unique_id[i]=unique_id[j]

            for i in range (0,len(list1)):
                for j in range (i+1,len(list1)):
                    if list2[j] == list2[i]:
                        if (unique_id[j]>unique_id[i]):
                          unique_id[j]=unique_id[i]
                        else:
                            unique_id[i]=unique_id[j]


            for i in range (0,len(list1)):
                for j in range (len(list1)):
                    if list1[j] == list2[i]:
                        if (unique_id[j]>unique_id[i]):
                          unique_id[j]=unique_id[i]
                        else:
                            unique_id[i]=unique_id[j]


            for i in range (len(list1)):
                for j in range (len(list1)):
                    if list2[j] == list1[i]:
                        if (unique_id[j]>unique_id[i]):
                          unique_id[j]=unique_id[i]
                        else:
                            unique_id[i]=unique_id[j]   




        unique_c = list(set(unique_id))
        unique_c
        all_cluster = []
        for i in range (len(unique_c)):
            cluster = []
            for j in range (len(list1)):
                if unique_c[i] == unique_id[j]:
                    cluster.append(list1[j])
                    cluster.append(list2[j])
            all_cluster.append(list(set(cluster)))
  
        df_final=pd.DataFrame(all_cluster[0])
        df_final=pd.DataFrame(all_cluster[0])
        for i in range(len(all_cluster)-1):
            df_final=pd.concat([df_final,pd.DataFrame(all_cluster[i+1])])
 

        df_final = df_final.rename(columns = {0:'Comment'})
  


        control_len=[]
        for i in range (len(unique_c)):
            for j in range (len(all_cluster[i])):
                control_len.append(unique_c[i])
        df_final['cluster']=control_len




        final_df = pd.merge(df, 
                              df_final, 
                              on ='Comment', 
                              how ='inner')






        final_df = final_df.sort_values(by='cluster')
        result = final_df
        cc = str(df.iloc[-1,:]['Comment'])
        try:
           cluster_i = int(result.loc[result['Comment'] == cc,'cluster'])
        except:
           cluster_i = 999
        #print(cluster_i)
        final_result = result.loc[result['cluster'] == cluster_i,:]
        final_result = final_result[final_result['ID'] != ''].reset_index()
        #print("\n")
    # print("User: ", Fore.YELLOW + cc)
        #print("\n")
        #print(Fore.BLUE + "Below is the recommendation for similiar type of problem ")
        #print("\n")
        answers = []
        for i in range (len(final_result)):
            p = []
            s = []
            p.append(final_result['Comment'][i])
            s.append(final_result['Description'][i])
            answers.append(p+s)
        return answers
        # print(Fore.GREEN + "Suggestion "+ str(i+1) + ": "+ final_result['Description'][i])
        #result.to_excel(r'C:\Users\surdas\OneDrive - vscocorp\Desktop\LLM\cluster.xlsx', index = False)


#final_df.to_excel(r'C:\Users\surdas\OneDrive - vscocorp\Desktop\LLM\output_s3.xlsx',index = False)


import streamlit as st

# Streamlit app layout
def main():
    # Apply HTML/CSS styling for the title
    

    # Add your company logo with fixed height and width
    st.markdown(
        """
        <style>
        .logo-container {
            max-width: 700px;
            max-height: 150px;
            display: flex;
            justify-content: center;
        }
        .logo-container img {
            width: 100%;
            height: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="logo-container">
            <img src="https://logos-world.net/wp-content/uploads/2020/05/Victoria-Secret-Logo.png" alt="Company Logo">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center; color: #ff69b4;'>MAT Helpdesk</h1>", unsafe_allow_html=True)

    # User input for the problem
    problem = st.text_area("What's your Error Message", key="error_message", height=100)

    # Center and widen the input area
    st.markdown(
        """
        <style>
        .css-17exl3v {
            max-width: 800px;
            margin: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Apply HTML/CSS styling for the button-like appearance
    if st.button("Get Suggestions", key="get_suggestions_button", help="Submit your problem"):
        # Display answers when the button is clicked
        if problem:
            answers = algo(problem)
            st.header("Suggestions:")

            # Apply styles to the output
            for i, answer in enumerate(answers, start=1):
                st.markdown(f"**Suggestion {i}:**")
                st.success(f"**{answer[0]}**")
                st.info(f"{answer[1]}")
        else:
            st.warning("Please enter a problem first.")

if __name__ == "__main__":
    main()
