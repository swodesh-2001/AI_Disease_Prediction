import tkinter as tk
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
RANDOM_STATE = 7

class ChatbotApp(tk.Tk):
    def __init__(self,model_path,encoder_path):
        super().__init__()
        self.title("Chatbot App")
        self.geometry("450x500")
        self.resizable(width=False, height=False)
        self.flag = False
        self.symptoms = []
        file = open(model_path,'rb')
        self.model = pickle.load(file)
        file.close()
        file = open(encoder_path,'rb')
        self.encoder = pickle.load(file)
        file.close()

        # Set up color scheme
        self.bg_color = "#F7F9FC"
        self.chat_bg_color = "#0c0c0d"
        self.text_color = "#9ea3a8"
        self.accent_color = "#3E9FFF"
        self.title_color = "#0080ff"
         
        self.title_font = ("Helvetica", 20, "bold")
        self.subtitle_font = ("Helvetica", 12, "bold")
        self.text_font = ("Helvetica", 12)



        
        self.title_label = tk.Label(self, text="Mero Swasthya Chatbot", fg=self.title_color, font=self.title_font, bg=self.bg_color)
        self.title_label.pack(pady=20)


        self.chat_log = tk.Text(self, fg=self.text_color, bg=self.chat_bg_color, font=self.text_font, height=15, width=50, highlightthickness=0, borderwidth=0)
        self.chat_log.pack(padx=20, pady=20)
        self.welcome_message = "Hello,  I'm Mero Swasthya Chatbot. How can I assist you?"
        
        self.display_chatbot_response(self.welcome_message)

        self.user_input = tk.Entry(self, fg=self.text_color, bg=self.chat_bg_color, font=self.text_font,highlightthickness=0, borderwidth=0)
        self.user_input.pack(padx=20, pady=10, fill="x")
        self.user_input.bind("<Return>", self.handle_user_input)

    def find_similar_words(self,sentence , feature_list):
        sentence = sentence.lower()
        words = re.findall(r'\b\w+\b', sentence)
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        similar_words = []
        for word in filtered_words:
            if word in feature_list:
                similar_words.append(word)

            for feature_word in feature_list:
                if word in feature_word:
                    similar_words.append(feature_word)
        return list(set(similar_words))

    def feature_label(self,word_list,feature_list):
        label = []
        for word in feature_list:
            if word in word_list:
                label.append(1)
            else :
                label.append(0)
        return label    
    
    def predict_my_disease(self,message):
        print(message)
        extracted_feature = feature_label(message,symptoms_list)
        to_predict = extracted_feature
        to_predict = np.expand_dims(to_predict, axis=0) 
        predicted = self.model.predict(to_predict)
        if any(predicted[0]) != 1 :
            return("Can you give me more information")
        else :
            disease = self.encoder.inverse_transform(predicted)[0][0]
            return_message = "There is a possibility that you might have " + str(disease)
            return_message += "\n Disease Description"
            disease_row = df_description[df_description['Disease'] == disease]
            description = disease_row['Description'].values[0]
            return_message += "\n"+ str(description) +"\n"
            return_message += "\n Disease Precaution \n"
            disease_row = df_precaution[df_precaution['Disease'] == disease]
            precaution1 = disease_row['Precaution_1'].values[0]
            precaution2 = disease_row['Precaution_2'].values[0]
            precaution3 = disease_row['Precaution_3'].values[0]
            precaution4 = disease_row['Precaution_4'].values[0]
            
            return_message += "\n"+ str(precaution1)  
            return_message += "\n"+ str(precaution2) 
            return_message += "\n"+ str(precaution3) 
            return_message += "\n"+ str(precaution4) 
            return(return_message)    
        
    def handle_user_input(self, event):
        user_input_text = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        self.chat_log.insert(tk.END, "\n")
        self.chat_log.insert(tk.END, "\nYou: " + user_input_text + "\n")
        self.display_chatbot_response(user_input_text)

    def display_chatbot_response(self, response_text):
        if self.flag :
            flag_temp = True
            if len(self.symptoms) == 0:
                self.symptoms = self.find_similar_words(response_text,symptoms_list)
                message = "Can you please select the symptoms that you have and type out the respective index \n"
                for i in range(0,len(self.symptoms)) :    
                     message += "\n" + str(i+1) + ". " + self.symptoms[i]
                chatbot_response_text = message
                flag_temp = False
                if len(self.symptoms) == 0 :
                    chatbot_response_text = "Hello, can you please provide me with symptoms of your disease"
                    

            if len(self.symptoms) != 0 and flag_temp:
                index_list = re.findall('\d+', response_text)
                index_list = [int(x) for x in index_list]
                temp = []
                for i in index_list:
                    temp.append(self.symptoms[i-1])
                                
                self.symptoms = temp  
                symptom_string = ""
                for i in self.symptoms:
                    symptom_string += " " + str(i)
                    
                chatbot_response_text = self.predict_my_disease(str(symptom_string))
                self.symptoms = []   

        else :
            chatbot_response_text = response_text.upper()
            

        def display_letter_by_letter(index=0):

            if index < len(chatbot_response_text):
                self.chat_log.insert(tk.END, chatbot_response_text[index], "tag_right")
                self.chat_log.see(tk.END)
                self.after(20, lambda: display_letter_by_letter(index + 1))
            else:
                self.chat_log.see(tk.END)

        self.chat_log.tag_config("tag_left", justify="left", lmargin1=20, lmargin2=20, wrap="word")
        self.chat_log.tag_config("tag_right", justify="right", rmargin=20)
        self.chat_log.insert(tk.END, "\nChatbot: ", "tag_left")
        display_letter_by_letter()
        self.flag = True

if __name__ == "__main__":
    df= pd.read_csv("data/final_dataset.csv")
    df_description = pd.read_csv("data/symptom_Description.csv")
    df_precaution = pd.read_csv("data/symptom_precaution.csv")
    symptoms_list = df.drop('Diseases', axis = 1).columns
    disease_list = df['Diseases']
    model_path = 'xgb_classifier.pickle.dat'
    encoder_path = 'encoder.pickle.dat'
    app = ChatbotApp(model_path,encoder_path)
    app.mainloop()
