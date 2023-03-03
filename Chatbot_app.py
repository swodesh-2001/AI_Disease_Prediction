import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
RANDOM_STATE = 7

df= pd.read_csv("data/final_dataset.csv")
symptoms_list = df.drop('Diseases', axis = 1).columns
model_path = 'xgb_classifier.pickle.dat'
encoder_path = 'encoder.pickle.dat'


def find_similar_words(sentence , feature_list):
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

def feature_label(word_list,feature_list):
    label = []
    for word in feature_list:
        if word in word_list:
            label.append(1)
        else :
            label.append(0)
    return label

def predict_my_disease(message):
    word_list = find_similar_words(message,symptoms_list)
    extracted_feature = feature_label(word_list,symptoms_list)
    to_predict = extracted_feature
    to_predict = np.expand_dims(to_predict, axis=0)
    file = open(model_path,'rb')
    model = pickle.load(file)
    file.close()
    file = open(encoder_path ,'rb')
    encoder = pickle.load(file)
    file.close()
    predicted = model.predict(to_predict)
    if any(predicted[0]) != 1 :
        return("Can you give me more information")
    else :
        return("There is a possibility that you might have " ,encoder.inverse_transform(predicted)[0][0])

def get_insight_prediction(message):
    word_list = find_similar_words(message,symptoms_list)
    print(word_list)
    extracted_feature = feature_label(word_list,symptoms_list)
    print(extracted_feature)
    to_predict = extracted_feature
    to_predict = np.expand_dims(to_predict, axis=0)
    print(to_predict)
    file = open(model_path,'rb')
    model = pickle.load(file)
    file.close()
    file = open(encoder_path,'rb')
    encoder = pickle.load(file)
    file.close()
    predicted = model.predict(to_predict)
    print(predicted)
    if any(predicted[0]) != 1 :
        print("Can you give me more information")
    else :
        print("There is a possibility that you might have " ,encoder.inverse_transform(predicted))

class ChatbotApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chatbot App")
        self.geometry("450x500")
        self.resizable(width=False, height=False)
        self.flag = False

        # Set up color scheme
        self.bg_color = "#F7F9FC"
        self.chat_bg_color = "#0c0c0d"
        self.text_color = "#9ea3a8"
        self.accent_color = "#3E9FFF"
        self.title_color = "#0080ff"
        # Set up styles
        self.title_font = ("Helvetica", 20, "bold")
        self.subtitle_font = ("Helvetica", 12, "bold")
        self.text_font = ("Helvetica", 12)



        # Set up UI elements
        self.title_label = tk.Label(self, text="Mero Swasthya Chatbot", fg=self.title_color, font=self.title_font, bg=self.bg_color)
        self.title_label.pack(pady=20)


        self.chat_log = tk.Text(self, fg=self.text_color, bg=self.chat_bg_color, font=self.text_font, height=15, width=50, highlightthickness=0, borderwidth=0)
        self.chat_log.pack(padx=20, pady=20)
        self.welcome_message = "Hello, I'm Mero Swasthya Chatbot. How can I assist you?"

        self.display_chatbot_response(self.welcome_message)

        self.user_input = tk.Entry(self, fg=self.text_color, bg=self.chat_bg_color, font=self.text_font,highlightthickness=0, borderwidth=0)
        self.user_input.pack(padx=20, pady=10, fill="x")
        self.user_input.bind("<Return>", self.handle_user_input)

    def handle_user_input(self, event):
        user_input_text = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)
        self.chat_log.insert(tk.END, "\n")
        self.chat_log.insert(tk.END, "\nYou: " + user_input_text + "\n")
        self.display_chatbot_response(user_input_text)

    def display_chatbot_response(self, response_text):
        if self.flag :
            chatbot_response_text = predict_my_disease(str(response_text))
        else :
            chatbot_response_text = response_text.upper()

        def display_letter_by_letter(index=0):

            if index < len(chatbot_response_text):
                self.chat_log.insert(tk.END, chatbot_response_text[index], "tag_right")
                self.chat_log.see(tk.END)
                self.after(50, lambda: display_letter_by_letter(index + 1))
            else:
                self.chat_log.see(tk.END)

        self.chat_log.tag_config("tag_left", justify="left", lmargin1=20, lmargin2=20, wrap="word")
        self.chat_log.tag_config("tag_right", justify="right", rmargin=20)
        self.chat_log.insert(tk.END, "\nChatbot: ", "tag_left")
        display_letter_by_letter()
        self.flag = True

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()
