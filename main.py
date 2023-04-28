import tkinter as tk
from tkinter import ttk
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from collections import Counter
RANDOM_STATE = 7


class ChatbotApp(tk.Tk):
    def __init__(self,xgb_path,logistic_path,random_path,encoder_path,disease_description_path
    ,disease_precaution_path,disease_severity_path,disease_data_path,chatbot_symptom_path,):
        super().__init__()
        self.title("Chatbot App")
        self.geometry("450x500")
        self.resizable(width=False, height=False)

        self.df= pd.read_csv(disease_data_path)
        self.df_description = pd.read_csv(disease_description_path)
        self.df_precaution = pd.read_csv(disease_precaution_path)
        self.df_severity = pd.read_csv(disease_severity_path)
        self.df_severity['Symptom'] = self.df_severity['Symptom'].str.replace('_', ' ')
        file = open(chatbot_symptom_path,'rb')
        self.symptoms_list = pickle.load(file)
        file.close()
        self.disease_list = self.df['Diseases']


        self.flag = False
        self.symptoms = []
        file = open(xgb_path,'rb')
        self.model_xgb = pickle.load(file)
        file.close()
        file = open(logistic_path,'rb')
        self.model_log = pickle.load(file)
        file.close()
        file = open(random_path,'rb')
        self.model_random = pickle.load(file)
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

    def vote_of_majority(self,to_predict):
        disease_predicted = []
        predicted_xgb = self.model_xgb.predict(to_predict)
        if any(predicted_xgb[0]) != 1 :
            disease_predicted.append('none')
        else :
            disease_predicted.append(self.encoder.inverse_transform(predicted_xgb)[0][0])

        predicted_log = self.model_log.predict(to_predict)[0]
        disease_predicted.append(str(predicted_log))
        predicted_random = self.model_random.predict(to_predict)[0]
        disease_predicted.append(str(predicted_random))
        data = Counter(disease_predicted)
        if len(data) == 0:
            return "none"
        most_common = data.most_common(1)
        if len(most_common) == 0:
            return None

        insight = "\n XGBOOST Predicted : " + str(disease_predicted[0]) + "\n"
        insight += "Logistic Regression Predicted : " + str(disease_predicted[1]) + "\n"
        insight += "Random Classifier Predicted : " + str(disease_predicted[2]) + "\n"
        print(insight)
        return insight,most_common[0][0]

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
                disease_severity = self.df_severity.loc[self.df_severity['Symptom'] == str(word), 'weight'].values[0]
                label.append(disease_severity)
            else :
                label.append(0)

        return label

    def predict_my_disease(self,message):


        extracted_feature = self.feature_label(message,self.symptoms_list)
        to_predict = extracted_feature
        to_predict = np.expand_dims(to_predict, axis=0)
        insight,predicted = self.vote_of_majority(to_predict)

        if predicted == 'none' :
            return("Can you give me more information")
        else :
            return_message = insight + "\n"
            disease = predicted
            return_message += "There is a possibility that you might have " + str(disease)
            return_message += "\n Disease Description"
            disease_row = self.df_description[self.df_description['Disease'] == disease]
            description = disease_row['Description'].values[0]
            return_message += "\n"+ str(description) +"\n"
            return_message += "\n Disease Precaution \n"
            disease_row = self.df_precaution[self.df_precaution['Disease'] == disease]
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
                self.symptoms = self.find_similar_words(response_text,self.symptoms_list)
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
                self.after(10, lambda: display_letter_by_letter(index + 1))
            else:
                self.chat_log.see(tk.END)

        self.chat_log.tag_config("tag_left", justify="left", lmargin1=20, lmargin2=20, wrap="word")
        self.chat_log.tag_config("tag_right", justify="right", rmargin=20)
        self.chat_log.insert(tk.END, "\nChatbot: ", "tag_left")
        display_letter_by_letter()
        self.flag = True

class HeartDiseasePrediction:
    def __init__(self, root,heart_model_path,heart_scaler_path):
        self.root = root
        self.root.title("Heart Disease Prediction")
        with open(heart_model_path, 'rb') as file:
            self.model = pickle.load(file)
        with open(heart_scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        # create a frame for the input fields
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=20)


        self.age_label = tk.Label(self.input_frame, text="Age:")
        self.age_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.age_scale = tk.Scale(self.input_frame, from_=1, to=120, orient="horizontal", length=200)
        self.age_scale.grid(row=0, column=1, padx=10, pady=5)

        self.sex_label = tk.Label(self.input_frame, text="Sex:")
        self.sex_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.sex_entry = ttk.Combobox(self.input_frame, values=["Male", "Female"])
        self.sex_entry.grid(row=1, column=1, padx=10, pady=5)

        self.cp_label = tk.Label(self.input_frame, text="Chest Pain Type:")
        self.cp_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.cp_entry = ttk.Combobox(self.input_frame, values=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        self.cp_entry.grid(row=2, column=1, padx=10, pady=5)

        self.bp_label = tk.Label(self.input_frame, text="Blood Pressure:")
        self.bp_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.bp_scale = tk.Scale(self.input_frame, from_=0, to=200, orient="horizontal", length=200)
        self.bp_scale.grid(row=3, column=1, padx=10, pady=5)

        self.chol_label = tk.Label(self.input_frame, text="Serum Cholesterol:")
        self.chol_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.chol_scale = tk.Scale(self.input_frame, from_=100, to=600, orient="horizontal", length=200)
        self.chol_scale.grid(row=4, column=1, padx=10, pady=5)

        self.fbs_label = tk.Label(self.input_frame, text="Fasting Blood Sugar:")
        self.fbs_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.fbs_entry = ttk.Combobox(self.input_frame, values=["<= 120 mg/dl", "> 120 mg/dl"])
        self.fbs_entry.grid(row=5, column=1, padx=10, pady=5)

        self.restecg_label = tk.Label(self.input_frame, text="Resting ECG Results:")
        self.restecg_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.restecg_entry = ttk.Combobox(self.input_frame, values=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        self.restecg_entry.grid(row=6, column=1, padx=10, pady=5)

        self.thalach_label = tk.Label(self.input_frame, text="Maximum Heart Rate Achieved:")
        self.thalach_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.thalach_scale = tk.Scale(self.input_frame, from_=0, to=200, orient="horizontal", length=200)
        self.thalach_scale.grid(row=7, column=1, padx=10, pady=5)

        self.exang_label = tk.Label(self.input_frame, text="Exercise Induced Angina:")
        self.exang_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.exang_entry = ttk.Combobox(self.input_frame, values=["No", "Yes"])
        self.exang_entry.grid(row=8, column=1, padx=10, pady=5)

        self.oldpeak_label = tk.Label(self.input_frame, text="ST Depression:")
        self.oldpeak_label.grid(row=9, column=0, padx=10, pady=5, sticky="w")
        self.oldpeak_scale = tk.Scale(self.input_frame, from_=0, to=10, orient="horizontal", length=150, resolution=0.1)
        self.oldpeak_scale.grid(row=9, column=1, padx=10, pady=5)

        self.slope_label = tk.Label(self.input_frame, text="Slope:")
        self.slope_label.grid(row=10, column=0, padx=10, pady=5, sticky="w")
        self.slope_entry = ttk.Combobox(self.input_frame, values=["Upsloping", "Flat", "Downsloping"])
        self.slope_entry.grid(row=10, column=1, padx=10, pady=5)

        self.ca_label = tk.Label(self.input_frame, text="Number of Major Vessels:")
        self.ca_label.grid(row=11, column=0, padx=10, pady=5, sticky="w")
        self.ca_entry = ttk.Combobox(self.input_frame, values=["0", "1", "2", "3"])
        self.ca_entry.grid(row=11, column=1, padx=10, pady=5)

        self.thal_label = tk.Label(self.input_frame, text="Thalassemia:")
        self.thal_label.grid(row=12, column=0, padx=10, pady=5, sticky="w")
        self.thal_entry = ttk.Combobox(self.input_frame, values=["Normal Thal", "Fixed Defect", "Reversable Defect"])
        self.thal_entry.grid(row=12, column=1, padx=10, pady=5)


        # create a submit button
        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit_form)
        self.submit_button.pack(pady=10)

        # create a label for the form submission result
        self.result_label = tk.Label(self.root)
        self.result_label.pack()

    # create a function to handle form submission
    def submit_form(self):
        # get the values from the input fields
        age = self.age_scale.get()
        sex = self.sex_entry.get()
        cp = self.cp_entry.get()
        bp = self.bp_scale.get()
        chol = self.chol_scale.get()
        fbs = self.fbs_entry.get()
        restecg = self.restecg_entry.get()
        thalach = self.thalach_scale.get()
        exang = self.exang_entry.get()
        oldpeak = self.oldpeak_scale.get()
        slope = self.slope_entry.get()
        ca = self.ca_entry.get()
        thal = self.thal_entry.get()

        # create a list with the input data
        input_data = [age, sex, cp, bp, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        print(input_data)

        def label_to_value(input):
            label_to_value = {
                        'Male': 1,
                        'Female': 0,
                        'Typical Angina': 0,
                        'Atypical Angina': 1,
                        'Non-Anginal Pain': 2,
                        'Asymptomatic': 3,
                        'Normal': 0,
                        'ST-T Wave Abnormality': 1,
                        'Left Ventricular Hypertrophy': 2,
                        'Yes': 1,
                        'No': 0,
                        'Upsloping': 0,
                        'Flat': 1,
                        'Downsloping': 2,
                        'Normal Thal': 0,
                        'Fixed Defect': 1,
                        'Reversable Defect': 2,
                        '<= 120 mg/dl' : 0,
                        '> 120 mg/dl' : 1,
                        '1' : 1,
                        '2' : 2,
                        '0' : 0,
                        '' : 0
                    }

            temp_data = []
            for i in input:
                if isinstance(i, str):
                    temp_data.append(label_to_value[i])
                else :
                    temp_data.append(i)
            return temp_data

        input_data = label_to_value(input_data)
        print(input_data)

        # clear the input fields
        self.age_scale.set(0)
        self.sex_entry.set("")
        self.cp_entry.set("")
        self.bp_scale.set(0)
        self.chol_scale.set(0)
        self.fbs_entry.set("")
        self.restecg_entry.set("")
        self.thalach_scale.set(0)
        self.exang_entry.set("")
        self.oldpeak_scale.set(0)
        self.slope_entry.set("")
        self.ca_entry.set("")
        self.thal_entry.set("")

        # show a message that the form was submitted
        # input_data = pd.DataFrame({'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}, index=[0])
        input_scaled = self.scaler.transform([input_data])
        # Use the trained model to predict the presence of heart disease for the input data
        prediction = self.model.predict(input_scaled)

        if prediction[0] == 0:
            self.result_label.config(text="The model predicts no presence of heart disease.")

        else:
            self.result_label.config(text="The model predicts presence of heart disease.")

class cancer_detection:
    def __init__(self, root,cancer_model_path,cancer_scaler_path):
        self.root = root
        self.root.title("Breast Cancer Prediction")
        with open(cancer_model_path, 'rb') as file:
            self.model = pickle.load(file)
        with open(cancer_scaler_path, 'rb') as file:
            self.scaler = pickle.load(file)

        # create a frame for the input fields
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=20)

        # concave_points_worst
        self.cp_worst_label = tk.Label(self.input_frame, text="concave_points_worst:")
        self.cp_worst_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.cp_worst_entry = ttk.Entry(self.input_frame)
        self.cp_worst_entry.grid(row=0, column=1, padx=10, pady=5)

        # perimeter_worst
        self.p_worst_label = tk.Label(self.input_frame, text="perimeter_worst:")
        self.p_worst_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.p_worst_entry = ttk.Entry(self.input_frame)
        self.p_worst_entry.grid(row=1, column=1, padx=10, pady=5)

        # concave_points_mean
        self.cp_mean_label = tk.Label(self.input_frame, text="concave_points_mean:")
        self.cp_mean_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.cp_mean_entry = ttk.Entry(self.input_frame)
        self.cp_mean_entry.grid(row=2, column=1, padx=10, pady=5)

        # radius_worst
        self.r_worst_label = tk.Label(self.input_frame, text="radius_worst:")
        self.r_worst_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.r_worst_entry = ttk.Entry(self.input_frame)
        self.r_worst_entry.grid(row=3, column=1, padx=10, pady=5)

        # perimeter_mean
        self.p_mean_label = tk.Label(self.input_frame, text="perimeter_mean:")
        self.p_mean_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        self.p_mean_entry = ttk.Entry(self.input_frame)
        self.p_mean_entry.grid(row=4, column=1, padx=10, pady=5)

        # radius_mean
        self.r_mean_label = tk.Label(self.input_frame, text="radius_mean:")
        self.r_mean_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.r_mean_entry = ttk.Entry(self.input_frame)
        self.r_mean_entry.grid(row=5, column=1, padx=10, pady=5)

        # area_mean
        self.area_mean_label = tk.Label(self.input_frame, text="area_mean:")
        self.area_mean_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
        self.area_mean_entry = ttk.Entry(self.input_frame)
        self.area_mean_entry.grid(row=6, column=1, padx=10, pady=5)

        # concavity_mean
        self.concavity_mean_label = tk.Label(self.input_frame, text="concavity_mean:")
        self.concavity_mean_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
        self.concavity_mean_entry = ttk.Entry(self.input_frame)
        self.concavity_mean_entry.grid(row=7, column=1, padx=10, pady=5)


        # concavity_worst
        self.concavity_worst_label = tk.Label(self.input_frame, text="concavity_worst:")
        self.concavity_worst_label.grid(row=0, column=3, padx=10, pady=5, sticky="w")
        self.concavity_worst_entry = ttk.Entry(self.input_frame)
        self.concavity_worst_entry.grid(row=0, column=4, padx=10, pady=5)

        # compactness_mean
        self.compactness_mean_label = tk.Label(self.input_frame, text="compactness_mean:")
        self.compactness_mean_label.grid(row=1, column=3, padx=10, pady=5, sticky="w")
        self.compactness_mean_entry = ttk.Entry(self.input_frame)
        self.compactness_mean_entry.grid(row=1, column=4, padx=10, pady=5)

        # compactness_worst
        self.compactness_worst_label = tk.Label(self.input_frame, text="compactness_worst:")
        self.compactness_worst_label.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        self.compactness_worst_entry = ttk.Entry(self.input_frame)
        self.compactness_worst_entry.grid(row=2, column=4, padx=10, pady=5)

		#radius_se
        self.radius_se_label = tk.Label(self.input_frame, text="Radius_se:")
        self.radius_se_label.grid(row=3, column=3, padx=10, pady=5, sticky="w")
        self.radius_se_entry = ttk.Entry(self.input_frame)
        self.radius_se_entry.grid(row=3, column=4, padx=10, pady=5)

        #perimeter_se
        self.perimeter_se_label = tk.Label(self.input_frame, text="Perimeter_se:")
        self.perimeter_se_label.grid(row=4, column=3, padx=10, pady=5, sticky="w")
        self.perimeter_se_entry = ttk.Entry(self.input_frame)
        self.perimeter_se_entry.grid(row=4, column=4, padx=10, pady=5)

        #area_se
        self.area_se_label = tk.Label(self.input_frame, text="Area_se:")
        self.area_se_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
        self.area_se_entry = ttk.Entry(self.input_frame)
        self.area_se_entry.grid(row=8, column=1, padx=10, pady=5)

        #texture_worst
        self.texture_worst_label = tk.Label(self.input_frame, text="Texture_worst:")
        self.texture_worst_label.grid(row=5, column=3, padx=10, pady=5, sticky="w")
        self.texture_worst_entry = ttk.Entry(self.input_frame)
        self.texture_worst_entry.grid(row=5, column=4, padx=10, pady=5)

        #smoothness_worst
        self.smoothness_worst_label = tk.Label(self.input_frame, text="Smoothness_worst:")
        self.smoothness_worst_label.grid(row=6, column=3, padx=10, pady=5, sticky="w")
        self.smoothness_worst_entry = ttk.Entry(self.input_frame)
        self.smoothness_worst_entry.grid(row=6, column=4, padx=10, pady=5)

        #symmetry_worst
        self.symmetry_worst_label = tk.Label(self.input_frame, text="Symmetry_worst:")
        self.symmetry_worst_label.grid(row=7, column=3, padx=10, pady=5, sticky="w")
        self.symmetry_worst_entry = ttk.Entry(self.input_frame)
        self.symmetry_worst_entry.grid(row=7, column=4, padx=10, pady=5)


        # create a submit button
        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit_form)
        self.submit_button.pack(pady=10)

        # create a label for the form submission result
        self.result_label = tk.Label(self.root)
        self.result_label.pack()

    # create a function to handle form submission
    def submit_form(self):
        # get the values from the input fields
        concave_points_worst = self.cp_worst_entry.get()
        perimeter_worst = self.p_worst_entry.get()
        concave_points_mean = self.cp_mean_entry.get()
        radius_worst = self.r_worst_entry.get()
        perimeter_mean = self.p_mean_entry.get()
        radius_mean = self.r_mean_entry.get()
        area_mean = self.area_mean_entry.get()
        concavity_mean = self.concavity_mean_entry.get()
        concavity_worst = self.concavity_worst_entry.get()
        compactness_mean = self.compactness_mean_entry.get()
        compactness_worst = self.compactness_worst_entry.get()
        radius_se = self.radius_se_entry.get()
        perimeter_se = self.perimeter_se_entry.get()
        area_se = self.area_se_entry.get()
        texture_worst = self.texture_worst_entry.get()
        smoothness_worst = self.smoothness_worst_entry.get()
        symmetry_worst = self.symmetry_worst_entry.get()

        # create a list with the input data
        input_data = [concave_points_worst, perimeter_worst, concave_points_mean, radius_worst, perimeter_mean, radius_mean, area_mean,
        concavity_mean, concavity_worst, compactness_mean, compactness_worst, radius_se, perimeter_se, area_se, texture_worst,
        smoothness_worst, symmetry_worst]
        print(input_data)
        tempo = []
        for i in input_data:
            if i == '':
                tempo.append(0)
            else :
                tempo.append(float(i))
        input_data = tempo
        print(input_data)

        # clear the input fields
        self.cp_worst_entry.delete(0, tk.END)
        self.p_worst_entry.delete(0, tk.END)
        self.cp_mean_entry.delete(0, tk.END)
        self.r_worst_entry.delete(0, tk.END)
        self.p_mean_entry.delete(0, tk.END)
        self.r_mean_entry.delete(0, tk.END)
        self.area_mean_entry.delete(0, tk.END)
        self.concavity_mean_entry.delete(0, tk.END)
        self.concavity_worst_entry.delete(0, tk.END)
        self.compactness_mean_entry.delete(0, tk.END)
        self.compactness_worst_entry.delete(0, tk.END)
        self.radius_se_entry.delete(0, tk.END)
        self.perimeter_se_entry.delete(0, tk.END)
        self.area_se_entry.delete(0, tk.END)
        self.texture_worst_entry.delete(0, tk.END)
        self.smoothness_worst_entry.delete(0, tk.END)
        self.symmetry_worst_entry.delete(0, tk.END)


        input_scaled = self.scaler.transform([input_data])
        # Use the trained model to predict the presence of heart disease for the input data
        prediction = self.model.predict(input_scaled)

        if prediction[0] == 0:
            self.result_label.config(text="The model predicts no presence of breast cancer.")

        else:
            self.result_label.config(text="The model predicts presence of breast cancer.")

class MyApp:
    def __init__(self, root,heart_model,heart_scaler,cancer_model ,cancer_scaler,disease_description_path
    ,disease_precaution_path,disease_severity_path,disease_data_path,chatbot_symptom_path,
    chatbot_xgb_path,chatbot_logistic_path,chatbot_rfc_path,chatbot_encoder_path):
        self.root = root
        self.heart_model_path = heart_model
        self.heart_scaler_path = heart_scaler
        self.cancer_model_path = cancer_model
        self.cancer_scaler_path = cancer_scaler
        self.disease_description_path = disease_description_path
        self.disease_precaution_path = disease_precaution_path
        self.disease_severity_path = disease_severity_path
        self.disease_data_path = disease_data_path
        self.chatbot_symptom_path = chatbot_symptom_path
        self.xgb_path = chatbot_xgb_path
        self.logistic_path = chatbot_logistic_path
        self.random_path = chatbot_rfc_path
        self.encoder_path = chatbot_encoder_path

        root.configure(background="#191c23")
        self.root.geometry("640x480")

        # create a frame for the options
        self.options_frame = tk.Frame(self.root)
        self.options_frame.pack(side=tk.TOP, pady=50)
        self.options_frame.configure(background="#191c23")

        # create the option buttons
        self.image1 = tk.PhotoImage(file="graphics/Heart1.png").subsample(2)
        self.heart_btn = tk.Button(self.options_frame,  image=self.image1 , borderwidth=0, highlightthickness=0)
        self.heart_btn.pack(side=tk.LEFT, padx=20)
        self.heart_btn.bind("<Enter>", lambda event: self.highlight_btn(event, self.heart_btn))
        self.heart_btn.bind("<Leave>", lambda event: self.unhighlight_btn(event, self.heart_btn))
        self.heart_btn.bind("<Button-1>", lambda event: self.show_page(event, "Heart Page"))


        self.image2 = tk.PhotoImage(file="graphics/breast_cancer1.png").subsample(2)
        self.cancer_btn = tk.Button(self.options_frame,  image=self.image2 , borderwidth=0, highlightthickness=0)
        self.cancer_btn.pack(side=tk.LEFT, padx=20)
        self.cancer_btn.bind("<Enter>", lambda event: self.highlight_btn(event, self.cancer_btn))
        self.cancer_btn.bind("<Leave>", lambda event: self.unhighlight_btn(event, self.cancer_btn))
        self.cancer_btn.bind("<Button-1>", lambda event: self.show_page(event, "Breast Cancer Page"))

        self.image3 = tk.PhotoImage(file="graphics/chatbot1.png").subsample(2)
        self.chatbot_btn = tk.Button(self.options_frame,  image=self.image3 , borderwidth=0, highlightthickness=0)
        self.chatbot_btn.pack(side=tk.LEFT, padx=20)
        self.chatbot_btn.bind("<Enter>", lambda event: self.highlight_btn(event, self.chatbot_btn))
        self.chatbot_btn.bind("<Leave>", lambda event: self.unhighlight_btn(event, self.chatbot_btn))
        self.chatbot_btn.bind("<Button-1>", lambda event: self.show_page(event, "Chatbot Page"))

        self.image4 = tk.PhotoImage(file="graphics/about.png").subsample(2)
        self.about_btn = tk.Button(self.root,  image=self.image4 , borderwidth=0, highlightthickness=0)
        self.about_btn.pack(side=tk.TOP, pady=2)
        self.about_btn.bind("<Enter>", lambda event: self.highlight_btn(event, self.about_btn))
        self.about_btn.bind("<Leave>", lambda event: self.unhighlight_btn(event, self.about_btn))
        self.about_btn.bind("<Button-1>", lambda event: self.show_page(event, "About Page"))

    def highlight_btn(self, event, btn):
        btn.config(bg="#191c23")

    def unhighlight_btn(self, event, btn):
        btn.config(bg="#191c23")

    def start_heart(self):
        heart_root = tk.Tk()
        app = HeartDiseasePrediction(heart_root,self.heart_model_path,self.heart_scaler_path )
        heart_root.mainloop()

    def start_cancer(self):
        cancer_root = tk.Tk()
        app = cancer_detection(cancer_root,self.cancer_model_path,self.cancer_scaler_path )
        cancer_root.mainloop()

    def start_chatbot(self):
        app = ChatbotApp(self.xgb_path,self.logistic_path,self.random_path,self.encoder_path,self.disease_description_path
        ,self.disease_precaution_path,self.disease_severity_path,self.disease_data_path,self.chatbot_symptom_path,)
        app.mainloop()

    def show_page(self, event, title):
        if title == "Heart Page" :
            self.start_heart()

        if title == "Breast Cancer Page" :
            self.start_cancer()

        if title == "Chatbot Page" :
            self.start_chatbot()

        if title == "About Page" :
            page = tk.Toplevel(self.root)
            page.title(title)
            page.geometry("500x500")
            label = tk.Label(page, text="Welcome to Disease prediction APP \n This is created for \n Fusemachine Semester Project  \n Made by \n Swodesh Sharma \n Krisbin Poudel \n Nadika Poudel \n Krishant Timilsina ", font=("", 20))
            label.pack(pady=20)


if __name__ == "__main__":
    heart_model_path = 'models/heart disease/model.pkl'
    heart_scaler_path = 'models/heart disease/scaler.pkl'
    cancer_model_path = 'models/breast cancer/model.pkl'
    cancer_scaler_path = 'models/breast cancer/scaler.pkl'
    disease_description_path = 'models/chatbot/data/symptom_Description.csv'
    disease_precaution_path = 'models/chatbot/data/symptom_precaution.csv'
    disease_severity_path =  'models/chatbot/data/Symptom-severity.csv'
    disease_data_path = 'models/chatbot/data/final_dataset.csv'
    chatbot_symptom_path =  'models/chatbot/symptom_list.pickle.dat'
    chatbot_xgb_path =  'models/chatbot/xgb_classifier.pickle.dat'
    chatbot_logistic_path =  'models/chatbot/log_reg.pickle.dat'
    chatbot_rfc_path = 'models/chatbot/random_clf.pickle.dat'
    chatbot_encoder_path = 'models/chatbot/encoder.pickle.dat'

    root = tk.Tk()
    app = MyApp(root,heart_model_path,heart_scaler_path,cancer_model_path
    ,cancer_scaler_path,disease_description_path,disease_precaution_path,
    disease_severity_path,disease_data_path,chatbot_symptom_path,
    chatbot_xgb_path,chatbot_logistic_path,chatbot_rfc_path,chatbot_encoder_path)
    root.mainloop()
