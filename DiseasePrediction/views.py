from django.http import HttpResponse
from django.shortcuts import render
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def get_disease_data(name,algo,disease):
    df1=pd.read_csv('./dataset/symptom_Description.csv')
    df2=pd.read_csv('./dataset/symptom_precaution.csv')
    disc=df1[df1["Disease"]==disease]
    pre=df2[df2["Disease"]==disease]
    data={
            "Name":name,
            "algo":algo,
            "prediction":disease,
           "discription":disc["Description"].values[0],
            "pre1":pre["Precaution_1"].values[0],
            "pre2":pre["Precaution_2"].values[0],
            "pre3":pre["Precaution_3"].values[0],
            "pre4":pre["Precaution_4"].values[0],
          }
    return data
def input_to_model(user_list):
    sympt=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    symptoms = []
    for i in sympt:
        if i in user_list:
            symptoms.append(1)
        else:
            symptoms.append(0)
    return symptoms
def algo_predict(algo,choice_input):
    data=pd.read_csv('./dataset/training.csv')
    X = data.drop('prognosis', axis=1)  # Assuming 'prognosis' is the target variable
    y = data['prognosis']
    if algo=='DecisionTree':
        DTmodel=DecisionTreeClassifier()
        DTmodel.fit(X, y)
        pred=DTmodel.predict([choice_input])
        return pred
    elif algo=='RandomForest':
        RFmodel=RandomForestClassifier(n_estimators=200)
        RFmodel.fit(X, y)
        pred=RFmodel.predict([choice_input])
        return pred
    elif algo=='KNN':
        KNNmodel=KNeighborsClassifier(n_neighbors=3)
        KNNmodel.fit(X, y)
        pred=KNNmodel.predict([choice_input])
        return pred
    elif algo=='SVM':
        SVMmodel=SVC()
        SVMmodel.fit(X, y)
        pred=SVMmodel.predict([choice_input])
        return pred
    elif algo=='NaiveBayes':
        NBmodel=GaussianNB()
        NBmodel.fit(X, y)
        pred=NBmodel.predict([choice_input])
        return pred
    return None

def algouse(algo):
   
    if algo=='DecisionTree':
        # DTMODEL=modelAlgos()
        DTamodel=joblib.load('./trained_models/DTmodel.pkl')
        return DTamodel
    elif algo=='RandomForest':
        RFmodel=joblib.load('./trained_models/RFmodel.pkl')
        return RFmodel
    elif algo=='KNN':
        KNNmodel=joblib.load('./trained_models/KNNmodel.pkl')
        return KNNmodel
    elif algo=='SVM':
        SVMmodel=joblib.load('./trained_models/SVMmodel.pkl')
        return SVMmodel
    elif algo=='NaiveBayes':
        NBmodel=joblib.load('./trained_models/NBmodel.pkl')
        return NBmodel
# DTamodel=joblib.load('./trained_models/DTmodel.pkl')
def index(request):
    if request.method=='POST':
        name = request.POST.get('Name')
        choice1=request.POST.get('symptom1')
        choice2=request.POST.get('symptom2')
        choice3=request.POST.get('symptom3')
        choice4=request.POST.get('symptom4')
        choice5=request.POST.get('symptom5')
        algo= request.POST.get('algo')
        choice=[choice1,choice2,choice3,choice4,choice5]
        choice_input=input_to_model(choice)
        # model=algouse(algo)
        # prediction=model.predict([choice_input])
        # prediction=algouse(algo)
        # prediction=DTamodel.predict([choice_input])
        # print(name,choice,algo,prediction[0])
        prediction=algo_predict(algo,choice_input)
        dict_resut=get_disease_data(name,algo,prediction[0])


        # dict_resut={'Name':name,'algo':algo,'prediction':prediction[0]}
        return render(request, 'report.html',dict_resut)

    sympt=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

    return render(request, 'index.html',{'symptoms':sympt})

def report(request):
    return render(request, 'report.html')
