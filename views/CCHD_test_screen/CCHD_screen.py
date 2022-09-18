from kivymd.uix.screen import MDScreen
import time
from kivy.metrics import dp
from kivymd.uix.datatables import MDDataTable
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.widget import Widget
from kivymd.uix.button import MDRaisedButton

from threading import Thread

import os
import shutil
import pandas as pd
from pandas.core.frame import DataFrame
import statistics
import datetime
import time

from scipy.integrate import simps
from scipy import integrate
from scipy.signal import find_peaks
import matplotlib.pyplot as plt  
from dtaidistance import dtw

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, label_binarize
import matplotlib.pyplot as plt 
from sklearn.datasets import make_classification
import joblib
import configparser

class CCHDProgress(MDScreen):
    result_output = {}
    extracted_csv = 'ExtractedFeature.csv'
    CCHD_csv = 'CCHD_result.csv'
    extracted_features_path = os.environ["ROOT_DIR"]

    def start_test(self):

        t = Thread(target=self.cchd_exam_thread, args=(123,))
        t.daemon = True
        self.data_tables = MDDataTable(
            size_hint=(0.9, 0.4),
            pos_hint={"center_x": 0.5},
            column_data=[
                ("Case Name", dp(30)),
                ("Result", dp(30)),
                ("Extration time", dp(30)),
                ("CCHD time", dp(30)),
                ("Test Time", dp(30)),
            ],
            row_data=[
                ("No case", "waiting",
                 "waiting", "waiting", "waiting")
            ],
        )
        self.add_widget(self.data_tables)
        t.start()
        
    def go_back(self):
        if self.isfinished:
            self.parent.current = "start_screen"
        pass
        
    def cchd_exam_thread(self, a):
        self.isfinished = False
        self.ids.progress_determinate.color = 0, 0.9, 0.1, 1
        self.ids.progress_determinate.value = 25
        self.ids.cchd_status.text = "Reading Config Files"
        self.ids.backhome.text = "Running please wait"
        

        # read config from [root]/Config/config.ini
        config = configparser.ConfigParser()
        config.read(os.path.join(os.environ["ROOT_DIR"], 'Config', 'config.ini'))

        self.AWAD_model = os.path.join(os.environ["ROOT_DIR"], 'Config', config['AWAD_model']['model_name'])
        AWAD_features = config['AWAD_model']['model_features'].split(',')

        self.CCHD_model = os.path.join(os.environ["ROOT_DIR"], 'Config', config['CCHD_model']['model_name'])
        CCHD_features = config['CCHD_model']['model_features'].split(',') 

        self.AWAD_features_columes = self.AWAD_feature_select(AWAD_features)
        self.CCHD_features_columes = self.CCHD_feature_select(CCHD_features)

        self.path_global = os.path.join(os.environ["DATA_DIR"], 'tmp', 'Recorded')
        self.preprocessed_path_global = os.path.join(os.environ["DATA_DIR"], 'tmp', 'Preprocessed')
        self.tested_path = os.path.join(os.environ["DATA_DIR"], 'Tested')

        self.ids.progress_determinate.value = 30
        self.ids.cchd_status.text = "Analyzing cases..."

        self.file_into_folder(self.path_global, self.preprocessed_path_global)
        cases_list = sorted(os.listdir(self.preprocessed_path_global))
        print("There are ", len(cases_list), " cases")

        many_files_cases = []
        noABC_cases = []
        ABC_cases = []
        for case in cases_list:
            file_list = sorted(os.listdir(os.path.join(self.preprocessed_path_global, case)))
            if len(file_list) == 6:
                ABC_cases.append(case)
            elif len(file_list) > 6:
                many_files_cases.append(case)
            else:
                noABC_cases.append(case)
                
        print("There are ", len(ABC_cases), "with well-set ABCD takes")
        print("There are ", len(noABC_cases)+len(many_files_cases), "with different file setup. With ", len(many_files_cases), "being cases with too many files")

        self.ids.progress_determinate.value = 50

        cases_to_exclude = self.clean_folder(self.preprocessed_path_global)
        self.extract_all_features(self.preprocessed_path_global, self.extracted_features_path ,cases_to_exclude)

        self.ids.progress_determinate.value = 90
        self.ids.cchd_status.text = "Runing ML Detecting..."

        Complete_Features = os.path.join(self.extracted_features_path, self.extracted_csv)
        evaled_ds = self.ml_cchd_detection(Complete_Features)
        print(self.result_output)

        res_out = pd.DataFrame(self.result_output, index=[0])
        if not os.path.exists(os.path.join(self.extracted_features_path, self.CCHD_csv)):
            res_out.to_csv(os.path.join(self.extracted_features_path, self.CCHD_csv))
        else:
            res_out.to_csv(os.path.join(self.extracted_features_path, self.CCHD_csv), mode='a', header=False)

        self.clean_procssed_folder()        

        self.ids.progress_determinate.color = 0, 0.9, 0.1, 1
        self.ids.progress_determinate.value = 100
        self.ids.cchd_status.text = "Finished..."
        self.ids.spinner.active = False

        self.ids.backhome.text = "Go Back Home"
        self.isfinished = True

        self.data_tables.row_data = [(self.result_output['case_name'], self.result_output['result'], self.result_output['extract_time']
        ,str(self.result_output['cchd_running_time']), str(self.result_output['test_time']))]

        

    def take_division(self, fp,folder_path,filename,isPleth):
        if isPleth:
            col_names = ['id','dp1','dp2','dp3','dp4','dp5','dp6','dp7','dp8','dp9','dp10','dp11','dp12','dp13','dp14','dp15'
            ,'dp16','dp17','dp18','dp19','dp20','dp21','dp22','dp23','dp24','dp25','counter','date','takes']
        else:
            col_names = ['bytes','status','voltage','pai','counter','spo2','hr','time','takes']
        df = pd.read_csv(fp,names = col_names)
        df.dropna(how="all",inplace=True)
        # Check if there are miliseconds markers
        if isPleth:
            num_of_decimal = len(df.date[0].split('-')[-1].split('.'))
            if (num_of_decimal == 2):
                isMili = True
            else:
                isMili = False
        else:
            isMili = False
        # Proceed to do takes division
        index_0 = 0
        index = 0
        curr_take = str(df.takes[0])
        for take in df.takes:
            if str(take) != str(curr_take):
                if not os.path.exists(os.path.join(folder_path,curr_take)):
                    os.makedirs(os.path.join(folder_path,curr_take))
                print(os.path.join(folder_path,curr_take))
                sliced_df = df[index_0:index]
                sliced_df.to_csv(os.path.join(folder_path, curr_take, filename))
                index_0 = index
                curr_take = str(take)
                onlyOneTake = False
            index += 1
        if not os.path.exists(os.path.join(folder_path,curr_take)):
                os.makedirs(os.path.join(folder_path,curr_take))
        print(os.path.join(folder_path,curr_take))
        sliced_df = df[index_0:index]
        sliced_df.to_csv(os.path.join(folder_path, curr_take, filename))
        
        return isMili

    def clean_folder(self, path):
        col_names = ['id','dp1','dp2','dp3','dp4','dp5','dp6','dp7','dp8','dp9','dp10','dp11','dp12','dp13','dp14','dp15'
            ,'dp16','dp17','dp18','dp19','dp20','dp21','dp22','dp23','dp24','dp25','counter','date','takes']
        cases_list = sorted(os.listdir(path))
        no_Mili = []
        for case in cases_list:
            file_list = sorted(os.listdir(os.path.join(path, case)))
            if len(file_list) == 6:
                print("Processing ", case)
                self.data_tables.row_data = [(case, "waiting", "waiting"
        ,"waiting", "waiting")]
                folder_path = os.path.join(path, case)
                #Hand Pleth
                pleth_hand_path = os.path.join(path, case, file_list[3])
                isMili1 = self.take_division(pleth_hand_path,folder_path,str(case)+'_pleth_hand.csv',True)
                #Foot Pleth
                pleth_foot_path = os.path.join(path, case, file_list[2])
                isMili2 = self.take_division(pleth_foot_path,folder_path,str(case)+'_pleth_foot.csv',True)
                #Hand Pulseox
                pulseox_hand_path = os.path.join(path, case, file_list[5])
                _ = self.take_division(pulseox_hand_path,folder_path,str(case)+'_pulseox_hand.csv',False)
                #Pleth Pulseox
                pulseox_foot_path = os.path.join(path, case, file_list[4])
                _ = self.take_division(pulseox_foot_path,folder_path,str(case)+'_pulseox_foot.csv',False)
                print("Done with case: ", case)
                if isMili1 == False or isMili2 == False:
                    no_Mili.append(str(case))
                    print("Case ", case, "has no miliseconds marker")
            else:
                pass
                print(case," skipped due to not having abcd take division")
        
        return no_Mili    

    def file_into_folder(self, path, new_path):
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        file_list = sorted(os.listdir(path))
        ID = []
        for i in range(len(file_list)):
            case_ID = file_list[i].split('_')[0]
            ID.append(case_ID)
        unique_ID = sorted(list(set(ID)))
        
        for folder in unique_ID:
            new_folder = os.path.join(new_path, folder)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            
            for i in range(len(file_list)):
                if file_list[i].startswith(folder):
                    old_file_path = os.path.join(path, file_list[i])
                    new_file_path = os.path.join(new_folder, file_list[i])
                    shutil.move(old_file_path,new_file_path)

    def AWAD_feature_select(self, incoming_features):
        feature_columes = [False,False,False,False,False,False,False,False,False,False,False,
        False,False,False]
        for feature in incoming_features:
            if feature == "'dialen'":
                feature_columes[0] = True
            elif feature == "'Sys Len'":
                feature_columes[1] = True
            elif feature == "'Total Len'":
                feature_columes[2] = True
            elif feature == "'Sys Ratio'":
                feature_columes[3] = True
            elif feature == "'Dias Ratio'":
                feature_columes[4] = True
            elif feature == "'Sys Slope'":
                feature_columes[5] = True
            elif feature == "'Dias Slope'":
                feature_columes[6] = True
            elif feature == "'Dias/Sys Ratio'":
                feature_columes[7] = True
            elif feature == "'Onset-Amp Ratio'":
                feature_columes[8] = True
            elif feature == "'Mid Sys Val'":
                feature_columes[9] = True
            elif feature == "'Amplitude'":
                feature_columes[10] = True
            elif feature == "'Amplitude Ratio'":
                feature_columes[11] = True
            elif feature == "'DTW Dist'":
                feature_columes[12] = True
            elif feature == "'DTW Avg Dist'":
                feature_columes[13] = True
            else:
                print("No such feature(s)")
        return feature_columes
        
    def CCHD_feature_select(self, incoming_features):

        feature_columes = [False,False,False,False,False,False,False,False,False,False,False,
        False,False,False,False,False,False,False,False,False,False,False,False,False,False,
        False,False,False,False,False,False,False,False,False,False,False,False,False,False,
        False]
        for feature in incoming_features:
            if feature == "'Min_PAI_h'":
                feature_columes[0] = True
            elif feature == "'Max_PAI_h'":
                feature_columes[1] = True
            elif feature == "'Median_PAI_h'":
                feature_columes[2] = True
            elif feature == "'Mean_PAI_h'":
                feature_columes[3] = True
            elif feature == "'Variance_PAI_h'":
                feature_columes[4] = True
            elif feature == "'Min_SPO2_h'":
                feature_columes[5] = True
            elif feature == "'Max_SPO2_h'":
                feature_columes[6] = True
            elif feature == "'Median_SPO2_h'":
                feature_columes[7] = True
            elif feature == "'Mean_SPO2_h'":
                feature_columes[8] = True
            elif feature == "'Variance_SPO2_h'":
                feature_columes[9] = True
            elif feature == "'Min_HR_h'":
                feature_columes[10] = True
            elif feature == "'Max_HR_h'":
                feature_columes[11] = True
            elif feature == "'Median_HR_h'":
                feature_columes[12] = True
            elif feature == "'Mean_HR_h'":
                feature_columes[13] = True
            elif feature == "'Variance_HR_h'":
                feature_columes[14] = True
            elif feature == "'Min_PAI_f'":
                feature_columes[15] = True
            elif feature == "'Max_PAI_f'":
                feature_columes[16] = True
            elif feature == "'Median_PAI_f'":
                feature_columes[17] = True
            elif feature == "'Mean_PAI_f'":
                feature_columes[18] = True
            elif feature == "'Variance_PAI_f'":
                feature_columes[19] = True
            elif feature == "'Min_SPO2_f'":
                feature_columes[20] = True
            elif feature == "'Max_SPO2_f'":
                feature_columes[21] = True
            elif feature == "'Median_SPO2_f'":
                feature_columes[22] = True
            elif feature == "'Mean_SPO2_f'":
                feature_columes[23] = True
            elif feature == "'Variance_SPO2_f'":
                feature_columes[24] = True
            elif feature == "'Min_HR_f'":
                feature_columes[25] = True
            elif feature == "'Max_HR_f'":
                feature_columes[26] = True
            elif feature == "'Median_HR_f'":
                feature_columes[27] = True
            elif feature == "'Mean_HR_f'":
                feature_columes[28] = True
            elif feature == "'Variance_HR_f'":
                feature_columes[29] = True
            elif feature == "'Min_slope'":
                feature_columes[30] = True
            elif feature == "'Max_slope'":
                feature_columes[31] = True
            elif feature == "'Median_slope'":
                feature_columes[32] = True
            elif feature == "'Mean_slope'":
                feature_columes[33] = True
            elif feature == "'Variance_slope'":
                feature_columes[34] = True
            elif feature == "'Min_delay'":
                feature_columes[35] = True
            elif feature == "'Max_delay'":
                feature_columes[36] = True
            elif feature == "'Median_delay'":
                feature_columes[37] = True
            elif feature == "'Mean_delay'":
                feature_columes[38] = True
            elif feature == "'Variance_delay'":
                feature_columes[39] = True
            else:
                print("No such feature(s)")

        return feature_columes    

    def extract_all_features(self,fp,csv_path,cases_to_exclude):
        ## need to add reconstructed beats dict here
        reconstructed_beats = {}
        paths = self.grab_data_files(fp, True)
        ct = 0
        case_ct = 0
        processed = False
        rfe_cols = 0
        for path in paths:
            case_name = path.split(os.sep)[-3]
            take_name = path.split(os.sep)[-2]
            ForH = path.split(os.sep)[-1].split('_')[2][:4]
            if case_name not in cases_to_exclude:
                start_time = time.time()
                print("About to take features for case ", case_name)
                if ForH == 'hand':
                    foot_path = path[:len(path)-8]+'foot.csv'
                    pulseox_hand_path = os.path.dirname(path) + os.sep + path.split(os.sep)[-1].split('_')[0]+'_'+'pulseox'+'_'+path.split(os.sep)[-1].split('_')[2]
                    pulseox_foot_path = os.path.dirname(foot_path) + os.sep + foot_path.split(os.sep)[-1].split('_')[0]+'_'+'pulseox'+'_'+foot_path.split(os.sep)[-1].split('_')[2]
                    if os.path.exists(path) and os.path.exists(foot_path) and os.path.exists(pulseox_hand_path) and os.path.exists(pulseox_foot_path):
                        PAI_f,SPO2_f,HR_f = self.pulseox_features(pulseox_foot_path)
                        PAI_h,SPO2_h,HR_h = self.pulseox_features(pulseox_hand_path)
                        
                        slope, delay, rfe_cols, reconstructed_beats = self.pleth_features(path, foot_path,case_name,take_name,rfe_cols, reconstructed_beats)
                                            
                        print("Took all features for case ", case_name)

                        if (len(slope)>2 and len(delay)>2) and (len(PAI_f)>2 and len(SPO2_f)>2 and len(HR_f)>2) and (len(PAI_h)>2 and len(SPO2_h)>2 and len(HR_h)>2):
                            print("Processing ", case_name, "take: ", take_name)
                            featuresDCT = {'Case': str(case_name+'_'+take_name),
                                        'ID': str(case_name),
                                        'take': str(take_name),
                                        'Min_PAI_h': min(PAI_h),
                                        'Max_PAI_h': max(PAI_h),
                                        'Median_PAI_h': statistics.median(PAI_h),
                                        'Mean_PAI_h': statistics.mean(PAI_h),
                                        'Variance_PAI_h': statistics.variance(PAI_h),
                                        'Min_SPO2_h': min(SPO2_h),
                                        'Max_SPO2_h': max(SPO2_h),
                                        'Median_SPO2_h': statistics.median(SPO2_h),
                                        'Mean_SPO2_h': statistics.mean(SPO2_h),
                                        'Variance_SPO2_h': statistics.variance(SPO2_h),
                                        'Min_HR_h': min(HR_h),
                                        'Max_HR_h': max(HR_h),
                                        'Median_HR_h': statistics.median(HR_h),
                                        'Mean_HR_h': statistics.mean(HR_h),
                                        'Variance_HR_h': statistics.variance(HR_h),
                                        'Min_PAI_f': min(PAI_f),
                                        'Max_PAI_f': max(PAI_f),
                                        'Median_PAI_f': statistics.median(PAI_f),
                                        'Mean_PAI_f': statistics.mean(PAI_f),
                                        'Variance_PAI_f': statistics.variance(PAI_f),
                                        'Min_SPO2_f': min(SPO2_f),
                                        'Max_SPO2_f': max(SPO2_f),
                                        'Median_SPO2_f': statistics.median(SPO2_f),
                                        'Mean_SPO2_f': statistics.mean(SPO2_f),
                                        'Variance_SPO2_f': statistics.variance(SPO2_f),
                                        'Min_HR_f': min(HR_f),
                                        'Max_HR_f': max(HR_f),
                                        'Median_HR_f': statistics.median(HR_f),
                                        'Mean_HR_f': statistics.mean(HR_f),
                                        'Variance_HR_f': statistics.variance(HR_f),
                                        'Min_slope': min(slope),
                                        'Max_slope': max(slope),
                                        'Median_slope': statistics.median(slope),
                                        'Mean_slope': statistics.mean(slope),
                                        'Variance_slope': statistics.variance(slope),
                                        'Min_delay': min(delay),
                                        'Max_delay': max(delay),
                                        'Median_delay': statistics.median(delay),
                                        'Mean_delay': statistics.mean(delay),
                                        'Variance_delay': statistics.variance(delay)}
                            if ct == 0:
                                ct += 1
                                processed = True
                                features = DataFrame(featuresDCT,index=[0],columns=['Case','ID','take',
                                                                                    'Min_PAI_h','Max_PAI_h','Median_PAI_h',
                                                                                'Mean_PAI_h','Variance_PAI_h','Min_SPO2_h',
                                                                                'Max_SPO2_h','Median_SPO2_h','Mean_SPO2_h',
                                                                                'Variance_SPO2_h','Min_HR_h','Max_HR_h',
                                                                                'Median_HR_h','Mean_HR_h','Variance_HR_h',
                                                                                    'Min_PAI_f','Max_PAI_f','Median_PAI_f',
                                                                                'Mean_PAI_f','Variance_PAI_f','Min_SPO2_f',
                                                                                'Max_SPO2_f','Median_SPO2_f','Mean_SPO2_f',
                                                                                'Variance_SPO2_f','Min_HR_f','Max_HR_f',
                                                                                'Median_HR_f','Mean_HR_f','Variance_HR_f',
                                                                                'Min_slope','Max_slope','Median_slope',
                                                                                'Mean_slope','Variance_slope','Min_delay',
                                                                                'Max_delay','Median_delay','Mean_delay',
                                                                                'Variance_delay'])
                            else:
                                processed = True
                                features2 = DataFrame(featuresDCT,index=[0],columns=['Case','ID','take',
                                                                                    'Min_PAI_h','Max_PAI_h','Median_PAI_h',
                                                                                'Mean_PAI_h','Variance_PAI_h','Min_SPO2_h',
                                                                                'Max_SPO2_h','Median_SPO2_h','Mean_SPO2_h',
                                                                                'Variance_SPO2_h','Min_HR_h','Max_HR_h',
                                                                                'Median_HR_h','Mean_HR_h','Variance_HR_h',
                                                                                    'Min_PAI_f','Max_PAI_f','Median_PAI_f',
                                                                                'Mean_PAI_f','Variance_PAI_f','Min_SPO2_f',
                                                                                'Max_SPO2_f','Median_SPO2_f','Mean_SPO2_f',
                                                                                'Variance_SPO2_f','Min_HR_f','Max_HR_f',
                                                                                'Median_HR_f','Mean_HR_f','Variance_HR_f',
                                                                                'Min_slope','Max_slope','Median_slope',
                                                                                'Mean_slope','Variance_slope','Min_delay',
                                                                                'Max_delay','Median_delay','Mean_delay',
                                                                                'Variance_delay'])
                                features = pd.concat([features,features2], ignore_index=True)
            if processed:
                reconstructed_beats = {}
                processed = False
                case_ct += 1
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.result_output['extract_time'] = elapsed_time
                print("Got delay and slope features for: ", case_name, "in take: ", take_name)
                print("Calculated in: ", elapsed_time)
        if not os.path.exists(os.path.join(csv_path, self.extracted_csv)):
            features.to_csv(os.path.join(csv_path, self.extracted_csv))
        else:
            self.ids.cchd_status.text = "Error: did not find cases..."
            self.ids.spinner.active = False
            self.ids.backhome.text = "Go Back Home"
            self.ids.progress_determinate.color = 1, 0, 0, 1
            self.isfinished = True
            features.to_csv(os.path.join(csv_path, self.extracted_csv), mode='a', header=False)

    
    def grab_data_files(self, path, isPleth):
        if isPleth:
            name_search = 'pleth'
        else:
            name_search = 'pulseox'
            
        to_extract = []
        for case in sorted(os.listdir(path)):
            case_path = os.path.join(path, case)
            if len(os.listdir(case_path)) > 6:
                for take in sorted(os.listdir(case_path)):
                    take_path = os.path.join(case_path, take)
                    if os.path.isdir(take_path):
                        for file in sorted(os.listdir(take_path)):
                            if file.split('_')[1] == name_search:
                                new_path = os.path.join(take_path, file)
                                to_extract.append(new_path)
        return to_extract

    def pulseox_features(self, fp):
        df = pd.read_csv(fp)
        df.drop(columns='Unnamed: 0',inplace=True)
        
        all_PAI = df['pai'].values
        all_SPO2 = df['spo2'].values
        all_HR = df['hr'].values
        
        all_PAI = all_PAI[all_PAI < 2001]
        all_SPO2 = all_SPO2[all_SPO2 < 101]
        all_HR = all_HR[all_HR < 251]
        all_HR = all_HR[all_HR > 0]
        
        return all_PAI, all_SPO2, all_HR

    def change_df(self, path):
        df = pd.read_csv(path)
        df.drop(columns='Unnamed: 0',inplace=True)
        time = []
        values = []
        #increment = 1/75
        dp1 = df['dp1'].tolist()
        dp2 = df['dp2'].tolist()
        dp3 = df['dp3'].tolist()
        dp4 = df['dp4'].tolist()
        dp5 = df['dp5'].tolist()
        dp6 = df['dp6'].tolist()
        dp7 = df['dp7'].tolist()
        dp8 = df['dp8'].tolist()
        dp9 = df['dp9'].tolist()
        dp10 = df['dp10'].tolist()
        dp11 = df['dp11'].tolist()
        dp12 = df['dp12'].tolist()
        dp13 = df['dp13'].tolist()
        dp14 = df['dp14'].tolist()
        dp15 = df['dp15'].tolist()
        dp16 = df['dp16'].tolist()
        dp17 = df['dp17'].tolist()
        dp18 = df['dp18'].tolist()
        dp19 = df['dp19'].tolist()
        dp20 = df['dp20'].tolist()
        dp21 = df['dp21'].tolist()
        dp22 = df['dp22'].tolist()
        dp23 = df['dp23'].tolist()
        dp24 = df['dp24'].tolist()
        dp25 = df['dp25'].tolist()
        datapoints = [dp1,dp2,dp3,dp4,dp5,dp6,dp7,dp8,dp9,dp10,dp11,dp12,dp13,dp14,dp15,dp16,dp17,dp18,dp19,dp20,dp21,dp22,dp23,dp24,dp25]
        times = df['date'].tolist()
        time_increment = datetime.timedelta(microseconds=13333)
        for i in range(len(df)):
            ct = 0
            for datapoint in datapoints:
                if ct == 0:
                    time_str = times[i]
                    curr_time = datetime.datetime.strptime(time_str, '%Y-%m-%d-%H-%M-%S.%f')
                else:
                    time_str = curr_time.strftime('%Y-%m-%d-%H-%M-%S.%f')
                values.append(datapoint[i])
                time.append(time_str)
                curr_time = curr_time + time_increment
                ct += 1
        line_df = pd.DataFrame(
        {'value':values,
        'time':time
        })

        return line_df

    # %%
    def pad_df(self, df):
        if len(df) == 180:
            df_new = df
        else:    
            df.reset_index(drop=True)
            up_index = int((180 - len(df))/2)
            df.index += up_index
            df_new = df.reindex(range(180),fill_value=0)
        return df_new

    def center_crop(self, df):
        if len(df) == 180:
            df_new = df
        else:
            df.reset_index(drop=True)
            df_1st = df[:90]
            df_2nd = df[len(df)-90:len(df)]
            df_new = pd.concat([df_1st,df_2nd])
            if len(df_new) != 180:
                print("REDO DATASET!!!!!!!")
                print("Length of center cropped is: ", len(df_new))
        return df_new

    def make_df(self,df,artifacts,ID):
        
        
        if len(df) < 180:
            df = self.pad_df(df)
        elif len(df) > 180:
            df = self.center_crop(df)
            artifacts = 3
        elif len(df) == 180:
            pass
        
        x_axis = df['time'].tolist()
        y_axis = df['value'].tolist()
        
        x_axis.insert(0,artifacts)
        x_axis.insert(0,'time')
        x_axis.insert(0,ID)
        
        y_axis.insert(0,artifacts)
        y_axis.insert(0,'pleth')
        y_axis.insert(0,ID)
        
        df_new = pd.DataFrame()
        time_series = pd.Series(x_axis)
        pleth_series = pd.Series(y_axis)
        
        df_new = pd.concat([df_new, time_series])
        df_new = pd.concat([df_new, pleth_series], axis=1)
        df_new = df_new.transpose()
        
        return df_new

    def mount_df(self, beat_sequences):
        df_count = 0
        for key in beat_sequences:
            slice_df = beat_sequences[key].reset_index(drop=True)
            x_axis = slice_df['time'].tolist()
            y_axis = slice_df['value'].tolist()
            if df_count == 0:
                dataset_new = self.make_df(slice_df,'?',key)
                df_count = df_count + 1
            else:
                dataset_partial = self.make_df(slice_df,'?',key)
                dataset_new = pd.concat([dataset_new,dataset_partial])
        dataset_new.rename(columns={0: 'ID', 1: 'isPleth', 2: 'isArtifact'},inplace=True)

        
        return dataset_new


    #must first iterate through df with iloc
    def unpad(self, seq):
        gt = str(seq['isArtifact'])
        ID = seq['ID']
        ForH = seq
        b = seq[3:].tolist()
        off = False
        
        zeros_index = []
        
        for i in range(len(b)):
            if b[i] == 0:
                zeros_index.append(i)
                
        dissonances = 0
        for j in range(len(zeros_index)):
            if j == 0:
                curr = zeros_index[j]
            else:
                old = curr
                curr = zeros_index[j]
                
                if (old - curr > 1):
                    dissonances += 1
        if dissonances > 1:
            off = True
        while 0 in b:
            b.remove(0)

        return b, gt, ID, off

    # %%
    def GrabInfo(self, df):
        
        ft_counter = 0
        
        dias_len = {}
        dias_prev = {}
        sys_len = {}
        sys_prev = {}
        tot_len = {}
        dias_slope = {}
        sys_slope = {}
        dias_sys_ratio = {}
        onset_ratio = {}
        mid_sys_val = {}
        amplitude = {}
        amount_peaks = {}
        amplitude_prev_ratio = {}
        dia_prev_ratio = {}
        sys_prev_ratio = {}
        dtw_distance = {}
        dtw_avg_distance = {}
        
        for a in ['gt?']:
            dias_len[a] = []
            dias_prev[a] = []
            sys_len[a] = []
            sys_prev[a] = []
            tot_len[a] = []
            dias_slope[a] = []
            sys_slope[a] = []
            dias_sys_ratio[a] = []
            onset_ratio[a] = []
            mid_sys_val[a] = []
            amplitude[a] = []
            amount_peaks[a] = []
            amplitude_prev_ratio[a] = []
            dtw_distance[a] = []
            dtw_avg_distance[a] = []
        
        counter = 0
        for seq in df.iloc:
            #makes sure it is a pleth row and not a time row
            if seq['isPleth'] == 'time':
                time_vec = seq
            if seq['isPleth'] == 'pleth':
                if counter == 0:
                    bn, gt, ID, off = self.unpad(seq)
                    counter = 1
                elif counter != 0:
                    bn_1 = bn
                    ID_1 = ID
                    bn, gt, ID, off = self.unpad(seq)
                    
                    if gt == '?':
                        key = 'gt?'
                    
                    #grabs peaks
                    peaks, _ = find_peaks(bn)
                    peaks_1, _ = find_peaks(bn_1)
                    
                    #makes sure we are analyzing the same case
                    if (ID.split('_')[0]== ID_1.split('_')[0]) and (ID.split('_')[1]== ID_1.split('_')[1]) and (ID.split('_')[2]== ID_1.split('_')[2]):
                        if len(peaks)>0 and len(peaks_1)>0:
                            ##print("Processing, ", ID)
                            #check if its foot or hand
                            ForH = ID.split('_')[2]
                            ##print("ForH = ", ForH)
                            take = ID.split('_')[1]
                            ##print("take = ", take)
                            
                            #Feature Extraction
                            dialen = peaks[0]
                            syslen = len(bn) - peaks[0]
                            totlen = len(bn)
                            diasis = syslen/dialen
                            amp = bn[peaks[0]]
                            peakamt = len(peaks)
                            hlf_sys = int(syslen/2)
                            hlf_dia = int(peaks[0]/2)
                            hlf_sval = bn[(peaks[0]+hlf_sys)]
                            onset_rat = (bn[0]+bn[len(bn)-1])/bn[peaks[0]]
                            sys_sl = (bn[peaks[0]]-bn[hlf_sys])/(syslen/2)
                            dias_sl = (bn[peaks[0]]-bn[hlf_dia])/(dialen/2)

                            dialen_1 = peaks_1[0]
                            syslen_1 = len(bn_1) - peaks_1[0]
                            totlen_1 = len(bn_1)
                            diasis_1 = syslen_1/dialen_1
                            amp_1 = bn_1[peaks_1[0]]
                            peakamt_1 = len(peaks_1)

                            ampratio = amp/amp_1
                            diaratio = dialen/dialen_1
                            sysratio = syslen/syslen_1

                            avg_len = (len(bn)+len(bn_1))/2


                            dtwdist = dtw.distance(bn,bn_1)
                            per_point_dtw = dtwdist/avg_len

                            #Make dictionaries for each ground truth so you can analyze the values
                            dias_len[key].append(dialen)
                            dias_prev[key].append(diaratio)
                            sys_len[key].append(syslen)
                            sys_prev[key].append(sysratio)
                            tot_len[key].append(totlen)
                            dias_slope[key].append(dias_sl)
                            sys_slope[key].append(sys_sl)
                            dias_sys_ratio[key].append(diasis)
                            onset_ratio[key].append(onset_rat)
                            mid_sys_val[key].append(hlf_sval)
                            amplitude[key].append(amp)
                            amount_peaks[key].append(peakamt)
                            amplitude_prev_ratio[key].append(ampratio)
                            dtw_distance[key].append(dtwdist)
                            dtw_avg_distance[key].append(per_point_dtw)
                        
                            featuresDCT = {'ID': str(ID),
                                        'ForH': ForH,
                                        'take': take,
                                        'gt': gt,
                                        'Dias Len': dialen,
                                        'Sys Len': syslen,
                                        'Total Len': totlen,
                                        'Sys Ratio': sysratio,
                                        'Dias Ratio': diaratio,
                                            'Sys Slope': sys_sl,
                                        'Dias Slope': dias_sl,
                                        'Dias/Sys Ratio': diasis,
                                        'Onset-Amp Ratio': onset_rat,
                                        'Mid Sys Val': hlf_sval,
                                        'Amplitude': amp,
                                        'Amplitude Ratio': ampratio,
                                        'DTW Dist': dtwdist,
                                        'DTW Avg Dist': per_point_dtw}
                            if ft_counter == 0:
                                features = DataFrame(featuresDCT, index=[0], columns=['ID','ForH','take','gt', 'Dias Len', 'Sys Len', 'Total Len', 'Sys Ratio', 'Dias Ratio', 
                                                                                    'Sys Slope', 'Dias Slope', 'Dias/Sys Ratio', 'Onset-Amp Ratio', 'Mid Sys Val',
                                                                                    'Amplitude', 'Amplitude Ratio', 'DTW Dist', 'DTW Avg Dist'])
                                
                                ft_counter +=1
                            else:
                                features2 = DataFrame(featuresDCT, index=[0], columns=['ID','ForH','take','gt', 'Dias Len', 'Sys Len', 'Total Len', 'Sys Ratio', 'Dias Ratio', 
                                                                                    'Sys Slope', 'Dias Slope', 'Dias/Sys Ratio', 'Onset-Amp Ratio', 'Mid Sys Val',
                                                                                    'Amplitude', 'Amplitude Ratio', 'DTW Dist', 'DTW Avg Dist'])
                                
                                features = pd.concat([features,features2],ignore_index=True)
                            
                        
                    else:
                        print("New case: ", ID[:16])
                        counter = 0
        
        return features

    # %%
    def reconstruct_waveforms(self, beat_df,beat_sequences,awad_pred,reconstructed_beats):
        gt = []
        previous_pred = 1
        for case in beat_df.ID:
            df1 = awad_pred[awad_pred['ID'].str.fullmatch(case)].reset_index(drop=True)
            if len(df1) > 0:
                prediction = df1.awad_gt[0]
                gt.append(prediction)
                previous_pred = prediction
            else:
                gt.append(previous_pred)
        
        beat_df['awad_gt'] = gt
        
        beat_df = beat_df[beat_df.awad_gt != 1]

        for ID in beat_df.ID:
            reconstructed_beats[ID] = beat_sequences[ID]
        
        return reconstructed_beats

    def pleth_features(self,fp_hand,fp_foot,case,take,rfe_cols, reconstructed_beats):
        df_hand = self.change_df(fp_hand)
        df_foot = self.change_df(fp_foot)
        
        reconstructed_beats, rfe_cols = self.artifact_removed(df_hand,case,take,rfe_cols,True,reconstructed_beats)
        reconstructed_beats, rfe_cols = self.artifact_removed(df_foot,case,take,rfe_cols,False,reconstructed_beats)
        
        
        slope = self.grab_slope(reconstructed_beats)
        delay = self.grab_delay(reconstructed_beats)
        
        return slope, delay, rfe_cols,reconstructed_beats 

    def grab_slope(self, reconstructed_beats):
        slope = []
        for key in reconstructed_beats:
            beat = reconstructed_beats[key]
            seq_val = beat['value'].to_numpy()
            time_val = beat['time'].to_numpy()
            peak, _ = find_peaks(seq_val)
            if len(peak) == 1 and len(seq_val) > 23 and (seq_val[peak[0]] > seq_val[0]): ## add filter for beats too short or too long
                y_ax_diff = seq_val[peak[0]] - seq_val[0]
                time_diff = datetime.datetime.strptime(time_val[peak[0]], '%Y-%m-%d-%H-%M-%S.%f') - datetime.datetime.strptime(time_val[0], '%Y-%m-%d-%H-%M-%S.%f')
                time_diff = time_diff.total_seconds()
                slope_val = y_ax_diff / time_diff
                if slope_val > 0:
                    slope.append(slope_val)
        return slope

    def grab_delay(self, reconstructed_beats):
        pre_delay = []
        delay = []
        baseline_factor = 100
        #print('delay',reconstructed_beats)
        for key in reconstructed_beats:
            #print('delay key',key)
            #print(key.split('_')[2])
            if key.split('_')[2] == 'hand':
                hand_key = key
                hand_beat = reconstructed_beats[hand_key]
                hand_seq = hand_beat['value'].to_numpy()
                hand_time = hand_beat['time'].to_numpy()
                peak_hand, _ = find_peaks(hand_seq)
                #print("hand delay Seq val LEN: ", len(hand_seq))
                if len(peak_hand) == 1 and len(hand_seq) > 23:  ## add filter for beats too short or too long
                    for i in range(300):
                        hand_subtake = int(hand_key.split('_')[3])
                        if hand_subtake < baseline_factor:
                            factor = hand_subtake - 1
                        else:
                            factor = baseline_factor
                        foot_subtake = abs(hand_subtake - (factor) + i)
                        foot_key = hand_key.split('_')[0]+'_'+hand_key.split('_')[1]+'_'+'foot'+'_'+str(foot_subtake)
                        if foot_key in reconstructed_beats:
                            foot_beat = reconstructed_beats[foot_key]
                            foot_seq = foot_beat['value'].to_numpy()
                            foot_time = foot_beat['time'].to_numpy()
                            peak_foot, _ = find_peaks(foot_seq)
                            foot_start_time = datetime.datetime.strptime(foot_time[0], '%Y-%m-%d-%H-%M-%S.%f')
                            hand_start_time = datetime.datetime.strptime(hand_time[0], '%Y-%m-%d-%H-%M-%S.%f')
                            start_time_diff = foot_start_time - hand_start_time
                            #print("foot delay Seq val LEN: ", len(foot_seq))
                            if len(peak_foot) == 1 and len(foot_seq) > 23 and start_time_diff.total_seconds() < 5:
                                #print("!!!! close enough match to calculate delay")
                                hand_peak_time = datetime.datetime.strptime(hand_time[peak_hand[0]], '%Y-%m-%d-%H-%M-%S.%f')
                                foot_peak_time = datetime.datetime.strptime(foot_time[peak_foot[0]], '%Y-%m-%d-%H-%M-%S.%f')
                                if foot_peak_time > hand_peak_time:
                                    time_diff = foot_peak_time - hand_peak_time
                                    time_diff = time_diff.total_seconds()
                                    #print("Time diff = ", time_diff)
                                    if time_diff < 2.4:
                                        pre_delay.append(time_diff)
                                        #print("CALCULATED CALCULATED CALCULATED. Delay calculated")
                    if len(pre_delay) != 0:
                        delay.append(min(pre_delay))
                        #print("!!!!!!!Appended!!!!!!!!")
        return delay

    def artifact_removed(self, seq, case, take,rfe_cols,isHand,reconstructed_beats):
    
        beat_sequences = {}
        key_count = 0
        seq_val = seq['value'].to_numpy()
        #seq_time = seq['time'].to_numpy()
        peaks, _ = find_peaks(seq_val)
        onset, _ = find_peaks(-seq_val)
        if isHand:
            ForH = 'hand'
        else:
            ForH = 'foot'
        for i in range(len(onset)):
            key_name = case + '_' + take + '_' + ForH + '_' + str(key_count)
            if i == 0:
                beat_sequences[key_name] = seq[0:onset[i]]
            else:
                beat_sequences[key_name] = seq[onset[i-1]:onset[i]]
            key_count = key_count + 1
            
        # we have beat dictionary here
        beat_df = self.mount_df(beat_sequences)
        
        awad_features = self.GrabInfo(beat_df)
        
        ## up to here it is looking fine both awad features and beat_df
        
        awad_pred, rfe_cols = self.ml_awad_model_test(case, ForH ,awad_features, printFull = False, col=rfe_cols, isStd = True)
        
        reconstructed_beats = self.reconstruct_waveforms(beat_df,beat_sequences,awad_pred,reconstructed_beats)

        return reconstructed_beats, rfe_cols

    def clean_procssed_folder(self):
        file_list = sorted(os.listdir(self.preprocessed_path_global))
        for case in file_list:
            if os.path.exists(os.path.join(self.tested_path, case)):
                shutil.rmtree(os.path.join(self.tested_path, case))
            shutil.move(os.path.join(self.preprocessed_path_global, case), os.path.join(self.tested_path, case))

    def ml_cchd_detection(self, data_path, printFull = False, col=0, isStd = True):

        predictProb = 0.5
        ##load old and new data    
        eval_data = pd.read_csv(data_path)
        eval_data = eval_data[-1:]
        eval_data.drop(columns='Unnamed: 0',inplace=True)
        
        cchd_stime = time.time()
        ##set columns to use, we remove columns like filename, Coarc, stuff that our model won't use
        columns_to_use_eval = list(set(eval_data.columns).difference(['Case', 'ID','take']))
        
        ##chose scaler
        if isStd:
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        
        X_test = eval_data[columns_to_use_eval]

        ## scale it, label
        X_test = scaler.fit_transform(X_test)

        columns_to_bool = self.CCHD_features_columes
        
        X_test = X_test[:,columns_to_bool]
    
        model = joblib.load(self.CCHD_model)

        #load that files and do preditice
        predictions = model.predict_proba(X_test)
        ##lines for Metrics, here there is a change in prediction
        y_pred = (model.predict_proba(X_test)[:,1] >= predictProb).astype(bool)
        
        for prediction in y_pred:
            if prediction:
                case_name = eval_data.ID.tail(1).tolist()[0]
                self.result_output['case_name'] = case_name
                self.result_output['result'] = 'CCHD'
                print("Case ", case_name, "is diagnosed as CCHD")
            else:
                case_name = eval_data.ID.tail(1).tolist()[0]
                self.result_output['case_name'] = case_name
                self.result_output['result'] = 'Healthy'
                print("Case ", case_name, "is diagnosed as healthy")
        cchd_etime = time.time()
        cchd_time = cchd_etime - cchd_stime
        self.result_output['cchd_running_time'] = cchd_time
        time2 = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        self.result_output['test_time'] = time2
        ##finalize function, return values of spec, sens and aucs for user to tune model accordingly
        return eval_data

    def ml_awad_model_test(self, case, ForH, data_test, printFull = False, col=0, isStd = True):

        awad_stime = time.time()
        X_test = data_test 
        Xtest_og = X_test

        if isStd:
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()

        X_test = X_test.drop(labels='ForH', axis=1)
        X_test = X_test.drop(labels='ID', axis=1)
        X_test = X_test.drop(labels='gt', axis=1)
        X_test = X_test.drop(labels='take', axis=1)

        X_test = scaler.fit_transform(X_test)

        ## select features
        columns_to_bool = self.AWAD_features_columes
        X_test = X_test[:,columns_to_bool]   ## print the columns that were used
        model = joblib.load(self.AWAD_model)
        ##lines for AUC, here there is no change of prediction probability
        awad_pred = model.predict_proba(X_test)
        
        predictions=[]
        for pred in awad_pred:
            if pred[0] > pred[1]:
                predictions.append(0)
            else:
                predictions.append(1)
        
        res = float(predictions.count(1)/len(predictions))
        awad_etime = time.time()
        awad_time = awad_etime - awad_stime
        
        if (ForH == 'hand'):
            self.result_output['hand_awad_ratio'] = res
            self.result_output['hand_awad_time'] = awad_time
        if (ForH == 'foot'):
            self.result_output['foot_awad_ratio'] = res
            self.result_output['foot_awad_time'] = awad_time

        Xtest_og['awad_gt'] = predictions    
        return Xtest_og, columns_to_bool