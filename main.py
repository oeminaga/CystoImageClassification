#%%
'''
Copyright by Okyaz Eminaga 2023 - MIT License
'''
import os
import GUI
import utils
import cv2
from registeries import model_registry, preprocess_registry
#%%
#Configure and retrieve information from the video source.
import argparse
par=argparse.ArgumentParser()
par.add_argument("--video_source", type=str, default="", help="Video source either a file name or a camera index (e.g., 0).")
par.add_argument("-c","--ask_case_id", action='store_true' )
args = par.parse_args()
if __name__=="__main__":
    try:
        video_source=args.video_source
        cap = cv2.VideoCapture(video_source)
        fps_=cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.release()
    except:
        print("Failed to open the video source. Existed!")
        exit()
    #Manage the storage.
    manager = utils.VideoStreamManagement()
    if type(video_source) is int or video_source.isdigit() or args.ask_case_id:
        print(video_source)
        manager.new()
    else:
        manager.case_id= os.path.basename(video_source)[:-4].split("_")[0]
        manager.case_type= os.path.basename(video_source)[:-4].split("_")[1]
        manager.DateTime = os.path.basename(video_source)[:-4].split("_")[2]
        manager.case_path= f"{manager.path_to_save}/{manager.case_id}_{manager.case_type}_{manager.DateTime}/"
        
        os.makedirs(manager.case_path, exist_ok=True)
        for itm in ["Images", "Videos", "AI"]:
            os.makedirs(f"{manager.case_path}/{itm}", exist_ok=True)
        manager.screen_shot_filename_template= f"{manager.case_path}/Images/{manager.case_id}_{manager.case_type}_{manager.DateTime}"
        manager.video_filename= f"{manager.case_path}/Videos/{manager.case_id}_{manager.case_type}_{manager.DateTime}"
        manager.result_filename= f"{manager.case_path}/AI/{manager.case_id}_{manager.case_type}_{manager.DateTime}"
    
    edges = (
        (0,1),
        (1,2),
        (2,3),
        (3,0),
        )
    #Generate function blocks-commands for the GUI.
    ModelForDetection = model_registry[manager.model_infos["name"]](manager.model_infos["path"],
                                                                    preprocess_registry[manager.model_infos["ModelPreprocessFunction"]],
                                                                    manager.model_infos,
                                                                    positive_label=manager.model_infos["positive_label"],
                                                                    negative_label=manager.model_infos["negative_label"],
                                                                    labels=manager.model_infos["labels"],
                                                                    threshold=manager.model_infos["threshold"])

    cmd_lst = [
        utils.ModePrediction(name="AI_Detection",filename=manager.result_filename,edges=edges,model=ModelForDetection,preprocess=preprocess_registry[manager.model_infos["PreprocessFunction"]], input_size=manager.model_infos["input_size"], color=None, thickness=None, TypeOfDetectionProblem=manager.model_infos["TypeOfDetectionProblem"], ShowOnlyPositiveAlert=manager.model_infos["ShowOnlyPositiveAlert"], PlaySound=manager.model_infos["PlaySound"])
        #utils.VideoRecoder(name="VideoRecoder", filename=manager.video_filename,shape=(width,height), fps=fps_)
    ]
    #Initiate GUI
    GUI_mmanager=GUI.WindowVisualization(
        buttons=[], Commands=cmd_lst, video_source=video_source)
    #Run
    GUI_mmanager.Run()