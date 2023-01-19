import sys, getopt
import os,json
import pandas as pd
import numpy as np
from deepface import DeepFace 

class ML_Module:
    def __init__(self, p1, p2):
        self.hyperparameter={'p1':p1, 'p2':p2}
        self.model_name = 'Facenet512'
        self.path_db_embedding='database/data_deepface.json'
    
    def load_model(self):
        print ("loading model")
        payload={'model':{'status':'loaded'}}
        return payload
    def predict(self, ifile):
        print ('predicting...')
        print (f'ifile: {ifile}')

        with open(self.path_db_embedding, 'r') as f:
            dic_pre_computed_embedings=json.load(f)

        embeddings = DeepFace.represent(img_path = ifile, model_name = self.model_name,enforce_detection=False)
        df_score = pd.DataFrame(columns=['name','score'])
        for name in dic_pre_computed_embedings:
            aa=dic_pre_computed_embedings[name]
            for inst in range (len(aa)):
                pre_computed=aa[inst][1]
                score=DeepFace.dst.findCosineDistance(embeddings, pre_computed)
                df_score_row = pd.DataFrame([[name,score]], columns=['name','score'])
                df_score=pd.concat([ df_score,df_score_row])
                df_final=df_score.sort_values('score').head(3)
        #print (df_final) 
        #payload=df_final.reset_index().to_json()
        response_dic=df_final.to_dict('list')        
        payload=response_dic
        return payload
    def train(self, ifile):
        print ('training...')
        print (f'ifile: {ifile}')
        payload={'trained':{'acc':0.9, 'loss':0.001}}
        return payload

    def generate_embedding2(self, ipath):
        if not os.path.isdir(ipath):
            ipath=os.path.dirname(ipath)

        path_dir=ipath
        print (f'generating embeddings for {path_dir}...')
        label=os.path.basename(path_dir)
        p_dic={}
        listFiles= [ f for f in os.listdir(path_dir) if f if not f.startswith('.') ]
        #embeddings = DeepFace.represent(img_path = ifile, model_name = self.model_name,enforce_detection=False)
        encodedList=[]
        for f in listFiles:
            path_image=os.path.join(path_dir,f)
            embeddings=f"embedding of {path_image}"
            d={label:{f:embeddings}}
            encodedList.append([d]) 
        print (encodedList)
        with open(self.path_db_embedding, 'w') as f:
            json.dump(encodedList, f)
        
    def generate_embedding(self,ipath):

        path_dir=ipath if os.path.isdir(ipath) else os.path.dirname(ipath)
        listDir= [ d for d in os.listdir(path_dir) if d if not d.startswith('.') ]

        dic_item={}
        for d in listDir:
            path_sub_dir=os.path.join (path_dir,d)
            dic_item[d]=  os.listdir(path_sub_dir)
            listLabel=[]
            for f in dic_item[d]:
                listLabel.append(os.path.join(path_sub_dir,f))
            dic_item[d]=listLabel

        p_dic=dic_item
        for d in p_dic:
            files=p_dic[d]
            encodedList=[]
            for f in files:
                if not os.path.basename(f).startswith('.'):
                    embeddings = DeepFace.represent(img_path = f, model_name = self.model_name,enforce_detection=False)
                    encodedList.append ([f,embeddings])
            p_dic[d]=encodedList

        with open(self.path_db_embedding, 'w') as f:
            json.dump(p_dic, f)


def main(argv):
   opts, args = getopt.getopt(argv,"m:i:",["mode=","ifile="])
   for opt, arg in opts:
      if opt in ("-i", "--ifile"):
         ifile = arg
      if opt in ("-m", "--mode"):
         mode = arg
   print (f'Mode:{mode}, ifile:{ifile}')
   if mode=='train':
         ml=ML_Module(p1=1,p2=2)
         ml.train(ifile)
   elif mode=='predict':
         ml=ML_Module(p1=1,p2=2)
         ml.load_model()
         ml.predict(ifile)
         print ('predicted')
   elif mode=='embedding':
         ml=ML_Module(p1=1,p2=2)
         #ml.load_model()
         payload=ml.generate_embedding(ifile)
         
if __name__ == "__main__":
    main(sys.argv[1:])